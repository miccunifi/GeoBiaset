from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from models.base import BaseGeoModel, ModelOutput, ModelUnavailableError, candidate_path_keys, describe_runtime_device, resolve_runtime_device
from models.img2loc import find_local_clip_snapshot_dir, read_csv_rows


DEFAULT_G3_ROOT = Path("/home/nniccoli/G3")
GROUND_TRUTH_AUGMENTATION = "ground truth"


def resolve_g3_paths(args: Any) -> tuple[Path, Path, Path, Path]:
    root = args.g3_root.expanduser().resolve()
    checkpoint_path = (args.g3_checkpoint or root / "checkpoints" / "g3.pth").expanduser().resolve()
    index_path = (args.g3_index or root / "index" / "g3.index").expanduser().resolve()
    default_database = root / "data" / "MP16_Pro_filtered.csv"
    if not default_database.exists() and (root / "MP16_Pro_filtered.csv").exists():
        default_database = root / "MP16_Pro_filtered.csv"
    database_csv = (args.g3_database_csv or default_database).expanduser().resolve()
    results_csv = (args.g3_results_csv or root / "data" / "im2gps3k_g3_results.csv").expanduser().resolve()
    return checkpoint_path, index_path, database_csv, results_csv


def load_database_coords(database_csv: Path) -> list[tuple[float, float]]:
    if not database_csv.exists():
        raise ModelUnavailableError(f"G3 database CSV does not exist: {database_csv}")
    rows = read_csv_rows(database_csv)
    if not rows:
        raise ValueError(f"Database CSV is empty: {database_csv}")

    lat_col = "LAT" if "LAT" in rows[0] else "gt_lat"
    lon_col = "LON" if "LON" in rows[0] else "gt_lon"
    return [(float(row[lat_col]), float(row[lon_col])) for row in rows]


def load_g3_precomputed_predictions(results_csv: Path) -> dict[str, tuple[float, float, float | None]]:
    if not results_csv.exists():
        return {}

    lookup: dict[str, tuple[float, float, float | None]] = {}
    for row in read_csv_rows(results_csv):
        image_id = row.get("IMG_ID") or row.get("image_path") or ""
        if not image_id:
            continue
        lat = float(row["LAT_pred"])
        lon = float(row["LON_pred"])
        for key in candidate_path_keys(image_id):
            lookup[key] = (lat, lon, None)
    return lookup


class G3Model(BaseGeoModel):
    model_name = "G3"

    def __init__(
        self,
        root: str | Path = DEFAULT_G3_ROOT,
        checkpoint_path: str | Path | None = None,
        index_path: str | Path | None = None,
        database_csv: str | Path | None = None,
        results_csv: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else self.root / "checkpoints" / "g3.pth"
        )
        self.index_path = Path(index_path).expanduser().resolve() if index_path else self.root / "index" / "g3.index"
        default_database = self.root / "data" / "MP16_Pro_filtered.csv"
        if not default_database.exists() and (self.root / "MP16_Pro_filtered.csv").exists():
            default_database = self.root / "MP16_Pro_filtered.csv"
        self.database_csv = Path(database_csv).expanduser().resolve() if database_csv else default_database
        self.results_csv = (
            Path(results_csv).expanduser().resolve()
            if results_csv
            else self.root / "data" / "im2gps3k_g3_results.csv"
        )
        self.device = resolve_runtime_device(device)
        self.precomputed_lookup = load_g3_precomputed_predictions(self.results_csv)
        self._model: Any | None = None
        self._index: Any | None = None
        self._database_coords: list[tuple[float, float]] | None = None
        self._torch: Any | None = None

    @classmethod
    def from_args(cls, args: Any) -> "G3Model":
        checkpoint_path, index_path, database_csv, results_csv = resolve_g3_paths(args)
        return cls(
            root=args.g3_root,
            checkpoint_path=checkpoint_path,
            index_path=index_path,
            database_csv=database_csv,
            results_csv=results_csv,
            device=args.device,
        )

    def _load_live_components(self) -> tuple[Any, Any, list[tuple[float, float]], Any]:
        if self._model is None:
            try:
                import faiss
                import torch
                import torch.nn as nn
                from pyproj import Proj, Transformer
                from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
            except ModuleNotFoundError as exc:
                raise ModelUnavailableError("G3 requires faiss, torch, pyproj, and transformers.") from exc

            if not self.checkpoint_path.exists():
                raise ModelUnavailableError(f"G3 checkpoint does not exist: {self.checkpoint_path}")
            if not self.index_path.exists():
                raise ModelUnavailableError(f"G3 index does not exist: {self.index_path}")

            class GaussianEncoding(nn.Module):
                def __init__(self, sigma: float, input_size: int, encoded_size: int) -> None:
                    super().__init__()
                    b = torch.randn((encoded_size, input_size)) * sigma
                    self.b = nn.Parameter(b, requires_grad=False)

                def forward(self, values: Any) -> Any:
                    vp = 2 * math.pi * values @ self.b.T
                    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

            class LocationEncoderCapsule(nn.Module):
                def __init__(self, sigma: float) -> None:
                    super().__init__()
                    rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
                    self.capsule = nn.Sequential(
                        rff_encoding,
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                    )
                    self.head = nn.Sequential(nn.Linear(1024, 512))

                def forward(self, values: Any) -> Any:
                    return self.head(self.capsule(values))

            class CustomLocationEncoder(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.sigmas = [2**0, 2**4, 2**8]
                    for index_sigma, sigma in enumerate(self.sigmas):
                        self.add_module(f"LocEnc{index_sigma}", LocationEncoderCapsule(sigma=sigma))

                    proj_wgs84 = Proj("epsg:4326")
                    proj_mercator = Proj("epsg:3857")
                    self.transformer = Transformer.from_proj(proj_wgs84, proj_mercator, always_xy=True)

                def forward(self, values: Any) -> Any:
                    lat = values[:, 0].float().detach().cpu().numpy()
                    lon = values[:, 1].float().detach().cpu().numpy()
                    projected_lon_lat = self.transformer.transform(lon, lat)
                    location = [[coord[1], coord[0]] for coord in zip(*projected_lon_lat)]
                    location_tensor = torch.tensor(location, device=values.device, dtype=torch.float32)
                    location_tensor = location_tensor / 20037508.3427892

                    features = torch.zeros(location_tensor.shape[0], 512, device=values.device)
                    for index_sigma in range(len(self.sigmas)):
                        features = features + self._modules[f"LocEnc{index_sigma}"](location_tensor)
                    return features

            class LocalG3(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    clip_snapshot_dir = find_local_clip_snapshot_dir()
                    clip_source = str(clip_snapshot_dir) if clip_snapshot_dir is not None else "openai/clip-vit-large-patch14"
                    local_only = clip_snapshot_dir is not None

                    clip_model = CLIPModel.from_pretrained(clip_source, local_files_only=local_only)
                    self.vision_model = clip_model.vision_model
                    self.text_model = clip_model.text_model
                    self.vision_processor = CLIPImageProcessor.from_pretrained(clip_source, local_files_only=local_only)
                    self.text_processor = CLIPTokenizer.from_pretrained(clip_source, local_files_only=local_only)
                    self.vision_projection = clip_model.visual_projection
                    self.text_projection = clip_model.text_projection

                    self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
                    self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
                    self.logit_scale3 = nn.Parameter(torch.tensor(3.99))

                    self.location_encoder = CustomLocationEncoder()
                    self.vision_projection_else_1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
                    self.text_projection_else = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
                    self.vision_projection_else_2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
                    self.location_projection_else = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 768))

                    self.vision_model.requires_grad_(False)
                    self.vision_projection.requires_grad_(False)
                    self.text_model.requires_grad_(False)
                    self.text_projection.requires_grad_(False)

            model = LocalG3().to(self.device)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.requires_grad_(False)
            model.eval()
            print(f"{self.model_name} loaded on {describe_runtime_device(self.device, torch)}")
            self._model = model
            self._index = faiss.read_index(str(self.index_path))
            self._database_coords = load_database_coords(self.database_csv)
            self._torch = torch

        return self._model, self._index, self._database_coords or [], self._torch

    def _precomputed_prediction(self, source_path: str | Path | None, augmentation: str | None) -> tuple[float, float, float | None] | None:
        if not source_path or augmentation != GROUND_TRUTH_AUGMENTATION:
            return None
        for key in candidate_path_keys(source_path):
            cached = self.precomputed_lookup.get(key)
            if cached is not None:
                return cached
        return None

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        cached = self._precomputed_prediction(kwargs.get("source_path"), kwargs.get("augmentation"))
        if cached is not None:
            pred_lat, pred_lon, confidence = cached
            return ModelOutput(
                model_name=self.model_name,
                image_path=str(image_path),
                pred_lat=pred_lat,
                pred_lon=pred_lon,
                confidence=confidence,
                metadata=self._metadata(
                    dataset_path=kwargs.get("dataset_path"),
                    source_path=kwargs.get("source_path"),
                    augmentation=kwargs.get("augmentation"),
                    source="precomputed",
                ),
            )

        from PIL import Image

        model, index, coords, torch = self._load_live_components()
        image = Image.open(image_path).convert("RGB")
        pixel_values = model.vision_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            vision_output = model.vision_model(pixel_values)[1]
            image_embeds = model.vision_projection(vision_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
            image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

            image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
            image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

            query = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)

        scores, indices = index.search(query.cpu().detach().numpy().astype("float32"), 1)
        pred_lat, pred_lon = coords[int(indices[0][0])]
        return ModelOutput(
            model_name=self.model_name,
            image_path=str(image_path),
            pred_lat=pred_lat,
            pred_lon=pred_lon,
            confidence=float(scores[0][0]),
            metadata=self._metadata(
                dataset_path=kwargs.get("dataset_path"),
                source_path=kwargs.get("source_path"),
                augmentation=kwargs.get("augmentation"),
                source="live",
            ),
        )
