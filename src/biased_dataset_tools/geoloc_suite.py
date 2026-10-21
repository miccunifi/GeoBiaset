from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from biased_dataset_tools.clip_attention import (
    CLIP_ATTENTION_METHOD_NAME,
    CLIP_ATTENTION_METHOD_REFERENCE,
    load_attention_explainer as load_clip_attention_explainer,
    save_attention_images as save_clip_attention_images,
)
from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION, load_config, resolve_dataset_path
from biased_dataset_tools.geoloc_methods import (
    build_img2loc_search_index,
    find_local_clip_snapshot_dir,
    load_geoclip_model,
    resolve_g3_paths,
    resolve_img2loc_paths,
)
from biased_dataset_tools.geoloc_shared import (
    DEFAULT_G3_ROOT,
    DEFAULT_IMG2LOC_EMBEDDINGS,
    DEFAULT_IMG2LOC_ROOT,
    EVAL_METHOD_CHOICES,
    METHOD_DISPLAY_NAMES,
    candidate_path_keys,
    method_slug,
    read_csv_rows,
    resolve_runtime_device,
)
from models.base import DEFAULT_VLM_MAX_NEW_TOKENS
from models.img2loc import IMG2LOC_CLIP_DIMENSIONS, IMG2LOC_COORD_DIMENSIONS


DEFAULT_SUITE_METHODS = [
    "geoclip",
    "img2loc",
    "g3",
    "qwen2-vl-7b",
]
SUITE_METHOD_CHOICES = ["all", *[method for method in EVAL_METHOD_CHOICES if method != "qwen-custom"]]
@dataclass(frozen=True)
class SuiteMethodSpec:
    cli_method: str
    display_name: str
    slug: str
    vlm_model_id: str | None = None

    @property
    def supports_attention(self) -> bool:
        return self.cli_method in {"geoclip", "img2loc", "g3"}


@dataclass(frozen=True)
class MethodRunPaths:
    method_dir: Path
    image_metrics_csv: Path
    overall_metrics_csv: Path
    prediction_map_html: Path
    attention_dir: Path
    attention_index_csv: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tidy multi-method geolocation evaluation suite over a dataset manifest. "
            "Each method gets its own metrics CSVs, map HTML, log file, and optional attention maps."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV override. Defaults to the manifest in config.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional suite output root. Defaults to "
            "<dataset_root>/geoloc_suite/<manifest_stem>/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        choices=SUITE_METHOD_CHOICES,
        help=(
            "Methods to run. Defaults to a practical built-in suite: "
            f"{', '.join(DEFAULT_SUITE_METHODS)}. Use 'all' to expand every built-in method choice."
        ),
    )
    parser.add_argument(
        "--extra-vlm-model-id",
        action="append",
        default=[],
        help=(
            "Optional additional Qwen/VLM model id to run through --method qwen-custom. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Use 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--countries-data",
        type=Path,
        default=None,
        help="Optional local country boundary file forwarded to the per-method evaluator.",
    )
    parser.add_argument(
        "--country-name-column",
        default=None,
        help="Optional country-name column forwarded to the per-method evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit forwarded to the per-method evaluator.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=DEFAULT_VLM_MAX_NEW_TOKENS,
        help="Maximum new tokens forwarded to VLM geolocation methods.",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip attention-map generation.",
    )
    parser.add_argument(
        "--attention-limit",
        type=int,
        default=None,
        help="Optional number of evaluated rows to explain per supported method.",
    )
    parser.add_argument(
        "--img2loc-root",
        type=Path,
        default=DEFAULT_IMG2LOC_ROOT,
        help="Path to the local Img2Loc repository.",
    )
    parser.add_argument(
        "--img2loc-index",
        type=Path,
        default=None,
        help="Optional Img2Loc FAISS index override.",
    )
    parser.add_argument(
        "--img2loc-embeddings-with-coords",
        type=Path,
        default=DEFAULT_IMG2LOC_EMBEDDINGS,
        help="Optional Img2Loc embeddings-with-coordinates override.",
    )
    parser.add_argument(
        "--img2loc-search-results-csv",
        type=Path,
        default=None,
        help="Optional Img2Loc cached search-results CSV override.",
    )
    parser.add_argument(
        "--g3-root",
        type=Path,
        default=DEFAULT_G3_ROOT,
        help="Path to the local G3 repository.",
    )
    parser.add_argument(
        "--g3-index",
        type=Path,
        default=None,
        help="Optional G3 FAISS index override.",
    )
    parser.add_argument(
        "--g3-checkpoint",
        type=Path,
        default=None,
        help="Optional G3 checkpoint override.",
    )
    parser.add_argument(
        "--g3-database-csv",
        type=Path,
        default=None,
        help="Optional G3 retrieval database CSV override.",
    )
    parser.add_argument(
        "--g3-results-csv",
        type=Path,
        default=None,
        help="Optional G3 cached results CSV override.",
    )
    return parser.parse_args()


def expand_requested_methods(methods: list[str] | None) -> list[str]:
    if not methods:
        return list(DEFAULT_SUITE_METHODS)
    if "all" in methods:
        return [method for method in EVAL_METHOD_CHOICES if method != "qwen-custom"]
    return list(methods)


def build_method_specs(args: argparse.Namespace) -> list[SuiteMethodSpec]:
    requested_methods = expand_requested_methods(args.methods)
    specs: list[SuiteMethodSpec] = []

    for method in requested_methods:
        if method == "qwen-custom":
            continue
        specs.append(
            SuiteMethodSpec(
                cli_method=method,
                display_name=METHOD_DISPLAY_NAMES.get(method, method),
                slug=method_slug(method),
            )
        )

    for model_id in args.extra_vlm_model_id:
        display_name = model_id.split("/")[-1]
        specs.append(
            SuiteMethodSpec(
                cli_method="qwen-custom",
                display_name=display_name,
                slug=method_slug(display_name),
                vlm_model_id=model_id,
            )
        )

    return specs


def default_output_root(dataset_root: Path, manifest_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dataset_root / "geoloc_suite" / manifest_path.stem / timestamp


def build_method_paths(output_root: Path, method_spec: SuiteMethodSpec) -> MethodRunPaths:
    method_dir = output_root / method_spec.slug
    return MethodRunPaths(
        method_dir=method_dir,
        image_metrics_csv=method_dir / "image_metrics.csv",
        overall_metrics_csv=method_dir / "overall_metrics.csv",
        prediction_map_html=method_dir / "prediction_map.html",
        attention_dir=method_dir / "attention_maps",
        attention_index_csv=method_dir / "attention_index.csv",
        log_path=method_dir / "run.log",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_method_eval(
    args: argparse.Namespace,
    manifest_path: Path,
    method_spec: SuiteMethodSpec,
    paths: MethodRunPaths,
) -> tuple[bool, str]:
    paths.method_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "biased_dataset_tools.geoloc_eval",
        "--config",
        str(args.config.expanduser().resolve()),
        "--manifest",
        str(manifest_path),
        "--method",
        method_spec.cli_method,
        "--device",
        args.device,
        "--image-metrics-csv",
        str(paths.image_metrics_csv),
        "--overall-metrics-csv",
        str(paths.overall_metrics_csv),
        "--map-html",
        str(paths.prediction_map_html),
    ]
    if args.countries_data is not None:
        command.extend(["--countries-data", str(args.countries_data.expanduser().resolve())])
    if args.country_name_column:
        command.extend(["--country-name-column", args.country_name_column])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    command.extend(["--vlm-max-new-tokens", str(args.vlm_max_new_tokens)])
    if method_spec.vlm_model_id is not None:
        command.extend(["--vlm-model-id", method_spec.vlm_model_id])
    if args.img2loc_root is not None:
        command.extend(["--img2loc-root", str(args.img2loc_root.expanduser().resolve())])
    if args.img2loc_index is not None:
        command.extend(["--img2loc-index", str(args.img2loc_index.expanduser().resolve())])
    if args.img2loc_embeddings_with_coords is not None:
        command.extend(
            [
                "--img2loc-embeddings-with-coords",
                str(args.img2loc_embeddings_with_coords.expanduser().resolve()),
            ]
        )
    if args.img2loc_search_results_csv is not None:
        command.extend(
            ["--img2loc-search-results-csv", str(args.img2loc_search_results_csv.expanduser().resolve())]
        )
    if args.g3_root is not None:
        command.extend(["--g3-root", str(args.g3_root.expanduser().resolve())])
    if args.g3_index is not None:
        command.extend(["--g3-index", str(args.g3_index.expanduser().resolve())])
    if args.g3_checkpoint is not None:
        command.extend(["--g3-checkpoint", str(args.g3_checkpoint.expanduser().resolve())])
    if args.g3_database_csv is not None:
        command.extend(["--g3-database-csv", str(args.g3_database_csv.expanduser().resolve())])
    if args.g3_results_csv is not None:
        command.extend(["--g3-results-csv", str(args.g3_results_csv.expanduser().resolve())])

    captured_lines: list[str] = []
    with paths.log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n\n")
        log_handle.write("[combined_output]\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd(),
            bufsize=1,
        )
        assert process.stdout is not None

        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            captured_lines.append(line.rstrip())

        return_code = process.wait()
        log_handle.write(f"\nexit_code={return_code}\n")
        log_handle.flush()

    if return_code == 0:
        return True, ""

    error_message = next(
        (line for line in reversed(captured_lines) if line.strip()),
        f"Method failed with exit code {return_code}",
    )
    return False, error_message.splitlines()[-1]


def build_suite_metadata(
    args: argparse.Namespace,
    dataset_root: Path,
    manifest_path: Path,
    output_root: Path,
    method_specs: list[SuiteMethodSpec],
) -> dict[str, Any]:
    return {
        "config_path": str(args.config.expanduser().resolve()),
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "requested_methods": [spec.cli_method for spec in method_specs],
        "extra_vlm_model_ids": list(args.extra_vlm_model_id),
        "device": args.device,
        "attention_enabled": not args.skip_attention,
        "attention_limit": args.attention_limit,
        "attention_method": CLIP_ATTENTION_METHOD_NAME,
        "attention_method_reference": CLIP_ATTENTION_METHOD_REFERENCE,
        "countries_data": str(args.countries_data.expanduser().resolve()) if args.countries_data else "",
        "country_name_column": args.country_name_column or "",
        "row_limit": args.limit,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_rgb_image(image_path: Path) -> tuple[Any, np.ndarray]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    return image, image_rgb


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0.0)
    max_value = float(heatmap.max()) if heatmap.size else 0.0
    min_value = float(heatmap.min()) if heatmap.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32)


def build_overlay(image_rgb: np.ndarray, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, colored_rgb, 0.45, 0.0)
    grayscale_rgb = np.stack([heatmap_uint8, heatmap_uint8, heatmap_uint8], axis=-1)
    return overlay, grayscale_rgb


def save_attention_images(
    attention_dir: Path,
    dataset_path: str,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[Path, Path]:
    relative_path = Path(dataset_path)
    base_path = attention_dir / relative_path.parent / relative_path.stem
    base_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_rgb, grayscale_rgb = build_overlay(image_rgb, heatmap)
    overlay_path = base_path.with_name(f"{base_path.name}_overlay.jpg")
    heatmap_path = base_path.with_name(f"{base_path.name}_heatmap.png")

    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(grayscale_rgb, cv2.COLOR_RGB2BGR))
    return overlay_path.resolve(), heatmap_path.resolve()


def attention_rollout_from_gradients(attentions: tuple[Any, ...]) -> np.ndarray:
    if not attentions:
        raise ValueError("No attention tensors were returned by the vision model.")

    num_tokens = attentions[0].shape[-1]
    rollout = None
    eye = None

    for attention in attentions:
        if attention.grad is None:
            raise ValueError("Attention gradients were not retained for rollout generation.")

        layer_attention = (attention * attention.grad).clamp(min=0.0).mean(dim=1)
        if eye is None:
            eye = torch.eye(num_tokens, device=layer_attention.device, dtype=layer_attention.dtype).unsqueeze(0)
        layer_attention = layer_attention + eye
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = layer_attention if rollout is None else layer_attention @ rollout

    cls_to_patches = rollout[0, 0, 1:]
    side = int(math.sqrt(int(cls_to_patches.numel())))
    if side * side != int(cls_to_patches.numel()):
        raise ValueError(f"Could not reshape attention rollout into a square grid: {cls_to_patches.numel()} tokens")
    return cls_to_patches.reshape(side, side).detach().cpu().numpy()


def resize_heatmap(heatmap: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    resized = cv2.resize(heatmap.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return normalize_heatmap(resized)


def load_eager_clip_components(device: str) -> tuple[Any, Any]:
    from transformers import CLIPImageProcessor, CLIPModel

    clip_snapshot_dir = find_local_clip_snapshot_dir()
    clip_source = str(clip_snapshot_dir) if clip_snapshot_dir is not None else "openai/clip-vit-large-patch14"
    local_only = clip_snapshot_dir is not None

    model = CLIPModel.from_pretrained(
        clip_source,
        local_files_only=local_only,
        attn_implementation="eager",
    ).to(device)
    processor = CLIPImageProcessor.from_pretrained(clip_source, local_files_only=local_only)
    model.eval()
    return model, processor


class GeoCLIPAttentionExplainer:
    def __init__(self, device: str) -> None:
        import torch
        import torch.nn.functional as f

        self.torch = torch
        self.f = f
        self.device = device
        self.geoclip_model = load_geoclip_model().to(device)
        self.geoclip_model.eval()
        self.clip_model, self.processor = load_eager_clip_components(device)

    def explain(self, image_path: Path, metric_row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        image, image_rgb = load_rgb_image(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

        self.geoclip_model.zero_grad(set_to_none=True)
        self.clip_model.zero_grad(set_to_none=True)

        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, output_attentions=True)
        for attention in vision_outputs.attentions:
            attention.retain_grad()

        pooled_output = vision_outputs.pooler_output
        clip_features = self.clip_model.visual_projection(pooled_output)
        image_features = self.geoclip_model.image_encoder.mlp(clip_features)
        image_features = self.f.normalize(image_features, dim=-1)

        predicted_location = self.torch.tensor(
            [[float(metric_row["pred_lat"]), float(metric_row["pred_lon"])]],
            device=self.device,
            dtype=image_features.dtype,
        )
        location_features = self.geoclip_model.location_encoder(predicted_location)
        location_features = self.f.normalize(location_features, dim=-1)

        score = (
            self.geoclip_model.logit_scale.exp()
            * (image_features * location_features).sum(dim=-1)
        ).sum()
        score.backward()

        heatmap = attention_rollout_from_gradients(vision_outputs.attentions)
        return image_rgb, resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1])


class Img2LocAttentionExplainer:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        import faiss

        self.device = device
        self.clip_model, self.processor = load_eager_clip_components(device)
        index_path, embeddings_path, search_results_csv = resolve_img2loc_paths(args)
        self.index_path = index_path
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.index = faiss.read_index(str(index_path))
        self.index_requires_rebuild = self.index.d != IMG2LOC_CLIP_DIMENSIONS
        self.cached_top_indices = self._load_cached_top_indices(search_results_csv)

    def _load_cached_top_indices(self, search_results_csv: Path) -> dict[str, int]:
        if not search_results_csv.exists():
            return {}

        lookup: dict[str, int] = {}
        for row in read_csv_rows(search_results_csv):
            if row.get("match_type") != "top" or str(row.get("rank")) != "0":
                continue
            for key in candidate_path_keys(row["image_path"]):
                lookup[key] = int(row["index_id"])
        return lookup

    def _ensure_search_index(self) -> Any:
        if self.index_requires_rebuild:
            print(
                "Img2Loc attention: rebuilding a compatible "
                f"{IMG2LOC_CLIP_DIMENSIONS}-d flat search index from the first "
                f"{IMG2LOC_CLIP_DIMENSIONS} embedding dimensions. The final "
                f"{IMG2LOC_COORD_DIMENSIONS} stored dimensions are coordinates."
            )
            self.index = build_img2loc_search_index(self.embeddings)
            self.index_requires_rebuild = False
        return self.index

    def _top_vector(self, image_embeds: Any, metric_row: dict[str, str]) -> Any:
        top_index: int | None = None
        if metric_row["augmentation"] == GROUND_TRUTH_AUGMENTATION:
            for key in candidate_path_keys(metric_row["source_path"]):
                cached_index = self.cached_top_indices.get(key)
                if cached_index is not None:
                    top_index = cached_index
                    break

        if top_index is None:
            query = image_embeds.detach().cpu().numpy().astype("float32")
            index = self._ensure_search_index()
            _, indices = index.search(query, 1)
            top_index = int(indices[0][0])

        if self.embeddings.ndim == 3:
            vector = self.embeddings[top_index, 0, :IMG2LOC_CLIP_DIMENSIONS]
        else:
            vector = self.embeddings[top_index, :IMG2LOC_CLIP_DIMENSIONS]
        return image_embeds.new_tensor(np.array(vector, dtype=np.float32))

    def explain(self, image_path: Path, metric_row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        image, image_rgb = load_rgb_image(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

        self.clip_model.zero_grad(set_to_none=True)
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, output_attentions=True)
        for attention in vision_outputs.attentions:
            attention.retain_grad()

        pooled_output = vision_outputs.pooler_output
        image_embeds = self.clip_model.visual_projection(pooled_output)
        top_vector = self._top_vector(image_embeds, metric_row)
        score = (image_embeds[0] * top_vector).sum()
        score.backward()

        heatmap = attention_rollout_from_gradients(vision_outputs.attentions)
        return image_rgb, resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1])


def build_local_g3_model(checkpoint_path: Path, device: str) -> Any:
    import torch
    import torch.nn as nn
    from pyproj import Proj, Transformer
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

    clip_snapshot_dir = find_local_clip_snapshot_dir()
    clip_source = str(clip_snapshot_dir) if clip_snapshot_dir is not None else "openai/clip-vit-large-patch14"
    local_only = clip_snapshot_dir is not None

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
            clip_model = CLIPModel.from_pretrained(
                clip_source,
                local_files_only=local_only,
                attn_implementation="eager",
            )
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

    model = LocalG3().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class G3AttentionExplainer:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        checkpoint_path, _, _, _ = resolve_g3_paths(args)
        self.device = device
        self.model = build_local_g3_model(checkpoint_path, device)
        self.processor = self.model.vision_processor

    def explain(self, image_path: Path, metric_row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        image, image_rgb = load_rgb_image(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

        self.model.zero_grad(set_to_none=True)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values, output_attentions=True)
        for attention in vision_outputs.attentions:
            attention.retain_grad()

        pooled_output = vision_outputs.pooler_output
        base_projection = self.model.vision_projection(pooled_output)

        image_location_embeds = self.model.vision_projection_else_2(base_projection)
        image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

        predicted_location = torch.tensor(
            [[float(metric_row["pred_lat"]), float(metric_row["pred_lon"])]],
            device=self.device,
            dtype=base_projection.dtype,
        )
        location_features = self.model.location_encoder(predicted_location)
        location_features = self.model.location_projection_else(location_features)
        location_features = location_features / location_features.norm(p=2, dim=-1, keepdim=True)

        score = (image_location_embeds[0] * location_features[0]).sum()
        score.backward()

        heatmap = attention_rollout_from_gradients(vision_outputs.attentions)
        return image_rgb, resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1])


def load_attention_explainer(method_spec: SuiteMethodSpec, args: argparse.Namespace, device: str) -> Any | None:
    if method_spec.cli_method == "geoclip":
        return GeoCLIPAttentionExplainer(device)
    if method_spec.cli_method == "img2loc":
        return Img2LocAttentionExplainer(args, device)
    if method_spec.cli_method == "g3":
        return G3AttentionExplainer(args, device)
    return None


def generate_attention_maps(
    args: argparse.Namespace,
    method_spec: SuiteMethodSpec,
    method_paths: MethodRunPaths,
    dataset_root: Path,
    device: str,
) -> tuple[str, str, int]:
    if not method_spec.supports_attention:
        return "unsupported", "attention maps are implemented only for CLIP-backed methods", 0
    if not method_paths.image_metrics_csv.exists():
        return "missing", "per-image metrics CSV is missing", 0

    rows = read_csv_rows(method_paths.image_metrics_csv)
    if args.attention_limit is not None:
        rows = rows[: args.attention_limit]
    if not rows:
        return "skipped", "no evaluated rows were available for attention-map generation", 0

    explainer = load_clip_attention_explainer(method_spec.cli_method, args, device)
    if explainer is None:
        return "unsupported", "no attention explainer is available for this method", 0

    attention_rows: list[dict[str, str]] = []
    method_paths.attention_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        image_path = resolve_dataset_path(row["dataset_path"], dataset_root)
        image_rgb, heatmap = explainer.explain(image_path, row)
        overlay_path, heatmap_path = save_clip_attention_images(
            method_paths.attention_dir,
            row["dataset_path"],
            image_rgb,
            heatmap,
        )
        attention_rows.append(
            {
                "dataset_path": row["dataset_path"],
                "source_path": row["source_path"],
                "category": row["category"],
                "augmentation": row["augmentation"],
                "pred_lat": row["pred_lat"],
                "pred_lon": row["pred_lon"],
                "attention_method": CLIP_ATTENTION_METHOD_NAME,
                "attention_reference": CLIP_ATTENTION_METHOD_REFERENCE,
                "overlay_path": str(overlay_path),
                "heatmap_path": str(heatmap_path),
            }
        )
        print(f"  [{index}/{len(rows)}] attention map for {row['dataset_path']}")

    write_csv(
        method_paths.attention_index_csv,
        [
            "dataset_path",
            "source_path",
            "category",
            "augmentation",
            "pred_lat",
            "pred_lon",
            "attention_method",
            "attention_reference",
            "overlay_path",
            "heatmap_path",
        ],
        attention_rows,
    )
    return "ok", "", len(attention_rows)


def main() -> int:
    args = parse_args()
    _, dataset_root, config_manifest_path = load_config(args.config)
    manifest_path = (args.manifest or config_manifest_path).expanduser().resolve()
    output_root = (args.output_root.expanduser().resolve() if args.output_root else default_output_root(dataset_root, manifest_path))
    output_root.mkdir(parents=True, exist_ok=True)

    method_specs = build_method_specs(args)
    if not method_specs:
        raise ValueError("No methods were selected for the suite run.")

    device = resolve_runtime_device(args.device)
    suite_metadata = build_suite_metadata(args, dataset_root, manifest_path, output_root, method_specs)
    write_json(output_root / "suite_metadata.json", suite_metadata)

    print(f"Manifest CSV: {manifest_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Suite output root: {output_root}")
    print(f"Execution device: {device}")
    print(f"Methods: {', '.join(spec.display_name for spec in method_specs)}")

    summary_rows: list[dict[str, str]] = []
    for method_spec in method_specs:
        method_paths = build_method_paths(output_root, method_spec)
        print(f"\nRunning {method_spec.display_name}...")

        success, error_message = run_method_eval(args, manifest_path, method_spec, method_paths)
        attention_status = "not_run"
        attention_note = ""
        attention_count = 0

        if success and not args.skip_attention:
            print(f"Generating attention maps for {method_spec.display_name}...")
            try:
                attention_status, attention_note, attention_count = generate_attention_maps(
                    args=args,
                    method_spec=method_spec,
                    method_paths=method_paths,
                    dataset_root=dataset_root,
                    device=device,
                )
            except Exception as exc:
                attention_status = "failed"
                attention_note = str(exc) or repr(exc)
        elif success:
            attention_status = "skipped"
            attention_note = "attention generation disabled by --skip-attention"

        summary_rows.append(
            {
                "method": method_spec.cli_method,
                "display_name": method_spec.display_name,
                "vlm_model_id": method_spec.vlm_model_id or "",
                "status": "ok" if success else "failed",
                "error_message": error_message,
                "attention_status": attention_status,
                "attention_note": attention_note,
                "attention_count": str(attention_count),
                "image_metrics_csv": str(method_paths.image_metrics_csv),
                "overall_metrics_csv": str(method_paths.overall_metrics_csv),
                "prediction_map_html": str(method_paths.prediction_map_html),
                "attention_index_csv": str(method_paths.attention_index_csv) if attention_count else "",
                "attention_dir": str(method_paths.attention_dir) if attention_count else "",
                "log_path": str(method_paths.log_path),
            }
        )

    write_csv(
        output_root / "suite_summary.csv",
        [
            "method",
            "display_name",
            "vlm_model_id",
            "status",
            "error_message",
            "attention_status",
            "attention_note",
            "attention_count",
            "image_metrics_csv",
            "overall_metrics_csv",
            "prediction_map_html",
            "attention_index_csv",
            "attention_dir",
            "log_path",
        ],
        summary_rows,
    )

    completed = sum(row["status"] == "ok" for row in summary_rows)
    print(f"\nFinished suite run: {completed}/{len(summary_rows)} methods completed successfully.")
    print(f"Suite summary CSV: {output_root / 'suite_summary.csv'}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
