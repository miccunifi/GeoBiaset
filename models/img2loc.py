from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from models.base import (
    BaseGeoModel,
    ModelOutput,
    ModelUnavailableError,
    candidate_path_keys,
    describe_runtime_device,
    find_local_hf_snapshot_dir,
    resolve_runtime_device,
)


DEFAULT_IMG2LOC_ROOT = Path("/home/nniccoli/Img2Loc")
DEFAULT_IMG2LOC_EMBEDDINGS = Path("/mnt/nniccoli/geoclipplus_data/embeddings/embeddings_clip-vit-large-patch14.npy")
GROUND_TRUTH_AUGMENTATION = "ground truth"
IMG2LOC_CLIP_DIMENSIONS = 768
IMG2LOC_COORD_DIMENSIONS = 2
IMG2LOC_SEARCH_SCRIPT = "search_index_with_images.py"
IMG2LOC_LLAVA_SCRIPT = "llava_geo_from_search_results.py"
IMG2LOC_DEGENERATE_COORD_EPSILON = 1e-8


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_local_clip_snapshot_dir() -> Path | None:
    return find_local_hf_snapshot_dir("openai/clip-vit-large-patch14")


def resolve_img2loc_paths(args: Any) -> tuple[Path, Path, Path]:
    root = args.img2loc_root.expanduser().resolve()
    index_path = (args.img2loc_index or root / "index" / "clip.index").expanduser().resolve()
    embeddings_path = args.img2loc_embeddings_with_coords.expanduser().resolve()
    search_results_path = (args.img2loc_search_results_csv or root / "index" / "search_results.csv").expanduser().resolve()
    return index_path, embeddings_path, search_results_path


def load_img2loc_precomputed_predictions(search_results_csv: Path) -> dict[str, tuple[float, float, float | None]]:
    if not search_results_csv.exists():
        return {}

    rows = read_csv_rows(search_results_csv)
    if not rows:
        return {}

    if {"image_path", "latitude", "longitude"}.issubset(rows[0]):
        return load_img2loc_llava_predictions(rows)

    return load_img2loc_search_predictions(rows)


def is_degenerate_img2loc_prediction(lat: float, lon: float) -> bool:
    return abs(lat) <= IMG2LOC_DEGENERATE_COORD_EPSILON and abs(lon) <= IMG2LOC_DEGENERATE_COORD_EPSILON


def load_img2loc_llava_predictions(rows: list[dict[str, str]]) -> dict[str, tuple[float, float, float | None]]:
    lookup: dict[str, tuple[float, float, float | None]] = {}
    for row in rows:
        if row.get("latitude") in {"", None} or row.get("longitude") in {"", None}:
            continue

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        if is_degenerate_img2loc_prediction(lat, lon):
            continue
        for key in candidate_path_keys(row["image_path"]):
            lookup[key] = (lat, lon, None)
    return lookup


def load_img2loc_search_predictions(rows: list[dict[str, str]]) -> dict[str, tuple[float, float, float | None]]:
    lookup: dict[str, tuple[float, float, float | None]] = {}
    for row in rows:
        if row.get("match_type") != "top" or str(row.get("rank")) != "0":
            continue

        lat = float(row["coord_lat"])
        lon = float(row["coord_lon"])
        if is_degenerate_img2loc_prediction(lat, lon):
            continue
        score = float(row["score"]) if row.get("score") not in {"", None} else None
        for key in candidate_path_keys(row["image_path"]):
            lookup[key] = (lat, lon, score)
    return lookup


def fetch_embedding_coords(embeddings: Any, index_id: int) -> tuple[float, float]:
    if embeddings.ndim == 3:
        coords = embeddings[index_id, 0, -2:]
    else:
        coords = embeddings[index_id, -2:]
    return float(coords[0]), float(coords[1])


def build_img2loc_search_index(embeddings: Any, chunk_size: int = 200_000) -> Any:
    try:
        import faiss
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModelUnavailableError("Img2Loc requires faiss and numpy.") from exc

    if embeddings.ndim not in (2, 3):
        raise ValueError(f"Unsupported embeddings shape for Img2Loc index build: {embeddings.shape}")

    embedding_dim = embeddings.shape[-1]
    if embedding_dim < IMG2LOC_CLIP_DIMENSIONS:
        raise ValueError(
            "Img2Loc embeddings must contain at least "
            f"{IMG2LOC_CLIP_DIMENSIONS} CLIP dimensions; got shape {embeddings.shape}"
        )

    index = faiss.IndexFlatIP(IMG2LOC_CLIP_DIMENSIONS)
    total = embeddings.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if embeddings.ndim == 3:
            chunk = np.array(embeddings[start:end, 0, :IMG2LOC_CLIP_DIMENSIONS], dtype=np.float32, copy=False)
        else:
            chunk = np.array(embeddings[start:end, :IMG2LOC_CLIP_DIMENSIONS], dtype=np.float32, copy=False)
        index.add(np.ascontiguousarray(chunk))
    return index


class Img2LocModel(BaseGeoModel):
    model_name = "Img2Loc"
    prefer_full_batch = True

    def __init__(
        self,
        root: str | Path = DEFAULT_IMG2LOC_ROOT,
        index_path: str | Path | None = None,
        embeddings_with_coords: str | Path = DEFAULT_IMG2LOC_EMBEDDINGS,
        search_results_csv: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve() if index_path else self.root / "index" / "clip.index"
        self.embeddings_path = Path(embeddings_with_coords).expanduser().resolve()
        self.search_results_csv = (
            Path(search_results_csv).expanduser().resolve()
            if search_results_csv
            else self.root / "index" / "search_results.csv"
        )
        self.predictions_csv = self.search_results_csv.with_name("llava_predictions.csv")
        self.device = resolve_runtime_device(device)
        self.precomputed_lookup = load_img2loc_precomputed_predictions(self.predictions_csv)
        for key, prediction in load_img2loc_precomputed_predictions(self.search_results_csv).items():
            self.precomputed_lookup.setdefault(key, prediction)
        self._clip_model: Any | None = None
        self._vision_processor: Any | None = None
        self._index: Any | None = None
        self._embeddings: Any | None = None
        self._torch: Any | None = None

    @classmethod
    def from_args(cls, args: Any) -> "Img2LocModel":
        index_path, embeddings_path, search_results_path = resolve_img2loc_paths(args)
        return cls(
            root=args.img2loc_root,
            index_path=index_path,
            embeddings_with_coords=embeddings_path,
            search_results_csv=search_results_path,
            device=args.device,
        )

    def _load_live_components(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._clip_model is None:
            try:
                import faiss
                import numpy as np
                import torch
                from transformers import CLIPImageProcessor, CLIPModel
            except ModuleNotFoundError as exc:
                raise ModelUnavailableError("Img2Loc requires faiss, numpy, torch, and transformers.") from exc

            if not self.embeddings_path.exists():
                raise ModelUnavailableError(f"Img2Loc embeddings file does not exist: {self.embeddings_path}")
            if not self.index_path.exists():
                raise ModelUnavailableError(f"Img2Loc index file does not exist: {self.index_path}")

            clip_snapshot_dir = find_local_clip_snapshot_dir()
            clip_source = str(clip_snapshot_dir) if clip_snapshot_dir is not None else "openai/clip-vit-large-patch14"
            local_only = clip_snapshot_dir is not None

            embeddings = np.load(self.embeddings_path, mmap_mode="r")
            index = faiss.read_index(str(self.index_path))
            if index.d != IMG2LOC_CLIP_DIMENSIONS:
                print(
                    f"Img2Loc index dimension is {index.d}, rebuilding a compatible "
                    f"{IMG2LOC_CLIP_DIMENSIONS}-d search index from the first "
                    f"{IMG2LOC_CLIP_DIMENSIONS} embedding dimensions. The final "
                    f"{IMG2LOC_COORD_DIMENSIONS} stored dimensions are coordinates."
                )
                index = build_img2loc_search_index(embeddings)

            clip_model = CLIPModel.from_pretrained(clip_source, local_files_only=local_only).to(self.device)
            clip_model.requires_grad_(False)
            clip_model.eval()
            print(f"{self.model_name} loaded on {describe_runtime_device(self.device, torch)}")
            self._clip_model = clip_model
            self._vision_processor = CLIPImageProcessor.from_pretrained(clip_source, local_files_only=local_only)
            self._index = index
            self._embeddings = embeddings
            self._torch = torch

        return self._clip_model, self._vision_processor, self._index, self._embeddings, self._torch

    def _precomputed_prediction(
        self,
        image_path: str | Path,
        source_path: str | Path | None,
        augmentation: str | None,
    ) -> tuple[float, float, float | None] | None:
        lookup_paths: list[str | Path] = [image_path]
        if source_path and augmentation == GROUND_TRUTH_AUGMENTATION:
            lookup_paths.insert(0, source_path)

        for lookup_path in lookup_paths:
            for key in candidate_path_keys(lookup_path):
                cached = self.precomputed_lookup.get(key)
                if cached is not None:
                    return cached
        return None

    def _script_path(self, script_name: str) -> Path:
        script_path = self.root / script_name
        if not script_path.exists():
            raise ModelUnavailableError(f"Img2Loc script does not exist: {script_path}")
        return script_path

    def _run_command(self, command: list[str]) -> None:
        try:
            subprocess.run(command, cwd=self.root, check=True, text=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            command_text = " ".join(command)
            error_text = (exc.stderr or "").strip()
            if len(error_text) > 4000:
                error_text = error_text[-4000:]
            details = f"\n{error_text}" if error_text else ""
            raise ModelUnavailableError(f"Img2Loc pipeline command failed: {command_text}{details}") from exc

    def _llava_command(self, llava_script: Path, search_results_csv: Path, predictions_csv: Path) -> list[str]:
        compat_runner = Path(__file__).resolve().parents[1] / "src" / "biased_dataset_tools" / "img2loc_llava_compat.py"
        if not compat_runner.exists():
            raise ModelUnavailableError(f"Img2Loc LLaVA compatibility runner does not exist: {compat_runner}")
        return [
            sys.executable,
            str(compat_runner),
            str(llava_script),
            "--results_csv",
            str(search_results_csv),
            "--output_csv",
            str(predictions_csv),
        ]

    def _run_local_pipeline(
        self,
        image_paths: list[str | Path],
    ) -> list[tuple[float, float, float | None]]:
        if not self.root.exists():
            raise ModelUnavailableError(f"Img2Loc root does not exist: {self.root}")
        if not self.index_path.exists():
            raise ModelUnavailableError(f"Img2Loc index file does not exist: {self.index_path}")
        if not self.embeddings_path.exists():
            raise ModelUnavailableError(f"Img2Loc embeddings file does not exist: {self.embeddings_path}")

        search_script = self._script_path(IMG2LOC_SEARCH_SCRIPT)
        llava_script = self._script_path(IMG2LOC_LLAVA_SCRIPT)

        with tempfile.TemporaryDirectory(prefix="img2loc_", dir="/tmp") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            image_dir = temp_dir / "images"
            output_dir = temp_dir / "output"
            image_dir.mkdir()
            output_dir.mkdir()

            temp_index_by_key: dict[str, int] = {}
            for index, image_path in enumerate(image_paths):
                resolved_image_path = Path(image_path).expanduser().resolve()
                suffix = resolved_image_path.suffix if resolved_image_path.suffix else ".jpg"
                temp_image_path = image_dir / f"{index:06d}{suffix.lower()}"
                try:
                    os.symlink(resolved_image_path, temp_image_path)
                except OSError:
                    shutil.copy2(resolved_image_path, temp_image_path)

                for key in candidate_path_keys(temp_image_path):
                    temp_index_by_key[key] = index

            search_results_csv = output_dir / "search_results.csv"
            predictions_csv = output_dir / "llava_predictions.csv"

            self._run_command(
                [
                    sys.executable,
                    str(search_script),
                    "--index",
                    str(self.index_path),
                    "--image_dir",
                    str(image_dir),
                    "--embeddings_with_coords",
                    str(self.embeddings_path),
                    "--output",
                    str(search_results_csv),
                ]
            )
            self._run_command(self._llava_command(llava_script, search_results_csv, predictions_csv))

            search_fallback_lookup = load_img2loc_precomputed_predictions(search_results_csv)
            predictions: list[tuple[float, float, float | None] | None] = [None] * len(image_paths)
            for row in read_csv_rows(predictions_csv):
                if row.get("latitude") in {"", None} or row.get("longitude") in {"", None}:
                    continue

                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if is_degenerate_img2loc_prediction(lat, lon):
                    continue

                matched_index: int | None = None
                for key in candidate_path_keys(row["image_path"]):
                    matched_index = temp_index_by_key.get(key)
                    if matched_index is not None:
                        break
                if matched_index is None:
                    continue

                predictions[matched_index] = (lat, lon, None)

            for temp_key, matched_index in temp_index_by_key.items():
                if predictions[matched_index] is not None:
                    continue
                fallback = search_fallback_lookup.get(temp_key)
                if fallback is not None:
                    predictions[matched_index] = fallback

        missing = [index for index, prediction in enumerate(predictions) if prediction is None]
        if missing:
            raise RuntimeError(f"Img2Loc pipeline did not produce predictions for indices: {missing}")

        return [prediction for prediction in predictions if prediction is not None]

    def _output_from_prediction(
        self,
        image_path: str | Path,
        prediction: tuple[float, float, float | None],
        *,
        source: str,
        kwargs: dict[str, Any],
    ) -> ModelOutput:
        pred_lat, pred_lon, confidence = prediction
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
                source=source,
            ),
        )

    def batch_inference(
        self,
        image_paths: list[str | Path],
        *,
        item_kwargs: list[dict[str, Any]] | None = None,
        **shared_kwargs: Any,
    ) -> list[ModelOutput]:
        if item_kwargs is None:
            item_kwargs = [{} for _ in image_paths]
        if len(item_kwargs) != len(image_paths):
            raise ValueError("item_kwargs must have the same length as image_paths")

        outputs: list[ModelOutput | None] = [None] * len(image_paths)
        live_paths: list[str | Path] = []
        live_indices: list[int] = []
        live_kwargs: list[dict[str, Any]] = []

        for index, (image_path, current_item_kwargs) in enumerate(zip(image_paths, item_kwargs)):
            merged_kwargs = dict(shared_kwargs)
            merged_kwargs.update(current_item_kwargs)
            cached = self._precomputed_prediction(
                image_path,
                merged_kwargs.get("source_path"),
                merged_kwargs.get("augmentation"),
            )
            if cached is not None:
                source = "precomputed_search_fallback" if cached[2] is not None else "precomputed_llava"
                outputs[index] = self._output_from_prediction(
                    image_path,
                    cached,
                    source=source,
                    kwargs=merged_kwargs,
                )
                continue

            live_paths.append(image_path)
            live_indices.append(index)
            live_kwargs.append(merged_kwargs)

        if live_paths:
            live_predictions = self._run_local_pipeline(live_paths)
            for original_index, image_path, prediction, merged_kwargs in zip(
                live_indices,
                live_paths,
                live_predictions,
                live_kwargs,
            ):
                source = "img2loc_pipeline_search_fallback" if prediction[2] is not None else "img2loc_pipeline"
                outputs[original_index] = self._output_from_prediction(
                    image_path,
                    prediction,
                    source=source,
                    kwargs=merged_kwargs,
                )

        missing = [index for index, output in enumerate(outputs) if output is None]
        if missing:
            raise RuntimeError(f"Img2Loc did not produce outputs for indices: {missing}")
        return [output for output in outputs if output is not None]

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        return self.batch_inference([image_path], item_kwargs=[kwargs])[0]

    def retrieval_only_inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        cached = self._precomputed_prediction(image_path, kwargs.get("source_path"), kwargs.get("augmentation"))
        if cached is not None:
            return self._output_from_prediction(
                image_path,
                cached,
                source="precomputed",
                kwargs=kwargs,
            )

        from PIL import Image

        clip_model, processor, index, embeddings, torch = self._load_live_components()
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            vision_output = clip_model.vision_model(pixel_values)[1]
            image_embeds = clip_model.visual_projection(vision_output)
        scores, indices = index.search(image_embeds.cpu().detach().numpy(), 1)
        pred_lat, pred_lon = fetch_embedding_coords(embeddings, int(indices[0][0]))
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
                source="retrieval_only",
            ),
        )
