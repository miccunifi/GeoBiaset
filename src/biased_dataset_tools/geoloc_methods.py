from __future__ import annotations

from pathlib import Path
from typing import Any

from biased_dataset_tools.classifier import ManifestEntry
from biased_dataset_tools.geoloc_shared import ImagePrediction, method_display_name, method_slug
from models.base import (
    DEFAULT_COORDINATE_PROMPT as QWEN_COORDINATE_PROMPT,
    find_local_hf_snapshot_dir,
    parse_lat_lon_from_text,
)
from models.g3 import (
    DEFAULT_G3_ROOT,
    G3Model,
    load_database_coords,
    load_g3_precomputed_predictions,
    resolve_g3_paths,
)
from models.geoclip import (
    GeoCLIPModel,
    find_local_clip_snapshot_dir,
    load_geoclip_model,
    load_geoclip_model_offline,
    parse_top_prediction,
    try_load_geoclip_model_offline,
)
from models.img2loc import (
    DEFAULT_IMG2LOC_EMBEDDINGS,
    DEFAULT_IMG2LOC_ROOT,
    Img2LocModel,
    build_img2loc_search_index,
    fetch_embedding_coords,
    load_img2loc_precomputed_predictions,
    resolve_img2loc_paths,
)
from models.qwen3_vl import (
    DEFAULT_QWEN_MODEL_IDS,
    QwenVLModel,
    available_local_qwen_model_ids,
    resolve_qwen_model_id,
    validate_qwen_snapshot,
)


def _prediction_from_output(output: Any, dataset_path: str) -> ImagePrediction:
    return ImagePrediction(
        dataset_path=dataset_path,
        pred_lat=output.pred_lat,
        pred_lon=output.pred_lon,
        pred_prob=output.confidence,
    )


def _predictor_from_model(model: Any) -> Any:
    def predict(entry: ManifestEntry, image_path: Path) -> ImagePrediction:
        output = model(
            image_path,
            dataset_path=entry.dataset_path,
            source_path=entry.source_path,
            augmentation=entry.augmentation,
        )
        return _prediction_from_output(output, entry.dataset_path)

    def batch_predict(entries: list[ManifestEntry], image_paths: list[Path]) -> list[ImagePrediction]:
        outputs = model.batch_inference(
            image_paths,
            item_kwargs=[
                {
                    "dataset_path": entry.dataset_path,
                    "source_path": entry.source_path,
                    "augmentation": entry.augmentation,
                }
                for entry in entries
            ],
        )
        return [
            _prediction_from_output(output, entry.dataset_path)
            for output, entry in zip(outputs, entries)
        ]

    predict.batch_predict = batch_predict  # type: ignore[attr-defined]
    predict.prefer_full_batch = bool(getattr(model, "prefer_full_batch", False))  # type: ignore[attr-defined]
    return predict


def load_geoclip_predictor(args: Any) -> Any:
    return _predictor_from_model(GeoCLIPModel(device=args.device))


def load_img2loc_predictor(args: Any) -> Any:
    return _predictor_from_model(Img2LocModel.from_args(args))


def load_g3_predictor(args: Any) -> Any:
    return _predictor_from_model(G3Model.from_args(args))


def load_qwen_predictor(args: Any) -> Any:
    return _predictor_from_model(QwenVLModel.from_args(args))


def load_predictor(args: Any) -> tuple[Any, str, str]:
    if args.method == "geoclip":
        return load_geoclip_predictor(args), method_display_name(args.method), method_slug(args.method)
    if args.method == "img2loc":
        return load_img2loc_predictor(args), method_display_name(args.method), method_slug(args.method)
    if args.method == "g3":
        return load_g3_predictor(args), method_display_name(args.method), method_slug(args.method)
    if args.method.startswith("qwen"):
        return load_qwen_predictor(args), method_display_name(args.method, args.vlm_model_id), method_slug(args.method)
    raise ValueError(f"Unsupported method: {args.method}")
