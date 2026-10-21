from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.base import BaseGeoModel
from models.deepseek_vl import DeepSeekVLModel
from models.g3 import G3Model
from models.gemma4 import Gemma4Model
from models.geoclip import GeoCLIPModel
from models.glm41v import GLM41VModel
from models.img2loc import Img2LocModel
from models.internvl35 import InternVL35Model
from models.qwen3_vl import Qwen3VL8BModel, QwenVLModel
from models.qwen36_fp8 import Qwen36FP8Model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    cls: type[BaseGeoModel]
    default_kwargs: dict[str, Any]
    evaluate_by_default: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "geoclip": ModelSpec("geoclip", "GeoCLIP", GeoCLIPModel, {}),
    "img2loc": ModelSpec("img2loc", "Img2Loc", Img2LocModel, {}),
    "g3": ModelSpec("g3", "G3", G3Model, {}),
    "qwen3-vl-8b": ModelSpec("qwen3-vl-8b", "Qwen3-VL-8B", Qwen3VL8BModel, {}),
    "deepseek-vl": ModelSpec("deepseek-vl", "DeepSeek-VL", DeepSeekVLModel, {}),
    "gemma4": ModelSpec("gemma4", "Gemma-4-E4B-IT", Gemma4Model, {}),
    "glm41v": ModelSpec("glm41v", "GLM-4.1V-9B-Base", GLM41VModel, {}),
    "internvl35": ModelSpec("internvl35", "InternVL3.5-8B-Instruct", InternVL35Model, {}),
    "qwen36-fp8": ModelSpec("qwen36-fp8", "Qwen3.6-27B-FP8", Qwen36FP8Model, {}),
    "qwen2-vl-7b": ModelSpec(
        "qwen2-vl-7b",
        "Qwen2-VL-7B",
        QwenVLModel,
        {"model_id": "Qwen/Qwen2-VL-7B-Instruct", "model_name": "Qwen2-VL-7B"},
        evaluate_by_default=False,
    ),
    "qwen2.5-vl-7b": ModelSpec(
        "qwen2.5-vl-7b",
        "Qwen2.5-VL-7B",
        QwenVLModel,
        {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_name": "Qwen2.5-VL-7B"},
        evaluate_by_default=False,
    ),
}

ALIASES = {
    "qwen": "qwen3-vl-8b",
    "qwen3": "qwen3-vl-8b",
    "deepseek": "deepseek-vl",
    "gemma": "gemma4",
    "gemma3": "gemma4",
    "glm": "glm41v",
    "internvl": "internvl35",
    "qwen3.6": "qwen36-fp8",
}


def normalize_model_name(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def available_model_names(include_compat: bool = True) -> list[str]:
    names = [
        name
        for name, spec in MODEL_SPECS.items()
        if include_compat or spec.evaluate_by_default
    ]
    return sorted(names)


def default_evaluation_model_names() -> list[str]:
    return [name for name, spec in MODEL_SPECS.items() if spec.evaluate_by_default]


def create_model(name: str, **kwargs: Any) -> BaseGeoModel:
    normalized = normalize_model_name(name)
    if normalized not in MODEL_SPECS:
        known = ", ".join(available_model_names())
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")

    spec = MODEL_SPECS[normalized]
    init_kwargs = dict(spec.default_kwargs)
    init_kwargs.update({key: value for key, value in kwargs.items() if value is not None})
    return spec.cls(**init_kwargs)


def create_model_from_args(name: str, args: Any) -> BaseGeoModel:
    normalized = normalize_model_name(name)
    if normalized == "geoclip":
        return GeoCLIPModel(device=args.device)
    if normalized == "img2loc":
        return Img2LocModel(
            root=args.img2loc_root,
            index_path=args.img2loc_index,
            embeddings_with_coords=args.img2loc_embeddings_with_coords,
            search_results_csv=args.img2loc_search_results_csv,
            device=args.device,
        )
    if normalized == "g3":
        return G3Model(
            root=args.g3_root,
            checkpoint_path=args.g3_checkpoint,
            index_path=args.g3_index,
            database_csv=args.g3_database_csv,
            results_csv=args.g3_results_csv,
            device=args.device,
        )
    if normalized == "qwen-custom":
        return QwenVLModel(
            model_id=args.vlm_model_id,
            model_name=args.vlm_model_id.split("/")[-1] if args.vlm_model_id else "Qwen VL",
            device=args.device,
            max_new_tokens=args.vlm_max_new_tokens,
            attention_implementation=getattr(args, "attention_implementation", "auto"),
        )
    if normalized == "qwen36-fp8":
        return Qwen36FP8Model(
            device=args.device,
            max_new_tokens=args.vlm_max_new_tokens,
            disable_quantization_if_vram_over_50gb=getattr(args, "qwen36_disable_quantization_if_vram_over_50gb", False),
            attention_implementation=getattr(args, "attention_implementation", "auto"),
            openai_base_url=getattr(args, "openai_base_url", None),
            openai_api_key=getattr(args, "openai_api_key", None),
        )
    extra_kwargs: dict[str, Any] = {}
    if normalized in {
        "qwen3-vl-8b",
        "qwen2-vl-7b",
        "qwen2.5-vl-7b",
        "deepseek-vl",
        "gemma4",
        "glm41v",
        "internvl35",
    }:
        extra_kwargs = {
            "device": args.device,
            "max_new_tokens": args.vlm_max_new_tokens,
            "attention_implementation": getattr(args, "attention_implementation", "auto"),
        }
    return create_model(normalized, **extra_kwargs)
