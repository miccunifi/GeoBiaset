from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from models.base import (
    BaseGeoModel,
    ModelOutput,
    ModelUnavailableError,
    describe_runtime_device,
    find_local_hf_snapshot_dir,
    resolve_runtime_device,
)


def find_local_clip_snapshot_dir() -> Path | None:
    return find_local_hf_snapshot_dir("openai/clip-vit-large-patch14")


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def parse_top_prediction(top_pred_gps: Any, top_pred_prob: Any) -> tuple[float, float, float]:
    gps_values = top_pred_gps.tolist() if hasattr(top_pred_gps, "tolist") else top_pred_gps
    prob_values = top_pred_prob.tolist() if hasattr(top_pred_prob, "tolist") else top_pred_prob

    if isinstance(gps_values, (list, tuple)) and gps_values and isinstance(gps_values[0], (list, tuple)):
        gps_values = gps_values[0]
    prob_value = prob_values[0] if isinstance(prob_values, (list, tuple)) else prob_values
    return _to_float(gps_values[0]), _to_float(gps_values[1]), _to_float(prob_value)


def load_geoclip_model(device: str = "auto") -> Any:
    try:
        from geoclip import GeoCLIP  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModelUnavailableError("GeoCLIP is not installed. Run `uv sync` or install `geoclip`.") from exc

    clip_snapshot_dir = find_local_clip_snapshot_dir()
    if clip_snapshot_dir is not None:
        return load_geoclip_model_offline(clip_snapshot_dir, device=device)

    try:
        model = GeoCLIP()
    except Exception as exc:
        offline_model = try_load_geoclip_model_offline(exc, device=device)
        if offline_model is not None:
            return offline_model
        raise
    move_geoclip_model_to_device(model, resolve_runtime_device(device))
    return model


def move_geoclip_model_to_device(model: Any, device: str) -> None:
    if hasattr(model, "to"):
        try:
            model.to(device)
            return
        except Exception:
            pass

    if hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:
            pass

    for attribute_name in ("image_encoder", "gps_encoder", "model"):
        component = getattr(model, attribute_name, None)
        if component is None or not hasattr(component, "to"):
            continue
        try:
            component.to(device)
        except Exception:
            continue


def load_geoclip_model_offline(clip_snapshot_dir: Path, device: str = "auto") -> Any:
    try:
        from geoclip import GeoCLIP  # type: ignore
        import importlib
        import torch
        import torch.nn as nn
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError as exc:
        raise ModelUnavailableError("GeoCLIP offline loading requires geoclip, torch, and transformers.") from exc

    resolved_device = resolve_runtime_device(device)

    class OfflineImageEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.CLIP = CLIPModel.from_pretrained(str(clip_snapshot_dir), local_files_only=True)
            self.image_processor = AutoProcessor.from_pretrained(str(clip_snapshot_dir), local_files_only=True)
            self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))
            for param in self.CLIP.parameters():
                param.requires_grad = False

        def preprocess_image(self, image):
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
            return pixel_values.to(resolved_device)

        def forward(self, x):
            x = self.CLIP.get_image_features(pixel_values=x)
            if hasattr(x, "pooler_output"):
                x = x.pooler_output
            elif hasattr(x, "image_embeds"):
                x = x.image_embeds
            return self.mlp(x)

    geoclip_module = importlib.import_module("geoclip.model.GeoCLIP")
    geoclip_module.ImageEncoder = OfflineImageEncoder
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = GeoCLIP()
    move_geoclip_model_to_device(model, resolved_device)
    return model


def try_load_geoclip_model_offline(original_error: Exception, device: str = "auto") -> Any | None:
    error_text = str(original_error)
    offline_hints = (
        "Temporary failure in name resolution",
        "Cannot send a request, as the client has been closed.",
        "processor_config.json",
    )
    if not any(hint in error_text for hint in offline_hints):
        return None
    clip_snapshot_dir = find_local_clip_snapshot_dir()
    if clip_snapshot_dir is None:
        return None
    return load_geoclip_model_offline(clip_snapshot_dir, device=device)


class GeoCLIPModel(BaseGeoModel):
    model_name = "GeoCLIP"

    def __init__(self, model: Any | None = None, device: str = "auto") -> None:
        self._model = model
        self.device = resolve_runtime_device(device)

    def _load(self) -> Any:
        if self._model is None:
            self._model = load_geoclip_model(self.device)
            device_label = describe_runtime_device(self.device)
            try:
                import torch

                first_param = next(self._model.parameters())
                device_label = describe_runtime_device(str(first_param.device), torch)
            except Exception:
                pass
            print(f"{self.model_name} loaded on {device_label}")
        return self._model

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        model = self._load()
        top_pred_gps, top_pred_prob = model.predict(str(image_path), top_k=1)
        pred_lat, pred_lon, confidence = parse_top_prediction(top_pred_gps, top_pred_prob)
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
            ),
        )
