from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION
from biased_dataset_tools.geoloc_methods import (
    build_img2loc_search_index,
    find_local_clip_snapshot_dir,
    load_geoclip_model,
    resolve_g3_paths,
    resolve_img2loc_paths,
)
from biased_dataset_tools.geoloc_shared import candidate_path_keys, read_csv_rows
from models.img2loc import IMG2LOC_CLIP_DIMENSIONS, IMG2LOC_COORD_DIMENSIONS


CLIP_ATTENTION_METHOD_NAME = "gradient_attention_rollout"
CLIP_ATTENTION_METHOD_REFERENCE = (
    "Transformer-native gradient-weighted attention rollout inspired by attention rollout "
    "and transformer attribution methods."
)


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
    import cv2

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, colored_rgb, 0.45, 0.0)
    grayscale_rgb = np.stack([heatmap_uint8, heatmap_uint8, heatmap_uint8], axis=-1)
    return overlay, grayscale_rgb


def safe_relative_output_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.is_absolute():
        path = Path(*path.parts[1:])

    clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
    if not clean_parts:
        return Path("image")
    return Path(*clean_parts)


def attention_output_paths(attention_dir: Path, dataset_path: str) -> tuple[Path, Path]:
    relative_path = safe_relative_output_path(dataset_path)
    base_path = attention_dir / relative_path.parent / relative_path.stem
    overlay_path = base_path.with_name(f"{base_path.name}_overlay.jpg")
    heatmap_path = base_path.with_name(f"{base_path.name}_heatmap.png")
    return overlay_path.resolve(), heatmap_path.resolve()


def save_attention_images(
    attention_dir: Path,
    dataset_path: str,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[Path, Path]:
    import cv2

    overlay_path, heatmap_path = attention_output_paths(attention_dir, dataset_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_rgb, grayscale_rgb = build_overlay(image_rgb, heatmap)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(grayscale_rgb, cv2.COLOR_RGB2BGR))
    return overlay_path, heatmap_path


def attention_rollout_from_gradients(attentions: tuple[Any, ...]) -> np.ndarray:
    import torch

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
    import cv2

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

        score = (self.geoclip_model.logit_scale.exp() * (image_features * location_features).sum(dim=-1)).sum()
        score.backward()

        heatmap = attention_rollout_from_gradients(vision_outputs.attentions)
        return image_rgb, resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1])


class Img2LocAttentionExplainer:
    def __init__(self, args: Any, device: str) -> None:
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
    def __init__(self, args: Any, device: str) -> None:
        import torch

        checkpoint_path, _, _, _ = resolve_g3_paths(args)
        self.torch = torch
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

        predicted_location = self.torch.tensor(
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


def load_attention_explainer(method: str, args: Any, device: str) -> Any | None:
    if method == "geoclip":
        return GeoCLIPAttentionExplainer(device)
    if method == "img2loc":
        return Img2LocAttentionExplainer(args, device)
    if method == "g3":
        return G3AttentionExplainer(args, device)
    return None
