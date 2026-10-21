from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from biased_dataset_tools.clip_attention import (
    build_overlay,
    load_eager_clip_components,
    load_rgb_image,
    normalize_heatmap,
)
from biased_dataset_tools.geoclip_xai import validate_columns
from biased_dataset_tools.geoloc_shared import resolve_runtime_device
from models.geoclip import load_geoclip_model, parse_top_prediction


DEFAULT_OUTPUT_DIR = Path("figures/geoclip_gradcam")
DEFAULT_IMAGE_SPECS = ("image_path:anonymized", "gt_path:standard")
DEFAULT_TARGET_LAYER = "vision_model.encoder.layers.-1.layer_norm1"
INDEX_COLUMNS = [
    "manifest_csv",
    "row_index",
    "image_kind",
    "image_column",
    "image_path",
    "resolved_image_path",
    "target_lat",
    "target_lon",
    "target_prob",
    "target_location",
    "gradcam_layer",
    "overlay_path",
    "heatmap_path",
]


@dataclass(frozen=True)
class ImageSpec:
    column: str
    kind: str


@dataclass(frozen=True)
class GradCamInput:
    manifest_csv: Path
    row_index: int
    image_kind: str
    image_column: str
    image_path: str
    resolved_image_path: Path
    row: dict[str, str]


@dataclass
class GradCamStats:
    total_rows: int = 0
    selected_images: int = 0
    missing_path_values: int = 0
    missing_image_files: int = 0
    duplicate_images: int = 0
    generated_images: int = 0
    skipped_existing_images: int = 0


@dataclass(frozen=True)
class GradCamRunResult:
    output_root: Path
    gradcam_dir: Path
    index_csv: Path
    metadata_path: Path
    generated_images: int
    skipped_existing_images: int


def parse_image_spec(value: str) -> ImageSpec:
    if ":" in value:
        column, kind = value.split(":", 1)
    else:
        column = value
        kind = value
    column = column.strip()
    kind = kind.strip()
    if not column or not kind:
        raise argparse.ArgumentTypeError("Image specs must use COLUMN or COLUMN:KIND with non-empty values.")
    return ImageSpec(column=column, kind=kind)


def read_config_manifest(config_path: Path) -> tuple[Path | None, Path | None]:
    import yaml

    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset = config.get("dataset") or {}
    manifest = dataset.get("manifest_csv")
    root_dir = dataset.get("root_dir")
    manifest_path = Path(manifest).expanduser() if manifest else None
    dataset_root = Path(root_dir).expanduser() if root_dir else None
    return manifest_path, dataset_root


def resolve_image_path(raw_path: str, manifest_csv: Path, dataset_root: Path | None) -> Path | None:
    path = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(manifest_csv.parent / path)
        if dataset_root is not None:
            candidates.append(dataset_root / path)
            candidates.append(dataset_root.parent / path)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return None


def safe_output_stem(image_path: str, image_kind: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        path = Path(*path.parts[1:])
    clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
    if not clean_parts:
        clean_parts = ["image"]
    relative = Path(*clean_parts)
    return relative.parent / f"{relative.stem}_{image_kind}"


def gradcam_output_paths(gradcam_dir: Path, image_path: str, image_kind: str) -> tuple[Path, Path]:
    base_path = gradcam_dir / safe_output_stem(image_path, image_kind)
    overlay_path = base_path.with_name(f"{base_path.name}_gradcam_overlay.jpg")
    heatmap_path = base_path.with_name(f"{base_path.name}_gradcam_heatmap.png")
    return overlay_path.resolve(), heatmap_path.resolve()


def save_gradcam_images(
    gradcam_dir: Path,
    image_path: str,
    image_kind: str,
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[Path, Path]:
    import cv2

    overlay_path, heatmap_path = gradcam_output_paths(gradcam_dir, image_path, image_kind)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_rgb, grayscale_rgb = build_overlay(image_rgb, heatmap)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(grayscale_rgb, cv2.COLOR_RGB2BGR))
    return overlay_path, heatmap_path


def collect_gradcam_inputs(
    manifest_csv: Path,
    image_specs: Sequence[ImageSpec],
    *,
    dataset_root: Path | None,
    include_duplicate_images: bool,
    limit: int | None,
) -> tuple[list[GradCamInput], GradCamStats]:
    manifest_csv = manifest_csv.expanduser().resolve()
    stats = GradCamStats()
    selected: list[GradCamInput] = []
    seen_images: set[tuple[str, Path]] = set()

    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, [spec.column for spec in image_specs], manifest_csv)
        for row_index, row in enumerate(reader, start=1):
            stats.total_rows += 1
            for spec in image_specs:
                raw_path = (row.get(spec.column) or "").strip()
                if not raw_path:
                    stats.missing_path_values += 1
                    continue
                resolved = resolve_image_path(raw_path, manifest_csv, dataset_root)
                if resolved is None:
                    stats.missing_image_files += 1
                    continue
                dedupe_key = (spec.kind, resolved)
                if not include_duplicate_images and dedupe_key in seen_images:
                    stats.duplicate_images += 1
                    continue
                seen_images.add(dedupe_key)
                selected.append(
                    GradCamInput(
                        manifest_csv=manifest_csv,
                        row_index=row_index,
                        image_kind=spec.kind,
                        image_column=spec.column,
                        image_path=raw_path,
                        resolved_image_path=resolved,
                        row=dict(row),
                    )
                )
                stats.selected_images += 1
                if limit is not None and len(selected) >= limit:
                    return selected, stats

    return selected, stats


def resolve_module(root: Any, module_path: str) -> Any:
    current = root
    for part in module_path.split("."):
        if not part:
            continue
        if part.lstrip("-").isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _tensor_from_hook_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def gradcam_from_activations(activations: Any, gradients: Any) -> np.ndarray:
    import torch

    if activations.ndim == 3:
        patch_activations = activations[:, 1:, :]
        patch_gradients = gradients[:, 1:, :]
        weights = patch_gradients.mean(dim=1, keepdim=True)
        cam = (patch_activations * weights).sum(dim=-1).clamp(min=0.0)
        patch_count = int(cam.shape[-1])
        side = int(math.sqrt(patch_count))
        if side * side != patch_count:
            raise ValueError(f"Could not reshape Grad-CAM patches into a square grid: {patch_count} tokens")
        return cam[0].reshape(side, side).detach().cpu().numpy()

    if activations.ndim == 4:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (activations * weights).sum(dim=1).clamp(min=0.0)
        return cam[0].detach().cpu().numpy()

    raise ValueError(f"Unsupported Grad-CAM activation shape: {tuple(activations.shape)}")


def resize_heatmap(heatmap: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(heatmap.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return normalize_heatmap(resized)


def geoclip_location_encoder(model: Any) -> Any:
    for attribute_name in ("location_encoder", "gps_encoder"):
        encoder = getattr(model, attribute_name, None)
        if encoder is not None:
            return encoder
    raise AttributeError("GeoCLIP model does not expose location_encoder or gps_encoder.")


class GeoCLIPGradCamExplainer:
    def __init__(self, device: str, target_layer: str = DEFAULT_TARGET_LAYER) -> None:
        import torch.nn.functional as f

        self.f = f
        self.device = device
        self.target_layer = target_layer
        self.geoclip_model = load_geoclip_model(device)
        if hasattr(self.geoclip_model, "to"):
            moved_model = self.geoclip_model.to(device)
            if moved_model is not None:
                self.geoclip_model = moved_model
        self.geoclip_model.eval()
        self.clip_model, self.processor = load_eager_clip_components(device)

    def predict_location(self, image_path: Path) -> tuple[float, float, float]:
        top_pred_gps, top_pred_prob = self.geoclip_model.predict(str(image_path), top_k=1)
        return parse_top_prediction(top_pred_gps, top_pred_prob)

    def explain(
        self,
        image_path: Path,
        *,
        target_location: str,
        row: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray, float, float, float | None]:
        import torch

        image, image_rgb = load_rgb_image(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

        if target_location == "ground-truth":
            target_lat = float(row["lat"])
            target_lon = float(row["lon"])
            target_prob = None
        else:
            target_lat, target_lon, target_prob = self.predict_location(image_path)

        activations: dict[str, Any] = {}
        gradients: dict[str, Any] = {}
        layer = resolve_module(self.clip_model, self.target_layer)

        def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations["value"] = _tensor_from_hook_value(output)

        def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
            gradients["value"] = _tensor_from_hook_value(grad_output)

        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_full_backward_hook(backward_hook)
        try:
            self.geoclip_model.zero_grad(set_to_none=True)
            self.clip_model.zero_grad(set_to_none=True)

            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            clip_features = self.clip_model.visual_projection(pooled_output)
            image_features = self.geoclip_model.image_encoder.mlp(clip_features)
            image_features = self.f.normalize(image_features, dim=-1)

            location = torch.tensor(
                [[target_lat, target_lon]],
                device=self.device,
                dtype=image_features.dtype,
            )
            location_features = geoclip_location_encoder(self.geoclip_model)(location)
            location_features = self.f.normalize(location_features, dim=-1)

            score = (self.geoclip_model.logit_scale.exp() * (image_features * location_features).sum(dim=-1)).sum()
            score.backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        if "value" not in activations or "value" not in gradients:
            raise ValueError(f"Grad-CAM hooks did not capture activations/gradients for {self.target_layer}")

        heatmap = gradcam_from_activations(activations["value"], gradients["value"])
        return image_rgb, resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1]), target_lat, target_lon, target_prob


def index_row_for(
    item: GradCamInput,
    *,
    target_lat: float,
    target_lon: float,
    target_prob: float | None,
    target_location: str,
    target_layer: str,
    overlay_path: Path,
    heatmap_path: Path,
) -> dict[str, str]:
    return {
        "manifest_csv": str(item.manifest_csv),
        "row_index": str(item.row_index),
        "image_kind": item.image_kind,
        "image_column": item.image_column,
        "image_path": item.image_path,
        "resolved_image_path": str(item.resolved_image_path),
        "target_lat": f"{target_lat:.12g}",
        "target_lon": f"{target_lon:.12g}",
        "target_prob": "" if target_prob is None else f"{target_prob:.12g}",
        "target_location": target_location,
        "gradcam_layer": target_layer,
        "overlay_path": str(overlay_path),
        "heatmap_path": str(heatmap_path),
    }


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_manifest_and_root(args: argparse.Namespace) -> tuple[Path, Path | None]:
    config_manifest = None
    config_dataset_root = None
    if args.config is not None:
        config_manifest, config_dataset_root = read_config_manifest(args.config)
    manifest_csv = (args.manifest_csv or config_manifest)
    if manifest_csv is None:
        raise ValueError("Provide --manifest-csv or a --config file with dataset.manifest_csv.")
    dataset_root = args.dataset_root or config_dataset_root
    return manifest_csv.expanduser().resolve(), dataset_root.expanduser().resolve() if dataset_root is not None else None


def run(args: argparse.Namespace, explainer: Any | None = None) -> GradCamRunResult:
    manifest_csv, dataset_root = resolve_manifest_and_root(args)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    image_values = list(args.image) if args.image is not None else list(DEFAULT_IMAGE_SPECS)
    image_specs = [parse_image_spec(value) for value in image_values]
    output_root = args.output_dir.expanduser().resolve() / manifest_csv.stem
    gradcam_dir = output_root / "gradcam_maps"
    index_csv = output_root / "gradcam_index.csv"
    metadata_path = output_root / "geoclip_gradcam_metadata.json"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    items, stats = collect_gradcam_inputs(
        manifest_csv,
        image_specs,
        dataset_root=dataset_root,
        include_duplicate_images=args.include_duplicate_images,
        limit=args.limit,
    )
    print(f"Selected {stats.selected_images} image(s) for GeoCLIP Grad-CAM.")

    index_rows: list[dict[str, str]] = []
    pending: list[GradCamInput] = []
    for item in items:
        overlay_path, heatmap_path = gradcam_output_paths(gradcam_dir, item.image_path, item.image_kind)
        if args.skip_existing and overlay_path.exists() and heatmap_path.exists():
            stats.skipped_existing_images += 1
            index_rows.append(
                index_row_for(
                    item,
                    target_lat=float("nan"),
                    target_lon=float("nan"),
                    target_prob=None,
                    target_location=args.target_location,
                    target_layer=args.target_layer,
                    overlay_path=overlay_path,
                    heatmap_path=heatmap_path,
                )
            )
            continue
        pending.append(item)

    if pending and explainer is None:
        device = resolve_runtime_device(args.device)
        explainer = GeoCLIPGradCamExplainer(device=device, target_layer=args.target_layer)

    for index, item in enumerate(pending, start=1):
        assert explainer is not None
        image_rgb, heatmap, target_lat, target_lon, target_prob = explainer.explain(
            item.resolved_image_path,
            target_location=args.target_location,
            row=item.row,
        )
        overlay_path, heatmap_path = save_gradcam_images(
            gradcam_dir,
            item.image_path,
            item.image_kind,
            image_rgb,
            heatmap,
        )
        index_rows.append(
            index_row_for(
                item,
                target_lat=target_lat,
                target_lon=target_lon,
                target_prob=target_prob,
                target_location=args.target_location,
                target_layer=args.target_layer,
                overlay_path=overlay_path,
                heatmap_path=heatmap_path,
            )
        )
        stats.generated_images += 1
        print(f"  [{index}/{len(pending)}] Grad-CAM [{item.image_kind}] for {item.image_path}")

    write_index(index_csv, index_rows)
    metadata = {
        "manifest_csv": str(manifest_csv),
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "output_root": str(output_root),
        "gradcam_dir": str(gradcam_dir),
        "index_csv": str(index_csv),
        "device": args.device,
        "image_specs": [asdict(spec) for spec in image_specs],
        "target_layer": args.target_layer,
        "target_location": args.target_location,
        "skip_existing": args.skip_existing,
        "include_duplicate_images": args.include_duplicate_images,
        "limit": args.limit,
        "stats": asdict(stats),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return GradCamRunResult(
        output_root=output_root.resolve(),
        gradcam_dir=gradcam_dir.resolve(),
        index_csv=index_csv.resolve(),
        metadata_path=metadata_path.resolve(),
        generated_images=stats.generated_images,
        skipped_existing_images=stats.skipped_existing_images,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GeoCLIP Grad-CAM heatmaps for anonymized/generated and standard images."
    )
    parser.add_argument("--manifest-csv", type=Path, default=None, help="Dataset manifest CSV.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config containing dataset.manifest_csv/root_dir.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional root for resolving relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--device", default="auto", help="Execution device. Use 'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images across all image specs.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing heatmap and overlay files.")
    parser.add_argument(
        "--include-duplicate-images",
        action="store_true",
        help="Do not deduplicate repeated image paths, especially repeated standard/gt_path images.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help="Image column to process, optionally COLUMN:KIND. Defaults to image_path:anonymized and gt_path:standard.",
    )
    parser.add_argument(
        "--target-location",
        choices=("prediction", "ground-truth"),
        default="prediction",
        help="Backpropagate either the GeoCLIP predicted location score or the row lat/lon score.",
    )
    parser.add_argument(
        "--target-layer",
        default=DEFAULT_TARGET_LAYER,
        help="Dotted module path to hook inside CLIPModel. Integers index module lists; -1 selects the final layer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(f"GeoCLIP Grad-CAM index: {result.index_csv}")
    print(f"GeoCLIP Grad-CAM metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
