from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION


EVAL_METHOD_CHOICES = [
    "geoclip",
    "img2loc",
    "g3",
    "qwen2-vl-7b",
    "qwen2.5-vl-2b",
    "qwen2.5-vl-8b",
    "qwen2.5-vl-7b",
    "qwen3-vl-8b",
    "qwen-custom",
]
METHOD_DISPLAY_NAMES = {
    "geoclip": "GeoCLIP",
    "img2loc": "Img2Loc",
    "g3": "G3",
    "qwen2-vl-7b": "Qwen2-VL-7B",
    "qwen2.5-vl-2b": "Qwen2.5-VL-2B",
    "qwen2.5-vl-8b": "Qwen2.5-VL-8B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "qwen-custom": "Qwen VL",
}
DEFAULT_QWEN_MODEL_IDS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}
DEFAULT_IMG2LOC_ROOT = Path("/home/nniccoli/Img2Loc")
DEFAULT_G3_ROOT = Path("/home/nniccoli/G3")
DEFAULT_IMG2LOC_EMBEDDINGS = Path("/mnt/nniccoli/geoclipplus_data/embeddings/embeddings_clip-vit-large-patch14.npy")
QWEN_COORDINATE_PROMPT = (
    "You are a geolocalization champion. Focus on the details of the picture with extreme care, including "
    "architecture, vegetation, terrain, road markings, signs, weather, vehicles, shadows, coastlines, "
    "mountains, water, text, and every small visual clue. You MUST infer a best-guess geographic location. "
    'You MUST return only valid JSON with exactly two keys: "lat" and "lon". '
    "Use signed decimal degrees: south and west are negative, north and east are positive. "
    "Do not copy coordinates from examples or instructions; there are no reference coordinates in this prompt."
)
ACCURACY_THRESHOLDS_KM = [1, 25, 200, 750, 2500]
PLOT_GROUP_ORDER = [
    GROUND_TRUTH_AUGMENTATION,
    "east_asian",
    "southeast_asian",
    "middle_eastern",
    "latino_hispanic",
    "indian",
    "black",
    "white",
    "asian",
    "european",
    "african",
]
PLOT_GROUP_STYLES = {
    GROUND_TRUTH_AUGMENTATION: {"label": "GT images", "color": "#7E57C2"},
    "east_asian": {"label": "East Asian generated", "color": "#F1C40F"},
    "southeast_asian": {"label": "Southeast Asian generated", "color": "#F39C12"},
    "middle_eastern": {"label": "Middle Eastern generated", "color": "#8E44AD"},
    "latino_hispanic": {"label": "Latino/Hispanic generated", "color": "#E67E22"},
    "indian": {"label": "Indian augmented", "color": "#2CA02C"},
    "black": {"label": "Black generated", "color": "#111111"},
    "white": {"label": "White generated", "color": "#1F77B4"},
    "asian": {"label": "Asian augmented", "color": "#F1C40F"},
    "european": {"label": "European augmented", "color": "#1F77B4"},
    "african": {"label": "African augmented", "color": "#111111"},
    "other": {"label": "Other augmented", "color": "#6B7280"},
}
COUNTRY_BOUNDARY_GLOBS = [
    "data/ne_*admin_0*countries*.shp",
    "data/ne_*admin_0*countries*.geojson",
    "data/ne_*admin_0*countries*.gpkg",
    "data/naturalearth_lowres/*.shp",
    "data/naturalearth_lowres/*.geojson",
    "data/naturalearth_lowres/*.gpkg",
    "data/*countries*.shp",
    "data/*countries*.geojson",
    "data/*countries*.gpkg",
    "data/**/*.shp",
    "data/**/*.geojson",
    "data/**/*.gpkg",
]
IMAGE_METRIC_COLUMNS = [
    "source_path",
    "dataset_path",
    "category",
    "augmentation",
    "group_source_path",
    "clean_dataset_path",
    "gt_lat",
    "gt_lon",
    "pred_lat",
    "pred_lon",
    "pred_prob",
    "haversine_to_gt_km",
    *(f"within_{threshold}km" for threshold in ACCURACY_THRESHOLDS_KM),
    "gt_country",
    "pred_country",
    "same_country_as_gt",
    "clean_pred_lat",
    "clean_pred_lon",
    "clean_to_aug_prediction_km",
]
OVERALL_METRIC_COLUMNS = [
    "metric_name",
    "group_by",
    "group_value",
    "count",
    "mean_km",
    "median_km",
    "min_km",
    "max_km",
    "accuracy",
]


@dataclass(frozen=True)
class ImagePrediction:
    dataset_path: str
    pred_lat: float
    pred_lon: float
    pred_prob: float | None


def derive_output_paths(
    manifest_path: Path,
    image_metrics_csv: Path | None,
    overall_metrics_csv: Path | None,
    map_html: Path | None,
    method_slug: str,
) -> tuple[Path, Path, Path]:
    if method_slug == "geoclip":
        default_image = manifest_path.with_name(f"{manifest_path.stem}_geoclip_image_metrics.csv")
        default_overall = manifest_path.with_name(f"{manifest_path.stem}_geoclip_overall_metrics.csv")
        default_map = manifest_path.with_name(f"{manifest_path.stem}_geoclip_prediction_map.html")
    else:
        default_image = manifest_path.with_name(f"{manifest_path.stem}_{method_slug}_image_metrics.csv")
        default_overall = manifest_path.with_name(f"{manifest_path.stem}_{method_slug}_overall_metrics.csv")
        default_map = manifest_path.with_name(f"{manifest_path.stem}_{method_slug}_prediction_map.html")

    image_path = image_metrics_csv or default_image
    overall_path = overall_metrics_csv or default_overall
    map_path = map_html or default_map
    return image_path.resolve(), overall_path.resolve(), map_path.resolve()


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def method_display_name(method: str, vlm_model_id: str | None = None) -> str:
    if method == "qwen-custom" and vlm_model_id:
        return vlm_model_id.split("/")[-1]
    return METHOD_DISPLAY_NAMES.get(method, method)


def resolve_runtime_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_top_prediction(top_pred_gps: Any, top_pred_prob: Any) -> tuple[float, float, float]:
    if hasattr(top_pred_gps, "tolist"):
        gps_values = top_pred_gps.tolist()
    else:
        gps_values = top_pred_gps

    if hasattr(top_pred_prob, "tolist"):
        prob_values = top_pred_prob.tolist()
    else:
        prob_values = top_pred_prob

    if isinstance(gps_values, (list, tuple)) and gps_values and isinstance(gps_values[0], (list, tuple)):
        gps_values = gps_values[0]
    if isinstance(prob_values, (list, tuple)):
        prob_value = prob_values[0]
    else:
        prob_value = prob_values

    lat = to_float(gps_values[0])
    lon = to_float(gps_values[1])
    prob = to_float(prob_value)
    return lat, lon, prob


def candidate_path_keys(path_value: str) -> list[str]:
    path = Path(path_value)
    keys = [str(path), path.name]
    try:
        keys.append(str(path.expanduser().resolve(strict=False)))
    except Exception:
        pass

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def format_method_from_metrics_path(image_metrics_csv: Path) -> str:
    stem = image_metrics_csv.stem
    for method in EVAL_METHOD_CHOICES:
        if method_slug(method) in stem:
            return method_display_name(method)
    return "Geolocation"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def normalize_plot_group(augmentation: str) -> str:
    normalized = augmentation.strip().lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    aliases = {
        "east asian": "east_asian",
        "southeast asian": "southeast_asian",
        "middle eastern": "middle_eastern",
        "latino hispanic": "latino_hispanic",
    }
    if normalized in aliases:
        return aliases[normalized]
    return normalized.replace(" ", "_")


def parse_bool_cell(value: str | None) -> bool | None:
    if value in {"1", "true", "True", "yes", "Yes"}:
        return True
    if value in {"0", "false", "False", "no", "No"}:
        return False
    return None


def build_threshold_hits(distance_km: float) -> dict[str, str]:
    return {
        f"within_{threshold}km": "1" if distance_km <= threshold else "0"
        for threshold in ACCURACY_THRESHOLDS_KM
    }


def format_optional_text(value: str | None) -> str:
    return value if value is not None else ""


def format_same_country(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_fieldnames_for_rows(rows: list[dict[str, str]]) -> list[str]:
    existing_fieldnames = list(rows[0].keys()) if rows else []
    fieldnames = list(IMAGE_METRIC_COLUMNS)
    fieldnames.extend(column for column in existing_fieldnames if column not in fieldnames)
    return fieldnames


def normalize_rows_for_fieldnames(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    return [{column: row.get(column, "") for column in fieldnames} for row in rows]
