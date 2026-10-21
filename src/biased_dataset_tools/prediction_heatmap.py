from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION
from biased_dataset_tools.geoloc_shared import normalize_plot_group
from biased_dataset_tools.gt_heatmap import (
    DEFAULT_BINS_X,
    DEFAULT_BINS_Y,
    DEFAULT_DENSITY_CLIP_PERCENTILE,
    DEFAULT_DISPLAY_LAT_MAX,
    DEFAULT_DISPLAY_LAT_MIN,
    DEFAULT_DPI,
    DEFAULT_MASK_FRACTION,
    DEFAULT_SIGMA,
    DEFAULT_THEME,
    THEME_CHOICES,
    CoordinateStats,
    RenderedHeatmap,
    build_histogram_from_coordinates,
    compute_density_vmax,
    default_boundary_path,
    metadata_for_render,
    parse_coordinate_values,
    render_theme_set,
    resolve_themes,
    validate_lat_bounds,
)


DEFAULT_INPUT_CSV = Path("dataset/img2loc_g3__geoclip_evaluation_results_with_extra_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("figures/prediction_heatmaps")
DEFAULT_MODEL_COLUMN = "model_name"
DEFAULT_AUGMENTATION_COLUMN = "augmentation"
DEFAULT_DATASET_PATH_COLUMN = "dataset_path"
DEFAULT_LAT_COLUMN = "pred_lat"
DEFAULT_LON_COLUMN = "pred_lon"
DENSITY_SCALE_CHOICES = ("per-map", "per-model", "global")
DEFAULT_DENSITY_SCALE = "per-model"
BACKGROUND_GROUP_KEY = "background_images"
GENDER_MALE = "male"
GENDER_FEMALE = "female"
ATTIRE_TRADITIONAL = "traditional_attire"
ATTIRE_WITHOUT_TRADITIONAL = "without_traditional_attire"
GROUP_VARIANT_ORDER = {
    "all": 0,
    GENDER_MALE: 1,
    GENDER_FEMALE: 2,
    ATTIRE_TRADITIONAL: 3,
    ATTIRE_WITHOUT_TRADITIONAL: 4,
}
RACE_GROUP_ORDER = [
    "east_asian",
    "southeast_asian",
    "middle_eastern",
    "latino_hispanic",
    "indian",
    "black",
    "white",
]


@dataclass(frozen=True)
class PredictionGroupSpec:
    key: str
    label: str
    augmentation: str
    augmentation_slug: str
    gender: str | None = None
    attire: str | None = None


@dataclass
class PredictionGroup:
    spec: PredictionGroupSpec
    coordinates: list[tuple[float, float]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.coordinates)


@dataclass
class ModelPredictionGroups:
    model_name: str
    model_slug: str
    source_rows: int = 0
    accepted_rows: int = 0
    duplicate_rows: int = 0
    grouped_assignments: int = 0
    groups: dict[str, PredictionGroup] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionHeatmapRunResult:
    metadata_path: Path
    output_root: Path


def slugify(value: str, default: str = "unknown") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or default


def validate_columns(fieldnames: Sequence[str] | None, required_columns: Sequence[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {csv_path}")
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")


def is_background_prediction(row: dict[str, str], augmentation_column: str, dataset_path_column: str) -> bool:
    augmentation_slug = normalize_plot_group(row.get(augmentation_column, ""))
    dataset_name = Path(row.get(dataset_path_column, "")).name.lower()
    return augmentation_slug == normalize_plot_group(GROUND_TRUTH_AUGMENTATION) or dataset_name.endswith("_gt.jpg")


def normalized_tokens(value: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def gender_from_row(row: dict[str, str], dataset_path_column: str) -> str | None:
    explicit = (row.get("gender") or row.get("person_gender") or row.get("subject_gender") or "").strip().lower()
    explicit = re.sub(r"[/_-]+", " ", explicit)
    explicit = re.sub(r"\s+", " ", explicit).strip()
    if explicit in {"female", "woman", "feminine"}:
        return GENDER_FEMALE
    if explicit in {"male", "man", "masculine"}:
        return GENDER_MALE

    tokens = normalized_tokens(Path(row.get(dataset_path_column, "")).stem)
    if "female" in tokens:
        return GENDER_FEMALE
    if "male" in tokens:
        return GENDER_MALE
    return None


def attire_from_dataset_path(dataset_path: str) -> str | None:
    dataset_name = Path(dataset_path).name.lower()
    if dataset_name.endswith("_01.jpg"):
        return ATTIRE_TRADITIONAL
    if dataset_name.endswith("_00.jpg"):
        return ATTIRE_WITHOUT_TRADITIONAL
    return None


def attire_from_row(row: dict[str, str], dataset_path_column: str) -> str | None:
    explicit = (row.get("clothing_type") or row.get("attire") or row.get("attire_type") or "").strip().lower()
    explicit = re.sub(r"[/_-]+", " ", explicit)
    explicit = re.sub(r"\s+", " ", explicit).strip()
    if explicit in {"traditional", "typical", "cultural", "cultural attire", "traditional attire"}:
        return ATTIRE_TRADITIONAL
    if explicit in {"non traditional", "nontraditional", "neutral", "casual", "ordinary", "whatever", "regular", "none"}:
        return ATTIRE_WITHOUT_TRADITIONAL
    return attire_from_dataset_path(row.get(dataset_path_column, ""))


def display_augmentation_name(augmentation: str) -> str:
    cleaned = re.sub(r"\s+", " ", augmentation.replace("_", " ").strip())
    if not cleaned:
        return "Unknown"
    if cleaned.lower() == "latino hispanic":
        return "Latino/Hispanic"
    return cleaned.title()


def group_specs_for_row(
    row: dict[str, str],
    augmentation_column: str = DEFAULT_AUGMENTATION_COLUMN,
    dataset_path_column: str = DEFAULT_DATASET_PATH_COLUMN,
) -> list[PredictionGroupSpec]:
    if is_background_prediction(row, augmentation_column, dataset_path_column):
        return [
            PredictionGroupSpec(
                key=BACKGROUND_GROUP_KEY,
                label="Background images",
                augmentation=GROUND_TRUTH_AUGMENTATION,
                augmentation_slug=normalize_plot_group(GROUND_TRUTH_AUGMENTATION),
                gender=None,
                attire=None,
            )
        ]

    augmentation = (row.get(augmentation_column) or "").strip()
    augmentation_slug = normalize_plot_group(augmentation)
    if not augmentation_slug:
        return []

    display_name = display_augmentation_name(augmentation)
    specs = [
        PredictionGroupSpec(
            key=f"{augmentation_slug}_all",
            label=f"{display_name} predictions",
            augmentation=augmentation,
            augmentation_slug=augmentation_slug,
            gender=None,
            attire=None,
        )
    ]

    gender = gender_from_row(row, dataset_path_column)
    if gender is not None:
        specs.append(
            PredictionGroupSpec(
                key=f"{augmentation_slug}_{gender}",
                label=f"{display_name} predictions, {gender}",
                augmentation=augmentation,
                augmentation_slug=augmentation_slug,
                gender=gender,
                attire=None,
            )
        )

    attire = attire_from_row(row, dataset_path_column)
    if attire is not None:
        attire_label = "traditional attire" if attire == ATTIRE_TRADITIONAL else "without traditional attire"
        specs.append(
            PredictionGroupSpec(
                key=f"{augmentation_slug}_{attire}",
                label=f"{display_name} predictions, {attire_label}",
                augmentation=augmentation,
                augmentation_slug=augmentation_slug,
                gender=None,
                attire=attire,
            )
        )
    return specs


def parse_prediction_coordinate(
    row: dict[str, str],
    lat_column: str,
    lon_column: str,
    stats: CoordinateStats,
) -> tuple[float, float] | None:
    return parse_coordinate_values(
        row.get(lat_column) or "",
        row.get(lon_column) or "",
        stats=stats,
    )


def uniquify_model_slug(model_name: str, used_slugs: dict[str, str]) -> str:
    base_slug = slugify(model_name, default="model")
    existing_model = used_slugs.get(base_slug)
    if existing_model is None or existing_model == model_name:
        used_slugs[base_slug] = model_name
        return base_slug

    suffix = 2
    while True:
        candidate = f"{base_slug}_{suffix}"
        existing_candidate = used_slugs.get(candidate)
        if existing_candidate is None or existing_candidate == model_name:
            used_slugs[candidate] = model_name
            return candidate
        suffix += 1


def collect_prediction_groups(
    csv_path: Path,
    *,
    model_column: str = DEFAULT_MODEL_COLUMN,
    augmentation_column: str = DEFAULT_AUGMENTATION_COLUMN,
    dataset_path_column: str = DEFAULT_DATASET_PATH_COLUMN,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    include_duplicate_rows: bool = False,
    model_names: Sequence[str] | None = None,
) -> tuple[dict[str, ModelPredictionGroups], CoordinateStats]:
    requested_models = {name.strip() for name in model_names or [] if name.strip()}
    coordinate_stats = CoordinateStats()
    models: dict[str, ModelPredictionGroups] = {}
    model_slugs_by_name: dict[str, str] = {}
    used_slugs: dict[str, str] = {}
    seen_predictions: set[tuple[str, str, str, str]] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(
            reader.fieldnames,
            [model_column, augmentation_column, dataset_path_column, lat_column, lon_column],
            csv_path,
        )
        for row in reader:
            coordinate_stats.total_rows += 1
            model_name = (row.get(model_column) or "").strip() or "unknown"
            if requested_models and model_name not in requested_models:
                continue

            model_slug = model_slugs_by_name.get(model_name)
            if model_slug is None:
                model_slug = uniquify_model_slug(model_name, used_slugs)
                model_slugs_by_name[model_name] = model_slug
                models[model_slug] = ModelPredictionGroups(model_name=model_name, model_slug=model_slug)

            model_groups = models[model_slug]
            model_groups.source_rows += 1
            coordinate = parse_prediction_coordinate(row, lat_column, lon_column, coordinate_stats)
            if coordinate is None:
                continue

            lat, lon = coordinate
            dedupe_key = (
                model_name,
                row.get(dataset_path_column, ""),
                f"{lat:.12g}",
                f"{lon:.12g}",
            )
            if not include_duplicate_rows and dedupe_key in seen_predictions:
                coordinate_stats.duplicate_rows += 1
                model_groups.duplicate_rows += 1
                continue
            seen_predictions.add(dedupe_key)

            specs = group_specs_for_row(row, augmentation_column, dataset_path_column)
            if not specs:
                continue

            coordinate_stats.plotted_rows += 1
            model_groups.accepted_rows += 1
            for spec in specs:
                group = model_groups.groups.setdefault(spec.key, PredictionGroup(spec=spec))
                group.coordinates.append(coordinate)
                model_groups.grouped_assignments += 1

    return models, coordinate_stats


def split_group_key(group_key: str) -> tuple[str, str]:
    if group_key == BACKGROUND_GROUP_KEY:
        return BACKGROUND_GROUP_KEY, "all"
    for suffix in (
        f"_{ATTIRE_WITHOUT_TRADITIONAL}",
        f"_{ATTIRE_TRADITIONAL}",
        f"_{GENDER_FEMALE}",
        f"_{GENDER_MALE}",
        "_all",
    ):
        if group_key.endswith(suffix):
            variant = suffix[1:]
            return group_key[: -len(suffix)], variant
    return group_key, "all"


def group_sort_key(group_key: str) -> tuple[int, int, str]:
    if group_key == BACKGROUND_GROUP_KEY:
        return (-1, 0, group_key)
    race_slug, variant = split_group_key(group_key)
    try:
        race_index = RACE_GROUP_ORDER.index(race_slug)
    except ValueError:
        race_index = len(RACE_GROUP_ORDER)
    return (race_index, GROUP_VARIANT_ORDER.get(variant, 99), group_key)


def ordered_groups(groups: dict[str, PredictionGroup]) -> list[PredictionGroup]:
    return [groups[key] for key in sorted(groups, key=group_sort_key)]


def model_density_vmax(
    model_groups: ModelPredictionGroups,
    *,
    bins_x: int,
    bins_y: int,
    sigma: float,
    density_clip_percentile: float,
) -> float | None:
    vmax = 0.0
    for group in model_groups.groups.values():
        histogram, _extent = build_histogram_from_coordinates(group.coordinates, bins_x=bins_x, bins_y=bins_y)
        vmax = max(vmax, compute_density_vmax(histogram, sigma=sigma, clip_percentile=density_clip_percentile))
    return vmax if vmax > 0 else None


def global_density_vmax(
    models: dict[str, ModelPredictionGroups],
    *,
    bins_x: int,
    bins_y: int,
    sigma: float,
    density_clip_percentile: float,
) -> float | None:
    vmax = 0.0
    for model_groups in models.values():
        model_vmax = model_density_vmax(
            model_groups,
            bins_x=bins_x,
            bins_y=bins_y,
            sigma=sigma,
            density_clip_percentile=density_clip_percentile,
        )
        if model_vmax is not None:
            vmax = max(vmax, model_vmax)
    return vmax if vmax > 0 else None


def render_group(
    group: PredictionGroup,
    output_base: Path,
    args: argparse.Namespace,
    themes: Sequence[str],
    boundary_path: Path | None,
    density_vmax: float | None,
) -> dict[str, RenderedHeatmap]:
    histogram, extent = build_histogram_from_coordinates(
        group.coordinates,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
    )
    return render_theme_set(
        histogram,
        extent,
        output_base,
        themes,
        dpi=args.dpi,
        sigma=args.sigma,
        boundary_path=boundary_path,
        include_colorbar=not args.no_colorbar,
        display_lat_min=args.display_lat_min,
        display_lat_max=args.display_lat_max,
        density_vmax=density_vmax,
        density_clip_percentile=args.density_clip_percentile,
        mask_fraction=args.mask_fraction,
    )


def render_model_groups(
    model_groups: ModelPredictionGroups,
    output_root: Path,
    args: argparse.Namespace,
    themes: Sequence[str],
    boundary_path: Path | None,
    density_vmax: float | None,
) -> dict[str, object]:
    model_output_dir = output_root / model_groups.model_slug
    rendered_groups: dict[str, object] = {}

    for group in ordered_groups(model_groups.groups):
        output_base = model_output_dir / f"{group.spec.key}_prediction_heatmap"
        renders = render_group(group, output_base, args, themes, boundary_path, density_vmax)
        rendered_groups[group.spec.key] = {
            "label": group.spec.label,
            "augmentation": group.spec.augmentation,
            "augmentation_slug": group.spec.augmentation_slug,
            "gender": group.spec.gender,
            "attire": group.spec.attire,
            "row_count": group.count,
            "themes": {theme: metadata_for_render(rendered) for theme, rendered in renders.items()},
        }

    return rendered_groups


def run(args: argparse.Namespace) -> PredictionHeatmapRunResult:
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input evaluation CSV does not exist: {input_csv}")
    if args.density_scale not in DENSITY_SCALE_CHOICES:
        raise ValueError(f"Unsupported density scale: {args.density_scale}")
    if not (0 < args.density_clip_percentile <= 100):
        raise ValueError("--density-clip-percentile must be in the interval (0, 100]")
    if args.mask_fraction < 0:
        raise ValueError("--mask-fraction must be non-negative")
    if args.bins_x <= 0 or args.bins_y <= 0:
        raise ValueError("--bins-x and --bins-y must be positive")
    if args.sigma < 0 or not math.isfinite(args.sigma):
        raise ValueError("--sigma must be finite and non-negative")
    validate_lat_bounds(args.display_lat_min, args.display_lat_max, "display")

    models, coordinate_stats = collect_prediction_groups(
        input_csv,
        model_column=args.model_column,
        augmentation_column=args.augmentation_column,
        dataset_path_column=args.dataset_path_column,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        include_duplicate_rows=args.include_duplicate_rows,
        model_names=args.model_name,
    )
    models = {model_slug: model_groups for model_slug, model_groups in models.items() if model_groups.groups}
    if not models:
        raise ValueError("No plottable prediction rows were found.")

    output_root = args.output_dir.expanduser().resolve() / input_csv.stem
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "prediction_heatmap_metadata.json"
    themes = resolve_themes(args.theme)
    boundary_path = None if args.no_boundaries else args.boundary_data
    boundary_path = boundary_path.expanduser().resolve() if boundary_path is not None else None

    global_vmax = None
    if args.density_scale == "global":
        global_vmax = global_density_vmax(
            models,
            bins_x=args.bins_x,
            bins_y=args.bins_y,
            sigma=args.sigma,
            density_clip_percentile=args.density_clip_percentile,
        )

    model_metadata: dict[str, object] = {}
    for model_slug, model_groups in models.items():
        density_vmax = None
        if args.density_scale == "per-model":
            density_vmax = model_density_vmax(
                model_groups,
                bins_x=args.bins_x,
                bins_y=args.bins_y,
                sigma=args.sigma,
                density_clip_percentile=args.density_clip_percentile,
            )
        elif args.density_scale == "global":
            density_vmax = global_vmax

        print(f"Rendering {model_groups.model_name}: {len(model_groups.groups)} heatmap group(s)")
        model_metadata[model_slug] = {
            "model_name": model_groups.model_name,
            "output_dir": str((output_root / model_slug).resolve()),
            "source_rows": model_groups.source_rows,
            "accepted_rows": model_groups.accepted_rows,
            "duplicate_rows": model_groups.duplicate_rows,
            "grouped_assignments": model_groups.grouped_assignments,
            "density_vmax": density_vmax,
            "groups": render_model_groups(
                model_groups,
                output_root,
                args,
                themes,
                boundary_path,
                density_vmax,
            ),
        }

    metadata: dict[str, object] = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "columns": {
            "model": args.model_column,
            "augmentation": args.augmentation_column,
            "dataset_path": args.dataset_path_column,
            "lat": args.lat_column,
            "lon": args.lon_column,
        },
        "duplicate_policy": "include" if args.include_duplicate_rows else "dedupe_by_model_path_prediction",
        "coordinate_stats": asdict(coordinate_stats),
        "render_settings": {
            "projection": "EPSG:3857 / Web Mercator",
            "display_latitude_degrees": [args.display_lat_min, args.display_lat_max],
            "bins": {"x": args.bins_x, "y": args.bins_y},
            "sigma": args.sigma,
            "dpi": args.dpi,
            "theme": args.theme,
            "themes_rendered": themes,
            "include_colorbar": not args.no_colorbar,
            "boundary_data": str(boundary_path) if boundary_path is not None else None,
        },
        "density_scale": {
            "mode": args.density_scale,
            "vmin": 0.0,
            "global_vmax": global_vmax,
            "clip_percentile": args.density_clip_percentile,
            "mask_fraction": args.mask_fraction,
        },
        "models": model_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return PredictionHeatmapRunResult(metadata_path=metadata_path.resolve(), output_root=output_root.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render prediction heatmaps grouped by model, augmentation, gender, and attire suffix.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Evaluation CSV containing prediction coordinates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for prediction heatmaps.")
    parser.add_argument("--model-column", default=DEFAULT_MODEL_COLUMN, help="Model-name column.")
    parser.add_argument("--augmentation-column", default=DEFAULT_AUGMENTATION_COLUMN, help="Augmentation/race column.")
    parser.add_argument("--dataset-path-column", default=DEFAULT_DATASET_PATH_COLUMN, help="Image path column.")
    parser.add_argument("--lat-column", default=DEFAULT_LAT_COLUMN, help="Prediction latitude column.")
    parser.add_argument("--lon-column", default=DEFAULT_LON_COLUMN, help="Prediction longitude column.")
    parser.add_argument("--model-name", action="append", default=[], help="Optional model name filter. Can be passed multiple times.")
    parser.add_argument("--include-duplicate-rows", action="store_true", help="Do not deduplicate identical model/path/prediction rows.")
    parser.add_argument("--density-scale", choices=DENSITY_SCALE_CHOICES, default=DEFAULT_DENSITY_SCALE, help="How to share heatmap color scale.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG output DPI.")
    parser.add_argument("--bins-x", type=int, default=DEFAULT_BINS_X, help="Horizontal density-bin count.")
    parser.add_argument("--bins-y", type=int, default=DEFAULT_BINS_Y, help="Vertical density-bin count.")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Gaussian smoothing sigma in density-bin pixels.")
    parser.add_argument("--boundary-data", type=Path, default=default_boundary_path(), help="Country boundary file for map context.")
    parser.add_argument("--no-boundaries", action="store_true", help="Render without country boundary polygons.")
    parser.add_argument("--theme", choices=THEME_CHOICES, default=DEFAULT_THEME, help="Render the light theme, dark theme, or both.")
    parser.add_argument(
        "--display-lat-min",
        type=float,
        default=DEFAULT_DISPLAY_LAT_MIN,
        help="Minimum latitude shown in the rendered map.",
    )
    parser.add_argument(
        "--display-lat-max",
        type=float,
        default=DEFAULT_DISPLAY_LAT_MAX,
        help="Maximum latitude shown in the rendered map.",
    )
    parser.add_argument(
        "--density-clip-percentile",
        type=float,
        default=DEFAULT_DENSITY_CLIP_PERCENTILE,
        help="Auto color-scale clipping percentile.",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=DEFAULT_MASK_FRACTION,
        help="Hide density below this fraction of the active color-scale maximum.",
    )
    parser.add_argument("--no-colorbar", action="store_true", help="Render without the density colorbar.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(f"Prediction heatmap metadata: {result.metadata_path}")
    print(f"Output root: {result.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
