from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from biased_dataset_tools.clip_attention import (
    CLIP_ATTENTION_METHOD_NAME,
    CLIP_ATTENTION_METHOD_REFERENCE,
    GeoCLIPAttentionExplainer,
    attention_output_paths,
    save_attention_images,
)
from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION
from biased_dataset_tools.gt_heatmap import CoordinateStats, parse_coordinate_values
from biased_dataset_tools.geoloc_shared import resolve_runtime_device


DEFAULT_INPUT_CSV = Path("dataset/img2loc_g3__geoclip_evaluation_results_with_extra_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("figures/geoclip_xai")
DEFAULT_MODEL_COLUMN = "model_name"
DEFAULT_DATASET_PATH_COLUMN = "dataset_path"
DEFAULT_SOURCE_PATH_COLUMN = "source_path"
DEFAULT_CATEGORY_COLUMN = "category"
DEFAULT_AUGMENTATION_COLUMN = "augmentation"
DEFAULT_LAT_COLUMN = "pred_lat"
DEFAULT_LON_COLUMN = "pred_lon"
DEFAULT_STATUS_COLUMN = "status"
GEOCLIP_MODEL_NAME = "geoclip"
INDEX_COLUMNS = [
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
]


@dataclass
class GeoCLIPXaiStats:
    total_rows: int = 0
    model_rows: int = 0
    status_skipped_rows: int = 0
    duplicate_rows: int = 0
    missing_image_rows: int = 0
    selected_rows: int = 0
    background_rows_selected: int = 0
    background_rows_added: int = 0
    explained_rows: int = 0
    skipped_existing_rows: int = 0


@dataclass(frozen=True)
class GeoCLIPXaiRunResult:
    output_root: Path
    attention_dir: Path
    attention_index_csv: Path
    metadata_path: Path
    explained_rows: int
    skipped_existing_rows: int


def validate_columns(fieldnames: Sequence[str] | None, required_columns: Sequence[str], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {csv_path}")
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")


def is_geoclip_row(row: dict[str, str], model_column: str) -> bool:
    return (row.get(model_column) or "").strip().lower() == GEOCLIP_MODEL_NAME


def has_ok_status(row: dict[str, str], fieldnames: Sequence[str], status_column: str) -> bool:
    if status_column not in fieldnames:
        return True
    return (row.get(status_column) or "").strip().lower() == "ok"


def is_background_row(row: dict[str, str], augmentation_column: str) -> bool:
    return (row.get(augmentation_column) or "").strip().lower() == GROUND_TRUTH_AUGMENTATION


def normalize_xai_row(
    row: dict[str, str],
    *,
    dataset_path_column: str,
    source_path_column: str,
    category_column: str,
    augmentation_column: str,
    lat: float,
    lon: float,
) -> dict[str, str]:
    normalized = dict(row)
    normalized["dataset_path"] = row.get(dataset_path_column, "")
    normalized["source_path"] = row.get(source_path_column, "")
    normalized["category"] = row.get(category_column, "")
    normalized["augmentation"] = row.get(augmentation_column, "")
    normalized["pred_lat"] = f"{lat:.12g}"
    normalized["pred_lon"] = f"{lon:.12g}"
    return normalized


def resolve_image_path(dataset_path: str, input_csv: Path, dataset_root: Path | None = None) -> Path | None:
    path = Path(dataset_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(input_csv.parent / path)
        if dataset_root is not None:
            candidates.append(dataset_root / path)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return None


def append_selected_row(
    rows: list[dict[str, str]],
    row: dict[str, str],
    *,
    selected_keys: set[tuple[str, str, str]],
    coordinate_stats: CoordinateStats,
    stats: GeoCLIPXaiStats,
    is_added_background: bool = False,
) -> bool:
    selection_key = (row["dataset_path"], row["pred_lat"], row["pred_lon"])
    if selection_key in selected_keys:
        return False
    selected_keys.add(selection_key)
    rows.append(row)
    coordinate_stats.plotted_rows += 1
    stats.selected_rows += 1
    if is_background_row(row, "augmentation"):
        stats.background_rows_selected += 1
    if is_added_background:
        stats.background_rows_added += 1
    return True


def select_geoclip_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], GeoCLIPXaiStats, CoordinateStats]:
    input_csv = args.input_csv.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root is not None else None
    candidate_rows: list[dict[str, str]] = []
    stats = GeoCLIPXaiStats()
    coordinate_stats = CoordinateStats()
    seen_predictions: set[tuple[str, str, str]] = set()
    background_rows_by_source: dict[str, list[dict[str, str]]] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        validate_columns(
            reader.fieldnames,
            [args.model_column, args.dataset_path_column, args.lat_column, args.lon_column],
            input_csv,
        )
        for row in reader:
            stats.total_rows += 1
            coordinate_stats.total_rows += 1
            if not is_geoclip_row(row, args.model_column):
                continue
            stats.model_rows += 1
            if not has_ok_status(row, fieldnames, args.status_column):
                stats.status_skipped_rows += 1
                continue

            coordinate = parse_coordinate_values(
                row.get(args.lat_column) or "",
                row.get(args.lon_column) or "",
                stats=coordinate_stats,
            )
            if coordinate is None:
                continue

            lat, lon = coordinate
            dataset_path = row.get(args.dataset_path_column, "")
            dedupe_key = (dataset_path, f"{lat:.12g}", f"{lon:.12g}")
            if not args.include_duplicate_rows and dedupe_key in seen_predictions:
                stats.duplicate_rows += 1
                coordinate_stats.duplicate_rows += 1
                continue
            seen_predictions.add(dedupe_key)

            image_path = resolve_image_path(dataset_path, input_csv, dataset_root=dataset_root)
            if image_path is None:
                stats.missing_image_rows += 1
                continue

            normalized = normalize_xai_row(
                row,
                dataset_path_column=args.dataset_path_column,
                source_path_column=args.source_path_column,
                category_column=args.category_column,
                augmentation_column=args.augmentation_column,
                lat=lat,
                lon=lon,
            )
            normalized["_resolved_image_path"] = str(image_path)
            candidate_rows.append(normalized)
            if is_background_row(row, args.augmentation_column):
                background_rows_by_source.setdefault(normalized["source_path"], []).append(normalized)
            if not args.include_background_images and args.limit is not None and len(candidate_rows) >= args.limit:
                break

    rows: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    limited_rows = candidate_rows[: args.limit] if args.limit is not None else candidate_rows
    for row in limited_rows:
        append_selected_row(
            rows,
            row,
            selected_keys=selected_keys,
            coordinate_stats=coordinate_stats,
            stats=stats,
        )

    if args.include_background_images and args.limit is not None:
        selected_source_paths: list[str] = []
        seen_source_paths: set[str] = set()
        for row in rows:
            source_path = row.get("source_path")
            if source_path and source_path not in seen_source_paths:
                seen_source_paths.add(source_path)
                selected_source_paths.append(source_path)
        for source_path in selected_source_paths:
            for background_row in background_rows_by_source.get(source_path, []):
                append_selected_row(
                    rows,
                    background_row,
                    selected_keys=selected_keys,
                    coordinate_stats=coordinate_stats,
                    stats=stats,
                    is_added_background=True,
                )

    return rows, stats, coordinate_stats


def write_attention_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(
    args: argparse.Namespace,
    explainer_factory: Callable[[str], Any] | None = None,
) -> GeoCLIPXaiRunResult:
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input evaluation CSV does not exist: {input_csv}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    output_root = args.output_dir.expanduser().resolve() / input_csv.stem
    attention_dir = output_root / "attention_maps"
    attention_index_csv = output_root / "attention_index.csv"
    metadata_path = output_root / "geoclip_xai_metadata.json"
    attention_dir.mkdir(parents=True, exist_ok=True)

    rows, stats, coordinate_stats = select_geoclip_rows(args)
    index_rows: list[dict[str, str]] = []
    pending_rows: list[dict[str, str]] = []

    print(
        "Selected "
        f"{stats.selected_rows} GeoCLIP row(s), including "
        f"{stats.background_rows_selected} background image row(s)"
        f" ({stats.background_rows_added} added beyond --limit)."
    )

    for row in rows:
        overlay_path, heatmap_path = attention_output_paths(attention_dir, row["dataset_path"])
        if args.skip_existing and overlay_path.exists() and heatmap_path.exists():
            stats.skipped_existing_rows += 1
            index_rows.append(index_row_for(row, overlay_path, heatmap_path))
            continue
        pending_rows.append(row)

    explainer = None
    if pending_rows:
        device = resolve_runtime_device(args.device)
        factory = explainer_factory or (lambda resolved_device: GeoCLIPAttentionExplainer(resolved_device))
        explainer = factory(device)

    for index, row in enumerate(pending_rows, start=1):
        assert explainer is not None
        image_path = Path(row["_resolved_image_path"])
        image_rgb, heatmap = explainer.explain(image_path, row)
        overlay_path, heatmap_path = save_attention_images(attention_dir, row["dataset_path"], image_rgb, heatmap)
        index_rows.append(index_row_for(row, overlay_path, heatmap_path))
        stats.explained_rows += 1
        augmentation = row.get("augmentation") or "unknown"
        print(f"  [{index}/{len(pending_rows)}] GeoCLIP XAI [{augmentation}] for {row['dataset_path']}")

    write_attention_index(attention_index_csv, index_rows)
    metadata = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "attention_dir": str(attention_dir),
        "attention_index_csv": str(attention_index_csv),
        "attention_method": CLIP_ATTENTION_METHOD_NAME,
        "attention_reference": CLIP_ATTENTION_METHOD_REFERENCE,
        "device": args.device,
        "limit": args.limit,
        "skip_existing": args.skip_existing,
        "include_background_images": args.include_background_images,
        "include_duplicate_rows": args.include_duplicate_rows,
        "stats": asdict(stats),
        "coordinate_stats": asdict(coordinate_stats),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return GeoCLIPXaiRunResult(
        output_root=output_root.resolve(),
        attention_dir=attention_dir.resolve(),
        attention_index_csv=attention_index_csv.resolve(),
        metadata_path=metadata_path.resolve(),
        explained_rows=stats.explained_rows,
        skipped_existing_rows=stats.skipped_existing_rows,
    )


def index_row_for(row: dict[str, str], overlay_path: Path, heatmap_path: Path) -> dict[str, str]:
    return {
        "dataset_path": row.get("dataset_path", ""),
        "source_path": row.get("source_path", ""),
        "category": row.get("category", ""),
        "augmentation": row.get("augmentation", ""),
        "pred_lat": row.get("pred_lat", ""),
        "pred_lon": row.get("pred_lon", ""),
        "attention_method": CLIP_ATTENTION_METHOD_NAME,
        "attention_reference": CLIP_ATTENTION_METHOD_REFERENCE,
        "overlay_path": str(overlay_path),
        "heatmap_path": str(heatmap_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GeoCLIP gradient attention rollout overlays from an evaluation CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Evaluation CSV containing GeoCLIP prediction rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for GeoCLIP XAI artifacts.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional root used to resolve relative dataset paths.")
    parser.add_argument("--device", default="auto", help="Execution device. Use 'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of unique GeoCLIP rows to explain.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing overlay and heatmap files when both are present.")
    parser.add_argument(
        "--include-background-images",
        dest="include_background_images",
        action="store_true",
        help="With --limit, also include matching ground-truth/background rows for selected source images. This is the default.",
    )
    parser.add_argument(
        "--no-background-images",
        dest="include_background_images",
        action="store_false",
        help="With --limit, do not also include matching ground-truth/background rows for selected source images.",
    )
    parser.add_argument("--include-duplicate-rows", action="store_true", help="Do not deduplicate identical image/prediction rows.")
    parser.add_argument("--model-column", default=DEFAULT_MODEL_COLUMN, help="Model-name column.")
    parser.add_argument("--dataset-path-column", default=DEFAULT_DATASET_PATH_COLUMN, help="Image path column.")
    parser.add_argument("--source-path-column", default=DEFAULT_SOURCE_PATH_COLUMN, help="Source path column.")
    parser.add_argument("--category-column", default=DEFAULT_CATEGORY_COLUMN, help="Category column.")
    parser.add_argument("--augmentation-column", default=DEFAULT_AUGMENTATION_COLUMN, help="Augmentation/race column.")
    parser.add_argument("--lat-column", default=DEFAULT_LAT_COLUMN, help="Prediction latitude column.")
    parser.add_argument("--lon-column", default=DEFAULT_LON_COLUMN, help="Prediction longitude column.")
    parser.add_argument("--status-column", default=DEFAULT_STATUS_COLUMN, help="Status column used to keep only 'ok' rows when present.")
    parser.set_defaults(include_background_images=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(f"GeoCLIP XAI attention index: {result.attention_index_csv}")
    print(f"GeoCLIP XAI metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
