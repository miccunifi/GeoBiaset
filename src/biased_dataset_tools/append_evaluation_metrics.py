from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from metrics.clip_iqa import CLIPIqaMetric
from metrics.hierarchical_popi import HierarchicalPopulationOriginPullIndex
from metrics.popi import normalize_group
from metrics.qualiclip import QualiCLIPMetric


HIERARCHICAL_POPI_COLUMNS = [
    "hpopi_group",
    "hpopi_d_true_orig",
    "hpopi_d_true_inj",
    "hpopi_d_target_orig",
    "hpopi_d_target_inj",
    "hpopi_d_true_to_target",
    "hpopi_delta_target_km",
    "hpopi_delta_true_km",
    "hpopi_steering_rate",
    "hpopi_closest_group_orig",
    "hpopi_closest_group_inj",
    "hpopi_dist_to_closest_orig",
    "hpopi_dist_to_closest_inj",
    "hpopi_target_rank_orig",
    "hpopi_target_rank_inj",
    "hpopi_delta_rank",
    "hpopi_tree_dist_orig",
    "hpopi_tree_dist_inj",
    "hpopi_delta_tree_hops",
    "hpopi_moved_toward_target",
    "hpopi_became_less_accurate",
    "hpopi_targeted_success",
    "hpopi_target_hit_orig",
    "hpopi_target_hit_inj",
    "hpopi_target_region_capture",
    "hpopi_rank_capture",
    "hpopi_topological_improvement",
    "hpopi_orig_correct",
    "hpopi_inj_wrong",
    "hpopi_targeted_flip",
    "hpopi_error",
]

HIERARCHICAL_FIELD_TO_COLUMN = {
    "group": "hpopi_group",
    "d_true_orig": "hpopi_d_true_orig",
    "d_true_inj": "hpopi_d_true_inj",
    "d_target_orig": "hpopi_d_target_orig",
    "d_target_inj": "hpopi_d_target_inj",
    "d_true_to_target": "hpopi_d_true_to_target",
    "delta_target_km": "hpopi_delta_target_km",
    "delta_true_km": "hpopi_delta_true_km",
    "steering_rate": "hpopi_steering_rate",
    "closest_group_orig": "hpopi_closest_group_orig",
    "closest_group_inj": "hpopi_closest_group_inj",
    "dist_to_closest_orig": "hpopi_dist_to_closest_orig",
    "dist_to_closest_inj": "hpopi_dist_to_closest_inj",
    "target_rank_orig": "hpopi_target_rank_orig",
    "target_rank_inj": "hpopi_target_rank_inj",
    "delta_rank": "hpopi_delta_rank",
    "tree_dist_orig": "hpopi_tree_dist_orig",
    "tree_dist_inj": "hpopi_tree_dist_inj",
    "delta_tree_hops": "hpopi_delta_tree_hops",
    "moved_toward_target": "hpopi_moved_toward_target",
    "became_less_accurate": "hpopi_became_less_accurate",
    "targeted_success": "hpopi_targeted_success",
    "target_hit_orig": "hpopi_target_hit_orig",
    "target_hit_inj": "hpopi_target_hit_inj",
    "target_region_capture": "hpopi_target_region_capture",
    "rank_capture": "hpopi_rank_capture",
    "topological_improvement": "hpopi_topological_improvement",
    "orig_correct": "hpopi_orig_correct",
    "inj_wrong": "hpopi_inj_wrong",
    "targeted_flip": "hpopi_targeted_flip",
}

IQA_COLUMNS = ["qualiclip", "qualiclip_error", "clip_iqa", "clip_iqa_error"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append POPI, QualiCLIP, and CLIP-IQA columns to an evaluation CSV."
    )
    parser.add_argument("csv_file", type=Path, help="Evaluation CSV to process.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV path. Defaults to '<input>_with_extra_metrics.csv' unless --in-place is used.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input CSV atomically.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute columns that already have values.")
    parser.add_argument(
        "--image-column",
        default="dataset_path",
        help="CSV column containing the image path used for QualiCLIP and CLIP-IQA.",
    )
    parser.add_argument(
        "--boundaries-data",
        type=Path,
        help="Optional Natural Earth-style country boundaries file for POPI region polygons.",
    )
    parser.add_argument("--strict-iso", action="store_true", help="Fail if region mappings contain missing ISO_A3 codes.")
    parser.add_argument("--wrong-threshold-km", type=float, default=100.0)
    parser.add_argument("--device", help="Device forwarded to pyiqa, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--skip-hierarchical-popi", action="store_true")
    parser.add_argument("--skip-qualiclip", action="store_true")
    parser.add_argument("--skip-clipiqa", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--allow-missing-iqa",
        action="store_true",
        help="If pyiqa is not installed, leave IQA columns blank instead of failing.",
    )
    return parser.parse_args(argv)


def progress_bar(
    iterable: Any,
    *,
    description: str,
    unit: str,
    total: int | None = None,
    show_progress: bool = True,
) -> Any:
    if not show_progress:
        return iterable
    return tqdm(iterable, desc=description, unit=unit, total=total, disable=None)


def text(value: Any) -> str:
    return str(value or "").strip()


def is_truth_row(row: dict[str, str]) -> bool:
    return text(row.get("augmentation")).lower().replace("_", " ") in {
        "ground truth",
        "groundtruth",
        "gt",
        "original",
        "clean",
        "source",
    }


def is_usable_status(row: dict[str, str]) -> bool:
    status = text(row.get("status")).lower()
    return not status or status == "ok"


def parse_float(value: Any) -> float | None:
    value_text = text(value)
    if not value_text:
        return None
    try:
        return float(value_text)
    except ValueError:
        return None


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def output_path_for(input_path: Path, output_path: Path | None, in_place: bool) -> Path:
    resolved_input = input_path.expanduser().resolve()
    if in_place:
        return resolved_input
    if output_path is not None:
        return output_path.expanduser().resolve()
    return resolved_input.with_name(f"{resolved_input.stem}_with_extra_metrics{resolved_input.suffix}")


def read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader.fieldnames), list(reader)


def write_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, csv_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def append_missing_columns(fieldnames: list[str], columns: list[str]) -> list[str]:
    updated = list(fieldnames)
    for column in columns:
        if column not in updated:
            updated.append(column)
    return updated


def row_needs_columns(row: dict[str, str], columns: list[str], overwrite: bool) -> bool:
    return overwrite or any(not text(row.get(column)) for column in columns)


def ground_truth_lookup_key(row: dict[str, str]) -> tuple[str, str]:
    return (text(row.get("model_name")), text(row.get("source_path")))


def build_ground_truth_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if not is_truth_row(row) or not is_usable_status(row):
            continue
        key = ground_truth_lookup_key(row)
        if key[1] and key not in lookup:
            lookup[key] = row
    return lookup


def append_hierarchical_popi(
    rows: list[dict[str, str]],
    *,
    boundaries_path: Path | None,
    strict_iso: bool,
    wrong_threshold_km: float,
    overwrite: bool,
    show_progress: bool,
) -> dict[str, int]:
    metric = HierarchicalPopulationOriginPullIndex(
        boundaries_path=boundaries_path,
        strict_iso=strict_iso,
        wrong_threshold_km=wrong_threshold_km,
    )
    ground_truth_by_key = build_ground_truth_lookup(rows)
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    counts = {"computed": 0, "skipped": 0, "errors": 0}

    for row in progress_bar(
        rows,
        description="POPI",
        unit="row",
        total=len(rows),
        show_progress=show_progress,
    ):
        if is_truth_row(row) or normalize_group(row.get("augmentation")) is None:
            counts["skipped"] += 1
            continue
        if not row_needs_columns(row, HIERARCHICAL_POPI_COLUMNS, overwrite):
            counts["skipped"] += 1
            continue
        if not is_usable_status(row):
            row["hpopi_error"] = "row status is not ok"
            counts["errors"] += 1
            continue

        truth_row = ground_truth_by_key.get(ground_truth_lookup_key(row))
        if truth_row is None:
            row["hpopi_error"] = "missing matching ground truth row"
            counts["errors"] += 1
            continue

        gt_lat = parse_float(row.get("gt_lat"))
        gt_lon = parse_float(row.get("gt_lon"))
        pred_inj_lat = parse_float(row.get("pred_lat"))
        pred_inj_lon = parse_float(row.get("pred_lon"))
        pred_orig_lat = parse_float(truth_row.get("pred_lat"))
        pred_orig_lon = parse_float(truth_row.get("pred_lon"))
        if None in (gt_lat, gt_lon, pred_inj_lat, pred_inj_lon, pred_orig_lat, pred_orig_lon):
            row["hpopi_error"] = "missing numeric coordinates"
            counts["errors"] += 1
            continue

        cache_key = (
            normalize_group(row.get("augmentation")),
            gt_lat,
            gt_lon,
            pred_orig_lat,
            pred_orig_lon,
            pred_inj_lat,
            pred_inj_lon,
        )
        try:
            result = cache.get(cache_key)
            if result is None:
                result = metric.compute_sample(
                    {
                        "sample_id": row.get("dataset_path"),
                        "augmentation": row.get("augmentation"),
                        "gt_lat": gt_lat,
                        "gt_lon": gt_lon,
                        "clean_pred_lat": pred_orig_lat,
                        "clean_pred_lon": pred_orig_lon,
                        "pred_lat": pred_inj_lat,
                        "pred_lon": pred_inj_lon,
                    }
                )
                cache[cache_key] = result
            for source_field, output_column in HIERARCHICAL_FIELD_TO_COLUMN.items():
                row[output_column] = format_metric_value(result.get(source_field))
            row["hpopi_error"] = ""
            counts["computed"] += 1
        except Exception as exc:
            row["hpopi_error"] = str(exc)
            counts["errors"] += 1

    return counts


def resolve_image_path(raw_path: str, csv_path: Path) -> Path | None:
    if not text(raw_path):
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    csv_candidate = (csv_path.parent / candidate).resolve()
    if csv_candidate.exists():
        return csv_candidate
    return None


def score_single_image(metric: Any, image_path: Path) -> float:
    score = metric(str(image_path))
    if hasattr(score, "item"):
        score = score.item()
    return float(score)


def append_iqa_scores(
    rows: list[dict[str, str]],
    *,
    csv_path: Path,
    image_column: str,
    output_column: str,
    error_column: str,
    metric_class: type[QualiCLIPMetric],
    device: str | None,
    overwrite: bool,
    allow_missing_iqa: bool,
    show_progress: bool,
) -> dict[str, int]:
    counts = {"computed": 0, "skipped": 0, "errors": 0}
    needed_paths: dict[str, Path | None] = {}
    for row in rows:
        if not row_needs_columns(row, [output_column], overwrite):
            counts["skipped"] += 1
            continue
        raw_path = text(row.get(image_column))
        if raw_path not in needed_paths:
            needed_paths[raw_path] = resolve_image_path(raw_path, csv_path)

    if not needed_paths:
        return counts

    metric_wrapper = metric_class(device=device)
    try:
        metric = metric_wrapper._load_metric()
    except ModuleNotFoundError as exc:
        if allow_missing_iqa:
            message = str(exc)
            for row in rows:
                if row_needs_columns(row, [output_column], overwrite):
                    row[output_column] = ""
                    row[error_column] = message
                    counts["errors"] += 1
            return counts
        raise

    score_cache: dict[str, tuple[str, str]] = {}
    for raw_path, image_path in progress_bar(
        needed_paths.items(),
        description=f"{output_column} scoring",
        unit="image",
        total=len(needed_paths),
        show_progress=show_progress,
    ):
        if image_path is None:
            score_cache[raw_path] = ("", "image path not found")
            continue
        try:
            score_cache[raw_path] = (format_metric_value(score_single_image(metric, image_path)), "")
        except Exception as exc:
            score_cache[raw_path] = ("", str(exc))

    for row in rows:
        if not row_needs_columns(row, [output_column], overwrite):
            continue
        score, error = score_cache.get(text(row.get(image_column)), ("", "image path not found"))
        row[output_column] = score
        row[error_column] = error
        if error:
            counts["errors"] += 1
        else:
            counts["computed"] += 1
    return counts


def process_csv(
    input_path: Path,
    output_path: Path,
    *,
    image_column: str = "dataset_path",
    boundaries_path: Path | None = None,
    strict_iso: bool = False,
    wrong_threshold_km: float = 100.0,
    device: str | None = None,
    overwrite: bool = False,
    skip_hierarchical_popi: bool = False,
    skip_qualiclip: bool = False,
    skip_clipiqa: bool = False,
    allow_missing_iqa: bool = False,
    show_progress: bool = True,
) -> dict[str, dict[str, int]]:
    input_path = input_path.expanduser().resolve()
    fieldnames, rows = read_csv(input_path)
    if image_column not in fieldnames:
        raise ValueError(f"Image column '{image_column}' was not found in {input_path}")

    if not skip_hierarchical_popi:
        fieldnames = append_missing_columns(fieldnames, HIERARCHICAL_POPI_COLUMNS)
    if not skip_qualiclip:
        fieldnames = append_missing_columns(fieldnames, ["qualiclip", "qualiclip_error"])
    if not skip_clipiqa:
        fieldnames = append_missing_columns(fieldnames, ["clip_iqa", "clip_iqa_error"])

    summaries: dict[str, dict[str, int]] = {}
    if not skip_hierarchical_popi:
        summaries["hierarchical_popi"] = append_hierarchical_popi(
            rows,
            boundaries_path=boundaries_path,
            strict_iso=strict_iso,
            wrong_threshold_km=wrong_threshold_km,
            overwrite=overwrite,
            show_progress=show_progress,
        )
    if not skip_qualiclip:
        summaries["qualiclip"] = append_iqa_scores(
            rows,
            csv_path=input_path,
            image_column=image_column,
            output_column="qualiclip",
            error_column="qualiclip_error",
            metric_class=QualiCLIPMetric,
            device=device,
            overwrite=overwrite,
            allow_missing_iqa=allow_missing_iqa,
            show_progress=show_progress,
        )
    if not skip_clipiqa:
        summaries["clip_iqa"] = append_iqa_scores(
            rows,
            csv_path=input_path,
            image_column=image_column,
            output_column="clip_iqa",
            error_column="clip_iqa_error",
            metric_class=CLIPIqaMetric,
            device=device,
            overwrite=overwrite,
            allow_missing_iqa=allow_missing_iqa,
            show_progress=show_progress,
        )

    write_csv(output_path, fieldnames, rows)
    return summaries


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(str(args.csv_file).removeprefix("@"))
    output_path = output_path_for(input_path, args.output, args.in_place)
    summaries = process_csv(
        input_path,
        output_path,
        image_column=args.image_column,
        boundaries_path=args.boundaries_data,
        strict_iso=args.strict_iso,
        wrong_threshold_km=args.wrong_threshold_km,
        device=args.device,
        overwrite=args.overwrite,
        skip_hierarchical_popi=args.skip_hierarchical_popi,
        skip_qualiclip=args.skip_qualiclip,
        skip_clipiqa=args.skip_clipiqa,
        allow_missing_iqa=args.allow_missing_iqa,
        show_progress=not args.no_progress,
    )
    print(f"Wrote enriched CSV to {output_path}")
    for metric_name, counts in summaries.items():
        print(
            f"{metric_name}: "
            f"{counts['computed']} computed, {counts['skipped']} skipped, {counts['errors']} errors"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
