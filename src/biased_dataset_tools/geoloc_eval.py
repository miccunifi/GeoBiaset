from __future__ import annotations

import argparse
from pathlib import Path

from biased_dataset_tools.classifier import (
    GROUND_TRUTH_AUGMENTATION,
    load_config,
    read_manifest_entries,
    resolve_dataset_path,
)
from biased_dataset_tools.geoloc_methods import (
    DEFAULT_G3_ROOT,
    DEFAULT_IMG2LOC_EMBEDDINGS,
    DEFAULT_IMG2LOC_ROOT,
    load_predictor,
)
from models.base import DEFAULT_VLM_MAX_NEW_TOKENS
from biased_dataset_tools.geoloc_reporting import (
    generate_map_from_image_metrics_csv,
    load_country_lookup,
    safe_country_lookup,
    summarize_accuracy_rows,
    summarize_metric_rows,
    write_prediction_map_html,
)
from biased_dataset_tools.geoloc_shared import (
    ACCURACY_THRESHOLDS_KM,
    EVAL_METHOD_CHOICES,
    IMAGE_METRIC_COLUMNS,
    OVERALL_METRIC_COLUMNS,
    build_threshold_hits,
    derive_output_paths,
    format_float,
    format_optional_text,
    format_same_country,
    haversine_km,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a geolocation method on the dataset and compute per-image and aggregate metrics."
    )
    parser.add_argument(
        "--method",
        choices=EVAL_METHOD_CHOICES,
        default="geoclip",
        help="Prediction method to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV override. Defaults to the manifest in config.yaml.",
    )
    parser.add_argument(
        "--image-metrics-csv",
        type=Path,
        default=None,
        help="Optional output CSV for per-image metrics.",
    )
    parser.add_argument(
        "--overall-metrics-csv",
        type=Path,
        default=None,
        help="Optional output CSV for aggregate metrics.",
    )
    parser.add_argument(
        "--map-html",
        type=Path,
        default=None,
        help="Optional output HTML file for the prediction world-map plot.",
    )
    parser.add_argument(
        "--map-from-image-metrics-csv",
        type=Path,
        default=None,
        help=(
            "Regenerate the map from an existing per-image metrics CSV instead of rerunning inference. "
            "The CSV is also rewritten with country columns when country boundaries are available."
        ),
    )
    parser.add_argument(
        "--countries-data",
        type=Path,
        default=None,
        help=(
            "Optional local country boundary file for country-match metrics and map marker shapes. "
            "If omitted, common data/*countries* Natural Earth paths are tried."
        ),
    )
    parser.add_argument(
        "--country-name-column",
        default=None,
        help="Optional country-name field in the boundary dataset. If omitted, common fields are auto-detected.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device. Use 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--img2loc-root",
        type=Path,
        default=DEFAULT_IMG2LOC_ROOT,
        help="Path to the local Img2Loc repository.",
    )
    parser.add_argument(
        "--img2loc-index",
        type=Path,
        default=None,
        help="Optional Img2Loc FAISS index override.",
    )
    parser.add_argument(
        "--img2loc-embeddings-with-coords",
        type=Path,
        default=DEFAULT_IMG2LOC_EMBEDDINGS,
        help="NumPy array containing retrieval embeddings with latitude/longitude in the last two values.",
    )
    parser.add_argument(
        "--img2loc-search-results-csv",
        type=Path,
        default=None,
        help="Optional precomputed Img2Loc search results CSV for source-image cache hits.",
    )
    parser.add_argument(
        "--g3-root",
        type=Path,
        default=DEFAULT_G3_ROOT,
        help="Path to the local G3 repository.",
    )
    parser.add_argument(
        "--g3-index",
        type=Path,
        default=None,
        help="Optional G3 FAISS index override.",
    )
    parser.add_argument(
        "--g3-checkpoint",
        type=Path,
        default=None,
        help="Optional G3 checkpoint override.",
    )
    parser.add_argument(
        "--g3-database-csv",
        type=Path,
        default=None,
        help="Optional G3 retrieval database CSV override.",
    )
    parser.add_argument(
        "--g3-results-csv",
        type=Path,
        default=None,
        help="Optional precomputed G3 results CSV for source-image cache hits.",
    )
    parser.add_argument(
        "--vlm-model-id",
        default=None,
        help="Optional Hugging Face model id override for Qwen VLM methods.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=DEFAULT_VLM_MAX_NEW_TOKENS,
        help="Maximum new tokens for Qwen VLM generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.map_from_image_metrics_csv is not None:
        return generate_map_from_image_metrics_csv(args)

    _, dataset_root, config_manifest_path = load_config(args.config)
    manifest_path = (args.manifest or config_manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")

    predictor, method_label, method_slug_value = load_predictor(args)
    image_metrics_csv, overall_metrics_csv, map_html = derive_output_paths(
        manifest_path=manifest_path,
        image_metrics_csv=args.image_metrics_csv,
        overall_metrics_csv=args.overall_metrics_csv,
        map_html=args.map_html,
        method_slug=method_slug_value,
    )

    manifest_entries = read_manifest_entries(manifest_path)
    if args.limit is not None:
        manifest_entries = manifest_entries[: args.limit]
    if not manifest_entries:
        raise ValueError(f"No manifest rows found in {manifest_path}")

    country_lookup = load_country_lookup(args.countries_data, args.country_name_column)

    print(f"Method: {method_label}")
    print(f"Manifest CSV: {manifest_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Per-image metrics CSV: {image_metrics_csv}")
    print(f"Overall metrics CSV: {overall_metrics_csv}")
    print(f"Prediction map HTML: {map_html}")
    print(f"Rows to evaluate: {len(manifest_entries)}")

    predictions_by_dataset_path = {}
    image_metric_rows: list[dict[str, str]] = []

    gt_error_all: list[float] = []
    gt_error_ground_truth: list[float] = []
    gt_error_augmented: list[float] = []
    gt_error_by_category: dict[str, list[float]] = {}
    gt_error_by_augmentation: dict[str, list[float]] = {}

    clean_prediction_gap_all: list[float] = []
    clean_prediction_gap_by_category: dict[str, list[float]] = {}
    clean_prediction_gap_by_augmentation: dict[str, list[float]] = {}

    clean_entries_by_source_path = {
        entry.source_path: entry
        for entry in manifest_entries
        if entry.augmentation == GROUND_TRUTH_AUGMENTATION
    }
    has_ground_truth_rows = bool(clean_entries_by_source_path)

    prediction_entries = []
    prediction_paths = []
    for entry in manifest_entries:
        image_path = resolve_dataset_path(entry.dataset_path, dataset_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Dataset image does not exist: {image_path}")
        prediction_entries.append(entry)
        prediction_paths.append(image_path)

    if getattr(predictor, "prefer_full_batch", False):
        batch_predictions = predictor.batch_predict(prediction_entries, prediction_paths)
        if len(batch_predictions) != len(prediction_entries):
            raise RuntimeError(
                f"Batch prediction returned {len(batch_predictions)} outputs for {len(prediction_entries)} inputs."
            )
        for index, (entry, prediction) in enumerate(zip(prediction_entries, batch_predictions), start=1):
            predictions_by_dataset_path[entry.dataset_path] = prediction
            print(f"[{index}/{len(manifest_entries)}] predicted {entry.dataset_path}")
    else:
        for index, (entry, image_path) in enumerate(zip(prediction_entries, prediction_paths), start=1):
            predictions_by_dataset_path[entry.dataset_path] = predictor(entry, image_path)
            print(f"[{index}/{len(manifest_entries)}] predicted {entry.dataset_path}")

    for entry in manifest_entries:
        prediction = predictions_by_dataset_path[entry.dataset_path]
        gt_lat = float(entry.lat)
        gt_lon = float(entry.lon)
        gt_error_km = haversine_km(gt_lat, gt_lon, prediction.pred_lat, prediction.pred_lon)
        gt_country = safe_country_lookup(country_lookup, gt_lat, gt_lon)
        pred_country = safe_country_lookup(country_lookup, prediction.pred_lat, prediction.pred_lon)
        same_country_as_gt: bool | None = None
        if country_lookup is not None:
            same_country_as_gt = gt_country is not None and pred_country is not None and gt_country == pred_country

        gt_error_all.append(gt_error_km)
        gt_error_by_category.setdefault(entry.category, []).append(gt_error_km)
        gt_error_by_augmentation.setdefault(entry.augmentation, []).append(gt_error_km)
        if has_ground_truth_rows and entry.augmentation == GROUND_TRUTH_AUGMENTATION:
            gt_error_ground_truth.append(gt_error_km)
        elif has_ground_truth_rows:
            gt_error_augmented.append(gt_error_km)

        clean_entry = clean_entries_by_source_path.get(entry.source_path)
        clean_dataset_path = clean_entry.dataset_path if clean_entry is not None else ""
        clean_pred_lat: float | None = None
        clean_pred_lon: float | None = None
        clean_to_aug_prediction_km: float | None = None

        if clean_entry is not None:
            clean_prediction = predictions_by_dataset_path.get(clean_entry.dataset_path)
            if clean_prediction is not None:
                clean_pred_lat = clean_prediction.pred_lat
                clean_pred_lon = clean_prediction.pred_lon
                if entry.augmentation != GROUND_TRUTH_AUGMENTATION:
                    clean_to_aug_prediction_km = haversine_km(
                        clean_prediction.pred_lat,
                        clean_prediction.pred_lon,
                        prediction.pred_lat,
                        prediction.pred_lon,
                    )
                    clean_prediction_gap_all.append(clean_to_aug_prediction_km)
                    clean_prediction_gap_by_category.setdefault(entry.category, []).append(
                        clean_to_aug_prediction_km
                    )
                    clean_prediction_gap_by_augmentation.setdefault(entry.augmentation, []).append(
                        clean_to_aug_prediction_km
                    )

        threshold_hits = build_threshold_hits(gt_error_km)
        image_metric_rows.append(
            {
                "source_path": entry.source_path,
                "dataset_path": entry.dataset_path,
                "category": entry.category,
                "augmentation": entry.augmentation,
                "group_source_path": entry.source_path,
                "clean_dataset_path": clean_dataset_path,
                "gt_lat": entry.lat,
                "gt_lon": entry.lon,
                "pred_lat": format_float(prediction.pred_lat),
                "pred_lon": format_float(prediction.pred_lon),
                "pred_prob": format_float(prediction.pred_prob),
                "haversine_to_gt_km": format_float(gt_error_km),
                **threshold_hits,
                "gt_country": format_optional_text(gt_country),
                "pred_country": format_optional_text(pred_country),
                "same_country_as_gt": format_same_country(same_country_as_gt),
                "clean_pred_lat": format_float(clean_pred_lat),
                "clean_pred_lon": format_float(clean_pred_lon),
                "clean_to_aug_prediction_km": format_float(clean_to_aug_prediction_km),
            }
        )

    overall_groups = {"all": gt_error_all}
    if has_ground_truth_rows:
        overall_groups["ground_truth"] = gt_error_ground_truth
        overall_groups["augmented"] = gt_error_augmented

    overall_metric_rows = summarize_metric_rows(
        "haversine_to_gt_km",
        {
            "overall": overall_groups,
            "category": gt_error_by_category,
            "augmentation": gt_error_by_augmentation,
        },
    )
    if has_ground_truth_rows:
        overall_metric_rows.extend(
            summarize_metric_rows(
                "clean_to_aug_prediction_km",
                {
                    "overall": {"augmented": clean_prediction_gap_all},
                    "category": clean_prediction_gap_by_category,
                    "augmentation": clean_prediction_gap_by_augmentation,
                },
            )
        )
    accuracy_values_by_group = {
        "overall": overall_groups,
        "category": gt_error_by_category,
        "augmentation": gt_error_by_augmentation,
    }
    for threshold in ACCURACY_THRESHOLDS_KM:
        overall_metric_rows.extend(
            summarize_accuracy_rows(
                metric_name=f"accuracy_at_{threshold}km",
                threshold_km=threshold,
                values_by_group=accuracy_values_by_group,
            )
        )

    write_csv(image_metrics_csv, IMAGE_METRIC_COLUMNS, image_metric_rows)
    write_csv(overall_metrics_csv, OVERALL_METRIC_COLUMNS, overall_metric_rows)
    write_prediction_map_html(map_html, image_metric_rows, method_label)

    print(f"Wrote {len(image_metric_rows)} per-image metric rows.")
    print(f"Wrote {len(overall_metric_rows)} aggregate metric rows.")
    print(f"Wrote prediction map to {map_html}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
