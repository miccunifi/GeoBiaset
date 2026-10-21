from __future__ import annotations

"""
Backward-compatible wrapper for the geolocation evaluation stack.

The implementation now lives across smaller modules:
- `geoloc_shared.py` for constants and generic helpers
- `geoloc_methods.py` for model loading and prediction backends
- `geoloc_reporting.py` for country lookup, map writing, and aggregation helpers
- `geoloc_eval.py` for the evaluator CLI entrypoint

Older imports still targeting `biased_dataset_tools.geoclip_eval` continue to work
through these re-exports.
"""

from biased_dataset_tools.classifier import GROUND_TRUTH_AUGMENTATION
from biased_dataset_tools.geoloc_eval import main, parse_args
from biased_dataset_tools.geoloc_methods import (
    available_local_qwen_model_ids,
    build_img2loc_search_index,
    extract_json_block,
    fetch_embedding_coords,
    find_local_clip_snapshot_dir,
    find_local_hf_snapshot_dir,
    load_database_coords,
    load_g3_predictor,
    load_g3_precomputed_predictions,
    load_geoclip_model,
    load_geoclip_model_offline,
    load_geoclip_predictor,
    load_img2loc_predictor,
    load_img2loc_precomputed_predictions,
    load_predictor,
    load_qwen_predictor,
    parse_lat_lon_from_text,
    resolve_g3_paths,
    resolve_img2loc_paths,
    resolve_qwen_model_id,
    try_load_geoclip_model_offline,
    validate_qwen_snapshot,
)
from biased_dataset_tools.geoloc_reporting import (
    build_country_match_summary_text,
    derive_map_path_from_image_metrics,
    enrich_rows_with_country_lookup,
    find_default_country_boundaries_path,
    generate_map_from_image_metrics_csv,
    load_country_lookup,
    safe_country_lookup,
    summarize_accuracy_rows,
    summarize_metric_rows,
    write_prediction_map_html,
)
from biased_dataset_tools.geoloc_shared import (
    ACCURACY_THRESHOLDS_KM,
    COUNTRY_BOUNDARY_GLOBS,
    DEFAULT_G3_ROOT,
    DEFAULT_IMG2LOC_EMBEDDINGS,
    DEFAULT_IMG2LOC_ROOT,
    DEFAULT_QWEN_MODEL_IDS,
    EVAL_METHOD_CHOICES,
    IMAGE_METRIC_COLUMNS,
    METHOD_DISPLAY_NAMES,
    OVERALL_METRIC_COLUMNS,
    PLOT_GROUP_ORDER,
    PLOT_GROUP_STYLES,
    QWEN_COORDINATE_PROMPT,
    ImagePrediction,
    build_threshold_hits,
    candidate_path_keys,
    derive_output_paths,
    format_float,
    format_method_from_metrics_path,
    format_optional_text,
    format_same_country,
    haversine_km,
    metric_fieldnames_for_rows,
    method_display_name,
    method_slug,
    normalize_plot_group,
    normalize_rows_for_fieldnames,
    parse_bool_cell,
    parse_top_prediction,
    read_csv_rows,
    resolve_runtime_device,
    to_float,
    write_csv,
)


if __name__ == "__main__":
    raise SystemExit(main())
