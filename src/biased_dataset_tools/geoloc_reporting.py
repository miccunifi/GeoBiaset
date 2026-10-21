from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from biased_dataset_tools.geoloc_shared import (
    COUNTRY_BOUNDARY_GLOBS,
    IMAGE_METRIC_COLUMNS,
    OVERALL_METRIC_COLUMNS,
    PLOT_GROUP_ORDER,
    PLOT_GROUP_STYLES,
    format_float,
    format_method_from_metrics_path,
    format_optional_text,
    format_same_country,
    metric_fieldnames_for_rows,
    normalize_plot_group,
    normalize_rows_for_fieldnames,
    parse_bool_cell,
    read_csv_rows,
    write_csv,
)


def find_default_country_boundaries_path(project_root: Path) -> Path | None:
    for pattern in COUNTRY_BOUNDARY_GLOBS:
        matches = sorted(project_root.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def load_country_lookup(
    countries_data: Path | None,
    country_name_column: str | None,
) -> Any | None:
    country_path = countries_data.expanduser().resolve() if countries_data is not None else None
    if country_path is None:
        country_path = find_default_country_boundaries_path(Path.cwd())
    if country_path is None:
        print("Country boundary file not found under data/. Country columns and map symbols will be unknown.")
        return None

    from country_lookup import CountryLookup

    print(f"Country boundaries: {country_path}")
    return CountryLookup(country_path, name_column=country_name_column)


def safe_country_lookup(country_lookup: Any | None, lat: float, lon: float) -> str | None:
    if country_lookup is None:
        return None
    try:
        return country_lookup.lookup(lat, lon)
    except ValueError:
        return None


def summarize_metric_rows(
    metric_name: str,
    values_by_group: dict[str, dict[str, list[float]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_by, grouped_values in values_by_group.items():
        for group_value, values in grouped_values.items():
            if not values:
                continue
            rows.append(
                {
                    "metric_name": metric_name,
                    "group_by": group_by,
                    "group_value": group_value,
                    "count": str(len(values)),
                    "mean_km": format_float(statistics.fmean(values)),
                    "median_km": format_float(statistics.median(values)),
                    "min_km": format_float(min(values)),
                    "max_km": format_float(max(values)),
                    "accuracy": "",
                }
            )
    return rows


def summarize_accuracy_rows(
    metric_name: str,
    threshold_km: float,
    values_by_group: dict[str, dict[str, list[float]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_by, grouped_values in values_by_group.items():
        for group_value, values in grouped_values.items():
            if not values:
                continue

            hits = sum(value <= threshold_km for value in values)
            rows.append(
                {
                    "metric_name": metric_name,
                    "group_by": group_by,
                    "group_value": group_value,
                    "count": str(len(values)),
                    "mean_km": "",
                    "median_km": "",
                    "min_km": "",
                    "max_km": "",
                    "accuracy": format_float(hits / len(values)),
                }
            )
    return rows


def enrich_rows_with_country_lookup(
    rows: list[dict[str, str]],
    country_lookup: Any | None,
) -> list[dict[str, str]]:
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        gt_country = enriched.get("gt_country") or None
        pred_country = enriched.get("pred_country") or None
        same_country = parse_bool_cell(enriched.get("same_country_as_gt"))

        if country_lookup is not None:
            gt_country = safe_country_lookup(country_lookup, float(enriched["gt_lat"]), float(enriched["gt_lon"]))
            pred_country = safe_country_lookup(country_lookup, float(enriched["pred_lat"]), float(enriched["pred_lon"]))
            same_country = gt_country is not None and pred_country is not None and gt_country == pred_country

        enriched["gt_country"] = format_optional_text(gt_country)
        enriched["pred_country"] = format_optional_text(pred_country)
        enriched["same_country_as_gt"] = format_same_country(same_country)
        enriched_rows.append(enriched)

    return enriched_rows


def derive_map_path_from_image_metrics(image_metrics_csv: Path, map_html: Path | None) -> Path:
    if map_html is not None:
        return map_html.expanduser().resolve()

    stem = image_metrics_csv.stem
    if stem.endswith("_image_metrics"):
        stem = f"{stem[: -len('_image_metrics')]}_prediction_map"
    else:
        stem = f"{stem}_prediction_map"
    return image_metrics_csv.with_name(f"{stem}.html").resolve()


def generate_map_from_image_metrics_csv(args: Any) -> int:
    image_metrics_csv = args.map_from_image_metrics_csv.expanduser().resolve()
    if not image_metrics_csv.exists():
        raise FileNotFoundError(f"Per-image metrics CSV does not exist: {image_metrics_csv}")

    rows = read_csv_rows(image_metrics_csv)
    if not rows:
        raise ValueError(f"No rows found in per-image metrics CSV: {image_metrics_csv}")

    country_lookup = load_country_lookup(args.countries_data, args.country_name_column)
    method_label = format_method_from_metrics_path(image_metrics_csv)
    enriched_rows = enrich_rows_with_country_lookup(rows, country_lookup)
    fieldnames = metric_fieldnames_for_rows(enriched_rows)
    normalized_rows = normalize_rows_for_fieldnames(enriched_rows, fieldnames)
    map_html = derive_map_path_from_image_metrics(image_metrics_csv, args.map_html)

    write_csv(image_metrics_csv, fieldnames, normalized_rows)
    write_prediction_map_html(map_html, normalized_rows, method_label)

    print(f"Updated per-image metrics CSV: {image_metrics_csv}")
    print(f"Wrote prediction map to {map_html}.")
    print(f"Rows plotted: {len(normalized_rows)}")
    return 0


def build_country_match_summary_text(image_metric_rows: list[dict[str, str]]) -> str:
    counts_by_group: dict[str, dict[bool | None, int]] = {}
    for row in image_metric_rows:
        group_key = normalize_plot_group(row["augmentation"])
        if group_key not in PLOT_GROUP_STYLES:
            group_key = "other"
        match_state = parse_bool_cell(row.get("same_country_as_gt"))
        counts_by_group.setdefault(group_key, {True: 0, False: 0, None: 0})[match_state] += 1

    lines = ["<b>Country prediction counts</b>"]
    for group_key in [*PLOT_GROUP_ORDER, "other"]:
        counts = counts_by_group.get(group_key)
        if not counts:
            continue

        correct_count = counts[True]
        wrong_count = counts[False]
        unknown_count = counts[None]
        label = PLOT_GROUP_STYLES[group_key]["label"]
        line = f"<b>{label}</b>: {correct_count} correct / {wrong_count} wrong"
        if unknown_count:
            line = f"{line} / {unknown_count} unknown"
        lines.append(line)

    return "<br>".join(lines)


def write_prediction_map_html(
    path: Path,
    image_metric_rows: list[dict[str, str]],
    method_label: str = "Geolocation",
) -> None:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotly is not installed. Run `uv sync` to install the plotting dependency."
        ) from exc

    grouped_rows: dict[str, dict[bool | None, list[dict[str, str]]]] = {}
    for row in image_metric_rows:
        plot_group = normalize_plot_group(row["augmentation"])
        if plot_group not in PLOT_GROUP_STYLES:
            plot_group = "other"
        match_state = parse_bool_cell(row.get("same_country_as_gt"))
        grouped_rows.setdefault(plot_group, {}).setdefault(match_state, []).append(row)

    traces = []

    def add_trace(group_key: str, match_state: bool | None, rows: list[dict[str, str]]) -> None:
        style = PLOT_GROUP_STYLES[group_key]
        match_label = {
            True: "same GT country",
            False: "different GT country",
            None: "country unknown",
        }[match_state]
        marker_symbol = "circle" if match_state is not False else "square"
        traces.append(
            go.Scattergeo(
                lon=[float(row["pred_lon"]) for row in rows],
                lat=[float(row["pred_lat"]) for row in rows],
                mode="markers",
                name=f"{style['label']} - {match_label}",
                marker={
                    "size": 7,
                    "symbol": marker_symbol,
                    "color": style["color"],
                    "opacity": 0.8,
                    "line": {"width": 0.4, "color": "#FFFFFF"},
                },
                text=[
                    (
                        f"{row['dataset_path']}<br>"
                        f"category: {row['category']}<br>"
                        f"augmentation: {row['augmentation']}<br>"
                        f"gt country: {row.get('gt_country', '') or 'unknown'}<br>"
                        f"pred country: {row.get('pred_country', '') or 'unknown'}<br>"
                        f"gt: ({row['gt_lat']}, {row['gt_lon']})<br>"
                        f"pred: ({row['pred_lat']}, {row['pred_lon']})<br>"
                        f"error: {row['haversine_to_gt_km']} km"
                    )
                    for row in rows
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for group_key in [*PLOT_GROUP_ORDER, "other"]:
        rows_by_match = grouped_rows.pop(group_key, {})
        for match_state in (True, False, None):
            rows = rows_by_match.get(match_state, [])
            if rows:
                add_trace(group_key, match_state, rows)

    summary_text = build_country_match_summary_text(image_metric_rows)
    figure = go.Figure(traces)
    figure.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#F4F1E8",
        showocean=True,
        oceancolor="#DDEEFF",
        showcountries=True,
        countrycolor="#A3A3A3",
        coastlinecolor="#7A7A7A",
        lataxis={"showgrid": True, "gridcolor": "#D9D9D9"},
        lonaxis={"showgrid": True, "gridcolor": "#D9D9D9"},
    )
    figure.update_layout(
        title=f"{method_label} Predicted Coordinates by Image Group",
        legend_title_text="Image group",
        annotations=[
            {
                "text": summary_text,
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.99,
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255, 255, 255, 0.86)",
                "bordercolor": "#A3A3A3",
                "borderwidth": 1,
                "borderpad": 6,
            }
        ],
        margin={"l": 20, "r": 20, "t": 120, "b": 20},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
