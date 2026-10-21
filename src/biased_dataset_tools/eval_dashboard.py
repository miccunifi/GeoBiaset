from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("evaluation_dashboard.html")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static interactive dashboard for model evaluation CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Optional evaluation CSV files to embed in the generated dashboard.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the generated HTML dashboard. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args(argv)


def read_initial_files(csv_paths: list[Path]) -> list[dict[str, str]]:
    initial_files: list[dict[str, str]] = []
    for csv_path in csv_paths:
        resolved_path = csv_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise IsADirectoryError(f"CSV path is not a file: {resolved_path}")

        initial_files.append(
            {
                "name": csv_path.name,
                "path": str(resolved_path),
                "content": resolved_path.read_text(encoding="utf-8"),
            }
        )
    return initial_files


def safe_json_for_script(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False)
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def get_plotly_bundle() -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotly is required to generate the dashboard. Install project dependencies with `uv sync`."
        ) from exc

    return get_plotlyjs().replace("</script>", "<\\/script>")


def render_dashboard_html(
    initial_files: list[dict[str, str]] | None = None,
    *,
    plotly_js: str | None = None,
) -> str:
    embedded_files_json = safe_json_for_script(initial_files or [])
    plotly_payload = get_plotly_bundle() if plotly_js is None else plotly_js
    return (
        DASHBOARD_HTML_TEMPLATE.replace("%%PLOTLY_JS%%", plotly_payload)
        .replace("%%INITIAL_FILES_JSON%%", embedded_files_json)
    )


def write_dashboard(output_path: Path, csv_paths: list[Path]) -> Path:
    initial_files = read_initial_files(csv_paths)
    html = render_dashboard_html(initial_files)
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(html, encoding="utf-8")
    return resolved_output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = write_dashboard(args.output, args.csv_files)
    print(f"Wrote evaluation dashboard to {output_path}")
    return 0


DASHBOARD_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --accent: #1f77b4;
      --warning-bg: #fff4d6;
      --warning-line: #e4b64f;
      --warning-text: #5b4510;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }

    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    h1 {
      margin: 0 0 14px;
      font-size: 26px;
      font-weight: 720;
      letter-spacing: 0;
    }

    h2 {
      margin: 0 0 12px;
      font-size: 17px;
      font-weight: 680;
      letter-spacing: 0;
    }

    .metric-heading {
      display: flex;
      align-items: center;
      gap: 7px;
    }

    .info-tooltip {
      position: relative;
      display: inline-grid;
      place-items: center;
      flex: 0 0 auto;
      width: 18px;
      height: 18px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #ffffff;
      font-size: 12px;
      font-weight: 780;
      line-height: 1;
      cursor: help;
    }

    .info-tooltip::after {
      position: absolute;
      bottom: calc(100% + 8px);
      left: 50%;
      z-index: 20;
      width: min(360px, calc(100vw - 56px));
      padding: 10px 12px;
      border: 1px solid #a8b3c2;
      border-radius: 6px;
      background: #17202a;
      color: #ffffff;
      box-shadow: 0 8px 24px rgba(23, 32, 42, 0.18);
      content: attr(data-tooltip);
      font-size: 12px;
      font-weight: 560;
      line-height: 1.35;
      letter-spacing: 0;
      opacity: 0;
      pointer-events: none;
      text-transform: none;
      transform: translateX(-50%);
      transition: opacity 120ms ease;
      visibility: hidden;
      white-space: normal;
    }

    .info-tooltip:hover::after,
    .info-tooltip:focus::after {
      opacity: 1;
      visibility: visible;
    }

    main {
      padding: 20px 28px 28px;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(260px, 1.35fr) repeat(3, minmax(170px, 0.55fr)) auto;
      gap: 12px;
      align-items: end;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    input[type="file"],
    select,
    button {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
    }

    input[type="file"] {
      padding: 8px;
    }

    select {
      padding: 8px 10px;
    }

    button {
      padding: 8px 12px;
      cursor: pointer;
      font-weight: 650;
    }

    button:hover,
    select:hover,
    input[type="file"]:hover {
      border-color: #a8b3c2;
    }

    .status {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0 0;
    }

    .stat-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #ffffff;
      color: var(--muted);
      padding: 5px 10px;
      font-size: 13px;
    }

    .warning {
      display: none;
      margin: 0 0 16px;
      padding: 12px 14px;
      border: 1px solid var(--warning-line);
      border-radius: 8px;
      background: var(--warning-bg);
      color: var(--warning-text);
      font-weight: 620;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
      min-width: 0;
    }

    .wide-panel {
      grid-column: 1 / -1;
    }

    .panel-heading-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .panel-heading-row h2 {
      margin-bottom: 0;
    }

    .compact-control {
      min-width: 170px;
    }

    .table-wrap {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th,
    td {
      padding: 8px 9px;
      border-bottom: 1px solid #edf0f4;
      text-align: right;
      white-space: nowrap;
    }

    th:first-child,
    td:first-child {
      text-align: left;
    }

    th {
      color: #394758;
      background: #f2f5f8;
      font-weight: 720;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    .map-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 230px;
      gap: 16px;
      align-items: stretch;
    }

    #map {
      min-height: 610px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .legend {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fafbfc;
    }

    .legend-title {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .legend-row {
      display: grid;
      grid-template-columns: 20px 14px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 6px 0;
      font-size: 13px;
    }

    .legend-row input {
      width: 16px;
      height: 16px;
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.22);
    }

    .legend-count {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 220px;
      color: var(--muted);
      text-align: center;
      padding: 24px;
    }

    @media (max-width: 1050px) {
      .controls,
      .metrics-grid,
      .map-layout {
        grid-template-columns: 1fr;
      }

      #map {
        min-height: 520px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Evaluation Dashboard</h1>
    <div class="controls">
      <label>
        CSV files
        <input id="fileInput" type="file" accept=".csv,text/csv" multiple>
      </label>
      <label>
        File
        <select id="fileSelect"></select>
      </label>
      <label>
        Model
        <select id="modelSelect"></select>
      </label>
      <label>
        Split
        <select id="splitSelect"></select>
      </label>
      <button id="clearButton" type="button">Clear</button>
    </div>
    <div id="status" class="status"></div>
  </header>

  <main>
    <div id="warning" class="warning"></div>
    <div class="dashboard-grid">
      <section class="metrics-grid">
        <div class="panel">
          <h2>Average Haversine Error</h2>
          <div id="haversineTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2>Accuracy by Distance</h2>
          <div id="accuracyTable" class="table-wrap"></div>
        </div>
        <div class="panel wide-panel">
          <div class="panel-heading-row">
            <h2>Background Race Accuracy Matrix</h2>
            <label class="compact-control">
              Threshold
              <select id="matrixAccuracySelect"></select>
            </label>
          </div>
          <div id="raceAccuracyMatrixTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2 class="metric-heading">
            POPI Pull Norm
            <span class="info-tooltip" tabindex="0" title="POPI measures whether geolocation predictions are pulled toward the population-origin centroid associated with an augmentation group." data-tooltip="POPI (Population-Origin Pull Index) measures whether a geolocation prediction for an augmented image shifts from the true location toward the population-origin centroid associated with that augmentation group. Pull norm is the signed pull normalized by the true-location-to-origin distance.">?</span>
          </h2>
          <div id="popiPullTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2 class="metric-heading">
            POPI Moved Toward Origin
            <span class="info-tooltip" tabindex="0" title="POPI measures whether geolocation predictions are pulled toward the population-origin centroid associated with an augmentation group." data-tooltip="POPI (Population-Origin Pull Index) measures whether a geolocation prediction for an augmented image shifts from the true location toward the population-origin centroid associated with that augmentation group. Moved toward origin is the share of predictions closer to that centroid than the true location is.">?</span>
          </h2>
          <div id="popiMovedTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2 class="metric-heading">
            POPI Movement
            <span class="info-tooltip" tabindex="0" title="POPI compares original/background and augmented predictions." data-tooltip="POPI compares each augmented prediction with the matching original/background prediction. It measures movement toward the target population-origin region, loss of true-location accuracy, target-rank changes, and tree-distance changes in the group taxonomy.">?</span>
          </h2>
          <div id="hpopiMovementTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2 class="metric-heading">
            POPI Events
            <span class="info-tooltip" tabindex="0" title="POPI event rates summarize targeted steering." data-tooltip="Targeted success means the augmented prediction moved closer to the target region and farther from the true location. Targeted flip means the original prediction was within the correctness threshold, the augmented prediction became wrong, and it moved toward the target region.">?</span>
          </h2>
          <div id="hpopiEventsTable" class="table-wrap"></div>
        </div>
        <div class="panel">
          <h2>Image Quality</h2>
          <div id="imageQualityTable" class="table-wrap"></div>
        </div>
      </section>

      <section class="panel">
        <h2>Predicted Coordinates</h2>
        <div class="map-layout">
          <div id="map"></div>
          <aside id="groupLegend" class="legend"></aside>
        </div>
      </section>
    </div>
  </main>

  <script id="initial-files-data" type="application/json">%%INITIAL_FILES_JSON%%</script>
  <script>
%%PLOTLY_JS%%
  </script>
  <script>
    const ALL_VALUE = "__all__";
    const INITIAL_FILES = JSON.parse(document.getElementById("initial-files-data").textContent || "[]");
    const GROUND_TRUTH_GROUP = "Ground Truth";
    const BACKGROUND_IMAGES_GROUP = "Background Images";
    const DEFAULT_SPLIT_ID = "augmentation";
    const GROUP_ORDER = [
      GROUND_TRUTH_GROUP,
      BACKGROUND_IMAGES_GROUP,
      "White",
      "Black",
      "East Asian",
      "Indian",
      "Latino",
      "Middle Eastern",
      "Southeast Asian",
      "Other"
    ];
    const RACE_GROUPS = GROUP_ORDER.filter((group) => ![GROUND_TRUTH_GROUP, BACKGROUND_IMAGES_GROUP].includes(group));
    const LANDMARK_SPLIT_ORDER = [
      GROUND_TRUTH_GROUP,
      "Landmark",
      "Non-Landmark",
      "Unknown"
    ];
    const GENDER_SPLIT_ORDER = [
      GROUND_TRUTH_GROUP,
      "Male",
      "Female",
      "Unknown"
    ];
    const ATTIRE_SPLIT_ORDER = [
      GROUND_TRUTH_GROUP,
      "Traditional Attire",
      "Non-Traditional Attire",
      "Unknown"
    ];
    const RACE_GENDER_VARIANTS = ["Male", "Female", "Unknown"];
    const RACE_ATTIRE_VARIANTS = ["Traditional Attire", "Non-Traditional Attire", "Unknown"];
    const RACE_GENDER_SPLIT_ORDER = compositeSplitOrder(RACE_GENDER_VARIANTS);
    const RACE_ATTIRE_SPLIT_ORDER = compositeSplitOrder(RACE_ATTIRE_VARIANTS);
    const GROUP_STYLES = {
      "Ground Truth": { color: "#dc2626" },
      "Background Images": { color: "#6b7280" },
      "White": { color: "#1f77b4" },
      "Black": { color: "#111827" },
      "East Asian": { color: "#f1c40f" },
      "Indian": { color: "#8b5e34" },
      "Latino": { color: "#d62728" },
      "Middle Eastern": { color: "#2ca02c" },
      "Southeast Asian": { color: "#9467bd" },
      "Other": { color: "#9ca3af" }
    };
    const LANDMARK_SPLIT_STYLES = {
      "Ground Truth": GROUP_STYLES["Ground Truth"],
      "Landmark": { color: "#d62728" },
      "Non-Landmark": { color: "#1f77b4" },
      "Unknown": { color: "#9ca3af" }
    };
    const GENDER_SPLIT_STYLES = {
      "Ground Truth": GROUP_STYLES["Ground Truth"],
      "Male": { color: "#1f77b4" },
      "Female": { color: "#e377c2" },
      "Unknown": { color: "#9ca3af" }
    };
    const ATTIRE_SPLIT_STYLES = {
      "Ground Truth": GROUP_STYLES["Ground Truth"],
      "Traditional Attire": { color: "#2ca02c" },
      "Non-Traditional Attire": { color: "#ff7f0e" },
      "Unknown": { color: "#9ca3af" }
    };
    const RACE_GENDER_SPLIT_STYLES = compositeSplitStyles(RACE_GENDER_VARIANTS);
    const RACE_ATTIRE_SPLIT_STYLES = compositeSplitStyles(RACE_ATTIRE_VARIANTS);
    const SPLIT_CONFIGS = [
      {
        id: DEFAULT_SPLIT_ID,
        label: "Augmentation",
        tableLabel: "Group",
        groups: GROUP_ORDER,
        styles: GROUP_STYLES,
        groupForRow: (row) => normalizeGroup(row.augmentation)
      },
      {
        id: "landmark",
        label: "Landmark Status",
        tableLabel: "Split",
        groups: LANDMARK_SPLIT_ORDER,
        styles: LANDMARK_SPLIT_STYLES,
        groupForRow: landmarkSplit
      },
      {
        id: "gender",
        label: "Gender",
        tableLabel: "Split",
        groups: GENDER_SPLIT_ORDER,
        styles: GENDER_SPLIT_STYLES,
        groupForRow: genderSplit
      },
      {
        id: "race_gender",
        label: "Race + Gender",
        tableLabel: "Race + Gender",
        groups: RACE_GENDER_SPLIT_ORDER,
        styles: RACE_GENDER_SPLIT_STYLES,
        groupForRow: raceGenderSplit
      },
      {
        id: "attire",
        label: "Attire",
        tableLabel: "Split",
        groups: ATTIRE_SPLIT_ORDER,
        styles: ATTIRE_SPLIT_STYLES,
        groupForRow: attireSplit
      },
      {
        id: "race_attire",
        label: "Race + Attire",
        tableLabel: "Race + Attire",
        groups: RACE_ATTIRE_SPLIT_ORDER,
        styles: RACE_ATTIRE_SPLIT_STYLES,
        groupForRow: raceAttireSplit
      }
    ];

    const state = {
      files: [],
      selectedSplitId: DEFAULT_SPLIT_ID,
      selectedMatrixAccuracyField: "",
      selectedGroups: new Set(splitConfigById(DEFAULT_SPLIT_ID).groups),
      nextFileId: 1
    };

    function text(value) {
      return String(value ?? "").trim();
    }

    function key(value) {
      return text(value)
        .toLowerCase()
        .replace(/[/_-]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function parseNumber(value) {
      const valueText = text(value);
      if (!valueText) {
        return null;
      }
      const parsed = Number(valueText);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function parseBinary(value) {
      const normalized = key(value);
      if (["1", "true", "yes", "y"].includes(normalized)) {
        return 1;
      }
      if (["0", "false", "no", "n"].includes(normalized)) {
        return 0;
      }
      const parsed = parseNumber(value);
      if (parsed === 0 || parsed === 1) {
        return parsed;
      }
      return null;
    }

    function normalizeGroup(value) {
      const normalized = key(value);
      const compact = normalized.replace(/\s+/g, "");
      if (["ground truth", "groundtruth", "gt", "original", "clean", "source"].includes(normalized) || compact === "groundtruth") {
        return BACKGROUND_IMAGES_GROUP;
      }
      if (["white", "european"].includes(normalized)) {
        return "White";
      }
      if (["black", "african"].includes(normalized)) {
        return "Black";
      }
      if (["east asian", "asian", "chinese"].includes(normalized)) {
        return "East Asian";
      }
      if (normalized === "indian") {
        return "Indian";
      }
      if (["latino", "latino hispanic", "hispanic"].includes(normalized)) {
        return "Latino";
      }
      if (["middle eastern", "middle east"].includes(normalized)) {
        return "Middle Eastern";
      }
      if (["southeast asian", "south east asian"].includes(normalized)) {
        return "Southeast Asian";
      }
      return "Other";
    }

    function normalizeKnownRace(value) {
      const normalized = key(value);
      if (!normalized || ["unknown", "unclear", "n a", "na"].includes(normalized)) {
        return "";
      }
      const race = normalizeGroup(normalized);
      if ([GROUND_TRUTH_GROUP, BACKGROUND_IMAGES_GROUP].includes(race)) {
        return "";
      }
      if (race === "Other" && normalized !== "other") {
        return "";
      }
      return race;
    }

    function compositeSplitOrder(variants) {
      return [
        GROUND_TRUTH_GROUP,
        BACKGROUND_IMAGES_GROUP,
        ...RACE_GROUPS.flatMap((race) => variants.map((variant) => `${race} ${variant}`))
      ];
    }

    function hexToRgb(hexValue) {
      const clean = String(hexValue || "").replace("#", "");
      if (!/^[0-9a-f]{6}$/i.test(clean)) {
        return null;
      }
      return {
        r: parseInt(clean.slice(0, 2), 16),
        g: parseInt(clean.slice(2, 4), 16),
        b: parseInt(clean.slice(4, 6), 16)
      };
    }

    function channelToHex(value) {
      return Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, "0");
    }

    function mixColor(baseColor, overlayColor, amount) {
      const base = hexToRgb(baseColor);
      const overlay = hexToRgb(overlayColor);
      if (!base || !overlay) {
        return baseColor || "#9ca3af";
      }
      const red = channelToHex(base.r * (1 - amount) + overlay.r * amount);
      const green = channelToHex(base.g * (1 - amount) + overlay.g * amount);
      const blue = channelToHex(base.b * (1 - amount) + overlay.b * amount);
      return `#${red}${green}${blue}`;
    }

    function colorForVariant(baseColor, variant) {
      if (variant === "Male") {
        return mixColor(baseColor, "#2563eb", 0.34);
      }
      if (variant === "Female") {
        return mixColor(baseColor, "#db2777", 0.34);
      }
      if (variant === "Traditional Attire") {
        return mixColor(baseColor, "#16a34a", 0.34);
      }
      if (variant === "Non-Traditional Attire") {
        return mixColor(baseColor, "#f97316", 0.34);
      }
      return mixColor(baseColor, "#9ca3af", 0.45);
    }

    function compositeSplitStyles(variants) {
      const styles = {
        [GROUND_TRUTH_GROUP]: GROUP_STYLES[GROUND_TRUTH_GROUP],
        [BACKGROUND_IMAGES_GROUP]: GROUP_STYLES[BACKGROUND_IMAGES_GROUP]
      };
      RACE_GROUPS.forEach((race) => {
        const baseColor = GROUP_STYLES[race]?.color || GROUP_STYLES.Other.color;
        variants.forEach((variant) => {
          styles[`${race} ${variant}`] = { color: colorForVariant(baseColor, variant) };
        });
      });
      return styles;
    }

    function imageFileName(row) {
      const pathValue = text(row.dataset_path) || text(row.image_path) || text(row.filename) || text(row.file_name) || text(row.path) || text(row.source_path);
      const withoutQuery = pathValue.split(/[?#]/, 1)[0];
      return withoutQuery.split(/[\\/]/).pop() || "";
    }

    function imageFileStem(row) {
      return imageFileName(row).replace(/\.[^.]*$/, "");
    }

    function raceFromPath(pathValue) {
      const parts = text(pathValue).split(/[\\/]+/).filter(Boolean);
      for (let index = 0; index < parts.length - 1; index += 1) {
        if (key(parts[index]) === "balanced dataset") {
          const race = normalizeKnownRace(parts[index + 1]);
          if (race) {
            return race;
          }
        }
      }
      for (const part of parts.slice(0, 3)) {
        const race = normalizeKnownRace(part);
        if (race) {
          return race;
        }
      }
      return "";
    }

    function backgroundRace(row) {
      const explicit = normalizeKnownRace(row.background_race || row.source_race || row.original_race || row.attribute);
      if (explicit) {
        return explicit;
      }

      const pathFields = [row.source_path, row.gt_path, row.clean_dataset_path, row.background_path, row.original_path];
      for (const pathValue of pathFields) {
        const race = raceFromPath(pathValue);
        if (race) {
          return race;
        }
      }
      return "Unknown";
    }

    function inpaintedPersonRace(row) {
      if (isGroundTruth(row)) {
        return BACKGROUND_IMAGES_GROUP;
      }
      return normalizeKnownRace(row.augmentation) || normalizeKnownRace(row.race || row.person_race || row.subject_race) || "Other";
    }

    function landmarkSplit(row) {
      const normalized = key(row.category);
      if (!normalized || ["unknown", "unclear", "n a", "na"].includes(normalized)) {
        return "Unknown";
      }
      return normalized === "landmark" ? "Landmark" : "Non-Landmark";
    }

    function genderSplit(row) {
      const explicit = key(row.gender || row.person_gender || row.subject_gender);
      if (["male", "man", "masculine"].includes(explicit)) {
        return "Male";
      }
      if (["female", "woman", "feminine"].includes(explicit)) {
        return "Female";
      }

      const tokens = key(imageFileStem(row)).split(" ").filter(Boolean);
      if (tokens.includes("male")) {
        return "Male";
      }
      if (tokens.includes("female")) {
        return "Female";
      }
      return "Unknown";
    }

    function attireSplit(row) {
      const explicit = key(row.clothing_type || row.attire || row.attire_type);
      if (["traditional", "typical", "cultural", "cultural attire", "traditional attire"].includes(explicit)) {
        return "Traditional Attire";
      }
      if (["non traditional", "nontraditional", "neutral", "casual", "ordinary", "whatever", "regular"].includes(explicit)) {
        return "Non-Traditional Attire";
      }

      const stem = imageFileStem(row).toLowerCase();
      if (/(^|[_\s-])01$/.test(stem)) {
        return "Traditional Attire";
      }
      if (/(^|[_\s-])00$/.test(stem)) {
        return "Non-Traditional Attire";
      }
      return "Unknown";
    }

    function raceGenderSplit(row) {
      const race = normalizeGroup(row.augmentation);
      if (race === BACKGROUND_IMAGES_GROUP) {
        return BACKGROUND_IMAGES_GROUP;
      }
      return `${race} ${genderSplit(row)}`;
    }

    function raceAttireSplit(row) {
      const race = normalizeGroup(row.augmentation);
      if (race === BACKGROUND_IMAGES_GROUP) {
        return BACKGROUND_IMAGES_GROUP;
      }
      return `${race} ${attireSplit(row)}`;
    }

    function splitConfigById(splitId) {
      return SPLIT_CONFIGS.find((config) => config.id === splitId) || SPLIT_CONFIGS[0];
    }

    function activeSplitConfig() {
      return splitConfigById(state.selectedSplitId);
    }

    function resetSelectedGroups() {
      state.selectedGroups = new Set(activeSplitConfig().groups);
    }

    function styleForGroup(group) {
      return activeSplitConfig().styles[group] || { color: "#9ca3af" };
    }

    function isGroundTruthAugmentation(value) {
      return normalizeGroup(value) === BACKGROUND_IMAGES_GROUP;
    }

    function isUsableStatus(row) {
      const status = key(row.status);
      return !status || status === "ok";
    }

    function isLandmark(row) {
      return key(row.category) === "landmark";
    }

    function isGroundTruth(row) {
      return isGroundTruthAugmentation(row.augmentation);
    }

    function parseCSV(content) {
      const rows = [];
      let currentRow = [];
      let currentField = "";
      let inQuotes = false;

      function finishField() {
        currentRow.push(currentField);
        currentField = "";
      }

      function finishRow() {
        const hasContent = currentRow.some((cell) => text(cell) !== "");
        if (hasContent) {
          rows.push(currentRow);
        }
        currentRow = [];
      }

      for (let index = 0; index < content.length; index += 1) {
        const char = content[index];
        const nextChar = content[index + 1];

        if (char === '"') {
          if (inQuotes && nextChar === '"') {
            currentField += '"';
            index += 1;
          } else {
            inQuotes = !inQuotes;
          }
          continue;
        }

        if (char === "," && !inQuotes) {
          finishField();
          continue;
        }

        if ((char === "\n" || char === "\r") && !inQuotes) {
          finishField();
          finishRow();
          if (char === "\r" && nextChar === "\n") {
            index += 1;
          }
          continue;
        }

        currentField += char;
      }

      finishField();
      finishRow();

      if (rows.length === 0) {
        return [];
      }

      const headers = rows[0].map((header, index) => {
        const cleanHeader = text(header).replace(/^\ufeff/, "");
        return cleanHeader || `column_${index + 1}`;
      });

      return rows.slice(1).map((row) => {
        const item = {};
        headers.forEach((header, index) => {
          item[header] = row[index] ?? "";
        });
        return item;
      });
    }

    function makeFileRecord(name, content, sourcePath = "") {
      const id = `file-${state.nextFileId}`;
      state.nextFileId += 1;
      try {
        const rows = parseCSV(content);
        rows.forEach((row, index) => {
          row.__fileId = id;
          row.__fileName = name;
          row.__rowNumber = index + 2;
        });
        return { id, name, path: sourcePath, rows, rowCount: rows.length, parseError: "" };
      } catch (error) {
        return { id, name, path: sourcePath, rows: [], rowCount: 0, parseError: String(error) };
      }
    }

    function readFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(makeFileRecord(file.name, String(reader.result || "")));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
      });
    }

    function option(value, label) {
      const element = document.createElement("option");
      element.value = value;
      element.textContent = label;
      return element;
    }

    function collectRowsForFile(fileValue) {
      if (fileValue === ALL_VALUE) {
        return state.files.flatMap((file) => file.rows);
      }
      const selectedFile = state.files.find((file) => file.id === fileValue);
      return selectedFile ? selectedFile.rows : [];
    }

    function collectRowsForSelection() {
      const fileSelect = document.getElementById("fileSelect");
      const modelSelect = document.getElementById("modelSelect");
      const fileRows = collectRowsForFile(fileSelect.value || ALL_VALUE);
      const modelValue = modelSelect.value || ALL_VALUE;
      if (modelValue === ALL_VALUE) {
        return fileRows;
      }
      return fileRows.filter((row) => text(row.model_name) === modelValue);
    }

    function rebuildFileSelector() {
      const fileSelect = document.getElementById("fileSelect");
      const previousValue = fileSelect.value || ALL_VALUE;
      fileSelect.innerHTML = "";
      fileSelect.appendChild(option(ALL_VALUE, "All Files"));
      state.files.forEach((file) => {
        const label = `${file.name} (${file.rowCount.toLocaleString()} rows)`;
        fileSelect.appendChild(option(file.id, label));
      });
      const values = new Set(Array.from(fileSelect.options).map((item) => item.value));
      fileSelect.value = values.has(previousValue) ? previousValue : ALL_VALUE;
    }

    function rebuildModelSelector() {
      const fileSelect = document.getElementById("fileSelect");
      const modelSelect = document.getElementById("modelSelect");
      const previousValue = modelSelect.value || ALL_VALUE;
      const models = Array.from(
        new Set(collectRowsForFile(fileSelect.value || ALL_VALUE).map((row) => text(row.model_name)).filter(Boolean))
      ).sort((a, b) => a.localeCompare(b));

      modelSelect.innerHTML = "";
      modelSelect.appendChild(option(ALL_VALUE, "All Models"));
      models.forEach((modelName) => {
        modelSelect.appendChild(option(modelName, modelName));
      });

      const values = new Set(Array.from(modelSelect.options).map((item) => item.value));
      modelSelect.value = values.has(previousValue) ? previousValue : ALL_VALUE;
    }

    function rebuildSplitSelector() {
      const splitSelect = document.getElementById("splitSelect");
      const selectedValue = splitConfigById(state.selectedSplitId).id;
      splitSelect.innerHTML = "";
      SPLIT_CONFIGS.forEach((config) => {
        splitSelect.appendChild(option(config.id, config.label));
      });
      state.selectedSplitId = selectedValue;
      splitSelect.value = state.selectedSplitId;
    }

    function renderStatus(rows) {
      const status = document.getElementById("status");
      const totalRows = rows.length;
      const usableRows = rows.filter(isUsableStatus).length;
      const statusIssues = totalRows - usableRows;
      const missingCoordinates = rows.filter((row) => {
        return isUsableStatus(row) && (parseNumber(row.pred_lat) === null || parseNumber(row.pred_lon) === null);
      }).length;
      const groundTruthPointCount = groundTruthPoints(rows).length;
      const backgroundImageCount = rows.filter((row) => isGroundTruthAugmentation(row.augmentation)).length;
      const parseErrors = state.files.filter((file) => file.parseError).length;
      const filesText = `${state.files.length.toLocaleString()} file${state.files.length === 1 ? "" : "s"}`;
      status.innerHTML = [
        `${filesText} loaded`,
        `${totalRows.toLocaleString()} selected rows`,
        `${usableRows.toLocaleString()} usable rows`,
        `${groundTruthPointCount.toLocaleString()} ground truth points`,
        `${backgroundImageCount.toLocaleString()} background images`,
        `${statusIssues.toLocaleString()} skipped status rows`,
        `${missingCoordinates.toLocaleString()} skipped map rows`,
        `${parseErrors.toLocaleString()} parse errors`
      ].map((item) => `<span class="stat-pill">${escapeHtml(item)}</span>`).join("");
    }

    function renderWarning(message) {
      const warning = document.getElementById("warning");
      if (!message) {
        warning.style.display = "none";
        warning.textContent = "";
        return;
      }
      warning.style.display = "block";
      warning.textContent = message;
    }

    function table(headers, rows, emptyMessage) {
      if (!rows.length) {
        return `<div class="empty">${escapeHtml(emptyMessage)}</div>`;
      }
      const head = `<thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>`;
      const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`)
        .join("");
      return `<table>${head}<tbody>${body}</tbody></table>`;
    }

    function formatNumber(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return "N/A";
      }
      return value.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      });
    }

    function formatPercent(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return "N/A";
      }
      return `${(value * 100).toLocaleString(undefined, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
      })}%`;
    }

    function formatYesNo(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return "N/A";
      }
      return value ? "yes" : "no";
    }

    function average(values) {
      if (!values.length) {
        return null;
      }
      return values.reduce((total, value) => total + value, 0) / values.length;
    }

    function groupsPresent(rows) {
      const splitConfig = activeSplitConfig();
      const present = new Set(rows.map((row) => splitConfig.groupForRow(row)));
      return splitConfig.groups.filter((group) => group !== GROUND_TRUTH_GROUP && present.has(group));
    }

    function valuesForGroup(rows, group, field, parser) {
      const splitConfig = activeSplitConfig();
      return rows
        .filter((row) => splitConfig.groupForRow(row) === group && isUsableStatus(row))
        .map((row) => parser(row[field]))
        .filter((value) => value !== null);
    }

    function summarizeList(values, maxItems = 3) {
      const items = Array.from(values).filter(Boolean);
      if (items.length <= maxItems) {
        return items.join(", ");
      }
      return `${items.slice(0, maxItems).join(", ")} +${items.length - maxItems} more`;
    }

    function groundTruthPoints(rows) {
      const pointsByCoordinate = new Map();
      rows.forEach((row) => {
        const lat = parseNumber(row.gt_lat);
        const lon = parseNumber(row.gt_lon);
        if (lat === null || lon === null) {
          return;
        }

        const coordinateKey = `${lat},${lon}`;
        if (!pointsByCoordinate.has(coordinateKey)) {
          pointsByCoordinate.set(coordinateKey, {
            lat,
            lon,
            rowCount: 0,
            files: new Set(),
            datasets: new Set(),
            sources: new Set()
          });
        }
        const point = pointsByCoordinate.get(coordinateKey);
        point.rowCount += 1;
        point.files.add(text(row.__fileName));
        point.datasets.add(text(row.dataset_path));
        point.sources.add(text(row.source_path));
      });
      return Array.from(pointsByCoordinate.values());
    }

    function renderHaversineTable(rows) {
      const tableRows = groupsPresent(rows).map((group) => {
        const values = valuesForGroup(rows, group, "haversine_to_gt_km", parseNumber);
        return [group, values.length.toLocaleString(), formatNumber(average(values), 2)];
      });
      document.getElementById("haversineTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Avg km"],
        tableRows,
        "No haversine metrics are available."
      );
    }

    function accuracyFieldDistance(field) {
      const match = field.match(/^within_(\d+(?:\.\d+)?)km$/);
      return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
    }

    function accuracyFields(rows) {
      const fields = new Set();
      rows.forEach((row) => {
        Object.keys(row).forEach((field) => {
          if (/^within_\d+(?:\.\d+)?km$/.test(field)) {
            fields.add(field);
          }
        });
      });
      return Array.from(fields).sort((a, b) => accuracyFieldDistance(a) - accuracyFieldDistance(b));
    }

    function formatDistanceField(field) {
      const distance = accuracyFieldDistance(field);
      return Number.isFinite(distance) ? `${distance} km` : field;
    }

    function renderAccuracyTable(rows) {
      const groups = groupsPresent(rows);
      const fields = accuracyFields(rows);
      const tableRows = fields.map((field) => {
        const row = [formatDistanceField(field)];
        groups.forEach((group) => {
          const values = valuesForGroup(rows, group, field, parseBinary);
          row.push(formatPercent(average(values)));
        });
        return row;
      });
      document.getElementById("accuracyTable").innerHTML = table(
        ["Threshold", ...groups],
        tableRows,
        "No accuracy metrics are available."
      );
    }

    function preferredMatrixAccuracyField(fields) {
      if (!fields.length) {
        state.selectedMatrixAccuracyField = "";
        return "";
      }
      if (fields.includes(state.selectedMatrixAccuracyField)) {
        return state.selectedMatrixAccuracyField;
      }
      state.selectedMatrixAccuracyField = fields.includes("within_25km") ? "within_25km" : fields[0];
      return state.selectedMatrixAccuracyField;
    }

    function syncMatrixAccuracySelector(fields) {
      const select = document.getElementById("matrixAccuracySelect");
      const selectedField = preferredMatrixAccuracyField(fields);
      select.innerHTML = "";
      if (!fields.length) {
        select.appendChild(option("", "No accuracy fields"));
        select.disabled = true;
        return "";
      }
      fields.forEach((field) => {
        select.appendChild(option(field, formatDistanceField(field)));
      });
      select.disabled = false;
      select.value = selectedField;
      return selectedField;
    }

    function matrixCell(rows, backgroundGroup, personGroup, field) {
      const values = rows
        .filter((row) => {
          if (!isUsableStatus(row) || backgroundRace(row) !== backgroundGroup) {
            return false;
          }
          if (personGroup === BACKGROUND_IMAGES_GROUP) {
            return isGroundTruth(row);
          }
          return !isGroundTruth(row) && inpaintedPersonRace(row) === personGroup;
        })
        .map((row) => parseBinary(row[field]))
        .filter((value) => value !== null);

      if (!values.length) {
        return "N/A";
      }
      return `${formatPercent(average(values))} (n=${values.length.toLocaleString()})`;
    }

    function backgroundRacesPresent(rows) {
      const present = new Set(rows.map(backgroundRace));
      const ordered = RACE_GROUPS.filter((race) => present.has(race));
      if (present.has("Unknown")) {
        ordered.push("Unknown");
      }
      return ordered;
    }

    function renderRaceAccuracyMatrix(rows) {
      const fields = accuracyFields(rows);
      const field = syncMatrixAccuracySelector(fields);
      const matrix = document.getElementById("raceAccuracyMatrixTable");
      if (!field) {
        matrix.innerHTML = table([], [], "No accuracy metrics are available.");
        return;
      }

      const backgroundRaces = backgroundRacesPresent(rows);
      const personRaces = RACE_GROUPS;
      const tableRows = backgroundRaces.map((backgroundGroup) => [
        backgroundGroup,
        matrixCell(rows, backgroundGroup, BACKGROUND_IMAGES_GROUP, field),
        ...personRaces.map((personGroup) => matrixCell(rows, backgroundGroup, personGroup, field))
      ]);

      matrix.innerHTML = table(
        ["Background Race", "Background Only", ...personRaces.map((race) => `${race} Person`)],
        tableRows,
        "No background race splits are available for the current selection."
      );
    }

    function renderPopiPullTable(rows) {
      const tableRows = groupsPresent(rows).map((group) => {
        const values = valuesForGroup(rows, group, "popi_pull_norm", parseNumber);
        return [group, values.length.toLocaleString(), formatNumber(average(values), 4)];
      });
      document.getElementById("popiPullTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Avg pull norm"],
        tableRows,
        "No POPI pull metrics are available."
      );
    }

    function renderPopiMovedTable(rows) {
      const tableRows = groupsPresent(rows).map((group) => {
        const values = valuesForGroup(rows, group, "popi_moved_toward_origin", parseBinary);
        return [group, values.length.toLocaleString(), formatPercent(average(values))];
      });
      document.getElementById("popiMovedTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Moved toward origin"],
        tableRows,
        "No POPI moved-toward-origin metrics are available."
      );
    }

    function renderHierarchicalPopiMovementTable(rows) {
      const tableRows = groupsPresent(rows)
        .map((group) => {
          const targetValues = valuesForGroup(rows, group, "hpopi_delta_target_km", parseNumber);
          if (!targetValues.length) {
            return null;
          }
          const trueValues = valuesForGroup(rows, group, "hpopi_delta_true_km", parseNumber);
          const rankValues = valuesForGroup(rows, group, "hpopi_delta_rank", parseNumber);
          const treeValues = valuesForGroup(rows, group, "hpopi_delta_tree_hops", parseNumber);
          return [
            group,
            targetValues.length.toLocaleString(),
            formatNumber(average(targetValues), 2),
            formatNumber(average(trueValues), 2),
            formatNumber(average(rankValues), 2),
            formatNumber(average(treeValues), 2)
          ];
        })
        .filter(Boolean);
      document.getElementById("hpopiMovementTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Avg target km", "Avg true km", "Avg rank", "Avg tree hops"],
        tableRows,
        "No POPI movement metrics are available."
      );
    }

    function renderHierarchicalPopiEventsTable(rows) {
      const tableRows = groupsPresent(rows)
        .map((group) => {
          const movedValues = valuesForGroup(rows, group, "hpopi_moved_toward_target", parseBinary);
          if (!movedValues.length) {
            return null;
          }
          const lessAccurateValues = valuesForGroup(rows, group, "hpopi_became_less_accurate", parseBinary);
          const successValues = valuesForGroup(rows, group, "hpopi_targeted_success", parseBinary);
          const flipValues = valuesForGroup(rows, group, "hpopi_targeted_flip", parseBinary);
          return [
            group,
            movedValues.length.toLocaleString(),
            formatPercent(average(movedValues)),
            formatPercent(average(lessAccurateValues)),
            formatPercent(average(successValues)),
            formatPercent(average(flipValues))
          ];
        })
        .filter(Boolean);
      document.getElementById("hpopiEventsTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Moved target", "Less accurate", "Targeted success", "Targeted flip"],
        tableRows,
        "No POPI event metrics are available."
      );
    }

    function renderImageQualityTable(rows) {
      const tableRows = groupsPresent(rows)
        .map((group) => {
          const qualiValues = valuesForGroup(rows, group, "qualiclip", parseNumber);
          const clipIqaValues = valuesForGroup(rows, group, "clip_iqa", parseNumber);
          if (!qualiValues.length && !clipIqaValues.length) {
            return null;
          }
          const rowCount = Math.max(qualiValues.length, clipIqaValues.length);
          return [
            group,
            rowCount.toLocaleString(),
            formatNumber(average(qualiValues), 4),
            formatNumber(average(clipIqaValues), 4)
          ];
        })
        .filter(Boolean);
      document.getElementById("imageQualityTable").innerHTML = table(
        [activeSplitConfig().tableLabel, "Rows", "Avg QualiCLIP", "Avg CLIP-IQA"],
        tableRows,
        "No image quality metrics are available."
      );
    }

    function renderTables(rows) {
      renderHaversineTable(rows);
      renderAccuracyTable(rows);
      renderRaceAccuracyMatrix(rows);
      renderPopiPullTable(rows);
      renderPopiMovedTable(rows);
      renderHierarchicalPopiMovementTable(rows);
      renderHierarchicalPopiEventsTable(rows);
      renderImageQualityTable(rows);
    }

    function clearTables() {
      document.getElementById("haversineTable").innerHTML = "";
      document.getElementById("accuracyTable").innerHTML = "";
      document.getElementById("raceAccuracyMatrixTable").innerHTML = "";
      const matrixSelect = document.getElementById("matrixAccuracySelect");
      matrixSelect.innerHTML = "";
      matrixSelect.appendChild(option("", "No accuracy fields"));
      matrixSelect.disabled = true;
      document.getElementById("popiPullTable").innerHTML = "";
      document.getElementById("popiMovedTable").innerHTML = "";
      document.getElementById("hpopiMovementTable").innerHTML = "";
      document.getElementById("hpopiEventsTable").innerHTML = "";
      document.getElementById("imageQualityTable").innerHTML = "";
    }

    function renderLegend(rows) {
      const legend = document.getElementById("groupLegend");
      const splitConfig = activeSplitConfig();
      const counts = new Map(splitConfig.groups.map((group) => [group, 0]));
      counts.set(GROUND_TRUTH_GROUP, groundTruthPoints(rows).length);
      rows.forEach((row) => {
        const group = splitConfig.groupForRow(row);
        counts.set(group, (counts.get(group) || 0) + 1);
      });

      const rowsHtml = splitConfig.groups.map((group) => {
        const count = counts.get(group) || 0;
        const checked = state.selectedGroups.has(group) ? "checked" : "";
        const disabled = count === 0 ? "disabled" : "";
        const color = styleForGroup(group).color;
        return `
          <label class="legend-row">
            <input type="checkbox" data-group="${escapeHtml(group)}" ${checked} ${disabled}>
            <span class="swatch" style="background: ${escapeHtml(color)}"></span>
            <span>${escapeHtml(group)}</span>
            <span class="legend-count">${count.toLocaleString()}</span>
          </label>
        `;
      }).join("");

      legend.innerHTML = `<div class="legend-title">${escapeHtml(splitConfig.label)} Layers</div>${rowsHtml}`;
      legend.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
        checkbox.addEventListener("change", () => {
          const group = checkbox.getAttribute("data-group");
          if (!group) {
            return;
          }
          if (checkbox.checked) {
            state.selectedGroups.add(group);
          } else {
            state.selectedGroups.delete(group);
          }
          renderMap(collectRowsForSelection());
        });
      });
    }

    function mapSymbol(row) {
      if (isGroundTruth(row) && isLandmark(row)) {
        return "square";
      }
      if (isGroundTruth(row)) {
        return "circle";
      }
      if (isLandmark(row)) {
        return "square-open";
      }
      return "circle-open";
    }

    function symbolLabel(symbol) {
      return {
        diamond: "True coordinate",
        circle: "Background non-landmark",
        "circle-open": "Augmented non-landmark",
        square: "Background landmark",
        "square-open": "Augmented landmark"
      }[symbol] || symbol;
    }

    function groundTruthHoverText(point) {
      const parts = [
        "<b>Ground truth coordinate</b>",
        `gt: (${escapeHtml(formatNumber(point.lat, 6))}, ${escapeHtml(formatNumber(point.lon, 6))})`,
        `rows: ${escapeHtml(point.rowCount.toLocaleString())}`
      ];
      const files = summarizeList(point.files);
      const datasets = summarizeList(point.datasets);
      const sources = summarizeList(point.sources);
      if (files) {
        parts.push(`files: ${escapeHtml(files)}`);
      }
      if (datasets) {
        parts.push(`datasets: ${escapeHtml(datasets)}`);
      }
      if (sources) {
        parts.push(`sources: ${escapeHtml(sources)}`);
      }
      return parts.join("<br>");
    }

    function hoverText(row, errorKm) {
      const parts = [
        `<b>${escapeHtml(row.__fileName)}</b>`,
        `model: ${escapeHtml(text(row.model_name) || "unknown")}`,
        `source: ${escapeHtml(text(row.source_path) || "unknown")}`,
        `dataset: ${escapeHtml(text(row.dataset_path) || "unknown")}`,
        `augmentation: ${escapeHtml(text(row.augmentation) || "unknown")}`,
        `category: ${escapeHtml(text(row.category) || "unknown")}`,
        `gender: ${escapeHtml(genderSplit(row).toLowerCase())}`,
        `attire: ${escapeHtml(attireSplit(row).toLowerCase())}`,
        `gt: (${escapeHtml(text(row.gt_lat))}, ${escapeHtml(text(row.gt_lon))})`,
        `pred: (${escapeHtml(text(row.pred_lat))}, ${escapeHtml(text(row.pred_lon))})`
      ];
      if (errorKm !== null) {
        parts.push(`error: ${escapeHtml(formatNumber(errorKm, 2))} km`);
      }
      const hpopiTargetDelta = parseNumber(row.hpopi_delta_target_km);
      if (hpopiTargetDelta !== null) {
        parts.push(`POPI target delta: ${escapeHtml(formatNumber(hpopiTargetDelta, 2))} km`);
      }
      const hpopiTrueDelta = parseNumber(row.hpopi_delta_true_km);
      if (hpopiTrueDelta !== null) {
        parts.push(`POPI true delta: ${escapeHtml(formatNumber(hpopiTrueDelta, 2))} km`);
      }
      const hpopiTargetedSuccess = parseBinary(row.hpopi_targeted_success);
      if (hpopiTargetedSuccess !== null) {
        parts.push(`POPI targeted success: ${escapeHtml(formatYesNo(hpopiTargetedSuccess))}`);
      }
      const hpopiTargetedFlip = parseBinary(row.hpopi_targeted_flip);
      if (hpopiTargetedFlip !== null) {
        parts.push(`POPI targeted flip: ${escapeHtml(formatYesNo(hpopiTargetedFlip))}`);
      }
      const qualiclip = parseNumber(row.qualiclip);
      if (qualiclip !== null) {
        parts.push(`QualiCLIP: ${escapeHtml(formatNumber(qualiclip, 4))}`);
      }
      const clipIqa = parseNumber(row.clip_iqa);
      if (clipIqa !== null) {
        parts.push(`CLIP-IQA: ${escapeHtml(formatNumber(clipIqa, 4))}`);
      }
      return parts.join("<br>");
    }

    function renderMap(rows) {
      const map = document.getElementById("map");
      const traceGroups = new Map();

      if (state.selectedGroups.has(GROUND_TRUTH_GROUP)) {
        const points = groundTruthPoints(rows);
        if (points.length) {
          traceGroups.set(`${GROUND_TRUTH_GROUP}|diamond`, {
            group: GROUND_TRUTH_GROUP,
            symbol: "diamond",
            lat: points.map((point) => point.lat),
            lon: points.map((point) => point.lon),
            text: points.map(groundTruthHoverText)
          });
        }
      }

      rows.filter(isUsableStatus).forEach((row) => {
        const group = activeSplitConfig().groupForRow(row);
        if (!state.selectedGroups.has(group)) {
          return;
        }

        const lat = parseNumber(row.pred_lat);
        const lon = parseNumber(row.pred_lon);
        if (lat === null || lon === null) {
          return;
        }

        const symbol = mapSymbol(row);
        const traceKey = `${group}|${symbol}`;
        if (!traceGroups.has(traceKey)) {
          traceGroups.set(traceKey, {
            group,
            symbol,
            lat: [],
            lon: [],
            text: []
          });
        }
        const trace = traceGroups.get(traceKey);
        trace.lat.push(lat);
        trace.lon.push(lon);
        trace.text.push(hoverText(row, parseNumber(row.haversine_to_gt_km)));
      });

      if (traceGroups.size === 0) {
        if (window.Plotly) {
          Plotly.purge(map);
        }
        map.innerHTML = '<div class="empty">No map points are available for the current selection.</div>';
        return;
      }

      const traces = Array.from(traceGroups.values()).map((trace) => {
        const color = styleForGroup(trace.group).color;
        const isOpen = trace.symbol.endsWith("-open");
        return {
          type: "scattergeo",
          mode: "markers",
          name: `${trace.group} - ${symbolLabel(trace.symbol)}`,
          lat: trace.lat,
          lon: trace.lon,
          text: trace.text,
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: 7,
            symbol: trace.symbol,
            color,
            opacity: 0.86,
            line: {
              color: isOpen ? color : "#ffffff",
              width: isOpen ? 1.6 : 0.5
            }
          }
        };
      });

      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        showlegend: false,
        geo: {
          projection: { type: "natural earth" },
          showland: true,
          landcolor: "#f0efe8",
          showocean: true,
          oceancolor: "#d9edf7",
          showcountries: true,
          countrycolor: "#a8b0ba",
          coastlinecolor: "#79828d",
          lataxis: { showgrid: true, gridcolor: "#d8dee8" },
          lonaxis: { showgrid: true, gridcolor: "#d8dee8" }
        }
      };

      map.innerHTML = "";
      Plotly.newPlot(map, traces, layout, { responsive: true, displaylogo: false });
    }

    function renderDashboard() {
      const rows = collectRowsForSelection();
      renderStatus(rows);
      renderLegend(rows);

      if (state.files.length === 0) {
        renderWarning("Upload one or more evaluation CSV files to populate the dashboard.");
        clearTables();
        renderMap([]);
        return;
      }

      if (rows.length === 0) {
        renderWarning("The current file and model filters have no rows.");
        clearTables();
        renderMap([]);
        return;
      }

      if (groundTruthPoints(rows).length === 0) {
        renderWarning("The current selection has no ground-truth coordinate pairs in gt_lat/gt_lon.");
      } else {
        renderWarning("");
      }
      renderTables(rows);
      renderMap(rows);
    }

    function refreshDashboard() {
      rebuildSplitSelector();
      rebuildFileSelector();
      rebuildModelSelector();
      renderDashboard();
    }

    document.getElementById("fileInput").addEventListener("change", async (event) => {
      const files = Array.from(event.target.files || []);
      if (!files.length) {
        return;
      }
      const records = await Promise.all(files.map(readFile));
      state.files.push(...records);
      event.target.value = "";
      refreshDashboard();
    });

    document.getElementById("fileSelect").addEventListener("change", () => {
      rebuildModelSelector();
      renderDashboard();
    });

    document.getElementById("modelSelect").addEventListener("change", renderDashboard);

    document.getElementById("matrixAccuracySelect").addEventListener("change", (event) => {
      state.selectedMatrixAccuracyField = event.target.value;
      renderRaceAccuracyMatrix(collectRowsForSelection());
    });

    document.getElementById("splitSelect").addEventListener("change", (event) => {
      state.selectedSplitId = splitConfigById(event.target.value).id;
      resetSelectedGroups();
      renderDashboard();
    });

    document.getElementById("clearButton").addEventListener("click", () => {
      state.files = [];
      state.nextFileId = 1;
      state.selectedSplitId = DEFAULT_SPLIT_ID;
      state.selectedMatrixAccuracyField = "";
      resetSelectedGroups();
      refreshDashboard();
    });

    INITIAL_FILES.forEach((file) => {
      state.files.push(makeFileRecord(file.name || file.path || "embedded.csv", file.content || "", file.path || ""));
    });
    refreshDashboard();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
