from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from biased_dataset_tools.classifier import ManifestEntry, load_config, read_manifest_entries, resolve_dataset_path
from metrics import AccuracyAtDistanceMetric, POPI
from models import ModelUnavailableError, create_model_from_args, default_evaluation_model_names
from models.base import DEFAULT_VLM_MAX_NEW_TOKENS
from models.g3 import DEFAULT_G3_ROOT
from models.img2loc import DEFAULT_IMG2LOC_EMBEDDINGS, DEFAULT_IMG2LOC_ROOT
from models.registry import MODEL_SPECS, normalize_model_name


CSV_COLUMNS = [
    "model_name",
    "dataset_path",
    "source_path",
    "category",
    "augmentation",
    "gt_lat",
    "gt_lon",
    "pred_lat",
    "pred_lon",
    "confidence",
    "haversine_to_gt_km",
    "within_1km",
    "within_25km",
    "within_200km",
    "within_750km",
    "within_2500km",
    "popi_pull",
    "popi_pull_norm",
    "popi_moved_toward_origin",
    "raw_output",
    "output_contract_retry",
    "status",
    "error",
]
ENHANCED_MANIFEST_REQUIRED_COLUMNS = {
    "image_path",
    "gt_path",
    "lat",
    "lon",
    "scene_type",
    "attribute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate geolocation models on the dataset manifest.")
    parser.add_argument("--config", type=Path, default=Path("config_geobiaset.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=["all"], help="Model names to run, or 'all'.")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Only evaluate manifest rows missing from the existing output CSV for the requested model(s).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vlm-max-new-tokens", type=int, default=DEFAULT_VLM_MAX_NEW_TOKENS)
    parser.add_argument("--openai-base-url", default=None, help="OpenAI-compatible base URL for served VLMs.")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI-compatible API key for served VLMs.")
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
        help="Attention backend for Hugging Face VLMs.",
    )
    parser.add_argument(
        "--qwen36-disable-quantization-if-vram-over-50gb",
        action="store_true",
        help=(
            "For qwen36-fp8 only: switch to Qwen/Qwen3.6-27B instead of the FP8 variant "
            "when the selected CUDA device has more than 50 GiB of VRAM."
        ),
    )
    parser.add_argument("--vlm-model-id", default=None, help="Model id used only with --models qwen-custom.")
    parser.add_argument("--img2loc-root", type=Path, default=DEFAULT_IMG2LOC_ROOT)
    parser.add_argument("--img2loc-index", type=Path, default=None)
    parser.add_argument("--img2loc-embeddings-with-coords", type=Path, default=DEFAULT_IMG2LOC_EMBEDDINGS)
    parser.add_argument("--img2loc-search-results-csv", type=Path, default=None)
    parser.add_argument("--g3-root", type=Path, default=DEFAULT_G3_ROOT)
    parser.add_argument("--g3-index", type=Path, default=None)
    parser.add_argument("--g3-checkpoint", type=Path, default=None)
    parser.add_argument("--g3-database-csv", type=Path, default=None)
    parser.add_argument("--g3-results-csv", type=Path, default=None)
    return parser.parse_args()


def chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def requested_models(model_args: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in model_args:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    if not expanded or "all" in expanded:
        return default_evaluation_model_names()
    return [normalize_model_name(name) for name in expanded]


def read_evaluation_entries(manifest_path: Path) -> list[ManifestEntry]:
    """Read legacy manifests plus metadata CSVs used by public dataset configs."""
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "race" in fieldnames or not ENHANCED_MANIFEST_REQUIRED_COLUMNS.issubset(fieldnames):
            return read_manifest_entries(manifest_path)

        entries: list[ManifestEntry] = []
        for row in reader:
            entries.append(
                ManifestEntry(
                    source_path=(row.get("gt_path") or "").strip(),
                    dataset_path=(row.get("image_path") or "").strip(),
                    lat=(row.get("lat") or "").strip(),
                    lon=(row.get("lon") or "").strip(),
                    category=(row.get("scene_type") or "").strip(),
                    augmentation=((row.get("attribute") or "").strip() or "ground truth"),
                )
            )
        return entries


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_raw_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def base_row(entry: Any, model_name: str, status: str, error: str = "") -> dict[str, str]:
    return {
        "model_name": model_name,
        "dataset_path": entry.dataset_path,
        "source_path": entry.source_path,
        "category": entry.category,
        "augmentation": entry.augmentation,
        "gt_lat": entry.lat,
        "gt_lon": entry.lon,
        "pred_lat": "",
        "pred_lon": "",
        "confidence": "",
        "haversine_to_gt_km": "",
        "within_1km": "",
        "within_25km": "",
        "within_200km": "",
        "within_750km": "",
        "within_2500km": "",
        "popi_pull": "",
        "popi_pull_norm": "",
        "popi_moved_toward_origin": "",
        "raw_output": "",
        "output_contract_retry": "",
        "status": status,
        "error": error,
    }


def unavailable_rows(entries: list[Any], model_name: str, error: str) -> list[dict[str, str]]:
    return [base_row(entry, model_name, "unavailable", error) for entry in entries]


def populate_prediction_row(
    row: dict[str, str],
    entry: Any,
    output: Any,
    accuracy: AccuracyAtDistanceMetric,
    popi: POPI,
) -> None:
    accuracy_values = accuracy.compute_sample(float(entry.lat), float(entry.lon), output.pred_lat, output.pred_lon)
    popi_values = popi.compute_sample(
        float(entry.lat),
        float(entry.lon),
        output.pred_lat,
        output.pred_lon,
        entry.augmentation,
    )
    output_metadata = getattr(output, "metadata", {})
    output_contract_retry = (
        output_metadata.get("output_contract_retry")
        if isinstance(output_metadata, dict)
        else None
    )
    row.update(
        {
            "pred_lat": format_value(output.pred_lat),
            "pred_lon": format_value(output.pred_lon),
            "confidence": format_value(output.confidence),
            "haversine_to_gt_km": format_value(accuracy_values["haversine_to_gt_km"]),
            "within_1km": format_value(accuracy_values["within_1km"]),
            "within_25km": format_value(accuracy_values["within_25km"]),
            "within_200km": format_value(accuracy_values["within_200km"]),
            "within_750km": format_value(accuracy_values["within_750km"]),
            "within_2500km": format_value(accuracy_values["within_2500km"]),
            "popi_pull": format_value(popi_values["popi_pull"]),
            "popi_pull_norm": format_value(popi_values["popi_pull_norm"]),
            "popi_moved_toward_origin": format_value(popi_values["popi_moved_toward_origin"]),
            "raw_output": format_raw_output(getattr(output, "raw_output", None)),
            "output_contract_retry": format_value(output_contract_retry),
        }
    )


def row_item_kwargs(entry: Any) -> dict[str, str]:
    return {
        "dataset_path": entry.dataset_path,
        "source_path": entry.source_path,
        "augmentation": entry.augmentation,
    }


def evaluate_model(model: Any, entries: list[Any], dataset_root: Path, batch_size: int = 1) -> list[dict[str, str]]:
    accuracy = AccuracyAtDistanceMetric()
    popi = POPI()
    rows: list[dict[str, str]] = []
    unavailable_error: str | None = None
    effective_batch_size = len(entries) if getattr(model, "prefer_full_batch", False) else batch_size

    for batch_entries in chunked(entries, max(1, effective_batch_size)):
        if unavailable_error is not None:
            rows.extend(base_row(entry, model.model_name, "unavailable", unavailable_error) for entry in batch_entries)
            continue

        batch_rows: list[dict[str, str]] = []
        ready_rows: list[dict[str, str]] = []
        ready_entries: list[Any] = []
        ready_paths: list[Path] = []
        ready_item_kwargs: list[dict[str, str]] = []

        for entry in batch_entries:
            row = base_row(entry, model.model_name, "ok")
            batch_rows.append(row)
            image_path = resolve_dataset_path(entry.dataset_path, dataset_root)
            if not image_path.exists():
                row["status"] = "error"
                row["error"] = f"Dataset image does not exist: {image_path}"
                continue

            ready_rows.append(row)
            ready_entries.append(entry)
            ready_paths.append(image_path)
            ready_item_kwargs.append(row_item_kwargs(entry))

        if not ready_entries:
            rows.extend(batch_rows)
            continue

        try:
            outputs = model.batch_inference(ready_paths, item_kwargs=ready_item_kwargs)
            if len(outputs) != len(ready_entries):
                raise RuntimeError(
                    f"Batch inference returned {len(outputs)} outputs for {len(ready_entries)} inputs."
                )
            for entry, row, output in zip(ready_entries, ready_rows, outputs):
                populate_prediction_row(row, entry, output, accuracy, popi)
        except ModelUnavailableError as exc:
            unavailable_error = str(exc)
            for row in ready_rows:
                row["status"] = "unavailable"
                row["error"] = unavailable_error
        except Exception:
            for entry, image_path, row, item_kwargs in zip(ready_entries, ready_paths, ready_rows, ready_item_kwargs):
                try:
                    output = model(image_path, **item_kwargs)
                    populate_prediction_row(row, entry, output, accuracy, popi)
                except ModelUnavailableError as exc:
                    unavailable_error = str(exc)
                    row["status"] = "unavailable"
                    row["error"] = unavailable_error
                except Exception as inner_exc:
                    row["status"] = "error"
                    row["error"] = str(inner_exc)
                if unavailable_error is not None:
                    break

            if unavailable_error is not None:
                exhausted = False
                for row in ready_rows:
                    if row["status"] == "unavailable" and row["error"] == unavailable_error:
                        exhausted = True
                        continue
                    if exhausted and row["status"] == "ok":
                        row["status"] = "unavailable"
                        row["error"] = unavailable_error

        batch_start = len(rows) + 1
        for offset, entry in enumerate(batch_entries, start=batch_start):
            print(f"[{offset}/{len(entries)}] {model.model_name}: {entry.dataset_path}")
        rows.extend(batch_rows)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{column: row.get(column, "") for column in CSV_COLUMNS} for row in csv.DictReader(handle)]


def result_key(model_name: str, dataset_path: str) -> tuple[str, str]:
    return (model_name, dataset_path)


def existing_result_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {
        result_key(row["model_name"], row["dataset_path"])
        for row in rows
        if row.get("model_name") and row.get("dataset_path")
    }


def expected_output_model_name(model_name: str, args: argparse.Namespace) -> str:
    normalized = normalize_model_name(model_name)
    if normalized == "qwen-custom":
        return args.vlm_model_id.split("/")[-1] if args.vlm_model_id else "Qwen VL"
    return MODEL_SPECS.get(normalized).display_name if normalized in MODEL_SPECS else model_name


def main() -> int:
    args = parse_args()
    _, dataset_root, config_manifest_path = load_config(args.config)
    manifest_path = (args.manifest or config_manifest_path).expanduser().resolve()
    output_csv = (args.output_csv or dataset_root / "model_evaluation_results.csv").expanduser().resolve()
    entries = read_evaluation_entries(manifest_path)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise ValueError(f"No manifest rows found in {manifest_path}")

    rerun_missing_only = getattr(args, "rerun", False)
    existing_rows = read_existing_rows(output_csv) if rerun_missing_only else []
    seen_keys = existing_result_keys(existing_rows)
    all_rows: list[dict[str, str]] = list(existing_rows)
    for model_name in requested_models(args.models):
        display_name = expected_output_model_name(model_name, args)
        model_entries = [
            entry
            for entry in entries
            if result_key(display_name, entry.dataset_path) not in seen_keys
        ]
        if rerun_missing_only:
            print(f"{display_name}: {len(model_entries)} missing rows out of {len(entries)} manifest rows.")
        if not model_entries:
            continue
        try:
            model = create_model_from_args(model_name, args)
        except Exception as exc:
            new_rows = unavailable_rows(model_entries, display_name, str(exc))
            all_rows.extend(new_rows)
            seen_keys.update(result_key(row["model_name"], row["dataset_path"]) for row in new_rows)
            continue
        new_rows = evaluate_model(model, model_entries, dataset_root, batch_size=getattr(args, "batch_size", 1))
        all_rows.extend(new_rows)
        seen_keys.update(result_key(row["model_name"], row["dataset_path"]) for row in new_rows)

    write_rows(output_csv, all_rows)
    if rerun_missing_only:
        print(f"Wrote {len(all_rows)} rows to {output_csv} ({len(all_rows) - len(existing_rows)} added).")
    else:
        print(f"Wrote {len(all_rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
