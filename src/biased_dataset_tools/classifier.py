from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import yaml


RAW_CATEGORY_BY_KEY = {
    2424832: "rural",
    2490368: "people",
    2555904: "landmark",
    2621440: "urban",
    65361: "rural",
    65362: "people",
    65363: "landmark",
    65364: "urban",
    81: "rural",
    82: "people",
    83: "landmark",
    84: "urban",
    ord("a"): "rural",
    ord("w"): "people",
    ord("d"): "landmark",
    ord("s"): "urban",
}
MASKED_ARROW_CATEGORY_BY_KEY = {
    81: "rural",
    82: "people",
    83: "landmark",
    84: "urban",
}
QUIT_KEYS = {27, ord("q"), ord("Q")}
SKIP_KEYS = {32}
UNDO_KEYS = {8, 127, 65288}
MANIFEST_COLUMNS = ["source_path", "dataset_path", "lat", "lon", "category", "augmentation"]
CATEGORY_ORDER = ["rural", "people", "landmark", "urban"]
GROUND_TRUTH_AUGMENTATION = "ground truth"
LEGACY_MANIFEST_REQUIRED_COLUMNS = {"source_path", "dataset_path", "lat", "lon", "category"}
METADATA_MANIFEST_REQUIRED_COLUMNS = {"image_path", "gt_path", "lat", "lon", "scene_type", "race"}


@dataclass(frozen=True)
class SourceRecord:
    source_path: Path
    lat: str
    lon: str


@dataclass(frozen=True)
class MetadataCandidate:
    csv_path: Path
    total_rows: int
    valid_rows: int
    absolute_img_ids: int

    @property
    def valid_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.valid_rows / self.total_rows


@dataclass(frozen=True)
class ManifestEntry:
    source_path: str
    dataset_path: str
    lat: str
    lon: str
    category: str
    augmentation: str


def require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for the interactive classifier. Install project dependencies with `uv sync`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Im2GPS3k images into rural, people, landmark, and urban."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[Path, Path, Path]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    try:
        source_root = Path(config["source"]["root_dir"]).expanduser().resolve()
        dataset_root = Path(config["dataset"]["root_dir"]).expanduser().resolve()
        manifest_path = Path(config["dataset"]["manifest_csv"]).expanduser().resolve()
    except KeyError as exc:
        raise KeyError(f"Missing required config key: {exc}") from exc

    return source_root, dataset_root, manifest_path


def find_images_dir(source_root: Path) -> Path:
    preferred = source_root / "im2gps3k_rgb_images"
    if preferred.is_dir():
        return preferred

    candidates = sorted(path for path in source_root.iterdir() if path.is_dir() and "rgb" in path.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No image directory found under {source_root}")
    return candidates[0]


def resolve_img_path(img_id: str, images_dir: Path) -> Path:
    candidate = Path(img_id)
    if candidate.is_absolute():
        return candidate
    return images_dir / candidate


def score_metadata_csv(csv_path: Path, images_dir: Path) -> MetadataCandidate | None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        required = {"IMG_ID", "LAT", "LON"}
        if not required.issubset(reader.fieldnames):
            return None

        total_rows = 0
        valid_rows = 0
        absolute_img_ids = 0

        for row in reader:
            img_id = (row.get("IMG_ID") or "").strip()
            if not img_id:
                continue

            total_rows += 1
            if Path(img_id).is_absolute():
                absolute_img_ids += 1

            if resolve_img_path(img_id, images_dir).exists():
                valid_rows += 1

    return MetadataCandidate(
        csv_path=csv_path,
        total_rows=total_rows,
        valid_rows=valid_rows,
        absolute_img_ids=absolute_img_ids,
    )


def choose_metadata_csv(source_root: Path, images_dir: Path) -> MetadataCandidate:
    candidates: list[MetadataCandidate] = []
    for csv_path in sorted(source_root.glob("*.csv")):
        candidate = score_metadata_csv(csv_path, images_dir)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No usable metadata CSV found under {source_root}")

    return max(
        candidates,
        key=lambda item: (
            item.valid_ratio,
            item.valid_rows,
            1 if item.absolute_img_ids == 0 else 0,
            1 if "updated" not in item.csv_path.name.lower() else 0,
        ),
    )


def load_source_records(metadata_csv: Path, images_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    missing_paths: list[Path] = []

    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_id = (row.get("IMG_ID") or "").strip()
            if not img_id:
                continue

            source_path = resolve_img_path(img_id, images_dir)
            if not source_path.exists():
                if len(missing_paths) < 5:
                    missing_paths.append(source_path)
                continue

            records.append(
                SourceRecord(
                    source_path=source_path.resolve(),
                    lat=(row.get("LAT") or "").strip(),
                    lon=(row.get("LON") or "").strip(),
                )
            )

    if missing_paths:
        missing_preview = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Some metadata image paths were invalid: {missing_preview}")

    return records


def ensure_dataset_layout(dataset_root: Path, manifest_path: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for category in CATEGORY_ORDER:
        (dataset_root / category).mkdir(parents=True, exist_ok=True)


def detect_manifest_schema(fieldnames: list[str] | None) -> str:
    available = set(fieldnames or [])
    if LEGACY_MANIFEST_REQUIRED_COLUMNS.issubset(available):
        return "legacy"
    if METADATA_MANIFEST_REQUIRED_COLUMNS.issubset(available):
        return "metadata"

    legacy_missing = ", ".join(sorted(LEGACY_MANIFEST_REQUIRED_COLUMNS - available))
    metadata_missing = ", ".join(sorted(METADATA_MANIFEST_REQUIRED_COLUMNS - available))
    raise ValueError(
        "Manifest schema is not supported. "
        f"Expected legacy columns missing: {legacy_missing or '<none>'}. "
        f"Expected metadata columns missing: {metadata_missing or '<none>'}."
    )


def read_manifest_entries(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        schema = detect_manifest_schema(reader.fieldnames)
        entries: list[ManifestEntry] = []
        for row in reader:
            if schema == "legacy":
                entries.append(
                    ManifestEntry(
                        source_path=(row.get("source_path") or "").strip(),
                        dataset_path=(row.get("dataset_path") or "").strip(),
                        lat=(row.get("lat") or "").strip(),
                        lon=(row.get("lon") or "").strip(),
                        category=(row.get("category") or "").strip(),
                        augmentation=((row.get("augmentation") or "").strip() or GROUND_TRUTH_AUGMENTATION),
                    )
                )
                continue

            entries.append(
                ManifestEntry(
                    source_path=(row.get("gt_path") or "").strip(),
                    dataset_path=(row.get("image_path") or "").strip(),
                    lat=(row.get("lat") or "").strip(),
                    lon=(row.get("lon") or "").strip(),
                    category=(row.get("scene_type") or "").strip(),
                    augmentation=((row.get("race") or "").strip() or GROUND_TRUTH_AUGMENTATION),
                )
            )
        return entries


def write_manifest_entries(manifest_path: Path, entries: list[ManifestEntry]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "source_path": entry.source_path,
                    "dataset_path": entry.dataset_path,
                    "lat": entry.lat,
                    "lon": entry.lon,
                    "category": entry.category,
                    "augmentation": entry.augmentation,
                }
            )


def load_manifest_state(manifest_path: Path) -> tuple[list[ManifestEntry], set[str], Counter]:
    entries = read_manifest_entries(manifest_path)
    seen: set[str] = set()
    counts: Counter = Counter({category: 0 for category in CATEGORY_ORDER})

    for entry in entries:
        if entry.augmentation == GROUND_TRUTH_AUGMENTATION and entry.source_path:
            seen.add(entry.source_path)
        if entry.augmentation == GROUND_TRUTH_AUGMENTATION and entry.category in CATEGORY_ORDER:
            counts[entry.category] += 1

    return entries, seen, counts


def append_manifest_row(manifest_path: Path, row: dict[str, str]) -> None:
    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_dataset_path_base(dataset_root: Path) -> Path:
    parent = dataset_root.parent
    return parent if parent != dataset_root else dataset_root


def to_relative_dataset_path(path: Path, dataset_root: Path) -> str:
    resolved_path = path.resolve()
    for base in (get_dataset_path_base(dataset_root), dataset_root):
        try:
            return resolved_path.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved_path.as_posix()


def resolve_dataset_path(path_value: str, dataset_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    base_candidates = [get_dataset_path_base(dataset_root), dataset_root]
    for base in base_candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (base_candidates[0] / path).resolve()


def normalize_manifest_dataset_paths(manifest_path: Path, dataset_root: Path) -> bool:
    schema = "legacy"
    has_augmentation_column = False
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            schema = detect_manifest_schema(reader.fieldnames)
            has_augmentation_column = "augmentation" in (reader.fieldnames or [])

    if schema != "legacy":
        return False

    entries = read_manifest_entries(manifest_path)
    updated = not has_augmentation_column
    normalized_entries: list[ManifestEntry] = []
    for entry in entries:
        dataset_path_value = entry.dataset_path
        normalized_entry = entry
        relative_path = dataset_path_value
        augmentation = entry.augmentation or GROUND_TRUTH_AUGMENTATION

        if dataset_path_value and Path(dataset_path_value).is_absolute():
            relative_path = to_relative_dataset_path(Path(dataset_path_value), dataset_root)
            if relative_path != dataset_path_value:
                updated = True

        if relative_path != entry.dataset_path or augmentation != entry.augmentation:
            normalized_entry = ManifestEntry(
                source_path=entry.source_path,
                dataset_path=relative_path,
                lat=entry.lat,
                lon=entry.lon,
                category=entry.category,
                augmentation=augmentation,
            )
            updated = True
        normalized_entries.append(normalized_entry)

    if not updated:
        return False

    write_manifest_entries(manifest_path, normalized_entries)
    return True


def resolve_pressed_category(key: int) -> str | None:
    category = RAW_CATEGORY_BY_KEY.get(key)
    if category is not None:
        return category

    if key > 0xFF:
        return MASKED_ARROW_CATEGORY_BY_KEY.get(key & 0xFF)

    return None


def is_quit_key(key: int) -> bool:
    return key in QUIT_KEYS


def is_skip_key(key: int) -> bool:
    return key in SKIP_KEYS


def is_undo_key(key: int) -> bool:
    return key in UNDO_KEYS or (key > 0xFF and (key & 0xFF) in UNDO_KEYS)


def ellipsize(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"...{value[-(max_chars - 3):]}"


def resize_image(image, max_width: int = 1600, max_height: int = 900):
    require_cv2()
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def render_frame(record: SourceRecord, counts: Counter, total_records: int):
    require_cv2()
    image = cv2.imread(str(record.source_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not load image: {record.source_path}")

    image = resize_image(image)
    frame = cv2.copyMakeBorder(
        image,
        top=110,
        bottom=60,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )

    classified = sum(counts.values())
    remaining = total_records - classified

    header_1 = f"Classified: {classified} / {total_records} | Remaining: {remaining}"
    header_2 = f"File: {ellipsize(record.source_path.name, 90)}"
    header_3 = f"LAT: {record.lat} | LON: {record.lon}"
    header_4 = (
        f"Rural: {counts['rural']} | People: {counts['people']} | "
        f"Landmark: {counts['landmark']} | Urban: {counts['urban']}"
    )
    footer = (
        "Left/a=rural  Up/w=people  Right/d=landmark  Down/s=urban  "
        "Space=skip  Backspace=undo  q/Esc=quit"
    )

    cv2.putText(frame, header_1, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, header_2, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, header_3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(frame, header_4, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, footer, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)

    return frame


def build_destination_path(dataset_root: Path, category: str, source_path: Path) -> Path:
    destination_dir = dataset_root / category
    destination = destination_dir / source_path.name
    if not destination.exists():
        return destination

    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def find_next_unclassified_index(records: list[SourceRecord], seen: set[str], start_index: int) -> int:
    index = max(start_index, 0)
    while index < len(records) and str(records[index].source_path) in seen:
        index += 1
    return index


def classify_records(records: list[SourceRecord], dataset_root: Path, manifest_path: Path) -> int:
    require_cv2()
    manifest_entries, seen, counts = load_manifest_state(manifest_path)
    total_records = len(records)
    source_index_by_path = {str(record.source_path): index for index, record in enumerate(records)}
    current_index = find_next_unclassified_index(records, seen, 0)
    session_history: list[str] = []

    if current_index >= len(records):
        print("All source images are already classified.")
        return 0

    window_name = "Im2GPS3k classifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)

    try:
        while current_index < len(records):
            record = records[current_index]
            frame = render_frame(record, counts, total_records)
            while True:
                cv2.imshow(window_name, frame)
                key = cv2.waitKeyEx(0)

                if is_quit_key(key):
                    print("Stopped by user.")
                    return 0

                if is_skip_key(key):
                    print(f"[skip] {record.source_path.name}")
                    current_index = find_next_unclassified_index(records, seen, current_index + 1)
                    break

                if is_undo_key(key):
                    if not session_history:
                        print("Nothing to undo.")
                        continue

                    last_source_path = session_history.pop()
                    undone_entry = None
                    for index in range(len(manifest_entries) - 1, -1, -1):
                        entry = manifest_entries[index]
                        if (
                            entry.source_path == last_source_path
                            and entry.augmentation == GROUND_TRUTH_AUGMENTATION
                        ):
                            undone_entry = manifest_entries.pop(index)
                            write_manifest_entries(manifest_path, manifest_entries)
                            break

                    if undone_entry is None:
                        print("Nothing to undo.")
                        continue

                    seen.discard(undone_entry.source_path)
                    if undone_entry.category in CATEGORY_ORDER and counts[undone_entry.category] > 0:
                        counts[undone_entry.category] -= 1

                    dataset_file = resolve_dataset_path(undone_entry.dataset_path, dataset_root)
                    if dataset_file.exists():
                        dataset_file.unlink()

                    undone_index = source_index_by_path.get(undone_entry.source_path, current_index)
                    current_index = undone_index
                    print(f"[undo] {Path(undone_entry.source_path).name}")
                    break

                category = resolve_pressed_category(key)
                if not category:
                    continue

                destination_path = build_destination_path(dataset_root, category, record.source_path)
                shutil.copy2(record.source_path, destination_path)
                append_manifest_row(
                    manifest_path,
                    {
                        "source_path": str(record.source_path),
                        "dataset_path": to_relative_dataset_path(destination_path, dataset_root),
                        "lat": record.lat,
                        "lon": record.lon,
                        "category": category,
                        "augmentation": GROUND_TRUTH_AUGMENTATION,
                    },
                )
                manifest_entries.append(
                    ManifestEntry(
                        source_path=str(record.source_path),
                        dataset_path=to_relative_dataset_path(destination_path, dataset_root),
                        lat=record.lat,
                        lon=record.lon,
                        category=category,
                        augmentation=GROUND_TRUTH_AUGMENTATION,
                    )
                )
                seen.add(str(record.source_path))
                session_history.append(str(record.source_path))
                counts[category] += 1
                print(
                    f"[{sum(counts.values())}/{total_records}] "
                    f"{record.source_path.name} -> {category}"
                )
                current_index = find_next_unclassified_index(records, seen, current_index + 1)
                break
    finally:
        cv2.destroyAllWindows()

    print("Finished classifying all available images.")
    return 0


def main() -> int:
    args = parse_args()
    require_cv2()

    source_root, dataset_root, manifest_path = load_config(args.config)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    ensure_dataset_layout(dataset_root, manifest_path)
    manifest_was_normalized = normalize_manifest_dataset_paths(manifest_path, dataset_root)
    images_dir = find_images_dir(source_root)
    metadata = choose_metadata_csv(source_root, images_dir)
    records = load_source_records(metadata.csv_path, images_dir)

    print(f"Source root: {source_root}")
    print(f"Images directory: {images_dir}")
    print(
        f"Using metadata CSV: {metadata.csv_path} "
        f"({metadata.valid_rows}/{metadata.total_rows} valid image paths)"
    )
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest CSV: {manifest_path}")
    if manifest_was_normalized:
        print("Normalized existing manifest dataset_path values to relative paths.")

    return classify_records(records, dataset_root, manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
