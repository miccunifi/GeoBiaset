from __future__ import annotations

import argparse
import csv
import io
import shutil
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ID = "HuggingFaceM4/FairFace"
DEFAULT_OUTPUT_ROOT = Path("/mnt/nniccoli/FairFace")
FALLBACK_CONFIG_NAMES = ("0.25", "1.25")
FALLBACK_SPLIT_NAMES = ("train", "validation")
IMAGE_SUFFIX_BY_FORMAT = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
METADATA_COLUMNS = [
    "dataset_id",
    "config",
    "split",
    "row_index",
    "image_path",
    "age_id",
    "age",
    "gender_id",
    "gender",
    "race_id",
    "race",
    "service_test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FairFace from Hugging Face and save the images plus metadata locally."
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where FairFace images and metadata will be saved.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional dataset config names. Defaults to all configs discovered from the dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split names. Defaults to the splits discovered for each config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to save per split.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Public FairFace downloads should not need one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output for a config/split before writing it again.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print a progress line every N saved rows.",
    )
    return parser.parse_args()


def import_hf_datasets() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import Image, get_dataset_config_names, get_dataset_split_names, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `datasets` package is required to download FairFace. "
            "Install it with `uv sync` or `pip install datasets` and run the script again."
        ) from exc

    return Image, get_dataset_config_names, get_dataset_split_names, load_dataset


def sanitize_cache_dir(cache_dir: Path | None) -> str | None:
    if cache_dir is None:
        return None
    return str(cache_dir.expanduser().resolve())


def resolve_config_names(
    dataset_id: str,
    requested_configs: list[str] | None,
    token: str | None,
    cache_dir: str | None,
    get_dataset_config_names: Any,
) -> list[str]:
    if requested_configs:
        return requested_configs

    try:
        return list(get_dataset_config_names(dataset_id, token=token, cache_dir=cache_dir))
    except Exception:
        return list(FALLBACK_CONFIG_NAMES)


def resolve_split_names(
    dataset_id: str,
    config_name: str,
    requested_splits: list[str] | None,
    token: str | None,
    cache_dir: str | None,
    get_dataset_split_names: Any,
) -> list[str]:
    if requested_splits:
        return requested_splits

    try:
        return list(get_dataset_split_names(dataset_id, config_name, token=token, cache_dir=cache_dir))
    except Exception:
        return list(FALLBACK_SPLIT_NAMES)


def decode_class_label(example: dict[str, Any], features: Any, column_name: str) -> tuple[str, str]:
    raw_value = example.get(column_name)
    feature = features.get(column_name)

    if raw_value is None:
        return "", ""

    raw_text = str(raw_value)
    if feature is None or not hasattr(feature, "int2str"):
        return raw_text, raw_text

    try:
        decoded = str(feature.int2str(int(raw_value)))
    except Exception:
        decoded = raw_text

    return raw_text, decoded


def infer_image_suffix(image_record: dict[str, Any]) -> str:
    source_path = image_record.get("path")
    if source_path:
        suffix = Path(str(source_path)).suffix.lower()
        if suffix:
            return suffix

    image_bytes = image_record.get("bytes")
    if image_bytes:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
        return IMAGE_SUFFIX_BY_FORMAT.get(image_format, ".jpg")

    return ".jpg"


def write_image(image_record: dict[str, Any], destination_path: Path) -> None:
    image_bytes = image_record.get("bytes")
    if image_bytes:
        destination_path.write_bytes(image_bytes)
        return

    source_path = image_record.get("path")
    if source_path:
        source = Path(str(source_path))
        if source.exists():
            shutil.copy2(source, destination_path)
            return

    raise ValueError(f"Could not materialize image payload for {destination_path}")


def iter_rows(
    dataset_id: str,
    config_name: str,
    split_name: str,
    token: str | None,
    cache_dir: str | None,
    Image: Any,
    load_dataset: Any,
) -> Any:
    dataset = load_dataset(
        dataset_id,
        name=config_name,
        split=split_name,
        streaming=True,
        token=token,
        cache_dir=cache_dir,
    )
    return dataset.cast_column("image", Image(decode=False))


def prepare_split_root(split_root: Path, overwrite: bool) -> tuple[Path, Path]:
    if overwrite and split_root.exists():
        shutil.rmtree(split_root)
    images_dir = split_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_root / "metadata.csv"
    return images_dir, metadata_path


def download_split(
    dataset_id: str,
    config_name: str,
    split_name: str,
    output_root: Path,
    token: str | None,
    cache_dir: str | None,
    limit: int | None,
    overwrite: bool,
    progress_every: int,
    Image: Any,
    get_dataset_split_names: Any,
    load_dataset: Any,
) -> int:
    split_root = output_root / config_name / split_name
    images_dir, metadata_path = prepare_split_root(split_root, overwrite=overwrite)
    dataset = iter_rows(
        dataset_id=dataset_id,
        config_name=config_name,
        split_name=split_name,
        token=token,
        cache_dir=cache_dir,
        Image=Image,
        load_dataset=load_dataset,
    )
    features = dataset.features

    saved_rows = 0
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()

        for row_index, example in enumerate(dataset):
            if limit is not None and row_index >= limit:
                break

            image_record = example["image"]
            suffix = infer_image_suffix(image_record)
            file_name = f"{row_index:08d}{suffix}"
            destination_path = images_dir / file_name

            if overwrite or not destination_path.exists():
                write_image(image_record, destination_path)

            age_id, age_name = decode_class_label(example, features, "age")
            gender_id, gender_name = decode_class_label(example, features, "gender")
            race_id, race_name = decode_class_label(example, features, "race")

            writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "config": config_name,
                    "split": split_name,
                    "row_index": row_index,
                    "image_path": str(destination_path),
                    "age_id": age_id,
                    "age": age_name,
                    "gender_id": gender_id,
                    "gender": gender_name,
                    "race_id": race_id,
                    "race": race_name,
                    "service_test": example.get("service_test", ""),
                }
            )
            saved_rows += 1

            if progress_every > 0 and saved_rows % progress_every == 0:
                print(f"[{config_name}/{split_name}] saved {saved_rows} images")

    print(f"[{config_name}/{split_name}] finished with {saved_rows} images")
    return saved_rows


def main() -> int:
    args = parse_args()
    Image, get_dataset_config_names, get_dataset_split_names, load_dataset = import_hf_datasets()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = sanitize_cache_dir(args.cache_dir)

    config_names = resolve_config_names(
        dataset_id=args.dataset_id,
        requested_configs=args.configs,
        token=args.token,
        cache_dir=cache_dir,
        get_dataset_config_names=get_dataset_config_names,
    )

    total_saved = 0
    for config_name in config_names:
        split_names = resolve_split_names(
            dataset_id=args.dataset_id,
            config_name=config_name,
            requested_splits=args.splits,
            token=args.token,
            cache_dir=cache_dir,
            get_dataset_split_names=get_dataset_split_names,
        )
        for split_name in split_names:
            total_saved += download_split(
                dataset_id=args.dataset_id,
                config_name=config_name,
                split_name=split_name,
                output_root=output_root,
                token=args.token,
                cache_dir=cache_dir,
                limit=args.limit,
                overwrite=args.overwrite,
                progress_every=args.progress_every,
                Image=Image,
                get_dataset_split_names=get_dataset_split_names,
                load_dataset=load_dataset,
            )

    print(f"Saved {total_saved} FairFace images under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
