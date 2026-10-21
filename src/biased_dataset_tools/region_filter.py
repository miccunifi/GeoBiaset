from __future__ import annotations

import argparse
import csv
import errno
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ATTRIBUTE_TO_REGION = {
    "white": "Europe, North America, and Oceania",
    "latino": "Latin America and the Caribbean",
    "indian": "Indian subcontinent",
    "east asian": "East Asia",
    "black": "Sub-Saharan Africa",
    "southeast asian": "Southeast Asia",
    "middle eastern": "Middle East and North Africa",
}
ATTRIBUTE_TO_SLUG = {
    "white": "white",
    "latino": "latino",
    "indian": "indian",
    "east asian": "east_asian",
    "black": "black",
    "southeast asian": "southeast_asian",
    "middle eastern": "middle_eastern",
}
ATTRIBUTE_ORDER = [
    "white",
    "latino",
    "indian",
    "east asian",
    "black",
    "southeast asian",
    "middle eastern",
]
MANIFEST_COLUMNS = [
    "source_path",
    "dataset_path",
    "lat",
    "lon",
    "attribute",
    "attribute_slug",
    "region",
    "source_dataset",
    "country",
    "continent",
]
SKIPPED_COLUMNS = ["source_dataset", "source_path", "lat", "lon", "reason"]
SUMMARY_COLUMNS = ["attribute", "attribute_slug", "source_dataset", "count"]


@dataclass(frozen=True)
class SourceImage:
    source_dataset: str
    source_path: Path
    lat: float
    lon: float


@dataclass(frozen=True)
class CountryInfo:
    name: str
    continent: str


@dataclass(frozen=True)
class ManifestRow:
    source_path: str
    dataset_path: str
    lat: str
    lon: str
    attribute: str
    attribute_slug: str
    region: str
    source_dataset: str
    country: str
    continent: str


@dataclass(frozen=True)
class SkippedRow:
    source_dataset: str
    source_path: str
    lat: str
    lon: str
    reason: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images_dir: Path
    metadata_path: Path
    metadata_kind: str


LATINO_NORTH_AMERICA = {
    "bahamas",
    "belize",
    "costarica",
    "cuba",
    "dominicanrep",
    "elsalvador",
    "guatemala",
    "haiti",
    "honduras",
    "jamaica",
    "mexico",
    "nicaragua",
    "panama",
    "puertorico",
    "trinidadandtobago",
}
INDIAN_SUBCONTINENT = {
    "afghanistan",
    "bangladesh",
    "bhutan",
    "india",
    "maldives",
    "nepal",
    "pakistan",
    "srilanka",
}
EAST_ASIA = {
    "china",
    "japan",
    "mongolia",
    "northkorea",
    "southkorea",
    "taiwan",
}
SOUTHEAST_ASIA = {
    "brunei",
    "cambodia",
    "indonesia",
    "laos",
    "malaysia",
    "myanmar",
    "philippines",
    "singapore",
    "thailand",
    "timorleste",
    "vietnam",
}
MIDDLE_EAST_AND_NORTH_AFRICA = {
    "algeria",
    "armenia",
    "azerbaijan",
    "bahrain",
    "cyprus",
    "egypt",
    "georgia",
    "iran",
    "iraq",
    "israel",
    "jordan",
    "kazakhstan",
    "kuwait",
    "kyrgyzstan",
    "lebanon",
    "libya",
    "morocco",
    "ncyprus",
    "oman",
    "palestine",
    "qatar",
    "saudiarabia",
    "syria",
    "tajikistan",
    "tunisia",
    "turkey",
    "turkmenistan",
    "unitedarabemirates",
    "uzbekistan",
    "westsahara",
    "yemen",
}


def normalize_country_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


class CountryBoundaryLookup:
    def __init__(self, boundaries_path: Path) -> None:
        import geopandas as gpd

        self.boundaries_path = boundaries_path.expanduser().resolve()
        countries = gpd.read_file(self.boundaries_path)
        countries = countries[~countries.geometry.isna() & ~countries.geometry.is_empty].copy()
        self._countries = countries[["name", "continent", countries.geometry.name]].reset_index(drop=True)
        self._spatial_index = self._countries.sindex

    def lookup(self, lat: float, lon: float) -> CountryInfo | None:
        from shapely import contains_xy
        from shapely.geometry import Point

        point = Point(lon, lat)
        candidate_positions = list(self._spatial_index.query(point))
        if not candidate_positions:
            return None

        candidates = self._countries.iloc[candidate_positions]
        for geometry, raw_name, raw_continent in zip(
            candidates.geometry,
            candidates["name"],
            candidates["continent"],
            strict=True,
        ):
            if contains_xy(geometry, lon, lat):
                return CountryInfo(name=str(raw_name), continent=str(raw_continent))

        for geometry, raw_name, raw_continent in zip(
            candidates.geometry,
            candidates["name"],
            candidates["continent"],
            strict=True,
        ):
            if geometry.covers(point):
                return CountryInfo(name=str(raw_name), continent=str(raw_continent))

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Im2GPS3k and YFCC4k into geographic bins linked to ethnicity-balance attributes."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("filtered_dataset"),
        help="Directory where the filtered dataset tree will be created.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to <output-root>/filtered_images.csv.",
    )
    parser.add_argument(
        "--skipped-csv",
        type=Path,
        default=None,
        help="Optional skipped-items path. Defaults to <output-root>/skipped_images.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary path. Defaults to <output-root>/summary.csv.",
    )
    parser.add_argument(
        "--country-boundaries",
        type=Path,
        default=Path("data/naturalearth_lowres/naturalearth_lowres.shp"),
        help="Local Natural Earth country boundaries shapefile.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=["auto", "copy", "symlink"],
        default="auto",
        help="How to place images into the output tree. 'auto' copies first, then falls back to symlinks on ENOSPC.",
    )
    return parser.parse_args()


def get_dataset_configs() -> list[DatasetConfig]:
    return [
        DatasetConfig(
            name="im2gps3k",
            images_dir=Path("/mnt/nniccoli/Im2GPS3k/im2gps3k_rgb_images"),
            metadata_path=Path("/mnt/nniccoli/Im2GPS3k/im2gps3k_places365_updated.csv"),
            metadata_kind="csv",
        ),
        DatasetConfig(
            name="yfcc4k",
            images_dir=Path("/mnt/nniccoli/YFCC4k/yfcc4k_rgb_images"),
            metadata_path=Path("/mnt/nniccoli/YFCC4k/coordinates_fixed.txt"),
            metadata_kind="coordinates_txt",
        ),
    ]


def parse_float(raw_value: str) -> float:
    value = float(raw_value)
    if not -90.0 <= value <= 90.0 and not -180.0 <= value <= 180.0:
        raise ValueError(f"Invalid coordinate value: {raw_value}")
    return value


def load_im2gps_records(config: DatasetConfig) -> tuple[list[SourceImage], list[SkippedRow]]:
    records_by_name: dict[str, SourceImage] = {}
    with config.metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_id = (row.get("IMG_ID") or "").strip()
            if not img_id:
                continue

            source_path = Path(img_id)
            if not source_path.is_absolute():
                source_path = config.images_dir / img_id
            if not source_path.exists():
                continue

            lat = float((row.get("LAT") or "").strip())
            lon = float((row.get("LON") or "").strip())
            records_by_name[source_path.name] = SourceImage(
                source_dataset=config.name,
                source_path=source_path.resolve(),
                lat=lat,
                lon=lon,
            )

    records: list[SourceImage] = []
    skipped: list[SkippedRow] = []
    for image_path in sorted(path for path in config.images_dir.iterdir() if path.is_file()):
        record = records_by_name.get(image_path.name)
        if record is None:
            skipped.append(
                SkippedRow(
                    source_dataset=config.name,
                    source_path=str(image_path.resolve()),
                    lat="",
                    lon="",
                    reason="missing_metadata",
                )
            )
            continue
        records.append(record)

    return records, skipped


def load_yfcc_records(config: DatasetConfig) -> tuple[list[SourceImage], list[SkippedRow]]:
    records_by_name: dict[str, SourceImage] = {}
    with config.metadata_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            source_path = Path(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            records_by_name[source_path.name] = SourceImage(
                source_dataset=config.name,
                source_path=(config.images_dir / source_path.name).resolve(),
                lat=lat,
                lon=lon,
            )

    records: list[SourceImage] = []
    skipped: list[SkippedRow] = []
    for image_path in sorted(path for path in config.images_dir.iterdir() if path.is_file()):
        record = records_by_name.get(image_path.name)
        if record is None:
            skipped.append(
                SkippedRow(
                    source_dataset=config.name,
                    source_path=str(image_path.resolve()),
                    lat="",
                    lon="",
                    reason="missing_metadata",
                )
            )
            continue
        records.append(record)

    return records, skipped


def load_dataset_records(config: DatasetConfig) -> tuple[list[SourceImage], list[SkippedRow]]:
    if config.metadata_kind == "csv":
        return load_im2gps_records(config)
    if config.metadata_kind == "coordinates_txt":
        return load_yfcc_records(config)
    raise ValueError(f"Unsupported metadata kind: {config.metadata_kind}")


def classify_from_country(country_info: CountryInfo) -> str | None:
    country_key = normalize_country_name(country_info.name)
    continent = country_info.continent

    if continent == "South America" or country_key in LATINO_NORTH_AMERICA:
        return "latino"
    if country_key in INDIAN_SUBCONTINENT:
        return "indian"
    if country_key in EAST_ASIA:
        return "east asian"
    if country_key in SOUTHEAST_ASIA:
        return "southeast asian"
    if country_key in MIDDLE_EAST_AND_NORTH_AFRICA:
        return "middle eastern"
    if continent == "Africa":
        return "black"
    if continent in {"Europe", "North America", "Oceania"}:
        return "white"
    if continent == "Asia":
        return "middle eastern"
    return None


def in_box(lat: float, lon: float, *, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> bool:
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def classify_from_coordinates(lat: float, lon: float) -> str | None:
    if in_box(lat, lon, lat_min=-60.0, lat_max=35.0, lon_min=-120.0, lon_max=-30.0):
        return "latino"
    if in_box(lat, lon, lat_min=-35.0, lat_max=20.0, lon_min=-25.0, lon_max=60.0):
        return "black"
    if in_box(lat, lon, lat_min=10.0, lat_max=48.0, lon_min=-18.0, lon_max=75.0):
        return "middle eastern"
    if in_box(lat, lon, lat_min=-2.0, lat_max=38.0, lon_min=60.0, lon_max=97.0):
        return "indian"
    if in_box(lat, lon, lat_min=-12.0, lat_max=28.0, lon_min=92.0, lon_max=141.0):
        return "southeast asian"
    if in_box(lat, lon, lat_min=20.0, lat_max=55.0, lon_min=100.0, lon_max=150.0):
        return "east asian"
    if in_box(lat, lon, lat_min=35.0, lat_max=75.0, lon_min=-32.0, lon_max=60.0):
        return "white"
    if in_box(lat, lon, lat_min=10.0, lat_max=85.0, lon_min=-170.0, lon_max=-50.0):
        return "white"
    if in_box(lat, lon, lat_min=-55.0, lat_max=10.0, lon_min=-180.0, lon_max=-130.0):
        return "white"
    if in_box(lat, lon, lat_min=-50.0, lat_max=10.0, lon_min=110.0, lon_max=180.0):
        return "white"
    return None


def classify_attribute(lat: float, lon: float, lookup: CountryBoundaryLookup) -> tuple[str | None, CountryInfo | None]:
    country_info = lookup.lookup(lat, lon)
    if country_info is not None:
        attribute = classify_from_country(country_info)
        if attribute is not None:
            return attribute, country_info

    return classify_from_coordinates(lat, lon), country_info


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Re-run with --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def format_coordinate(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_image(source_path: Path, destination_path: Path, mode: str) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        destination_path.symlink_to(source_path)
        return "symlink"

    if mode == "copy":
        shutil.copy2(source_path, destination_path)
        return "copy"

    try:
        shutil.copy2(source_path, destination_path)
        return "copy"
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        if destination_path.exists() or destination_path.is_symlink():
            destination_path.unlink()
        destination_path.symlink_to(source_path)
        return "symlink"


def build_summary_rows(counts: Counter[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for attribute in ATTRIBUTE_ORDER:
        slug = ATTRIBUTE_TO_SLUG[attribute]
        for source_dataset in ["im2gps3k", "yfcc4k"]:
            rows.append(
                {
                    "attribute": attribute,
                    "attribute_slug": slug,
                    "source_dataset": source_dataset,
                    "count": str(counts.get((attribute, source_dataset), 0)),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    manifest_csv = (
        args.manifest_csv.resolve()
        if args.manifest_csv is not None and args.manifest_csv.is_absolute()
        else (repo_root / args.manifest_csv).resolve()
        if args.manifest_csv is not None
        else output_root / "filtered_images.csv"
    )
    skipped_csv = (
        args.skipped_csv.resolve()
        if args.skipped_csv is not None and args.skipped_csv.is_absolute()
        else (repo_root / args.skipped_csv).resolve()
        if args.skipped_csv is not None
        else output_root / "skipped_images.csv"
    )
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None and args.summary_csv.is_absolute()
        else (repo_root / args.summary_csv).resolve()
        if args.summary_csv is not None
        else output_root / "summary.csv"
    )
    boundaries_path = (
        args.country_boundaries.resolve()
        if args.country_boundaries.is_absolute()
        else (repo_root / args.country_boundaries).resolve()
    )

    ensure_output_root(output_root, overwrite=args.overwrite)
    lookup = CountryBoundaryLookup(boundaries_path)

    manifest_rows: list[ManifestRow] = []
    skipped_rows: list[SkippedRow] = []
    summary_counts: Counter[tuple[str, str]] = Counter()
    materialization_counts: Counter[str] = Counter()

    for config in get_dataset_configs():
        records, dataset_skipped = load_dataset_records(config)
        skipped_rows.extend(dataset_skipped)

        for record in records:
            attribute, country_info = classify_attribute(record.lat, record.lon, lookup)
            if attribute is None:
                skipped_rows.append(
                    SkippedRow(
                        source_dataset=record.source_dataset,
                        source_path=str(record.source_path),
                        lat=format_coordinate(record.lat),
                        lon=format_coordinate(record.lon),
                        reason="unmatched_region",
                    )
                )
                continue

            attribute_slug = ATTRIBUTE_TO_SLUG[attribute]
            destination_dir = output_root / attribute_slug / record.source_dataset
            destination_path = destination_dir / record.source_path.name
            materialization_mode = materialize_image(record.source_path, destination_path, args.materialize_mode)
            materialization_counts[materialization_mode] += 1

            relative_output_path = destination_path.relative_to(repo_root)
            manifest_rows.append(
                ManifestRow(
                    source_path=str(record.source_path),
                    dataset_path=relative_output_path.as_posix(),
                    lat=format_coordinate(record.lat),
                    lon=format_coordinate(record.lon),
                    attribute=attribute,
                    attribute_slug=attribute_slug,
                    region=ATTRIBUTE_TO_REGION[attribute],
                    source_dataset=record.source_dataset,
                    country=country_info.name if country_info is not None else "",
                    continent=country_info.continent if country_info is not None else "",
                )
            )
            summary_counts[(attribute, record.source_dataset)] += 1

    write_csv(
        manifest_csv,
        MANIFEST_COLUMNS,
        [row.__dict__ for row in manifest_rows],
    )
    write_csv(
        skipped_csv,
        SKIPPED_COLUMNS,
        [row.__dict__ for row in skipped_rows],
    )
    write_csv(summary_csv, SUMMARY_COLUMNS, build_summary_rows(summary_counts))

    print(f"Created {len(manifest_rows)} filtered images in {output_root}")
    print(f"Wrote manifest to {manifest_csv}")
    print(f"Wrote skipped report to {skipped_csv}")
    print(f"Wrote summary to {summary_csv}")
    print(
        "Materialization: "
        + ", ".join(
            f"{mode}={materialization_counts.get(mode, 0)}" for mode in ["copy", "symlink"]
        )
    )
    for attribute in ATTRIBUTE_ORDER:
        counts = [
            f"{dataset}={summary_counts.get((attribute, dataset), 0)}"
            for dataset in ["im2gps3k", "yfcc4k"]
        ]
        print(f"{ATTRIBUTE_TO_SLUG[attribute]}: {', '.join(counts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
