from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


COMMON_COUNTRY_NAME_COLUMNS = (
    "NAME",
    "ADMIN",
    "NAME_EN",
    "NAME_LONG",
    "FORMAL_EN",
    "SOVEREIGNT",
    "GEOUNIT",
    "COUNTRY",
    "name",
    "admin",
    "country",
)


class CountryLookupError(RuntimeError):
    """Raised when the country boundary dataset cannot be loaded safely."""


class CountryLookup:
    """Offline country lookup for latitude/longitude coordinates.

    The public API accepts coordinates as ``(lat, lon)``. Internally, geometry
    predicates use ``(x, y) = (lon, lat)``, which is what Shapely expects.

    Example:
        lookup = CountryLookup("data/ne_110m_admin_0_countries.shp")
        country = lookup.lookup(43.7696, 11.2558)
        print(country)  # "Italy" with the standard Natural Earth naming
    """

    def __init__(
        self,
        boundaries_path: str | Path,
        *,
        name_column: str | None = None,
        target_crs: str = "EPSG:4326",
    ) -> None:
        self.boundaries_path = Path(boundaries_path).expanduser().resolve()
        self.target_crs = target_crs

        if not self.boundaries_path.exists():
            raise FileNotFoundError(f"Country boundaries file does not exist: {self.boundaries_path}")

        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:
            raise CountryLookupError(
                "geopandas is required for offline country lookup. Install it with `uv sync` "
                "or `pip install -r requirements.txt`."
            ) from exc

        try:
            countries = gpd.read_file(self.boundaries_path)
        except Exception as exc:
            raise CountryLookupError(f"Could not read country boundaries file {self.boundaries_path}: {exc}") from exc

        if countries.empty:
            raise ValueError(f"Country boundaries file has no rows: {self.boundaries_path}")
        if countries.geometry.name not in countries.columns:
            raise ValueError(f"Country boundaries file has no geometry column: {self.boundaries_path}")

        countries = countries[~countries.geometry.isna() & ~countries.geometry.is_empty].copy()
        if countries.empty:
            raise ValueError(f"Country boundaries file has no usable geometries: {self.boundaries_path}")

        self.name_column = self._resolve_name_column(countries.columns, name_column)

        if countries.crs is None:
            # Natural Earth country files are distributed in WGS84; setting the
            # CRS here keeps common local exports usable while still preserving
            # the public API as latitude/longitude.
            countries = countries.set_crs(target_crs, allow_override=True)
        elif countries.crs.to_string() != target_crs:
            countries = countries.to_crs(target_crs)

        self._countries = countries[[self.name_column, countries.geometry.name]].reset_index(drop=True)
        self._spatial_index = self._countries.sindex

    @staticmethod
    def _resolve_name_column(columns: Iterable[str], requested: str | None) -> str:
        columns_list = list(columns)
        if requested is not None:
            if requested in columns_list:
                return requested
            raise ValueError(
                f"Country-name column '{requested}' was not found. Available columns: {', '.join(columns_list)}"
            )

        lower_to_actual = {column.lower(): column for column in columns_list}
        for candidate in COMMON_COUNTRY_NAME_COLUMNS:
            if candidate in columns_list:
                return candidate
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                return actual

        raise ValueError(
            "Could not auto-detect a country-name column. Pass `name_column=` explicitly. "
            f"Tried: {', '.join(COMMON_COUNTRY_NAME_COLUMNS)}. "
            f"Available columns: {', '.join(columns_list)}"
        )

    @staticmethod
    def _validate_coordinates(lat: float, lon: float) -> None:
        if not math.isfinite(lat):
            raise ValueError(f"Latitude must be finite, got {lat!r}")
        if not math.isfinite(lon):
            raise ValueError(f"Longitude must be finite, got {lon!r}")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"Latitude must be between -90 and 90 degrees, got {lat!r}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"Longitude must be between -180 and 180 degrees, got {lon!r}")

    @staticmethod
    def _clean_country_name(value: object) -> str | None:
        if value is None:
            return None
        # NaN is not equal to itself.
        if value != value:
            return None
        name = str(value).strip()
        return name or None

    def lookup(self, lat: float, lon: float) -> str | None:
        """Return the country containing ``(lat, lon)``, or ``None`` if unmatched.

        Exact containment is tried first with ``shapely.contains_xy`` for speed.
        Border and coastline points then fall back to ``covers`` so points that
        lie exactly on a polygon boundary are handled gracefully.
        """

        self._validate_coordinates(lat, lon)

        from shapely import contains_xy
        from shapely.geometry import Point

        point = Point(lon, lat)
        candidate_positions = list(self._spatial_index.query(point))
        if not candidate_positions:
            return None

        candidates = self._countries.iloc[candidate_positions]

        for geometry, raw_name in zip(candidates.geometry, candidates[self.name_column], strict=True):
            if contains_xy(geometry, lon, lat):
                return self._clean_country_name(raw_name)

        for geometry, raw_name in zip(candidates.geometry, candidates[self.name_column], strict=True):
            if geometry.covers(point):
                return self._clean_country_name(raw_name)

        return None

    def lookup_many(self, coords: list[tuple[float, float]]) -> list[str | None]:
        """Return country names for many ``(lat, lon)`` pairs using the cached dataset."""

        return [self.lookup(lat, lon) for lat, lon in coords]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline country lookup for a latitude/longitude point.")
    parser.add_argument("--data", required=True, type=Path, help="Local country boundary file: .shp, .geojson, or .gpkg.")
    parser.add_argument("--lat", required=True, type=float, help="Latitude in degrees.")
    parser.add_argument("--lon", required=True, type=float, help="Longitude in degrees.")
    parser.add_argument(
        "--name-column",
        default=None,
        help="Optional country-name column. If omitted, common Natural Earth columns are auto-detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lookup = CountryLookup(args.data, name_column=args.name_column)
    country = lookup.lookup(args.lat, args.lon)
    print(country if country is not None else "None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
