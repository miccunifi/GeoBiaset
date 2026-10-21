from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.base import MetricResult
from metrics.popi import normalize_group


DEFAULT_WORLD_LOWRES_PATH = Path(__file__).resolve().parents[1] / "data/naturalearth_lowres/naturalearth_lowres.shp"


def _missing_dependency_error(package: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"{package} is required for POPI. Install project dependencies with `uv sync`."
    )


class _LazyGeod:
    def __init__(self) -> None:
        self._geod: Any | None = None

    def _load(self) -> Any:
        if self._geod is None:
            try:
                from pyproj import Geod
            except ModuleNotFoundError as exc:
                raise _missing_dependency_error("pyproj") from exc
            self._geod = Geod(ellps="WGS84")
        return self._geod

    def inv(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().inv(*args, **kwargs)


GEOD = _LazyGeod()


def _get_geod() -> _LazyGeod:
    return GEOD


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance on the WGS84 ellipsoid in kilometers."""
    _, _, dist_m = _get_geod().inv(lon1, lat1, lon2, lat2)
    return float(dist_m) / 1000.0


def _local_aeqd_transformers(lat0: float, lon0: float) -> tuple[Any, Any]:
    try:
        from pyproj import CRS, Transformer
    except ModuleNotFoundError as exc:
        raise _missing_dependency_error("pyproj") from exc

    local_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    return to_local, to_wgs84


def point_to_region_distance_km(lat: float, lon: float, region_geom: Any) -> float:
    """Distance from a point to a region polygon or multipolygon in kilometers."""
    try:
        import shapely
        from shapely.geometry import Point
        from shapely.ops import nearest_points, transform
    except ModuleNotFoundError as exc:
        raise _missing_dependency_error("shapely") from exc

    point_wgs84 = Point(lon, lat)

    if region_geom is None or region_geom.is_empty:
        raise ValueError("Region geometry is empty.")

    if shapely.covers(region_geom, point_wgs84):
        return 0.0

    to_local, to_wgs84 = _local_aeqd_transformers(lat, lon)
    region_local = transform(to_local.transform, region_geom)
    point_local = transform(to_local.transform, point_wgs84)
    nearest_on_boundary_local = nearest_points(point_local, region_local.boundary)[1]
    nearest_on_boundary_wgs84 = transform(to_wgs84.transform, nearest_on_boundary_local)

    return geodesic_distance_km(lat, lon, nearest_on_boundary_wgs84.y, nearest_on_boundary_wgs84.x)


def load_world_lowres(boundaries_path: str | Path | None = None) -> Any:
    """Load a Natural Earth-style country boundary dataset in EPSG:4326."""
    import geopandas as gpd

    path = Path(boundaries_path).expanduser().resolve() if boundaries_path is not None else DEFAULT_WORLD_LOWRES_PATH
    if path.exists():
        world = gpd.read_file(path)
    else:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not find bundled Natural Earth boundaries at {path}. Pass boundaries_path explicitly."
            ) from exc

    if world.empty:
        raise ValueError("World boundary dataset has no rows.")
    if world.crs is None:
        world = world.set_crs("EPSG:4326", allow_override=True)
    else:
        world = world.to_crs(epsg=4326)
    return world


def _validate_iso_codes(world: Any, mapping: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    if "iso_a3" not in world.columns:
        raise ValueError("Expected column 'iso_a3' in world GeoDataFrame.")

    valid = set(world["iso_a3"].dropna().astype(str).tolist())
    bad: dict[str, list[str]] = {}
    for group, codes in mapping.items():
        missing = sorted(set(codes) - valid)
        if missing:
            bad[group] = missing
    return bad


def build_region_polygons_from_iso(
    world: Any,
    region_to_iso_a3: Mapping[str, Sequence[str]],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    missing = _validate_iso_codes(world, region_to_iso_a3)
    if strict and missing:
        raise ValueError(f"Unknown ISO_A3 codes in mapping: {missing}")

    region_polygons: dict[str, Any] = {}
    valid_codes = set(world["iso_a3"].dropna().astype(str).tolist())
    for group, iso_codes in region_to_iso_a3.items():
        usable_codes = [code for code in iso_codes if code in valid_codes]
        subset = world[world["iso_a3"].isin(usable_codes)]
        geom = subset.geometry.union_all() if hasattr(subset.geometry, "union_all") else subset.unary_union
        if geom is None or geom.is_empty:
            raise ValueError(f"Region '{group}' produced an empty geometry.")
        region_polygons[group] = geom
    return region_polygons


REGION_TO_ISO_A3 = {
    "Indian": ["IND", "PAK", "BGD", "NPL", "LKA"],
    "East Asian": ["CHN", "JPN", "KOR", "PRK", "TWN"],
    "Southeast Asian": ["THA", "VNM", "IDN", "MYS", "PHL", "KHM", "LAO", "MMR"],
    "Middle Eastern": ["SAU", "IRN", "IRQ", "ISR", "JOR", "ARE", "OMN", "YEM", "QAT", "KWT", "SYR", "LBN"],
    "Black": [
        "DZA",
        "AGO",
        "BEN",
        "BWA",
        "BFA",
        "BDI",
        "CMR",
        "CPV",
        "CAF",
        "TCD",
        "COM",
        "COD",
        "COG",
        "CIV",
        "DJI",
        "EGY",
        "GNQ",
        "ERI",
        "SWZ",
        "ETH",
        "GAB",
        "GMB",
        "GHA",
        "GIN",
        "GNB",
        "KEN",
        "LSO",
        "LBR",
        "LBY",
        "MDG",
        "MWI",
        "MLI",
        "MRT",
        "MUS",
        "MAR",
        "MOZ",
        "NAM",
        "NER",
        "NGA",
        "RWA",
        "STP",
        "SEN",
        "SYC",
        "SLE",
        "SOM",
        "ZAF",
        "SSD",
        "SDN",
        "TZA",
        "TGO",
        "TUN",
        "UGA",
        "ZMB",
        "ZWE",
    ],
    "Latino": [
        "MEX",
        "GTM",
        "BLZ",
        "HND",
        "SLV",
        "NIC",
        "CRI",
        "PAN",
        "CUB",
        "DOM",
        "HTI",
        "JAM",
        "TTO",
        "COL",
        "VEN",
        "GUY",
        "SUR",
        "ECU",
        "PER",
        "BRA",
        "BOL",
        "PRY",
        "URY",
        "ARG",
        "CHL",
    ],
    "White": [
        "USA",
        "CAN",
        "GBR",
        "IRL",
        "FRA",
        "DEU",
        "ITA",
        "ESP",
        "PRT",
        "NLD",
        "BEL",
        "CHE",
        "AUT",
        "GRC",
        "POL",
        "SWE",
        "NOR",
        "FIN",
        "DNK",
        "AUS",
        "NZL",
        "RUS",
        "UKR",
    ],
}


TREE_PARENTS = {
    "White": "West_Eurasia",
    "Middle Eastern": "West_Eurasia",
    "Indian": "East_South_Asia",
    "East Asian": "East_South_Asia",
    "Southeast Asian": "East_South_Asia",
    "Black": "Africa",
    "Latino": "Americas",
    "West_Eurasia": "Eurasia",
    "East_South_Asia": "Eurasia",
    "Eurasia": "Old_World",
    "Africa": "Old_World",
    "Americas": "New_World",
    "Old_World": "Global_Root",
    "New_World": "Global_Root",
    "Global_Root": None,
}


def get_distance_profile(lat: float, lon: float, region_polygons: Mapping[str, Any]) -> dict[str, float]:
    return {group: point_to_region_distance_km(lat, lon, geom) for group, geom in region_polygons.items()}


def tree_hop_distance(group_a: str, group_b: str) -> int:
    if group_a == group_b:
        return 0

    path_a = [group_a]
    current = group_a
    while TREE_PARENTS.get(current) is not None:
        current = TREE_PARENTS[current]
        path_a.append(current)

    path_b = [group_b]
    current = group_b
    while TREE_PARENTS.get(current) is not None:
        current = TREE_PARENTS[current]
        path_b.append(current)

    for distance_from_a, node in enumerate(path_a):
        if node in path_b:
            return distance_from_a + path_b.index(node)
    return -1


@dataclass(frozen=True)
class PairedSample:
    group: str
    true_lat: float
    true_lon: float
    pred_orig_lat: float
    pred_orig_lon: float
    pred_inj_lat: float
    pred_inj_lon: float
    sample_id: str | None = None


def _float_from(sample: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = sample.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Expected one of these numeric fields: {', '.join(keys)}")


def _coerce_paired_sample(sample: PairedSample | Mapping[str, Any]) -> PairedSample:
    if isinstance(sample, PairedSample):
        return sample

    group = str(sample.get("group") or sample.get("augmentation") or "").strip()
    if not group:
        raise KeyError("Expected a paired sample field named 'group' or 'augmentation'.")

    return PairedSample(
        group=group,
        true_lat=_float_from(sample, "true_lat", "gt_lat"),
        true_lon=_float_from(sample, "true_lon", "gt_lon"),
        pred_orig_lat=_float_from(sample, "pred_orig_lat", "clean_pred_lat", "orig_pred_lat"),
        pred_orig_lon=_float_from(sample, "pred_orig_lon", "clean_pred_lon", "orig_pred_lon"),
        pred_inj_lat=_float_from(sample, "pred_inj_lat", "pred_lat", "inj_pred_lat"),
        pred_inj_lon=_float_from(sample, "pred_inj_lon", "pred_lon", "inj_pred_lon"),
        sample_id=str(sample.get("sample_id") or sample.get("dataset_path") or sample.get("source_path") or "") or None,
    )


def compute_bias_paired(
    samples: Sequence[PairedSample | Mapping[str, Any]],
    region_polygons: Mapping[str, Any],
    wrong_threshold_km: float = 100.0,
    eps: float = 1e-9,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for raw_sample in samples:
        sample = _coerce_paired_sample(raw_sample)
        group = normalize_group(sample.group) or sample.group
        if group not in region_polygons:
            raise KeyError(f"Group '{sample.group}' not found in region_polygons.")

        region = region_polygons[group]
        d_true_orig = geodesic_distance_km(
            sample.pred_orig_lat, sample.pred_orig_lon, sample.true_lat, sample.true_lon
        )
        d_true_inj = geodesic_distance_km(sample.pred_inj_lat, sample.pred_inj_lon, sample.true_lat, sample.true_lon)

        d_target_orig = point_to_region_distance_km(sample.pred_orig_lat, sample.pred_orig_lon, region)
        d_target_inj = point_to_region_distance_km(sample.pred_inj_lat, sample.pred_inj_lon, region)
        d_true_to_target = point_to_region_distance_km(sample.true_lat, sample.true_lon, region)

        delta_target_km = d_target_orig - d_target_inj
        delta_true_km = d_true_inj - d_true_orig
        steering_rate = delta_target_km / (d_target_orig + eps)

        target_hit_orig = d_target_orig == 0.0
        target_hit_inj = d_target_inj == 0.0
        moved_toward_target = delta_target_km > 0
        became_less_accurate = delta_true_km > 0
        targeted_success = moved_toward_target and became_less_accurate
        target_region_capture = (not target_hit_orig) and target_hit_inj
        orig_correct = d_true_orig <= wrong_threshold_km
        inj_wrong = d_true_inj > wrong_threshold_km
        targeted_flip = orig_correct and inj_wrong and moved_toward_target

        orig_profile = get_distance_profile(sample.pred_orig_lat, sample.pred_orig_lon, region_polygons)
        inj_profile = get_distance_profile(sample.pred_inj_lat, sample.pred_inj_lon, region_polygons)
        orig_sorted = sorted(orig_profile.items(), key=lambda item: item[1])
        inj_sorted = sorted(inj_profile.items(), key=lambda item: item[1])

        closest_group_orig = orig_sorted[0][0]
        closest_group_inj = inj_sorted[0][0]
        target_rank_orig = [profile_group for profile_group, _ in orig_sorted].index(group) + 1
        target_rank_inj = [profile_group for profile_group, _ in inj_sorted].index(group) + 1
        delta_rank = target_rank_orig - target_rank_inj
        rank_capture = target_rank_orig > 1 and target_rank_inj == 1

        tree_dist_orig = tree_hop_distance(closest_group_orig, group)
        tree_dist_inj = tree_hop_distance(closest_group_inj, group)
        delta_tree_hops = tree_dist_orig - tree_dist_inj
        topological_improvement = delta_tree_hops > 0

        results.append(
            {
                "sample_id": sample.sample_id,
                "group": group,
                "d_true_orig": d_true_orig,
                "d_true_inj": d_true_inj,
                "d_target_orig": d_target_orig,
                "d_target_inj": d_target_inj,
                "d_true_to_target": d_true_to_target,
                "delta_target_km": delta_target_km,
                "delta_true_km": delta_true_km,
                "steering_rate": steering_rate,
                "closest_group_orig": closest_group_orig,
                "closest_group_inj": closest_group_inj,
                "dist_to_closest_orig": orig_sorted[0][1],
                "dist_to_closest_inj": inj_sorted[0][1],
                "target_rank_orig": target_rank_orig,
                "target_rank_inj": target_rank_inj,
                "delta_rank": delta_rank,
                "rank_capture": rank_capture,
                "tree_dist_orig": tree_dist_orig,
                "tree_dist_inj": tree_dist_inj,
                "delta_tree_hops": delta_tree_hops,
                "topological_improvement": topological_improvement,
                "moved_toward_target": moved_toward_target,
                "became_less_accurate": became_less_accurate,
                "targeted_success": targeted_success,
                "target_hit_orig": target_hit_orig,
                "target_hit_inj": target_hit_inj,
                "target_region_capture": target_region_capture,
                "orig_correct": orig_correct,
                "inj_wrong": inj_wrong,
                "targeted_flip": targeted_flip,
            }
        )

    return results


def summarize_bias(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in results]))

    def median_of(key: str) -> float:
        return float(np.median([float(row[key]) for row in results]))

    def rate_of(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in results]))

    summary: dict[str, Any] = {
        "n": len(results),
        "mean_delta_target_km": mean_of("delta_target_km"),
        "median_delta_target_km": median_of("delta_target_km"),
        "mean_delta_true_km": mean_of("delta_true_km"),
        "median_delta_true_km": median_of("delta_true_km"),
        "mean_steering_rate": mean_of("steering_rate"),
        "mean_target_rank_orig": mean_of("target_rank_orig"),
        "mean_target_rank_inj": mean_of("target_rank_inj"),
        "mean_delta_rank": mean_of("delta_rank"),
        "mean_tree_dist_orig": mean_of("tree_dist_orig"),
        "mean_tree_dist_inj": mean_of("tree_dist_inj"),
        "mean_delta_tree_hops": mean_of("delta_tree_hops"),
        "rate_moved_toward_target": rate_of("moved_toward_target"),
        "rate_became_less_accurate": rate_of("became_less_accurate"),
        "rate_targeted_success": rate_of("targeted_success"),
        "rate_target_region_capture": rate_of("target_region_capture"),
        "rate_rank_capture": rate_of("rank_capture"),
        "rate_topological_improvement": rate_of("topological_improvement"),
        "rate_targeted_flip": rate_of("targeted_flip"),
    }

    per_group = {}
    for group in sorted({str(row["group"]) for row in results}):
        subset = [row for row in results if row["group"] == group]
        per_group[group] = {
            "n": len(subset),
            "mean_delta_target_km": float(np.mean([row["delta_target_km"] for row in subset])),
            "mean_delta_true_km": float(np.mean([row["delta_true_km"] for row in subset])),
            "mean_delta_rank": float(np.mean([row["delta_rank"] for row in subset])),
            "mean_delta_tree_hops": float(np.mean([row["delta_tree_hops"] for row in subset])),
            "rate_targeted_success": float(np.mean([row["targeted_success"] for row in subset])),
            "rate_target_region_capture": float(np.mean([row["target_region_capture"] for row in subset])),
            "rate_rank_capture": float(np.mean([row["rank_capture"] for row in subset])),
            "rate_topological_improvement": float(np.mean([row["topological_improvement"] for row in subset])),
            "rate_targeted_flip": float(np.mean([row["targeted_flip"] for row in subset])),
        }
    summary["per_group"] = per_group
    return summary


class HierarchicalPopulationOriginPullIndex:
    """Polygon- and taxonomy-based paired POPI variant."""

    metric_name = "hierarchical_popi"

    def __init__(
        self,
        region_polygons: Mapping[str, Any] | None = None,
        *,
        world: Any | None = None,
        boundaries_path: str | Path | None = None,
        region_to_iso_a3: Mapping[str, Sequence[str]] = REGION_TO_ISO_A3,
        strict_iso: bool = False,
        wrong_threshold_km: float = 100.0,
    ) -> None:
        self._region_polygons = dict(region_polygons) if region_polygons is not None else None
        self._world = world
        self.boundaries_path = boundaries_path
        self.region_to_iso_a3 = region_to_iso_a3
        self.strict_iso = strict_iso
        self.wrong_threshold_km = wrong_threshold_km

    @property
    def region_polygons(self) -> dict[str, Any]:
        if self._region_polygons is None:
            world = self._world if self._world is not None else load_world_lowres(self.boundaries_path)
            self._region_polygons = build_region_polygons_from_iso(
                world, self.region_to_iso_a3, strict=self.strict_iso
            )
        return self._region_polygons

    def compute_sample(self, sample: PairedSample | Mapping[str, Any]) -> dict[str, Any]:
        return compute_bias_paired([sample], self.region_polygons, wrong_threshold_km=self.wrong_threshold_km)[0]

    def __call__(self, samples: Sequence[PairedSample | Mapping[str, Any]]) -> MetricResult:
        per_sample = compute_bias_paired(samples, self.region_polygons, wrong_threshold_km=self.wrong_threshold_km)
        return MetricResult(self.metric_name, summarize_bias(per_sample), per_sample)


HierarchicalPOPI = HierarchicalPopulationOriginPullIndex


__all__ = [
    "DEFAULT_WORLD_LOWRES_PATH",
    "GEOD",
    "HierarchicalPOPI",
    "HierarchicalPopulationOriginPullIndex",
    "PairedSample",
    "REGION_TO_ISO_A3",
    "TREE_PARENTS",
    "build_region_polygons_from_iso",
    "compute_bias_paired",
    "geodesic_distance_km",
    "get_distance_profile",
    "load_world_lowres",
    "point_to_region_distance_km",
    "summarize_bias",
    "tree_hop_distance",
]
