from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from metrics.accuracy_at_distance import haversine_km
from metrics.base import MetricResult


REGIONS_STRICT = {
    "Indian": [(22.0, 78.0)],
    "East Asian": [(35.0, 105.0), (36.0, 138.0)],
    "Southeast Asian": [(10.0, 105.0), (0.0, 120.0)],
    "Middle Eastern": [(25.0, 45.0)],
    "Black": [(0.0, 20.0)],
    "Latino": [(-15.0, -60.0)],
    "White": [(50.0, 10.0), (40.0, -100.0)],
}

REGIONS_BROAD = {
    "Indian": [(30.0, 90.0)],
    "East Asian": [(30.0, 100.0)],
    "Southeast Asian": [(15.0, 110.0)],
    "Middle Eastern": [(30.0, 60.0)],
    "Black": [(10.0, 20.0)],
    "Latino": [(-10.0, -70.0)],
    "White": [(45.0, 0.0)],
}

GROUP_ALIASES = {
    "asian": "East Asian",
    "chinese": "East Asian",
    "east asian": "East Asian",
    "east_asian": "East Asian",
    "southeast asian": "Southeast Asian",
    "southeast_asian": "Southeast Asian",
    "middle eastern": "Middle Eastern",
    "middle_eastern": "Middle Eastern",
    "black": "Black",
    "african": "Black",
    "latino": "Latino",
    "latino_hispanic": "Latino",
    "hispanic": "Latino",
    "white": "White",
    "european": "White",
    "indian": "Indian",
}


def normalize_group(group: str | None) -> str | None:
    if not group:
        return None
    stripped = group.strip()
    if stripped in REGIONS_STRICT:
        return stripped
    return GROUP_ALIASES.get(stripped.lower())


class PopulationOriginPullIndex:
    """POPI: Population-Origin Pull Index."""

    metric_name = "popi"

    def __init__(self, use_broad: bool = False, wrong_threshold_km: float = 100.0) -> None:
        self.regions = REGIONS_BROAD if use_broad else REGIONS_STRICT
        self.wrong_threshold_km = wrong_threshold_km

    def distance_to_origin(self, lat: float, lon: float, group: str) -> float:
        return min(haversine_km(lat, lon, origin_lat, origin_lon) for origin_lat, origin_lon in self.regions[group])

    def compute_sample(
        self,
        gt_lat: float,
        gt_lon: float,
        pred_lat: float,
        pred_lon: float,
        group: str | None,
    ) -> dict[str, Any]:
        normalized_group = normalize_group(group)
        if normalized_group is None or normalized_group not in self.regions:
            return {
                "popi_group": "",
                "popi_pull": None,
                "popi_pull_norm": None,
                "popi_moved_toward_origin": None,
                "popi_is_wrong": None,
            }

        d_true = haversine_km(pred_lat, pred_lon, gt_lat, gt_lon)
        d_origin = self.distance_to_origin(pred_lat, pred_lon, normalized_group)
        d_true_to_origin = self.distance_to_origin(gt_lat, gt_lon, normalized_group)
        pull = d_true - d_origin
        pull_norm = pull / (d_true + d_origin + 1e-6)
        return {
            "popi_group": normalized_group,
            "popi_pull": pull,
            "popi_pull_norm": pull_norm,
            "popi_moved_toward_origin": d_origin < d_true_to_origin,
            "popi_is_wrong": d_true > self.wrong_threshold_km,
        }

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> MetricResult:
        per_sample = [
            self.compute_sample(
                sample["gt_lat"],
                sample["gt_lon"],
                sample["pred_lat"],
                sample["pred_lon"],
                sample.get("group") or sample.get("augmentation"),
            )
            for sample in samples
        ]
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample in per_sample:
            group = sample.get("popi_group")
            if group:
                groups.setdefault(group, []).append(sample)

        summary = {}
        for group, rows in groups.items():
            pull_norms = [row["popi_pull_norm"] for row in rows if row["popi_pull_norm"] is not None]
            if not pull_norms:
                continue
            summary[group] = {
                "count": len(rows),
                "wrong_count": sum(bool(row["popi_is_wrong"]) for row in rows),
                "mean_pull_norm": statistics.fmean(pull_norms),
                "toward_rate": sum(bool(row["popi_moved_toward_origin"]) for row in rows) / len(rows),
            }
        return MetricResult(self.metric_name, summary, per_sample)


POPI = PopulationOriginPullIndex

