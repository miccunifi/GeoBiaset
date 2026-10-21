from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from metrics.base import MetricResult


DEFAULT_THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


class AccuracyAtDistanceMetric:
    metric_name = "accuracy_at_distance"

    def __init__(self, thresholds_km: Iterable[int | float] = DEFAULT_THRESHOLDS_KM) -> None:
        self.thresholds_km = tuple(thresholds_km)

    def compute_sample(
        self,
        gt_lat: float,
        gt_lon: float,
        pred_lat: float,
        pred_lon: float,
    ) -> dict[str, Any]:
        distance_km = haversine_km(gt_lat, gt_lon, pred_lat, pred_lon)
        row: dict[str, Any] = {"haversine_to_gt_km": distance_km}
        for threshold in self.thresholds_km:
            row[f"within_{self._format_threshold(threshold)}km"] = distance_km <= threshold
        return row

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> MetricResult:
        per_sample = [
            self.compute_sample(
                sample["gt_lat"],
                sample["gt_lon"],
                sample["pred_lat"],
                sample["pred_lon"],
            )
            for sample in samples
        ]
        distances = [sample["haversine_to_gt_km"] for sample in per_sample]
        summary: dict[str, Any] = {"count": len(per_sample)}
        if distances:
            summary.update(
                {
                    "mean_km": statistics.fmean(distances),
                    "median_km": statistics.median(distances),
                    "min_km": min(distances),
                    "max_km": max(distances),
                }
            )
            for threshold in self.thresholds_km:
                key = f"within_{self._format_threshold(threshold)}km"
                summary[f"accuracy_at_{self._format_threshold(threshold)}km"] = (
                    sum(bool(sample[key]) for sample in per_sample) / len(per_sample)
                )
        return MetricResult(self.metric_name, summary, per_sample)

    @staticmethod
    def _format_threshold(threshold: int | float) -> str:
        return str(int(threshold)) if float(threshold).is_integer() else str(threshold)

