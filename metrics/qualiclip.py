from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from metrics.base import MetricResult


class QualiCLIPMetric:
    metric_name = "qualiclip"

    def __init__(self, device: str | None = None, metric_name: str = "qualiclip") -> None:
        self.device = device
        self.pyiqa_metric_name = metric_name
        self._metric: Any | None = None

    def _load_metric(self) -> Any:
        if self._metric is None:
            try:
                import pyiqa
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("QualiCLIP requires IQA-PyTorch: install the `pyiqa` package.") from exc

            kwargs = {"device": self.device} if self.device is not None else {}
            self._metric = pyiqa.create_metric(self.pyiqa_metric_name, **kwargs)
        return self._metric

    def __call__(self, image_paths: Iterable[str | Path]) -> MetricResult:
        metric = self._load_metric()
        per_sample = []
        for image_path in image_paths:
            score = metric(str(image_path))
            if hasattr(score, "item"):
                score = score.item()
            per_sample.append({"image_path": str(image_path), self.metric_name: float(score)})
        scores = [row[self.metric_name] for row in per_sample]
        summary = {"count": len(scores), "mean": sum(scores) / len(scores)} if scores else {"count": 0}
        return MetricResult(self.metric_name, summary, per_sample)
