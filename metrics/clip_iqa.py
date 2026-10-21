from __future__ import annotations

from metrics.qualiclip import QualiCLIPMetric


class CLIPIqaMetric(QualiCLIPMetric):
    metric_name = "clip_iqa"

    def __init__(self, device: str | None = None, metric_name: str = "clipiqa") -> None:
        super().__init__(device=device, metric_name=metric_name)

