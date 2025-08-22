#
# For licensing see accompanying LICENSE file.
# Copyright © 2025 Apple Inc. All Rights Reserved.
#
from collections import Counter

MetricName = str


def calculate_percentage(counts: dict[str, int]) -> float:
    """Calculate the percentage given a dict of counts."""
    if not counts:
        return 0.0
    try:
        assert len(counts) == 2
    except AssertionError:
        assert "total" in list(counts.keys())[0]
        return 0.0
    nominator, denominator = 0, 0
    for k, v in counts.items():
        if k.endswith("total"):
            denominator = v
        if k.endswith("cnt"):
            nominator = v
    if denominator > 0:
        return nominator * 100 / denominator
    return 0.0


class CountsAccumulator:
    def __init__(self):
        self._metrics: dict[MetricName, Counter] = {}

    def _accumulate(self, name: MetricName, counts: Counter):
        if name not in self._metrics:
            self._metrics[name] = counts
        else:
            self._metrics[name] += counts

    def accumulate(self, counts_data: dict[MetricName, Counter]):
        for key, cnts in counts_data.items():
            if isinstance(cnts, Counter):
                self._accumulate(key, cnts)

    def get_ratios(self) -> dict[str, float]:
        out = {}
        for key, counts in self._metrics.items():
            ratio = calculate_percentage(counts)
            out[f"{key}_perc"] = ratio

        return out
