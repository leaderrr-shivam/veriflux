"""Detailed multi-metric comparative analysis.

Computes absolute and relative percentages between baseline runs and candidate releases.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import Dict, List, Optional


class MetricStatus(enum.Enum):
    """Status classification of metric variance."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    STABLE = "stable"


@dataclass(slots=True, frozen=True)
class MetricDiff:
    """Detailed differential breakdown for a single metric.

    Attributes:
        metric: Name of metric.
        baseline_value: Previous benchmark score.
        current_value: Candidate benchmark score.
        delta: Signed absolute difference.
        delta_percent: Percentage change relative to baseline.
        status: Classification (IMPROVED, REGRESSED, STABLE).
    """

    metric: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    status: MetricStatus


class MetricComparator:
    """Computes fine-grained metric differences across test suites."""

    def compare(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
        tolerances: Optional[Dict[str, float]] = None,
        default_tolerance: float = 0.05,
    ) -> List[MetricDiff]:
        """Compute metric diffs between baseline and current runs."""
        diffs: List[MetricDiff] = []
        tol_dict = tolerances or {}

        for metric, base_val in baseline.items():
            if metric not in current:
                continue

            cur_val = current[metric]
            delta = cur_val - base_val
            pct = (delta / base_val * 100.0) if base_val != 0.0 else 0.0
            tol = tol_dict.get(metric, default_tolerance)

            if delta < -tol:
                status = MetricStatus.REGRESSED
            elif delta > tol:
                status = MetricStatus.IMPROVED
            else:
                status = MetricStatus.STABLE

            diffs.append(
                MetricDiff(
                    metric=metric,
                    baseline_value=round(base_val, 4),
                    current_value=round(cur_val, 4),
                    delta=round(delta, 4),
                    delta_percent=round(pct, 2),
                    status=status,
                )
            )

        return diffs

