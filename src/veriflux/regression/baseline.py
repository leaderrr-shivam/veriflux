"""Baseline snapshot management and regression detection.

Captures metric baseline snapshots and alerts on performance degradation beyond tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional


@dataclass(slots=True, frozen=True)
class ComparisonResult:
    """Outcome of comparing current run metrics against baseline.

    Attributes:
        has_regressions: True if any metric degraded beyond tolerance.
        regressions: List of degraded metric descriptions.
        improvements: List of improved metric descriptions.
        unchanged: List of stable metric descriptions.
        diff: Mapping of metric name to signed delta.
    """

    has_regressions: bool
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    diff: Dict[str, float] = field(default_factory=dict)


class BaselineManager:
    """Manages versioned metric baselines and regression detection."""

    def __init__(self, directory: str = ".veriflux_baselines", default_tolerance: float = 0.05) -> None:
        self.directory = directory
        self.default_tolerance = default_tolerance
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, f"{name}.json")

    def save(self, name: str, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save metric snapshot as a named baseline."""
        payload = {
            "name": name,
            "timestamp": time.time(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        with open(self._path(name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, name: str) -> Dict[str, float]:
        """Load baseline metrics by name."""
        path = self._path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline '{name}' not found in {self.directory}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("metrics", {})

    def compare(
        self,
        name: str,
        current_metrics: Dict[str, float],
        tolerance: Optional[float] = None,
    ) -> ComparisonResult:
        """Compare current metrics against saved baseline."""
        baseline = self.load(name)
        tol = tolerance if tolerance is not None else self.default_tolerance

        regressions: List[str] = []
        improvements: List[str] = []
        unchanged: List[str] = []
        diffs: Dict[str, float] = {}

        for metric, base_val in baseline.items():
            if metric not in current_metrics:
                continue

            cur_val = current_metrics[metric]
            delta = cur_val - base_val
            diffs[metric] = round(delta, 4)

            # Higher is better assumption for standard eval metrics (accuracy, groundedness)
            if delta < -tol:
                regressions.append(f"{metric}: degraded from {base_val:.3f} to {cur_val:.3f} (Δ {delta:+.3f})")
            elif delta > tol:
                improvements.append(f"{metric}: improved from {base_val:.3f} to {cur_val:.3f} (Δ {delta:+.3f})")
            else:
                unchanged.append(f"{metric}: stable at {cur_val:.3f}")

        return ComparisonResult(
            has_regressions=len(regressions) > 0,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            diff=diffs,
        )

