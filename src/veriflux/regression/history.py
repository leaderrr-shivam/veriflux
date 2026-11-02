"""Historical metrics time-series tracking and trend analysis.

Maintains longitudinal performance runs and calculates trajectory trends.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class MetricsHistory:
    """Stores and analyzes historical evaluation runs across CI/CD commits."""

    def __init__(self, storage_path: str = ".veriflux_history.json") -> None:
        self.storage_path = storage_path
        self._history: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = []

    def _save(self) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def record(self, run_id: str, metrics: Dict[str, float], timestamp: Optional[float] = None) -> None:
        """Record evaluation metrics for a run."""
        entry = {
            "run_id": run_id,
            "timestamp": timestamp or time.time(),
            "metrics": metrics,
        }
        self._history.append(entry)
        self._save()

    def get_history(self, metric: str, last_n: Optional[int] = None) -> List[Tuple[str, float, float]]:
        """Retrieve history for a metric as list of (run_id, value, timestamp)."""
        points: List[Tuple[str, float, float]] = []
        for entry in self._history:
            if metric in entry.get("metrics", {}):
                points.append((entry["run_id"], float(entry["metrics"][metric]), entry["timestamp"]))
        if last_n is not None:
            points = points[-last_n:]
        return points

    def trend(self, metric: str, window: int = 5) -> str:
        """Calculate recent metric trend: 'improving', 'degrading', or 'stable'."""
        points = self.get_history(metric, last_n=window)
        if len(points) < 2:
            return "stable"

        vals = [p[1] for p in points]
        delta = vals[-1] - vals[0]

        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "degrading"
        return "stable"

