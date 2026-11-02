"""Regression detection and historical baseline analytics for Veriflux.

Provides snapshot management, differential comparison, semantic drift monitoring, and historical trends.
"""
from __future__ import annotations

from veriflux.regression.baseline import BaselineManager, ComparisonResult
from veriflux.regression.comparator import MetricComparator, MetricDiff, MetricStatus
from veriflux.regression.drift_detector import DriftDetector, DriftResult
from veriflux.regression.history import MetricsHistory

__all__ = [
    # Baseline
    "BaselineManager",
    "ComparisonResult",
    # Comparator
    "MetricComparator",
    "MetricDiff",
    "MetricStatus",
    # Drift
    "DriftDetector",
    "DriftResult",
    # History
    "MetricsHistory",
]

