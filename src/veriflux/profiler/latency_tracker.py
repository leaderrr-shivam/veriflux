"""High-precision latency, TTFT, and ITL performance tracking.

Collects execution timestamps, calculates statistical distributions (mean, p50, p95, p99),
and supports context-manager and decorator profiling interfaces.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import functools
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(slots=True, frozen=True)
class LatencyStats:
    """Statistical distribution summary of latency observations in milliseconds.

    Attributes:
        count: Number of measurements.
        mean: Arithmetic average duration (ms).
        min: Minimum recorded duration (ms).
        max: Maximum recorded duration (ms).
        median: 50th percentile (ms).
        p95: 95th percentile (ms).
        p99: 99th percentile (ms).
        std_dev: Standard deviation.
    """

    count: int
    mean: float
    min: float
    max: float
    median: float
    p95: float
    p99: float
    std_dev: float


class LatencyTracker:
    """Thread-safe latency profiler for tracking execution times and token streaming delays."""

    def __init__(self) -> None:
        self._measurements: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, name: str, duration_ms: float) -> None:
        """Record a measured duration in milliseconds."""
        with self._lock:
            self._measurements[name].append(duration_ms)

    def stats(self, name: str) -> Optional[LatencyStats]:
        """Compute statistical percentiles for a named benchmark series."""
        with self._lock:
            vals = sorted(self._measurements.get(name, []))

        if not vals:
            return None

        n = len(vals)
        mean = sum(vals) / n
        variance = sum((x - mean) ** 2 for x in vals) / n
        std_dev = math.sqrt(variance)

        def _pct(p: float) -> float:
            idx = int(math.ceil((p / 100.0) * n)) - 1
            return vals[max(0, min(n - 1, idx))]

        return LatencyStats(
            count=n,
            mean=round(mean, 2),
            min=round(vals[0], 2),
            max=round(vals[-1], 2),
            median=round(_pct(50), 2),
            p95=round(_pct(95), 2),
            p99=round(_pct(99), 2),
            std_dev=round(std_dev, 2),
        )

    def measure(self, name: str) -> LatencyContext:
        """Context manager for timing a block of code."""
        return LatencyContext(self, name)

    def profile(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator timing execution of the decorated function."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            metric_name = name or fn.__name__
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.measure(metric_name):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    def reset(self) -> None:
        """Clear all stored latency recordings."""
        with self._lock:
            self._measurements.clear()


class LatencyContext:
    """Context manager for LatencyTracker timing blocks."""

    def __init__(self, tracker: LatencyTracker, name: str) -> None:
        self.tracker = tracker
        self.name = name
        self.start_time: float = 0.0

    def __enter__(self) -> LatencyContext:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.tracker.record(self.name, elapsed_ms)

