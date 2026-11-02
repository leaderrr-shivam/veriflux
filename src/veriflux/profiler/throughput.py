"""Token throughput and request velocity measurement.

Tracks tokens per second (TPS) and requests per minute (RPM) across continuous streams.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import List, Tuple


@dataclass(slots=True, frozen=True)
class ThroughputStats:
    """Aggregate throughput performance summary.

    Attributes:
        total_tokens: Total tokens processed.
        total_requests: Total discrete requests.
        total_seconds: Aggregate generation elapsed time.
        tokens_per_second: Average generation throughput.
        requests_per_minute: Average request velocity.
        peak_tokens_per_second: Peak instantaneous throughput observed.
    """

    total_tokens: int
    total_requests: int
    total_seconds: float
    tokens_per_second: float
    requests_per_minute: float
    peak_tokens_per_second: float


class ThroughputMeter:
    """Tracks token throughput and invocation rates."""

    def __init__(self) -> None:
        self._measurements: List[Tuple[int, float]] = []  # (tokens, duration_sec)
        self._lock = threading.Lock()

    def record(self, tokens: int, duration_seconds: float) -> None:
        """Record a completed completion event."""
        if duration_seconds <= 0.0 or tokens <= 0:
            return
        with self._lock:
            self._measurements.append((tokens, duration_seconds))

    def stats(self) -> ThroughputStats:
        """Compute aggregate throughput statistics."""
        with self._lock:
            data = list(self._measurements)

        if not data:
            return ThroughputStats(0, 0, 0.0, 0.0, 0.0, 0.0)

        total_tokens = sum(t for t, _ in data)
        total_seconds = sum(s for _, s in data)
        total_requests = len(data)

        avg_tps = (total_tokens / total_seconds) if total_seconds > 0 else 0.0
        rpm = (total_requests / (total_seconds / 60.0)) if total_seconds > 0 else 0.0
        peak_tps = max(t / s for t, s in data if s > 0)

        return ThroughputStats(
            total_tokens=total_tokens,
            total_requests=total_requests,
            total_seconds=round(total_seconds, 3),
            tokens_per_second=round(avg_tps, 2),
            requests_per_minute=round(rpm, 2),
            peak_tokens_per_second=round(peak_tps, 2),
        )

    def reset(self) -> None:
        """Clear all throughput recordings."""
        with self._lock:
            self._measurements.clear()

