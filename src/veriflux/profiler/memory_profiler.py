"""Peak memory allocation profiler using stdlib tracemalloc.

Measures RAM consumption, garbage collection pressure, and peak memory spikes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import tracemalloc
from typing import Any, List, Optional, Tuple


@dataclass(slots=True, frozen=True)
class MemorySnapshot:
    """RAM footprint metrics.

    Attributes:
        current_kb: Memory allocated at snapshot time in kilobytes.
        peak_kb: Peak memory allocated since tracing commenced in kilobytes.
        top_allocations: List of top allocating trace entries (file:line, size_kb).
    """

    current_kb: float
    peak_kb: float
    top_allocations: List[Tuple[str, float]] = field(default_factory=list)


class MemoryProfiler:
    """Context manager for measuring RAM memory consumption."""

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n
        self._was_tracing = False
        self.peak_kb: float = 0.0
        self.current_kb: float = 0.0

    def __enter__(self) -> MemoryProfiler:
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
        self.current_kb = round(cur_bytes / 1024.0, 2)
        self.peak_kb = round(peak_bytes / 1024.0, 2)

        if not self._was_tracing:
            tracemalloc.stop()

    def snapshot(self) -> MemorySnapshot:
        """Capture current and peak memory allocations."""
        cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
        top: List[Tuple[str, float]] = []

        try:
            snap = tracemalloc.take_snapshot()
            top_stats = snap.statistics("lineno")
            for stat in top_stats[: self.top_n]:
                top.append((str(stat.traceback), round(stat.size / 1024.0, 2)))
        except Exception:
            pass

        return MemorySnapshot(
            current_kb=round(cur_bytes / 1024.0, 2),
            peak_kb=round(peak_bytes / 1024.0, 2),
            top_allocations=top,
        )

