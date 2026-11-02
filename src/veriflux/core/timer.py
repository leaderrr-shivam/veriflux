"""High-precision timing utilities for Veriflux.

Provides a context-manager timer, a decorator timer, a statistics collector,
and a manual stopwatch — all based on :func:`time.perf_counter_ns` for
nanosecond resolution.
"""
from __future__ import annotations

import functools
import statistics as _stats
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------

def _format_ns(ns: int) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.4f} s"


# ---------------------------------------------------------------------------
# Timer (context manager)
# ---------------------------------------------------------------------------

class Timer:
    """High-precision context-manager timer.

    Example::

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_ns: int = 0
        self._end_ns: int = 0

    def __enter__(self) -> Timer:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end_ns = time.perf_counter_ns()

    # -- properties ---------------------------------------------------------

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        if self._end_ns:
            return self._end_ns - self._start_ns
        # Still running
        return time.perf_counter_ns() - self._start_ns if self._start_ns else 0

    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self.elapsed_ns / 1_000

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000

    @property
    def elapsed_human(self) -> str:
        """Human-readable elapsed time."""
        return _format_ns(self.elapsed_ns)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"Timer{label}({self.elapsed_human})"


# ---------------------------------------------------------------------------
# timer() decorator
# ---------------------------------------------------------------------------

def timer(fn: Optional[F] = None, *, callback: Optional[Callable[[str, int], None]] = None) -> Any:
    """Decorator that times function execution.

    Parameters:
        fn: The function to wrap (used when ``@timer`` is applied without
            parentheses).
        callback: Optional ``(name, elapsed_ns) -> None`` callback invoked
            after each call.  If *None*, timing is recorded as an attribute
            on the wrapper (``wrapper.last_elapsed_ns``).

    Returns:
        The decorated function.

    Examples::

        @timer
        def compute():
            ...

        @timer(callback=lambda name, ns: print(f"{name}: {ns}ns"))
        def compute():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - start
                wrapper.last_elapsed_ns = elapsed  # type: ignore[attr-defined]
                if callback is not None:
                    callback(func.__qualname__, elapsed)
            return result

        wrapper.last_elapsed_ns = 0  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        # @timer without parentheses
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# TimingStats
# ---------------------------------------------------------------------------

class TimingStats:
    """Collects a series of timing measurements and computes statistics.

    Example::

        ts = TimingStats("my_operation")
        for _ in range(100):
            with Timer() as t:
                do_work()
            ts.record(t.elapsed_ns)
        print(ts.summary())
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._measurements: List[int] = []

    def record(self, elapsed_ns: int) -> None:
        """Record a timing measurement in nanoseconds."""
        self._measurements.append(elapsed_ns)

    def record_ms(self, elapsed_ms: float) -> None:
        """Record a timing measurement in milliseconds."""
        self._measurements.append(int(elapsed_ms * 1_000_000))

    @property
    def count(self) -> int:
        """Number of recorded measurements."""
        return len(self._measurements)

    @property
    def total_ns(self) -> int:
        """Sum of all measurements."""
        return sum(self._measurements)

    @property
    def min_ns(self) -> int:
        """Minimum measurement."""
        return min(self._measurements) if self._measurements else 0

    @property
    def max_ns(self) -> int:
        """Maximum measurement."""
        return max(self._measurements) if self._measurements else 0

    @property
    def mean_ns(self) -> float:
        """Arithmetic mean."""
        if not self._measurements:
            return 0.0
        return sum(self._measurements) / len(self._measurements)

    @property
    def median_ns(self) -> float:
        """Median measurement."""
        if not self._measurements:
            return 0.0
        return float(_stats.median(self._measurements))

    @property
    def stdev_ns(self) -> float:
        """Standard deviation (0 if fewer than 2 samples)."""
        if len(self._measurements) < 2:
            return 0.0
        return _stats.stdev(self._measurements)

    def percentile(self, p: float) -> float:
        """Compute the *p*-th percentile (0–100).

        Uses linear interpolation between nearest ranks.
        """
        if not self._measurements:
            return 0.0
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"Percentile must be in [0, 100], got {p}")
        sorted_data = sorted(self._measurements)
        n = len(sorted_data)
        if n == 1:
            return float(sorted_data[0])
        # Rank (0-indexed fractional)
        rank = (p / 100.0) * (n - 1)
        low = int(rank)
        high = min(low + 1, n - 1)
        frac = rank - low
        return sorted_data[low] + frac * (sorted_data[high] - sorted_data[low])

    @property
    def p95_ns(self) -> float:
        """95th-percentile measurement."""
        return self.percentile(95.0)

    @property
    def p99_ns(self) -> float:
        """99th-percentile measurement."""
        return self.percentile(99.0)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict with all statistics in both ns and human-readable form."""
        return {
            "name": self.name,
            "count": self.count,
            "total": _format_ns(self.total_ns),
            "min": _format_ns(self.min_ns),
            "max": _format_ns(self.max_ns),
            "mean": _format_ns(int(self.mean_ns)),
            "median": _format_ns(int(self.median_ns)),
            "stdev": _format_ns(int(self.stdev_ns)),
            "p95": _format_ns(int(self.p95_ns)),
            "p99": _format_ns(int(self.p99_ns)),
            "total_ns": self.total_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "median_ns": self.median_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
        }

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._measurements.clear()

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"TimingStats{label}(count={self.count}, mean={_format_ns(int(self.mean_ns))})"


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------

class Stopwatch:
    """Manual start/stop/lap stopwatch for precise timing.

    Example::

        sw = Stopwatch()
        sw.start()
        do_phase_1()
        sw.lap("phase_1")
        do_phase_2()
        sw.lap("phase_2")
        sw.stop()
        print(sw.laps)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_ns: int = 0
        self._stop_ns: int = 0
        self._running: bool = False
        self._lap_start_ns: int = 0
        self._laps: List[Dict[str, Any]] = []

    def start(self) -> Stopwatch:
        """Start (or restart) the stopwatch."""
        now = time.perf_counter_ns()
        self._start_ns = now
        self._lap_start_ns = now
        self._stop_ns = 0
        self._running = True
        self._laps.clear()
        return self

    def stop(self) -> int:
        """Stop the stopwatch and return total elapsed nanoseconds."""
        if not self._running:
            raise RuntimeError("Stopwatch is not running")
        self._stop_ns = time.perf_counter_ns()
        self._running = False
        return self.elapsed_ns

    def lap(self, name: str = "") -> int:
        """Record a lap and return the lap duration in nanoseconds.

        The lap measures time since the previous lap (or start).
        """
        if not self._running:
            raise RuntimeError("Stopwatch is not running")
        now = time.perf_counter_ns()
        lap_ns = now - self._lap_start_ns
        self._laps.append({
            "name": name or f"lap_{len(self._laps) + 1}",
            "elapsed_ns": lap_ns,
            "elapsed_human": _format_ns(lap_ns),
            "cumulative_ns": now - self._start_ns,
        })
        self._lap_start_ns = now
        return lap_ns

    @property
    def elapsed_ns(self) -> int:
        """Total elapsed nanoseconds."""
        if self._running:
            return time.perf_counter_ns() - self._start_ns
        if self._stop_ns:
            return self._stop_ns - self._start_ns
        return 0

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Total elapsed seconds."""
        return self.elapsed_ns / 1_000_000_000

    @property
    def elapsed_human(self) -> str:
        """Human-readable total elapsed time."""
        return _format_ns(self.elapsed_ns)

    @property
    def is_running(self) -> bool:
        """Whether the stopwatch is currently running."""
        return self._running

    @property
    def laps(self) -> List[Dict[str, Any]]:
        """List of recorded laps."""
        return list(self._laps)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        state = "running" if self._running else "stopped"
        return f"Stopwatch{label}({state}, {self.elapsed_human}, laps={len(self._laps)})"
