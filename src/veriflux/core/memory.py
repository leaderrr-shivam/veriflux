"""Memory management utilities for Veriflux.

Provides arena allocation, buffer pooling, object pooling, and memory
tracking to reduce GC pressure in hot evaluation loops.
"""
from __future__ import annotations

import collections
import threading
import tracemalloc
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ArenaAllocator
# ---------------------------------------------------------------------------

class ArenaAllocator:
    """Pre-allocated byte arena that hands out :class:`memoryview` slices.

    The arena is a contiguous :class:`bytearray` from which callers request
    fixed-size slices.  When work is done, call :meth:`reset` to reclaim all
    space in O(1).

    Parameters:
        size: Total arena size in bytes.
        alignment: Allocation alignment in bytes (must be a power of 2).

    Example::

        arena = ArenaAllocator(1024)
        buf = arena.alloc(64)
        buf[:4] = b'test'
        arena.reset()
    """

    def __init__(self, size: int, alignment: int = 8) -> None:
        if size <= 0:
            raise ValueError(f"Arena size must be positive, got {size}")
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"Alignment must be a positive power of 2, got {alignment}")
        self._buf: bytearray = bytearray(size)
        self._view: memoryview = memoryview(self._buf)
        self._size: int = size
        self._alignment: int = alignment
        self._offset: int = 0
        self._lock: threading.Lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def alloc(self, nbytes: int) -> memoryview:
        """Allocate *nbytes* from the arena.

        Returns a :class:`memoryview` slice.  Raises :class:`MemoryError` if
        there is not enough space remaining.
        """
        if nbytes <= 0:
            raise ValueError(f"Allocation size must be positive, got {nbytes}")
        with self._lock:
            # Align the current offset
            aligned = self._align_up(self._offset)
            end = aligned + nbytes
            if end > self._size:
                raise MemoryError(
                    f"Arena exhausted: requested {nbytes} bytes at offset {aligned}, "
                    f"but only {self._size - aligned} bytes remain"
                )
            self._offset = end
            return self._view[aligned:end]

    def reset(self) -> None:
        """Reclaim all arena space (O(1)).

        Existing :class:`memoryview` slices remain valid but their content
        may be overwritten by subsequent allocations.
        """
        with self._lock:
            self._offset = 0

    @property
    def bytes_used(self) -> int:
        """Number of bytes currently allocated."""
        return self._offset

    @property
    def bytes_free(self) -> int:
        """Number of bytes remaining."""
        return self._size - self._offset

    @property
    def capacity(self) -> int:
        """Total arena capacity in bytes."""
        return self._size

    # -- internals ----------------------------------------------------------

    def _align_up(self, offset: int) -> int:
        """Round *offset* up to the next alignment boundary."""
        mask = self._alignment - 1
        return (offset + mask) & ~mask


# ---------------------------------------------------------------------------
# BufferPool
# ---------------------------------------------------------------------------

class BufferPool:
    """Pool of reusable :class:`bytearray` buffers, sized to powers of 2.

    Reduces GC pressure by keeping freed buffers for reuse instead of
    allocating new ones.

    Parameters:
        max_buffers_per_size: Maximum number of buffers to pool for each
            power-of-2 size class.

    Thread-safe.
    """

    def __init__(self, max_buffers_per_size: int = 32) -> None:
        if max_buffers_per_size <= 0:
            raise ValueError("max_buffers_per_size must be positive")
        self._max_per_size: int = max_buffers_per_size
        # size_class -> deque of bytearrays
        self._pools: Dict[int, Deque[bytearray]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._stats_acquired: int = 0
        self._stats_released: int = 0
        self._stats_created: int = 0

    def acquire(self, min_size: int) -> bytearray:
        """Acquire a buffer of at least *min_size* bytes.

        The returned buffer is a power-of-2 size ≥ *min_size*.  Its contents
        are **not** zeroed; use ``buf[:] = b'\\x00' * len(buf)`` if needed.
        """
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        size_class = self._next_power_of_2(min_size)
        with self._lock:
            self._stats_acquired += 1
            pool = self._pools.get(size_class)
            if pool:
                return pool.popleft()
        # Create outside the lock
        with self._lock:
            self._stats_created += 1
        return bytearray(size_class)

    def release(self, buf: bytearray) -> None:
        """Return a buffer to the pool for reuse."""
        size_class = len(buf)
        # Only pool power-of-2 sized buffers
        if size_class <= 0 or (size_class & (size_class - 1)) != 0:
            return  # Not a power-of-2; just drop it
        with self._lock:
            self._stats_released += 1
            pool = self._pools.setdefault(size_class, collections.deque())
            if len(pool) < self._max_per_size:
                pool.append(buf)

    def clear(self) -> None:
        """Discard all pooled buffers."""
        with self._lock:
            self._pools.clear()

    def stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            total_pooled = sum(len(d) for d in self._pools.values())
            return {
                "acquired": self._stats_acquired,
                "released": self._stats_released,
                "created": self._stats_created,
                "pooled": total_pooled,
                "size_classes": {k: len(v) for k, v in sorted(self._pools.items())},
            }

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Return the smallest power of 2 ≥ *n*."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1


# ---------------------------------------------------------------------------
# ObjectPool
# ---------------------------------------------------------------------------

class ObjectPool(Generic[T]):
    """Generic object pool backed by a :class:`collections.deque`.

    Parameters:
        factory: Callable that creates a new object when the pool is empty.
        reset_fn: Optional callable to reset an object before returning it
            to the pool.
        max_size: Maximum number of idle objects to retain.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        reset_fn: Optional[Callable[[T], None]] = None,
        max_size: int = 64,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._factory = factory
        self._reset_fn = reset_fn
        self._max_size = max_size
        self._pool: Deque[T] = collections.deque()
        self._lock: threading.Lock = threading.Lock()
        self._stats_acquired: int = 0
        self._stats_released: int = 0
        self._stats_created: int = 0

    def acquire(self) -> T:
        """Get an object from the pool, creating one if necessary."""
        with self._lock:
            self._stats_acquired += 1
            if self._pool:
                return self._pool.popleft()
            self._stats_created += 1
        return self._factory()

    def release(self, obj: T) -> None:
        """Return an object to the pool.

        If *reset_fn* was provided, it is called before the object is pooled.
        """
        if self._reset_fn is not None:
            try:
                self._reset_fn(obj)
            except Exception:
                # If reset fails, discard the object
                return
        with self._lock:
            self._stats_released += 1
            if len(self._pool) < self._max_size:
                self._pool.append(obj)

    def clear(self) -> None:
        """Discard all pooled objects."""
        with self._lock:
            self._pool.clear()

    def stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            return {
                "acquired": self._stats_acquired,
                "released": self._stats_released,
                "created": self._stats_created,
                "pooled": len(self._pool),
                "max_size": self._max_size,
            }


# ---------------------------------------------------------------------------
# MemoryTracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Context manager that tracks peak memory usage via :mod:`tracemalloc`.

    Example::

        with MemoryTracker() as tracker:
            data = [0] * 1_000_000
        print(f"Peak: {tracker.peak_mb:.2f} MB")
    """

    def __init__(self, nframe: int = 1) -> None:
        self._nframe = nframe
        self._peak_bytes: int = 0
        self._current_bytes: int = 0
        self._was_tracing: bool = False
        self._snapshot_before: Optional[tracemalloc.Snapshot] = None
        self._snapshot_after: Optional[tracemalloc.Snapshot] = None

    def __enter__(self) -> MemoryTracker:
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start(self._nframe)
        # Reset peak to get an accurate delta
        tracemalloc.reset_peak()
        self._snapshot_before = tracemalloc.take_snapshot()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._snapshot_after = tracemalloc.take_snapshot()
        self._current_bytes, self._peak_bytes = tracemalloc.get_traced_memory()
        if not self._was_tracing:
            tracemalloc.stop()

    @property
    def peak_bytes(self) -> int:
        """Peak memory usage in bytes during the tracked block."""
        return self._peak_bytes

    @property
    def peak_kb(self) -> float:
        """Peak memory usage in kilobytes."""
        return self._peak_bytes / 1024.0

    @property
    def peak_mb(self) -> float:
        """Peak memory usage in megabytes."""
        return self._peak_bytes / (1024.0 * 1024.0)

    @property
    def current_bytes(self) -> int:
        """Memory still allocated at the end of the tracked block."""
        return self._current_bytes

    def top_allocations(self, limit: int = 10) -> List[str]:
        """Return the top *limit* memory-allocating lines.

        Only available if a snapshot was taken (i.e. the context manager
        was used and has exited).
        """
        if self._snapshot_after is None or self._snapshot_before is None:
            return []
        stats = self._snapshot_after.compare_to(self._snapshot_before, "lineno")
        lines: List[str] = []
        for stat in stats[:limit]:
            lines.append(str(stat))
        return lines
