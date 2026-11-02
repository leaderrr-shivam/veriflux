"""Concurrency and worker pool utilities for Veriflux.

Provides thread pool management, parallel mapping, and batch processing
with both synchronous and asynchronous coordination.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import os
import threading
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class WorkerPool:
    """Thread pool manager with active tracking and graceful shutdown."""

    def __init__(self, max_workers: Optional[int] = None, thread_name_prefix: str = "veriflux-worker") -> None:
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.max_workers = max(1, max_workers)
        self.thread_name_prefix = thread_name_prefix
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.thread_name_prefix)
        self._active_futures: set[Future[Any]] = set()
        self._lock = threading.Lock()
        self._is_shutdown = False

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future[R]:
        """Submit a callable to be executed asynchronously."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Cannot submit to a shutdown WorkerPool")
            fut = self._executor.submit(fn, *args, **kwargs)
            self._active_futures.add(fut)
            fut.add_done_callback(self._on_future_done)
            return fut

    def _on_future_done(self, fut: Future[Any]) -> None:
        with self._lock:
            self._active_futures.discard(fut)

    @property
    def active_count(self) -> int:
        """Count of currently active or queued futures."""
        with self._lock:
            return len(self._active_futures)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shutdown the underlying thread pool."""
        with self._lock:
            self._is_shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown(wait=True)


class ParallelMapper(Generic[T, R]):
    """Utility to map a function across an iterable with bounded concurrency."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

    def map(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[R]:
        """Map `fn` over `items` preserving order.

        Args:
            fn: Callable taking item and returning result.
            items: Sequence of items.
            on_progress: Optional callback invoked with (completed_count, total_count).

        Returns:
            List of results matching input item ordering.
        """
        if not items:
            return []

        total = len(items)
        results: List[Optional[R]] = [None] * total
        completed = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(fn, item): idx for idx, item in enumerate(items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results[idx] = res
                with lock:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

        return [r for r in results if r is not None]  # type: ignore[return-value]

    async def map_async(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[R]:
        """Asynchronously map `fn` over `items` in background threads."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.map(fn, items, on_progress))


class BatchProcessor(Generic[T, R]):
    """Process large sequences in discrete batches, optionally in parallel."""

    def __init__(self, batch_size: int = 64, max_workers: Optional[int] = None) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1))

    def _chunk(self, items: Sequence[T]) -> Iterator[Sequence[T]]:
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def process_batches(
        self,
        batch_fn: Callable[[Sequence[T]], Sequence[R]],
        items: Sequence[T],
    ) -> List[R]:
        """Process items by splitting into chunks and passing each chunk to `batch_fn`."""
        chunks = list(self._chunk(items))
        if not chunks:
            return []

        if len(chunks) == 1:
            return list(batch_fn(chunks[0]))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch_results = list(executor.map(batch_fn, chunks))

        flattened: List[R] = []
        for batch in batch_results:
            flattened.extend(batch)
        return flattened

