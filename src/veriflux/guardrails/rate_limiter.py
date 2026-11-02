"""Thread-safe token bucket rate limiter for LLM requests and token budgets.

Prevents unexpected quota exhaustion and enforces throughput boundaries.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter supporting request and token consumption limits."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tokens = max_tokens

        self._requests_timestamps: list[float] = []
        self._tokens_consumed: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._requests_timestamps = [t for t in self._requests_timestamps if t > cutoff]
        if self.max_tokens is not None:
            self._tokens_consumed = [(t, c) for t, c in self._tokens_consumed if t > cutoff]

    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire permission for a request.

        Returns True if allowed, False if limit exceeded.
        """
        now = time.time()
        with self._lock:
            self._purge_expired(now)

            if len(self._requests_timestamps) >= self.max_requests:
                return False

            if self.max_tokens is not None:
                current_tokens = sum(c for _, c in self._tokens_consumed)
                if current_tokens + tokens > self.max_tokens:
                    return False
                self._tokens_consumed.append((now, tokens))

            self._requests_timestamps.append(now)
            return True

    def wait(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Block until a slot is available or timeout elapses."""
        start = time.time()
        while True:
            if self.acquire(tokens=tokens):
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(0.05)

    def remaining(self) -> int:
        """Count of remaining request slots in the active window."""
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            return max(0, self.max_requests - len(self._requests_timestamps))

