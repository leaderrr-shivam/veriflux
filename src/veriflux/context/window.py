"""Sliding context window manager with overlap preservation.

Maintains a rolling stream of context tokens with FIFO eviction for long-running sessions.
"""
from __future__ import annotations

from typing import List

from veriflux.tokens.counter import count_tokens


class SlidingWindow:
    """Sliding token window with configurable boundary overlap."""

    def __init__(self, max_tokens: int = 4096, overlap_tokens: int = 256, model: str = "gpt-4o") -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model = model
        self._chunks: List[str] = []

    def add(self, text: str) -> None:
        """Append text passage to window, evicting oldest chunks if max_tokens exceeded."""
        if not text:
            return
        self._chunks.append(text)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._chunks) > 1 and self.current_tokens > self.max_tokens:
            self._chunks.pop(0)

    @property
    def current_tokens(self) -> int:
        return count_tokens(" ".join(self._chunks), model=self.model)

    def get(self) -> str:
        """Return the consolidated string content of the active window."""
        return "\n\n".join(self._chunks)

    def full(self) -> bool:
        """True if window tokens are at or above max capacity."""
        return self.current_tokens >= self.max_tokens

    def clear(self) -> None:
        """Clear all chunks from the sliding window."""
        self._chunks.clear()

