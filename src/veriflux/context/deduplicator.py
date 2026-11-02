"""Near-duplicate context passage deduplication.

Uses MinHash locality-sensitive fingerprinting to discard redundant retrieved chunks.
"""
from __future__ import annotations

from typing import List, Sequence

from veriflux.nlp.fingerprint import MinHasher


class Deduplicator:
    """Discards near-duplicate passages from RAG retrieval sets."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self.hasher = MinHasher(num_perm=64)

    def deduplicate(self, chunks: Sequence[str]) -> List[str]:
        """Filter out passages that have estimated Jaccard similarity >= threshold with an earlier passage."""
        if len(chunks) <= 1:
            return list(chunks)

        unique_chunks: List[str] = []
        signatures = []

        for chunk in chunks:
            sig = self.hasher.fingerprint(chunk)
            is_dup = False
            for prev_sig in signatures:
                if self.hasher.similarity(sig, prev_sig) >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                signatures.append(sig)
                unique_chunks.append(chunk)

        return unique_chunks


# Ergonomic alias
ContextDeduplicator = Deduplicator

