"""Context chunk relevance ranking for retrieval and prompt assembly.

Ranks candidate context passages using BM25 lexical relevance and reciprocal rank fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from veriflux.similarity.bm25 import BM25


@dataclass(slots=True, frozen=True)
class RankedChunk:
    """A scored context chunk."""

    text: str
    score: float
    rank: int
    source_index: int


class ContextRanker:
    """Ranks context passages against a reference user query."""

    def rank(self, chunks: Sequence[str], query: str, top_k: int = 10) -> List[RankedChunk]:
        """Rank chunks by relevance to query.

        Args:
            chunks: List of context passages.
            query: User prompt or information query.
            top_k: Maximum ranked results to return.

        Returns:
            Sorted list of RankedChunk instances.
        """
        if not chunks or not query:
            return []

        bm25 = BM25()
        bm25.fit(list(chunks))
        ranked_indices = bm25.rank(query, top_k=top_k)

        results: List[RankedChunk] = []
        for rank, (idx, score) in enumerate(ranked_indices, start=1):
            results.append(
                RankedChunk(
                    text=chunks[idx],
                    score=round(score, 4),
                    rank=rank,
                    source_index=idx,
                )
            )

        return results

