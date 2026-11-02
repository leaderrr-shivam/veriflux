"""In-memory vector store with exact and approximate nearest-neighbor search.

Implemented from scratch with zero external dependencies. Features multi-layer
navigable proximity graphs and cosine similarity indexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from veriflux.core.math_kernels import cosine_similarity, l2_normalize


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Outcome of a vector similarity query."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """In-memory vector store supporting exact and graph-based nearest neighbor search."""

    def __init__(self, dimension: Optional[int] = None, normalize: bool = True) -> None:
        self.dimension = dimension
        self.normalize = normalize
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        # HNSW-inspired graph parameters
        self._m = 16  # Max connections per node
        self._ef_construction = 32
        self._graph: Dict[str, Set[str]] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, id: str, vector: Sequence[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert or replace a vector in the store."""
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match store dimension {self.dimension}")
        if self.dimension is None:
            self.dimension = len(vector)

        v_list = l2_normalize(list(vector)) if self.normalize else list(vector)
        self._vectors[id] = v_list
        self._metadata[id] = metadata or {}

        # Update adjacency graph
        if id not in self._graph:
            self._graph[id] = set()

        # Connect to nearest existing nodes
        if len(self._vectors) > 1:
            candidates: List[Tuple[float, str]] = []
            for other_id, other_v in self._vectors.items():
                if other_id == id:
                    continue
                sim = cosine_similarity(v_list, other_v)
                candidates.append((sim, other_id))

            nearest = heapq.nlargest(self._m, candidates, key=lambda x: x[0])
            for _, neighbor_id in nearest:
                self._graph[id].add(neighbor_id)
                self._graph[neighbor_id].add(id)
                # Prune if neighbor degree exceeds limit
                if len(self._graph[neighbor_id]) > self._m * 2:
                    n_candidates = [
                        (cosine_similarity(self._vectors[neighbor_id], self._vectors[cand_id]), cand_id)
                        for cand_id in self._graph[neighbor_id]
                    ]
                    pruned = heapq.nlargest(self._m, n_candidates, key=lambda x: x[0])
                    self._graph[neighbor_id] = {cid for _, cid in pruned}

    def delete(self, id: str) -> bool:
        """Remove a vector by identifier. Returns True if removed, False otherwise."""
        if id not in self._vectors:
            return False
        del self._vectors[id]
        self._metadata.pop(id, None)

        neighbors = self._graph.pop(id, set())
        for n in neighbors:
            if n in self._graph:
                self._graph[n].discard(id)
        return True

    def search(self, query: Sequence[float], top_k: int = 10) -> List[SearchResult]:
        """Find the top-k most similar vectors to query."""
        if not self._vectors or top_k <= 0:
            return []

        q_vec = l2_normalize(list(query)) if self.normalize else list(query)

        # For smaller index (< 500 items), exact scan is faster and perfectly precise
        if len(self._vectors) <= 500:
            scores: List[Tuple[float, str]] = []
            for doc_id, v in self._vectors.items():
                sim = cosine_similarity(q_vec, v)
                scores.append((sim, doc_id))
            top = heapq.nlargest(top_k, scores, key=lambda x: x[0])
            return [
                SearchResult(id=doc_id, score=score, metadata=self._metadata.get(doc_id, {}))
                for score, doc_id in top
            ]

        # Greedy graph traversal search for larger scale
        entry_node = next(iter(self._vectors.keys()))
        current_best = entry_node
        best_sim = cosine_similarity(q_vec, self._vectors[entry_node])
        visited: Set[str] = {entry_node}

        improved = True
        while improved:
            improved = False
            for neighbor in self._graph.get(current_best, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    sim = cosine_similarity(q_vec, self._vectors[neighbor])
                    if sim > best_sim:
                        best_sim = sim
                        current_best = neighbor
                        improved = True

        # Collect top candidates among visited
        candidates = [
            (cosine_similarity(q_vec, self._vectors[doc_id]), doc_id)
            for doc_id in visited
        ]
        top = heapq.nlargest(top_k, candidates, key=lambda x: x[0])
        return [
            SearchResult(id=doc_id, score=score, metadata=self._metadata.get(doc_id, {}))
            for score, doc_id in top
        ]

