"""Bridge interface connecting Veriflux to dense embedding providers.

Supports in-memory content-addressable caching and transparent fallback to
built-in lexical vector representation when external provider endpoints are not configured.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

from veriflux.core.cache import LRUCache
from veriflux.core.math_kernels import cosine_similarity
from veriflux.similarity.tfidf import TfidfVectorizer


class EmbeddingBridge:
    """Interface for generating embeddings and computing semantic similarity with caching."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        cache_size: int = 4096,
    ) -> None:
        self._embed_fn = embed_fn
        self._cache = LRUCache[str, List[float]](max_size=cache_size)
        self._fallback_vectorizer: Optional[TfidfVectorizer] = None

    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _fallback_embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate dense surrogate vectors using TF-IDF when no external embedder is registered."""
        if self._fallback_vectorizer is None:
            self._fallback_vectorizer = TfidfVectorizer(max_features=256)
            self._fallback_vectorizer.fit(list(texts) + ["the quick brown fox", "artificial intelligence machine learning"])

        sparse_vectors = self._fallback_vectorizer.transform(list(texts))
        dim = max(256, len(self._fallback_vectorizer.vocabulary_ or {}))
        dense_results: List[List[float]] = []

        for sv in sparse_vectors:
            row = [0.0] * dim
            for idx, val in sv.data.items():
                if idx < dim:
                    row[idx] = val
            dense_results.append(row)

        return dense_results

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Compute embeddings for a sequence of texts with cache lookup."""
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(texts):
            key = self._hash_key(text)
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            if self._embed_fn is not None:
                new_embeddings = self._embed_fn(uncached_texts)
            else:
                new_embeddings = self._fallback_embed(uncached_texts)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache.set(self._hash_key(text), emb)
                results[idx] = emb

        return [r for r in results if r is not None]  # type: ignore[return-value]

    def embed(self, text: str) -> List[float]:
        """Compute embedding for a single text string."""
        return self.embed_batch([text])[0]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic cosine similarity between two texts."""
        emb_a, emb_b = self.embed_batch([text_a, text_b])
        return cosine_similarity(emb_a, emb_b)

