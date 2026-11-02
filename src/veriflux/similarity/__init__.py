"""Veriflux similarity engine — text similarity and retrieval from scratch.

Provides TF-IDF vectorization, BM25+ ranking, cosine similarity,
Jaccard similarity, Levenshtein edit distance, score fusion,
embedding bridge, and in-memory vector storage.

Usage::

    from veriflux.similarity import TfidfVectorizer, BM25, cosine_similarity
    from veriflux.similarity import VectorStore, EmbeddingBridge
"""
from __future__ import annotations

from veriflux.similarity.tfidf import SparseVector, TfidfVectorizer
from veriflux.similarity.bm25 import BM25
from veriflux.similarity.cosine import (
    cosine_similarity,
    pairwise_cosine,
    sparse_cosine,
    top_k_similar,
)
from veriflux.similarity.jaccard import (
    jaccard_distance,
    jaccard_similarity,
    ngram_jaccard,
    token_jaccard,
    weighted_jaccard,
)
from veriflux.similarity.levenshtein import (
    EditOp,
    damerau_levenshtein,
    edit_operations,
    levenshtein_distance,
    levenshtein_similarity,
    normalized_levenshtein,
)
from veriflux.similarity.fusion import (
    combmnz,
    normalize_scores,
    reciprocal_rank_fusion,
    weighted_linear,
)
from veriflux.similarity.embedding_bridge import EmbeddingBridge
from veriflux.similarity.vector_store import SearchResult, VectorStore

__all__ = [
    # tfidf
    "SparseVector",
    "TfidfVectorizer",
    # bm25
    "BM25",
    # cosine
    "cosine_similarity",
    "sparse_cosine",
    "pairwise_cosine",
    "top_k_similar",
    # jaccard
    "jaccard_similarity",
    "jaccard_distance",
    "weighted_jaccard",
    "token_jaccard",
    "ngram_jaccard",
    # levenshtein
    "EditOp",
    "levenshtein_distance",
    "normalized_levenshtein",
    "damerau_levenshtein",
    "levenshtein_similarity",
    "edit_operations",
    # fusion
    "reciprocal_rank_fusion",
    "weighted_linear",
    "combmnz",
    "normalize_scores",
    # embedding bridge
    "EmbeddingBridge",
    # vector store
    "SearchResult",
    "VectorStore",
]
