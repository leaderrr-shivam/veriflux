"""Optimised cosine similarity functions for dense and sparse representations.

Supports single-pair, pairwise, batch, and top-k retrieval modes.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple, Union

from veriflux.core.math_kernels import cosine_similarity as _dense_cosine
from veriflux.similarity.tfidf import SparseVector


def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Compute cosine similarity between two dense vectors.

    Args:
        a: First dense vector.
        b: Second dense vector.
        eps: Epsilon constant for zero division protection.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    return _dense_cosine(a, b, eps=eps)


def sparse_cosine(a: SparseVector, b: SparseVector, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two SparseVector instances.

    Args:
        a: First SparseVector.
        b: Second SparseVector.
        eps: Epsilon for zero norm protection.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    if not a.data or not b.data:
        return 0.0

    # Iterate over smaller dict
    if len(a.data) > len(b.data):
        a, b = b, a

    dot = 0.0
    for idx, val_a in a.data.items():
        if idx in b.data:
            dot += val_a * b.data[idx]

    norm_a = math.sqrt(sum(v * v for v in a.data.values()))
    norm_b = math.sqrt(sum(v * v for v in b.data.values()))
    denom = norm_a * norm_b
    if denom < eps:
        return 0.0

    score = dot / denom
    return max(-1.0, min(1.0, score))


def pairwise_cosine(matrix: Sequence[Sequence[float]], eps: float = 1e-12) -> List[List[float]]:
    """Compute all-pairs cosine similarity matrix.

    Args:
        matrix: Sequence of N vectors of length D.
        eps: Epsilon for zero division protection.

    Returns:
        N x N symmetric matrix of similarity values.
    """
    n = len(matrix)
    if n == 0:
        return []

    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine_similarity(matrix[i], matrix[j], eps=eps)
            result[i][j] = sim
            result[j][i] = sim
    return result


def top_k_similar(
    query: Sequence[float],
    matrix: Sequence[Sequence[float]],
    k: int = 10,
    eps: float = 1e-12,
) -> List[Tuple[int, float]]:
    """Retrieve top-k indices and scores from a collection of candidate vectors.

    Args:
        query: Query vector.
        matrix: Candidate vectors.
        k: Maximum number of matches to return.
        eps: Numerical stability constant.

    Returns:
        List of (index, similarity_score) tuples sorted in descending order of similarity.
    """
    if not matrix or k <= 0:
        return []

    scores: List[Tuple[float, int]] = []
    for idx, cand in enumerate(matrix):
        sim = cosine_similarity(query, cand, eps=eps)
        scores.append((sim, idx))

    top = heapq.nlargest(k, scores, key=lambda x: x[0])
    return [(idx, sim) for sim, idx in top]

