"""SIMD-optimised mathematical kernels for Veriflux.

Provides fundamental vector and matrix operations with optional NumPy acceleration
and a robust, zero-dependency pure-Python fallback implementation.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def dot_product(a: Vector, b: Vector) -> float:
    """Compute the inner product of two real-valued vectors.

    Args:
        a: First input vector.
        b: Second input vector.

    Returns:
        The scalar dot product.

    Raises:
        ValueError: If vectors differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: len(a)={len(a)}, len(b)={len(b)}")
    if not a:
        raise ValueError("Cannot compute dot product of empty vectors")

    if _HAS_NUMPY:
        return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))

    return math.fsum(x * y for x, y in zip(a, b))


def l2_normalize(v: Vector, eps: float = 1e-12) -> List[float]:
    """Normalise a vector to unit L2 Euclidean norm.

    Args:
        v: Input vector.
        eps: Small epsilon to prevent division by zero.

    Returns:
        A new list representing the unit vector.
    """
    if not v:
        return []

    if _HAS_NUMPY:
        arr = np.asarray(v, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < eps:
            return [0.0] * len(v)
        return (arr / norm).tolist()

    norm_sq = math.fsum(x * x for x in v)
    norm = math.sqrt(norm_sq)
    if norm < eps:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(a: Vector, b: Vector, eps: float = 1e-12) -> float:
    """Compute the cosine similarity between two vectors in [-1.0, 1.0].

    Args:
        a: First vector.
        b: Second vector.
        eps: Numerical stability constant.

    Returns:
        Cosine similarity score.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")
    if not a:
        return 0.0

    if _HAS_NUMPY:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))
        if norm_a < eps or norm_b < eps:
            return 0.0
        val = float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, val))

    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    denom = norm_a * norm_b
    if denom < eps:
        return 0.0
    val = dot / denom
    return max(-1.0, min(1.0, val))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute the Euclidean L2 distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance (>= 0.0).
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")
    if not a:
        return 0.0

    if _HAS_NUMPY:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        return float(np.linalg.norm(arr_a - arr_b))

    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def softmax(x: Vector) -> List[float]:
    """Compute the numerically stable softmax over a real vector.

    Args:
        x: Input unnormalised log-probabilities / logits.

    Returns:
        Probability distribution summing to 1.0.
    """
    if not x:
        return []

    if _HAS_NUMPY:
        arr = np.asarray(x, dtype=np.float64)
        shift = arr - np.max(arr)
        exps = np.exp(shift)
        return (exps / np.sum(exps)).tolist()

    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    sum_exps = math.fsum(exps)
    if sum_exps == 0.0:
        uniform = 1.0 / len(x)
        return [uniform] * len(x)
    return [e / sum_exps for e in exps]


def sigmoid(x: float) -> float:
    """Compute numerically stable logistic sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def log_sum_exp(x: Vector) -> float:
    """Compute log(sum(exp(x_i))) in a numerically stable manner."""
    if not x:
        return float("-inf")

    if _HAS_NUMPY:
        arr = np.asarray(x, dtype=np.float64)
        max_val = float(np.max(arr))
        return float(max_val + np.log(np.sum(np.exp(arr - max_val))))

    max_val = max(x)
    if math.isinf(max_val):
        return max_val
    sum_exp = math.fsum(math.exp(v - max_val) for v in x)
    return max_val + math.log(sum_exp)


def sparse_dot(
    indices_a: Sequence[int],
    values_a: Sequence[float],
    indices_b: Sequence[int],
    values_b: Sequence[float],
) -> float:
    """Compute the dot product of two sparse vectors represented as sorted index-value pairs.

    Args:
        indices_a: Sorted indices of non-zero entries in vector a.
        values_a: Corresponding non-zero values for vector a.
        indices_b: Sorted indices of non-zero entries in vector b.
        values_b: Corresponding non-zero values for vector b.

    Returns:
        The inner product of the two sparse vectors.
    """
    total = 0.0
    i, j = 0, 0
    len_a, len_b = len(indices_a), len(indices_b)

    while i < len_a and j < len_b:
        idx_a, idx_b = indices_a[i], indices_b[j]
        if idx_a == idx_b:
            total += values_a[i] * values_b[j]
            i += 1
            j += 1
        elif idx_a < idx_b:
            i += 1
        else:
            j += 1

    return total


def batch_cosine_similarity(query: Vector, matrix: Matrix, eps: float = 1e-12) -> List[float]:
    """Compute cosine similarities between a single query vector and a matrix of vectors.

    Args:
        query: Query vector of dimension D.
        matrix: Sequence of vectors of dimension D.
        eps: Epsilon for zero division protection.

    Returns:
        List of similarity scores corresponding to each vector in matrix.
    """
    if not matrix or not query:
        return []

    if _HAS_NUMPY:
        q_arr = np.asarray(query, dtype=np.float64)
        m_arr = np.asarray(matrix, dtype=np.float64)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm < eps:
            return [0.0] * len(matrix)
        m_norms = np.linalg.norm(m_arr, axis=1)
        m_norms = np.where(m_norms < eps, eps, m_norms)
        dots = np.dot(m_arr, q_arr)
        sims = dots / (m_norms * q_norm)
        return np.clip(sims, -1.0, 1.0).tolist()

    q_norm = math.sqrt(math.fsum(x * x for x in query))
    if q_norm < eps:
        return [0.0] * len(matrix)

    results: List[float] = []
    for row in matrix:
        if len(row) != len(query):
            raise ValueError(f"Matrix row dimension {len(row)} does not match query {len(query)}")
        dot = math.fsum(q * r for q, r in zip(query, row))
        r_norm = math.sqrt(math.fsum(r * r for r in row))
        denom = q_norm * r_norm
        if denom < eps:
            results.append(0.0)
        else:
            results.append(max(-1.0, min(1.0, dot / denom)))

    return results

