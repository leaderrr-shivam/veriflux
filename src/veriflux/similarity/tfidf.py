"""TF-IDF vectorizer built from scratch.

Implements Term Frequency–Inverse Document Frequency vectorization with
sparse vector output, sublinear TF scaling, IDF smoothing, L2 normalization,
and configurable n-gram ranges, min/max document frequency filters, and
max feature limits.

All algorithms are implemented from first principles using only Python stdlib.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


# ─── Built-in stopwords (avoid external dependency) ─────────────────────────

_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "no", "nor",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "just", "don", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn", "of", "if",
})

# Regex pattern for basic tokenization
_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"(?u)\b\w\w+\b")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens (length >= 2).

    Args:
        text: Input text.

    Returns:
        List of lowercase token strings.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def _generate_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    """Generate n-grams from a list of tokens.

    Args:
        tokens: List of individual tokens.
        ngram_range: Tuple of (min_n, max_n) for n-gram sizes.

    Returns:
        List of n-gram strings (space-joined for multi-word).
    """
    min_n, max_n = ngram_range
    result: list[str] = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            result.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                result.append(" ".join(tokens[i : i + n]))
    return result


# ─── SparseVector ───────────────────────────────────────────────────────────


class SparseVector:
    """Dictionary-based sparse vector supporting math operations.

    Stores only non-zero entries keyed by integer index.
    Provides element-wise arithmetic, dot products, norms, and
    cosine similarity computation.

    Attributes:
        data: Dictionary mapping index to value.
    """

    __slots__ = ("data",)

    def __init__(self, data: Optional[Dict[int, float]] = None) -> None:
        """Initialise a SparseVector.

        Args:
            data: Optional dict mapping indices to values.
                  Defaults to empty vector.
        """
        self.data: Dict[int, float] = dict(data) if data else {}

    # ── Representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self.data)
        if n <= 5:
            entries = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(self.data.items()))
        else:
            items = sorted(self.data.items())
            entries = ", ".join(f"{k}: {v:.4f}" for k, v in items[:3])
            entries += f", ... ({n - 3} more)"
        return f"SparseVector({{{entries}}})"

    def __len__(self) -> int:
        """Number of non-zero entries."""
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseVector):
            return NotImplemented
        return self.data == other.data

    def __getitem__(self, index: int) -> float:
        """Get value at index (0.0 for missing entries)."""
        return self.data.get(index, 0.0)

    def __setitem__(self, index: int, value: float) -> None:
        """Set value at index. Removes entry if value is 0."""
        if value == 0.0:
            self.data.pop(index, None)
        else:
            self.data[index] = value

    def __contains__(self, index: int) -> bool:
        return index in self.data

    def __iter__(self) -> Iterator[int]:
        return iter(self.data)

    # ── Arithmetic ──────────────────────────────────────────────────────

    def __add__(self, other: SparseVector) -> SparseVector:
        """Element-wise addition of two sparse vectors.

        Args:
            other: Another SparseVector.

        Returns:
            New SparseVector with summed entries.
        """
        if not isinstance(other, SparseVector):
            return NotImplemented
        result: Dict[int, float] = dict(self.data)
        for k, v in other.data.items():
            result[k] = result.get(k, 0.0) + v
        # Remove zeros
        return SparseVector({k: v for k, v in result.items() if v != 0.0})

    def __sub__(self, other: SparseVector) -> SparseVector:
        """Element-wise subtraction.

        Args:
            other: Another SparseVector.

        Returns:
            New SparseVector with differences.
        """
        if not isinstance(other, SparseVector):
            return NotImplemented
        result: Dict[int, float] = dict(self.data)
        for k, v in other.data.items():
            result[k] = result.get(k, 0.0) - v
        return SparseVector({k: v for k, v in result.items() if v != 0.0})

    def __mul__(self, scalar: float) -> SparseVector:
        """Scalar multiplication.

        Args:
            scalar: Scalar value to multiply by.

        Returns:
            New SparseVector with scaled entries.
        """
        if isinstance(scalar, (int, float)):
            if scalar == 0.0:
                return SparseVector()
            return SparseVector({k: v * scalar for k, v in self.data.items()})
        return NotImplemented

    def __rmul__(self, scalar: float) -> SparseVector:
        return self.__mul__(scalar)

    # ── Vector operations ───────────────────────────────────────────────

    def dot(self, other: SparseVector) -> float:
        """Compute dot product with another sparse vector.

        Iterates over the smaller vector for efficiency.

        Args:
            other: Another SparseVector.

        Returns:
            Scalar dot product.
        """
        if not isinstance(other, SparseVector):
            raise TypeError(f"Expected SparseVector, got {type(other).__name__}")

        # Iterate over the smaller vector for O(min(nnz_a, nnz_b))
        if len(self.data) <= len(other.data):
            small, large = self.data, other.data
        else:
            small, large = other.data, self.data

        total: float = 0.0
        for k, v in small.items():
            if k in large:
                total += v * large[k]
        return total

    def norm(self) -> float:
        """Compute L2 norm.

        Returns:
            Euclidean norm of the vector.
        """
        if not self.data:
            return 0.0
        return math.sqrt(sum(v * v for v in self.data.values()))

    def normalize(self) -> SparseVector:
        """Return L2-normalized copy.

        Returns:
            New SparseVector with unit L2 norm, or empty vector if norm is 0.
        """
        n = self.norm()
        if n == 0.0:
            return SparseVector()
        inv_n = 1.0 / n
        return SparseVector({k: v * inv_n for k, v in self.data.items()})

    def cosine_with(self, other: SparseVector) -> float:
        """Compute cosine similarity with another sparse vector.

        Args:
            other: Another SparseVector.

        Returns:
            Cosine similarity in [-1, 1]. Returns 0.0 if either vector has
            zero norm.
        """
        dot_val = self.dot(other)
        if dot_val == 0.0:
            return 0.0
        norm_a = self.norm()
        norm_b = other.norm()
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_val / (norm_a * norm_b)

    def to_dense(self, size: int) -> list[float]:
        """Convert to dense list.

        Args:
            size: Dimension of the output dense vector.

        Returns:
            Dense list of floats with length ``size``.
        """
        dense = [0.0] * size
        for k, v in self.data.items():
            if 0 <= k < size:
                dense[k] = v
        return dense

    @classmethod
    def from_dense(cls, vector: list[float]) -> SparseVector:
        """Create from dense list, storing only non-zero entries.

        Args:
            vector: Dense float list.

        Returns:
            SparseVector with non-zero entries.
        """
        return cls({i: v for i, v in enumerate(vector) if v != 0.0})

    def keys(self) -> Iterator[int]:
        """Iterate over non-zero indices."""
        return iter(self.data.keys())

    def values(self) -> Iterator[float]:
        """Iterate over non-zero values."""
        return iter(self.data.values())

    def items(self) -> Iterator[tuple[int, float]]:
        """Iterate over (index, value) pairs."""
        return iter(self.data.items())

    def copy(self) -> SparseVector:
        """Return a shallow copy."""
        return SparseVector(dict(self.data))


# ─── TfidfVectorizer ───────────────────────────────────────────────────────


class TfidfVectorizer:
    """TF-IDF vectorizer built from scratch.

    Converts a collection of text documents into TF-IDF sparse vectors.

    Supports:
        - Sublinear TF: ``1 + log(tf)``
        - IDF smoothing: ``log((N+1)/(df+1)) + 1``
        - L2 normalization of output vectors
        - Configurable n-gram ranges, min/max document frequency,
          max features, stopword filtering

    Example::

        vectorizer = TfidfVectorizer(max_features=1000, sublinear_tf=True)
        vectors = vectorizer.fit_transform([
            "the quick brown fox",
            "the lazy brown dog",
        ])
        similarity = vectors[0].cosine_with(vectors[1])

    Attributes:
        max_features: Maximum vocabulary size (None for unlimited).
        min_df: Minimum document frequency (int) or fraction (float).
        max_df: Maximum document frequency (int) or fraction (float).
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        use_idf: Whether to apply IDF weighting.
        sublinear_tf: Whether to use sublinear TF (1 + log(tf)).
        remove_stopwords: Whether to remove English stopwords.
    """

    def __init__(
        self,
        *,
        max_features: Optional[int] = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        ngram_range: tuple[int, int] = (1, 1),
        use_idf: bool = True,
        sublinear_tf: bool = False,
        remove_stopwords: bool = True,
    ) -> None:
        """Initialise the TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features (vocabulary size).
                          If ``None``, all features are kept.
            min_df: Minimum document frequency. If int, the absolute count.
                    If float in [0.0, 1.0], the fraction of documents.
            max_df: Maximum document frequency. If int, the absolute count.
                    If float in [0.0, 1.0], the fraction of documents.
            ngram_range: Tuple of (min_n, max_n) for token n-gram sizes.
            use_idf: Apply inverse document frequency weighting.
            sublinear_tf: Apply sublinear TF scaling ``1 + log(tf)`` instead
                         of raw term frequency.
            remove_stopwords: Remove built-in English stopwords.
        """
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(
                f"Invalid ngram_range {ngram_range}: min_n must be >= 1 "
                f"and max_n must be >= min_n."
            )
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")

        self.max_features: Optional[int] = max_features
        self.min_df: int | float = min_df
        self.max_df: int | float = max_df
        self.ngram_range: tuple[int, int] = ngram_range
        self.use_idf: bool = use_idf
        self.sublinear_tf: bool = sublinear_tf
        self.remove_stopwords: bool = remove_stopwords

        # Fitted state
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[int, float] = {}
        self._fitted: bool = False

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def vocabulary_(self) -> Dict[str, int]:
        """Mapping of terms to feature indices.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")
        return dict(self._vocabulary)

    @property
    def idf_(self) -> Dict[int, float]:
        """Mapping of feature indices to IDF values.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")
        return dict(self._idf)

    @property
    def n_features(self) -> int:
        """Number of features in the vocabulary.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")
        return len(self._vocabulary)

    # ── Internal helpers ────────────────────────────────────────────────

    def _preprocess(self, text: str) -> list[str]:
        """Tokenize and optionally remove stopwords, then generate n-grams.

        Args:
            text: Raw input text.

        Returns:
            List of n-gram strings.
        """
        tokens = _tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _ENGLISH_STOPWORDS]
        return _generate_ngrams(tokens, self.ngram_range)

    def _compute_df_bounds(self, n_docs: int) -> tuple[int, int]:
        """Convert min_df/max_df to absolute document counts.

        Args:
            n_docs: Total number of documents.

        Returns:
            Tuple of (min_df_abs, max_df_abs).
        """
        if isinstance(self.min_df, float):
            min_df_abs = max(1, int(math.ceil(self.min_df * n_docs)))
        else:
            min_df_abs = self.min_df

        if isinstance(self.max_df, float):
            max_df_abs = max(1, int(math.floor(self.max_df * n_docs)))
        else:
            max_df_abs = self.max_df

        return min_df_abs, max_df_abs

    # ── Fit / Transform ─────────────────────────────────────────────────

    def fit(self, documents: list[str]) -> TfidfVectorizer:
        """Build vocabulary and IDF scores from a corpus.

        Args:
            documents: List of raw text documents.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list.")

        n_docs = len(documents)
        min_df_abs, max_df_abs = self._compute_df_bounds(n_docs)

        # Count document frequency for each term
        df: Dict[str, int] = {}
        for doc in documents:
            terms = self._preprocess(doc)
            seen: set[str] = set()
            for term in terms:
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)

        # Filter by document frequency bounds
        filtered_terms: list[tuple[str, int]] = []
        for term, doc_freq in df.items():
            if min_df_abs <= doc_freq <= max_df_abs:
                filtered_terms.append((term, doc_freq))

        # Sort by document frequency (descending), then alphabetically for ties
        filtered_terms.sort(key=lambda x: (-x[1], x[0]))

        # Apply max_features limit
        if self.max_features is not None and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[: self.max_features]

        # Build vocabulary
        # Sort alphabetically for deterministic ordering
        filtered_terms.sort(key=lambda x: x[0])
        self._vocabulary = {term: idx for idx, (term, _) in enumerate(filtered_terms)}

        # Compute IDF: log((N+1)/(df+1)) + 1  (smooth variant)
        self._idf = {}
        for term, idx in self._vocabulary.items():
            term_df = df[term]
            if self.use_idf:
                self._idf[idx] = math.log((n_docs + 1) / (term_df + 1)) + 1.0
            else:
                self._idf[idx] = 1.0

        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> list[SparseVector]:
        """Transform documents to TF-IDF sparse vectors.

        Args:
            documents: List of raw text documents to transform.

        Returns:
            List of L2-normalized TF-IDF SparseVectors.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        result: list[SparseVector] = []
        for doc in documents:
            terms = self._preprocess(doc)

            # Count term frequencies
            tf: Dict[str, int] = {}
            for term in terms:
                if term in self._vocabulary:
                    tf[term] = tf.get(term, 0) + 1

            # Build TF-IDF sparse vector
            vec_data: Dict[int, float] = {}
            for term, count in tf.items():
                idx = self._vocabulary[term]

                # Sublinear TF
                if self.sublinear_tf:
                    tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                else:
                    tf_val = float(count)

                # Multiply by IDF
                tfidf_val = tf_val * self._idf[idx]

                if tfidf_val != 0.0:
                    vec_data[idx] = tfidf_val

            # L2 normalize
            vec = SparseVector(vec_data)
            result.append(vec.normalize())

        return result

    def fit_transform(self, documents: list[str]) -> list[SparseVector]:
        """Fit the vectorizer and transform documents in one step.

        Args:
            documents: List of raw text documents.

        Returns:
            List of L2-normalized TF-IDF SparseVectors.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> list[str]:
        """Get the list of feature names ordered by index.

        Returns:
            Sorted list of feature names.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")
        names = [""] * len(self._vocabulary)
        for term, idx in self._vocabulary.items():
            names[idx] = term
        return names
