"""
N-gram generation and analysis for Veriflux NLP.

Provides functions for building token- and character-level n-grams,
frequency distributions, top-k extraction, skip-grams, and a simple
n-gram language model with add-k smoothing and perplexity computation.
"""
from __future__ import annotations

import math
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)


# ─── Core n-gram functions ─────────────────────────────────────────────────


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Generate token-level n-grams.

    Args:
        tokens: Ordered sequence of tokens.
        n: Size of each n-gram (must be ≥ 1).

    Returns:
        List of n-gram tuples.

    Raises:
        ValueError: If *n* < 1.

    Example::

        >>> ngrams(["a", "b", "c", "d"], 2)
        [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    length = len(tokens)
    if n > length:
        return []
    return [tuple(tokens[i : i + n]) for i in range(length - n + 1)]


def char_ngrams(text: str, n: int) -> List[str]:
    """Generate character-level n-grams.

    Args:
        text: Input string.
        n: Size of each character n-gram (must be ≥ 1).

    Returns:
        List of character n-gram strings.

    Raises:
        ValueError: If *n* < 1.

    Example::

        >>> char_ngrams("hello", 3)
        ['hel', 'ell', 'llo']
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    length = len(text)
    if n > length:
        return []
    return [text[i : i + n] for i in range(length - n + 1)]


def ngram_frequency(
    tokens: Sequence[str], n: int
) -> Dict[Tuple[str, ...], int]:
    """Compute the frequency distribution of n-grams.

    Args:
        tokens: Ordered sequence of tokens.
        n: N-gram size.

    Returns:
        Dictionary mapping each n-gram tuple to its count.
    """
    freq: Dict[Tuple[str, ...], int] = {}
    for gram in ngrams(tokens, n):
        freq[gram] = freq.get(gram, 0) + 1
    return freq


def top_ngrams(
    tokens: Sequence[str], n: int, k: int
) -> List[Tuple[Tuple[str, ...], int]]:
    """Return the *k* most frequent n-grams.

    Args:
        tokens: Ordered sequence of tokens.
        n: N-gram size.
        k: Number of top entries to return.

    Returns:
        List of ``(ngram_tuple, count)`` sorted by count descending,
        then lexicographically by n-gram for stability.
    """
    freq = ngram_frequency(tokens, n)
    # Sort by (-count, ngram) for deterministic ordering.
    sorted_items = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return sorted_items[:k]


def skip_grams(
    tokens: Sequence[str], n: int, skip: int
) -> List[Tuple[str, ...]]:
    """Generate skip-grams.

    A skip-gram of order *n* with *skip* skips selects *n* tokens from
    a window of size ``n + skip``, allowing up to *skip* tokens to be
    skipped between each selected token.

    For *n* = 2, this produces all pairs ``(tokens[i], tokens[j])`` where
    ``i < j`` and ``j - i <= skip + 1``.

    Args:
        tokens: Ordered sequence of tokens.
        n: Number of tokens in each skip-gram (must be ≥ 1).
        skip: Maximum number of tokens to skip (must be ≥ 0).

    Returns:
        List of skip-gram tuples.

    Raises:
        ValueError: If constraints on *n* or *skip* are violated.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")

    length = len(tokens)
    if length == 0 or n > length:
        return []

    results: List[Tuple[str, ...]] = []

    def _build(
        start: int, depth: int, current: List[str]
    ) -> None:
        if depth == n:
            results.append(tuple(current))
            return
        # The furthest index we can pick from
        max_idx = start + skip + 1
        for i in range(start, min(max_idx, length)):
            current.append(tokens[i])
            _build(i + 1, depth + 1, current)
            current.pop()

    for i in range(length):
        _build(i, 0, [])

    return results


# ─── N-gram Language Model ──────────────────────────────────────────────────


class NGramModel:
    """N-gram language model with add-k smoothing.

    Builds a simple n-gram language model from tokenised text and
    supports probability queries, sequence scoring, and perplexity
    computation.

    Args:
        n: Order of the model (e.g. 2 for bigrams).
        k: Smoothing constant for add-k smoothing (default ``1.0``).

    Example::

        model = NGramModel(n=2, k=1.0)
        model.fit([["the", "cat", "sat"], ["the", "cat", "ran"]])
        model.probability("sat", ("the", "cat"))
    """

    __slots__ = (
        "_n",
        "_k",
        "_ngram_counts",
        "_context_counts",
        "_vocab",
    )

    def __init__(self, n: int = 2, k: float = 1.0) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self._n = n
        self._k = k
        self._ngram_counts: Dict[Tuple[str, ...], int] = {}
        self._context_counts: Dict[Tuple[str, ...], int] = {}
        self._vocab: set[str] = set()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, token_sequences: Sequence[Sequence[str]]) -> None:
        """Train the model on a corpus of token sequences.

        Each sequence is padded with ``<s>`` (start) and ``</s>`` (end)
        boundary tokens.

        Args:
            token_sequences: List of token lists.
        """
        self._ngram_counts.clear()
        self._context_counts.clear()
        self._vocab.clear()

        pad_start = ["<s>"] * (self._n - 1)
        pad_end = ["</s>"]

        for seq in token_sequences:
            padded = pad_start + list(seq) + pad_end
            self._vocab.update(seq)
            self._vocab.add("</s>")
            for gram in ngrams(padded, self._n):
                self._ngram_counts[gram] = self._ngram_counts.get(gram, 0) + 1
                context = gram[:-1]
                self._context_counts[context] = (
                    self._context_counts.get(context, 0) + 1
                )

    def update(self, token_sequences: Sequence[Sequence[str]]) -> None:
        """Incrementally update model counts with new data.

        Args:
            token_sequences: Additional token lists.
        """
        pad_start = ["<s>"] * (self._n - 1)
        pad_end = ["</s>"]

        for seq in token_sequences:
            padded = pad_start + list(seq) + pad_end
            self._vocab.update(seq)
            self._vocab.add("</s>")
            for gram in ngrams(padded, self._n):
                self._ngram_counts[gram] = self._ngram_counts.get(gram, 0) + 1
                context = gram[:-1]
                self._context_counts[context] = (
                    self._context_counts.get(context, 0) + 1
                )

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute the smoothed probability P(word | context).

        Uses add-k smoothing:
            P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k * |V|)

        Args:
            word: Target word.
            context: Tuple of preceding context tokens (length = n-1).

        Returns:
            Smoothed probability (float in (0, 1]).
        """
        if len(context) != self._n - 1:
            raise ValueError(
                f"Context must have {self._n - 1} tokens, got {len(context)}"
            )
        gram = context + (word,)
        gram_count = self._ngram_counts.get(gram, 0)
        context_count = self._context_counts.get(context, 0)
        vocab_size = max(len(self._vocab), 1)
        return (gram_count + self._k) / (context_count + self._k * vocab_size)

    def log_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute log₂ P(word | context).

        Args:
            word: Target word.
            context: Context tuple.

        Returns:
            Log₂ probability.
        """
        return math.log2(self.probability(word, context))

    def sequence_log_probability(self, tokens: Sequence[str]) -> float:
        """Compute the total log₂ probability of a token sequence.

        The sequence is padded with start-of-sentence markers.

        Args:
            tokens: Sequence of tokens.

        Returns:
            Sum of log₂ probabilities.
        """
        pad_start = ["<s>"] * (self._n - 1)
        padded = pad_start + list(tokens) + ["</s>"]
        total = 0.0
        for i in range(self._n - 1, len(padded)):
            context = tuple(padded[i - self._n + 1 : i])
            word = padded[i]
            total += self.log_probability(word, context)
        return total

    def perplexity(self, tokens: Sequence[str]) -> float:
        """Compute the perplexity of a token sequence.

        Perplexity = 2^(-1/N * sum(log₂ P(wᵢ | context)))

        Args:
            tokens: Sequence of tokens.

        Returns:
            Perplexity (lower is better).

        Raises:
            ValueError: If the token sequence is empty.
        """
        if not tokens:
            raise ValueError("Cannot compute perplexity of an empty sequence")
        log_prob = self.sequence_log_probability(tokens)
        # N includes the </s> token
        n_tokens = len(tokens) + 1
        return 2.0 ** (-log_prob / n_tokens)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def most_likely(
        self, context: Tuple[str, ...], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return the top-k most likely next words given a context.

        Args:
            context: Tuple of preceding tokens (length = n-1).
            top_k: Number of candidates to return.

        Returns:
            List of ``(word, probability)`` tuples, sorted descending.
        """
        scored: List[Tuple[str, float]] = []
        for word in self._vocab:
            p = self.probability(word, context)
            scored.append((word, p))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Order of the model."""
        return self._n

    @property
    def k(self) -> float:
        """Smoothing constant."""
        return self._k

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def vocab(self) -> frozenset[str]:
        """Immutable snapshot of the vocabulary."""
        return frozenset(self._vocab)

    def __repr__(self) -> str:
        return (
            f"NGramModel(n={self._n}, k={self._k}, "
            f"vocab_size={len(self._vocab)})"
        )
