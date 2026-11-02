"""BM25+ ranking algorithm built from scratch.

Implements the BM25+ variant of the Okapi BM25 probabilistic relevance
model with an additive delta parameter to address the lower-bounding
issue of standard BM25.

Key formulas:
    - IDF: ``log((N - df + 0.5) / (df + 0.5) + 1)``
    - TF component: ``((k1 + 1) * tf) / (k1 * (1 - b + b * dl/avgdl) + tf) + delta``
    - Score: ``sum(IDF(t) * TF_component(t, d))`` for query terms ``t``

Uses an efficient inverted index for fast ranking.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── Tokenization (self-contained) ─────────────────────────────────────────

_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"(?u)\b\w\w+\b")

_STOPWORDS: frozenset[str] = frozenset({
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
    "too", "very", "s", "t", "just", "don", "now", "of", "if",
})


def _tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Args:
        text: Input text.
        remove_stopwords: Whether to remove stopwords.

    Returns:
        List of lowercase token strings.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


# ─── Inverted Index ─────────────────────────────────────────────────────────


@dataclass
class _PostingEntry:
    """A single posting in the inverted index.

    Attributes:
        doc_idx: Index of the document.
        term_freq: Frequency of the term in this document.
    """
    doc_idx: int
    term_freq: int


class _InvertedIndex:
    """Efficient inverted index mapping terms to document postings.

    Supports fast lookup of which documents contain a given term
    and the term frequency in each.

    Attributes:
        postings: Dict mapping term to list of posting entries.
        doc_lengths: List of document lengths (in tokens).
        n_docs: Total number of indexed documents.
        avg_dl: Average document length.
    """

    __slots__ = ("postings", "doc_lengths", "n_docs", "avg_dl", "doc_freq")

    def __init__(self) -> None:
        self.postings: Dict[str, list[_PostingEntry]] = {}
        self.doc_lengths: list[int] = []
        self.n_docs: int = 0
        self.avg_dl: float = 0.0
        self.doc_freq: Dict[str, int] = {}

    def build(self, tokenized_docs: list[list[str]]) -> None:
        """Build the inverted index from pre-tokenized documents.

        Args:
            tokenized_docs: List of tokenized documents (each a list of
                           token strings).
        """
        self.n_docs = len(tokenized_docs)
        self.doc_lengths = [len(doc) for doc in tokenized_docs]

        total_length = sum(self.doc_lengths)
        self.avg_dl = total_length / self.n_docs if self.n_docs > 0 else 0.0

        self.postings.clear()
        self.doc_freq.clear()

        for doc_idx, tokens in enumerate(tokenized_docs):
            # Count term frequencies in this document
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            # Add to postings
            for term, freq in tf.items():
                if term not in self.postings:
                    self.postings[term] = []
                self.postings[term].append(_PostingEntry(doc_idx, freq))
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def get_postings(self, term: str) -> list[_PostingEntry]:
        """Get posting list for a term.

        Args:
            term: The term to look up.

        Returns:
            List of PostingEntry objects, or empty list if not found.
        """
        return self.postings.get(term, [])

    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term.

        Args:
            term: The term to look up.

        Returns:
            Number of documents containing the term.
        """
        return self.doc_freq.get(term, 0)


# ─── BM25 ──────────────────────────────────────────────────────────────────


class BM25:
    """BM25+ ranking algorithm.

    An improved variant of BM25 that adds a delta parameter to the
    term frequency component, ensuring that the contribution of a term
    to the score is always positive even for very long documents.

    Example::

        bm25 = BM25(k1=1.5, b=0.75, delta=1.0)
        bm25.fit([
            "information retrieval systems",
            "machine learning algorithms",
            "information theory and coding",
        ])
        results = bm25.rank("information retrieval", top_k=2)
        # [(0, 3.45), (2, 1.23)]

    Attributes:
        k1: Term frequency saturation parameter. Controls how quickly
            TF saturates. Default 1.5.
        b: Length normalization parameter in [0, 1]. 0 = no normalization,
           1 = full normalization. Default 0.75.
        delta: BM25+ addition to the TF component. Default 1.0.
        remove_stopwords: Whether to remove stopwords during tokenization.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        remove_stopwords: bool = True,
    ) -> None:
        """Initialise BM25+.

        Args:
            k1: TF saturation parameter (>= 0). Default 1.5.
            b: Length normalization parameter in [0, 1]. Default 0.75.
            delta: BM25+ delta added to TF component. Default 1.0.
            remove_stopwords: Whether to remove stopwords. Default True.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")

        self.k1: float = k1
        self.b: float = b
        self.delta: float = delta
        self.remove_stopwords: bool = remove_stopwords

        # Internal state
        self._index: _InvertedIndex = _InvertedIndex()
        self._idf_cache: Dict[str, float] = {}
        self._tokenized_docs: list[list[str]] = []
        self._fitted: bool = False

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term.

        Formula: log((N - df + 0.5) / (df + 0.5) + 1)

        Args:
            term: The term.

        Returns:
            IDF value.
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        n = self._index.n_docs
        df = self._index.get_doc_freq(term)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = idf
        return idf

    def _tf_component(self, tf: int, doc_len: int) -> float:
        """Compute BM25+ TF component.

        Formula: ((k1 + 1) * tf) / (k1 * (1 - b + b * dl/avgdl) + tf) + delta

        Args:
            tf: Term frequency in the document.
            doc_len: Length of the document (in tokens).

        Returns:
            TF component value.
        """
        avg_dl = self._index.avg_dl
        if avg_dl == 0.0:
            return self.delta

        norm = 1.0 - self.b + self.b * (doc_len / avg_dl)
        numerator = (self.k1 + 1.0) * tf
        denominator = self.k1 * norm + tf
        return (numerator / denominator) + self.delta

    # ── Public API ──────────────────────────────────────────────────────

    def fit(self, documents: list[str]) -> BM25:
        """Index a collection of documents.

        Args:
            documents: List of raw text documents.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list.")

        self._tokenized_docs = [
            _tokenize(doc, self.remove_stopwords) for doc in documents
        ]
        self._index.build(self._tokenized_docs)
        self._idf_cache.clear()

        # Pre-compute IDF for all terms
        for term in self._index.postings:
            self._compute_idf(term)

        self._fitted = True
        return self

    def score(self, query: str, doc_idx: int) -> float:
        """Score a single document against a query.

        Args:
            query: Raw query text.
            doc_idx: Index of the document in the fitted collection.

        Returns:
            BM25+ score for the document.

        Raises:
            RuntimeError: If BM25 has not been fitted.
            IndexError: If doc_idx is out of range.
        """
        if not self._fitted:
            raise RuntimeError("BM25 has not been fitted. Call fit() first.")
        if doc_idx < 0 or doc_idx >= self._index.n_docs:
            raise IndexError(
                f"doc_idx {doc_idx} out of range [0, {self._index.n_docs})"
            )

        query_terms = _tokenize(query, self.remove_stopwords)
        doc_len = self._index.doc_lengths[doc_idx]

        # Get term frequencies for target document
        doc_tokens = self._tokenized_docs[doc_idx]
        doc_tf: Dict[str, int] = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1

        total_score: float = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            idf = self._compute_idf(term)
            tf_comp = self._tf_component(tf, doc_len)
            total_score += idf * tf_comp

        return total_score

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Rank all documents against a query.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_idx, score) tuples sorted by score descending.

        Raises:
            RuntimeError: If BM25 has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("BM25 has not been fitted. Call fit() first.")

        query_terms = _tokenize(query, self.remove_stopwords)
        if not query_terms:
            return []

        # Accumulate scores using the inverted index (efficient)
        scores: Dict[int, float] = {}
        for term in query_terms:
            idf = self._compute_idf(term)
            for posting in self._index.get_postings(term):
                doc_len = self._index.doc_lengths[posting.doc_idx]
                tf_comp = self._tf_component(posting.term_freq, doc_len)
                score_contrib = idf * tf_comp
                scores[posting.doc_idx] = scores.get(posting.doc_idx, 0.0) + score_contrib

        # Sort by score descending, then by doc_idx ascending for stability
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # Limit to top_k
        return ranked[:top_k]

    def batch_rank(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[tuple[int, float]]]:
        """Rank documents for multiple queries.

        Args:
            queries: List of raw query texts.
            top_k: Maximum number of results per query.

        Returns:
            List of ranking lists, one per query.

        Raises:
            RuntimeError: If BM25 has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("BM25 has not been fitted. Call fit() first.")
        return [self.rank(query, top_k) for query in queries]

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._index.n_docs

    @property
    def avg_doc_length(self) -> float:
        """Average document length in tokens."""
        return self._index.avg_dl

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index.postings)
