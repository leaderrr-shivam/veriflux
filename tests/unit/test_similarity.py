"""Unit tests for Similarity Engine."""
from __future__ import annotations

import pytest
from veriflux.similarity import (
    BM25,
    SparseVector,
    TfidfVectorizer,
    VectorStore,
    combmnz,
    cosine_similarity,
    damerau_levenshtein,
    jaccard_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    reciprocal_rank_fusion,
    token_jaccard,
    weighted_linear,
)


def test_tfidf_and_cosine() -> None:
    docs = [
        "Machine learning models require training data.",
        "Deep learning uses neural network architectures.",
        "Data science and statistical inference for predictive modeling.",
    ]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    assert len(vectors) == 3

    # Similarity of identical document
    sim_self = vectors[0].cosine_with(vectors[0])
    assert pytest.approx(sim_self, abs=1e-3) == 1.0

    # Cross-document similarity
    sim_diff = vectors[0].cosine_with(vectors[1])
    assert 0.0 <= sim_diff <= 1.0


def test_bm25_ranking() -> None:
    corpus = [
        "The Apollo 11 landed humans on the Moon in 1969.",
        "Mars rover Curiosity explores the Martian surface.",
        "Hubble Space Telescope captures distant galactic nebulae.",
    ]
    bm25 = BM25()
    bm25.fit(corpus)
    rankings = bm25.rank("Apollo Moon 1969", top_k=2)
    assert len(rankings) >= 1
    # Document 0 should be top ranked
    assert rankings[0][0] == 0


def test_jaccard_and_levenshtein() -> None:
    s1 = "kitten"
    s2 = "sitting"
    assert levenshtein_distance(s1, s2) == 3
    assert 0.0 < levenshtein_similarity(s1, s2) < 1.0

    # Damerau transposition
    assert damerau_levenshtein("ca", "ac") == 1

    # Jaccard
    assert token_jaccard("apple banana cherry", "banana cherry date") == 0.5


def test_fusion_algorithms() -> None:
    r1 = [(0, 0.9), (1, 0.8), (2, 0.5)]
    r2 = [(1, 0.95), (0, 0.7), (2, 0.4)]
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    assert len(fused) == 3
    # Items 0 and 1 should dominate
    top_ids = [doc_id for doc_id, _ in fused[:2]]
    assert 0 in top_ids and 1 in top_ids


def test_vector_store() -> None:
    store = VectorStore(dimension=3)
    store.add("doc1", [1.0, 0.0, 0.0], metadata={"text": "vector 1"})
    store.add("doc2", [0.0, 1.0, 0.0], metadata={"text": "vector 2"})
    store.add("doc3", [0.9, 0.1, 0.0], metadata={"text": "vector 3"})

    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[1].id == "doc3"
