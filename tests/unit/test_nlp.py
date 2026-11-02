"""Unit tests for NLP Engine."""
from __future__ import annotations

import pytest
from veriflux.nlp import (
    BPETokenizer,
    ClaimExtractor,
    EntityExtractor,
    MinHasher,
    PorterStemmer,
    SentenceSplitter,
    SimHasher,
    TextNormalizer,
    WordPieceTokenizer,
)


def test_text_normalizer() -> None:
    normalizer = TextNormalizer()
    text = "Hello   \t\n  World!  “Smart quotes” and — em-dash."
    normalized = normalizer.normalize(text)
    assert "  " not in normalized
    assert '"Smart quotes"' in normalized
    assert "-" in normalized


def test_sentence_splitter() -> None:
    splitter = SentenceSplitter()
    text = "Dr. Smith went to Washington, D.C. He bought 3.14 lbs of apples! Did he like it? Yes, he did."
    sentences = splitter.split(text)
    assert len(sentences) >= 3
    # Ensure Dr. and D.C. and 3.14 didn't cause false splits
    assert any("Dr. Smith" in s for s in sentences)
    assert any("3.14" in s for s in sentences)


def test_porter_stemmer() -> None:
    stemmer = PorterStemmer()
    assert stemmer.stem("connecting") == "connect"
    assert stemmer.stem("connections") == "connect"
    assert stemmer.stem("relational") == "relat"
    assert stemmer.stem("conditional") == "condit"


def test_bpe_tokenizer() -> None:
    tokenizer = BPETokenizer()
    corpus = [
        "low low low lower lowest",
        "new newer newest news",
        "wide wider widest",
    ]
    tokenizer.train(corpus, vocab_size=280)
    encoded = tokenizer.encode("lower news")
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert decoded.strip() == "lower news"


def test_wordpiece_tokenizer() -> None:
    tokenizer = WordPieceTokenizer()
    corpus = ["un", "affable"]
    tokenizer.fit(corpus, vocab_size=30)
    tokens = tokenizer.tokenize("unaffable")
    assert any(t.startswith("##") for t in tokens[1:])


def test_claim_extractor() -> None:
    extractor = ClaimExtractor()
    text = "Albert Einstein was born in 1879, and he developed the theory of relativity in 1905."
    claims = extractor.extract(text, clause_level=True)
    assert len(claims) >= 2
    claim_texts = [c.text for c in claims]
    assert any("1879" in c for c in claim_texts)
    assert any("relativity" in c for c in claim_texts)


def test_entity_extractor() -> None:
    extractor = EntityExtractor()
    text = "Elon Musk visited London on January 15, 2024 and paid $500 for tickets. Contact elon@example.com."
    entities = extractor.extract(text)
    types = {e.entity_type for e in entities}
    assert "EMAIL" in types
    assert "MONEY" in types
    assert "DATE" in types
    assert "LOCATION" in types


def test_fingerprinting() -> None:
    minhasher = MinHasher(num_perm=64)
    simhasher = SimHasher(hashbits=64)

    text_a = "The quick brown fox jumps over the lazy dog in the summer heat"
    text_b = "The quick brown fox jumps over the lazy dog in summer heat"
    text_c = "Quantum computing relies on qubits, superposition and entanglement"

    sig_a = minhasher.fingerprint(text_a)
    sig_b = minhasher.fingerprint(text_b)
    sig_c = minhasher.fingerprint(text_c)

    sim_ab = minhasher.similarity(sig_a, sig_b)
    sim_ac = minhasher.similarity(sig_a, sig_c)
    assert sim_ab > 0.6
    assert sim_ac < 0.2

    h_a = simhasher.fingerprint(text_a)
    h_b = simhasher.fingerprint(text_b)
    h_c = simhasher.fingerprint(text_c)

    assert simhasher.similarity(h_a, h_b) > simhasher.similarity(h_a, h_c)
