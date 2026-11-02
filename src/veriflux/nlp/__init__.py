"""Natural Language Processing Engine for Veriflux.

Provides tokenizers, sentence splitters, claim extractors, entity recognition,
stemming, text normalization, and fingerprinting built from first principles.
"""
from __future__ import annotations

from veriflux.nlp.claim_extractor import ClaimExtractor
from veriflux.nlp.entity_extractor import Entity, EntityExtractor
from veriflux.nlp.fingerprint import MinHasher, MinHashSignature, SimHasher
from veriflux.nlp.ngrams import (
    NGramModel,
    char_ngrams,
    ngram_frequency,
    ngrams,
    skip_grams,
    top_ngrams,
)
from veriflux.nlp.sentence_splitter import SentenceSplitter, Span, split_sentences
from veriflux.nlp.stemmer import PorterStemmer, stem_tokens
from veriflux.nlp.stopwords import StopwordFilter
from veriflux.nlp.text_normalizer import (
    TextNormalizer,
    normalize_case,
    normalize_dashes,
    normalize_full,
    normalize_quotes,
    normalize_unicode,
    normalize_whitespace,
    remove_control_chars,
)
from veriflux.nlp.tokenizer import (
    BPETokenizer,
    RegexTokenizer,
    SimpleTokenizer,
    TokenizerResult,
    WordPieceTokenizer,
)

__all__ = [
    # Claim extraction
    "ClaimExtractor",
    # Entity extraction
    "Entity",
    "EntityExtractor",
    # Fingerprinting
    "MinHasher",
    "MinHashSignature",
    "SimHasher",
    # N-grams
    "ngrams",
    "char_ngrams",
    "ngram_frequency",
    "top_ngrams",
    "skip_grams",
    "NGramModel",
    # Sentence splitting
    "SentenceSplitter",
    "Span",
    "split_sentences",
    # Stemmer
    "PorterStemmer",
    "stem_tokens",
    # Stopwords
    "StopwordFilter",
    # Normalizer
    "TextNormalizer",
    "normalize_unicode",
    "normalize_whitespace",
    "normalize_case",
    "remove_control_chars",
    "normalize_quotes",
    "normalize_dashes",
    "normalize_full",
    # Tokenizers
    "BPETokenizer",
    "WordPieceTokenizer",
    "SimpleTokenizer",
    "RegexTokenizer",
    "TokenizerResult",
]

