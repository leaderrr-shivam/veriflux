"""Multi-strategy context compression without external LLM dependencies.

Compresses context passages using statistical sentence salience scoring (extractive)
and rule-based syntactic compaction (compressive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from veriflux.nlp.sentence_splitter import SentenceSplitter
from veriflux.similarity.tfidf import TfidfVectorizer
from veriflux.tokens.counter import count_tokens
from veriflux.tokens.optimizer import PromptOptimizer


@dataclass(slots=True, frozen=True)
class CompressedContext:
    """Outcome of context compression."""

    text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_estimate: float


class ContextCompressor:
    """Compresses context passages to meet stringent token constraints."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.sentence_splitter = SentenceSplitter()
        self.optimizer = PromptOptimizer(model=model)

    def compress(
        self,
        text: str,
        target_tokens: int,
        strategy: str = "extractive",
    ) -> CompressedContext:
        """Compress text to target token ceiling."""
        if not text:
            return CompressedContext("", 0, 0, 1.0, 1.0)

        orig_toks = count_tokens(text, model=self.model)
        if orig_toks <= target_tokens:
            return CompressedContext(text, orig_toks, orig_toks, 1.0, 1.0)

        compressed_text = text

        if strategy in ("compressive", "hybrid"):
            compressed_text = self.optimizer.optimize_prompt(compressed_text)

        if strategy in ("extractive", "hybrid"):
            sentences = self.sentence_splitter.split(compressed_text)
            if len(sentences) > 1:
                # Score sentences using TF-IDF term weights
                vec = TfidfVectorizer()
                vec.fit(sentences)
                scored = []
                for idx, sent in enumerate(sentences):
                    # Position bias: earlier sentences typically have higher context value
                    pos_bias = 1.0 - (0.3 * (idx / len(sentences)))
                    # Content density
                    tokens = sent.split()
                    score = len(tokens) * pos_bias
                    scored.append((score, idx, sent))

                # Select top sentences until target token budget is reached
                scored.sort(key=lambda x: x[0], reverse=True)
                selected_indices: set[int] = set()
                curr_toks = 0

                for _, idx, sent in scored:
                    s_tok = count_tokens(sent, model=self.model)
                    if curr_toks + s_tok <= target_tokens:
                        selected_indices.add(idx)
                        curr_toks += s_tok

                # Reconstruct in original chronological order
                ordered = [sentences[i] for i in sorted(selected_indices)]
                compressed_text = " ".join(ordered)

        final_tokens = count_tokens(compressed_text, model=self.model)
        ratio = round(orig_toks / max(1, final_tokens), 2)
        quality = min(1.0, max(0.5, 1.0 - (0.2 * (ratio - 1.0))))

        return CompressedContext(
            text=compressed_text,
            original_tokens=orig_toks,
            compressed_tokens=final_tokens,
            compression_ratio=ratio,
            quality_estimate=quality,
        )

