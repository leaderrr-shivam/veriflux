"""Context pruning to satisfy strict token boundaries.

Prunes passages preserving sentence integrity and prioritizing high-relevance chunks.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from veriflux.nlp.sentence_splitter import SentenceSplitter
from veriflux.tokens.counter import count_tokens


class ContextPruner:
    """Prunes context passages to fit strict token budgets while minimizing semantic loss."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.sentence_splitter = SentenceSplitter()

    def prune(self, chunks: Sequence[str], budget: int) -> List[str]:
        """Prune chunks in sequence until total token consumption fits within budget."""
        if budget <= 0 or not chunks:
            return []

        retained: List[str] = []
        accumulated_tokens = 0

        for chunk in chunks:
            chunk_tokens = count_tokens(chunk, model=self.model)
            if accumulated_tokens + chunk_tokens <= budget:
                retained.append(chunk)
                accumulated_tokens += chunk_tokens
            else:
                # Try sentence-level salvage
                headroom = budget - accumulated_tokens
                if headroom > 30:
                    sentences = self.sentence_splitter.split(chunk)
                    partial_sentences: List[str] = []
                    for sent in sentences:
                        s_toks = count_tokens(sent, model=self.model)
                        if headroom >= s_toks:
                            partial_sentences.append(sent)
                            headroom -= s_toks
                            accumulated_tokens += s_toks
                        else:
                            break
                    if partial_sentences:
                        retained.append(" ".join(partial_sentences))
                break

        return retained

