"""Prompt token optimization without semantic distortion.

Compresses prompts through whitespace normalization, filler phrase pruning,
and syntactic simplification.
"""
from __future__ import annotations

import re
from typing import Sequence

from veriflux.tokens.counter import count_tokens


class PromptOptimizer:
    """Optimises prompt token efficiency while preserving core semantic meaning."""

    _FILLER_PHRASES: Sequence[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bplease\s+(?:make\s+sure\s+to|be\s+sure\s+to|note\s+that)\b", re.I), "ensure"),
        (re.compile(r"\bin\s+order\s+to\b", re.I), "to"),
        (re.compile(r"\bas\s+a\s+matter\s+of\s+fact\b", re.I), ""),
        (re.compile(r"\bat\s+the\s+present\s+time\b", re.I), "now"),
        (re.compile(r"\bdue\s+to\s+the\s+fact\s+that\b", re.I), "because"),
        (re.compile(r"\bfor\s+the\s+purpose\s+of\b", re.I), "for"),
        (re.compile(r"\bin\s+the\s+event\s+that\b", re.I), "if"),
        (re.compile(r"\bwith\s+regard\s+to\b", re.I), "regarding"),
        (re.compile(r"\bit\s+is\s+important\s+to\s+note\s+that\b", re.I), "note:"),
        (re.compile(r"\bkeep\s+in\s+mind\s+that\b", re.I), "remember:"),
    ]

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    def optimize_prompt(self, prompt: str, target_tokens: Optional[int] = None) -> str:
        """Compress prompt text to reduce token footprint.

        Args:
            prompt: Original prompt text.
            target_tokens: Optional token ceiling.

        Returns:
            Optimized prompt string.
        """
        if not prompt:
            return ""

        text = prompt

        # 1. Normalize excessive whitespace and consecutive blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 2. Prune verbose filler phrases
        for pat, repl in self._FILLER_PHRASES:
            text = pat.sub(repl, text)

        # 3. Clean up resulting punctuation artifacts
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)
        text = text.strip()

        # If a strict target token limit was set and we still exceed it, truncate safely
        if target_tokens is not None:
            current_tokens = count_tokens(text, model=self.model)
            if current_tokens > target_tokens:
                lines = text.splitlines(keepends=True)
                accumulated: list[str] = []
                total = 0
                for line in lines:
                    line_toks = count_tokens(line, model=self.model)
                    if total + line_toks <= target_tokens:
                        accumulated.append(line)
                        total += line_toks
                    else:
                        break
                text = "".join(accumulated).strip()

        return text

    def compression_ratio(self, original: str, optimized: str) -> float:
        """Compute the compression ratio (original_tokens / optimized_tokens)."""
        orig_toks = count_tokens(original, model=self.model)
        opt_toks = count_tokens(optimized, model=self.model)
        if opt_toks == 0:
            return 1.0
        return round(orig_toks / opt_toks, 3)

    # Convenient alias
    optimize = optimize_prompt

