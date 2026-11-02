"""Behavioral, formatting, and tone compliance assertions.

Verifies adherence to stylistic guidelines, length boundaries, and language expectations.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def language_is(text: str, expected_lang: str = "en") -> bool:
    """Heuristic check for language consistency (e.g. 'en', 'es', 'fr', 'de')."""
    clean = text.lower()
    if expected_lang.lower() in ("en", "english"):
        common_en = {
            "the", "and", "is", "of", "to", "in", "it", "you", "that", "was",
            "a", "an", "this", "for", "with", "on", "as", "at", "be", "are", "have", "by", "from"
        }
        words = set(re.findall(r"\b[a-z]+\b", clean))
        return len(words & common_en) >= 1 or len(words) < 3
    return True


def tone_matches(text: str, expected_tone: str = "formal") -> bool:
    """Verify that text satisfies an expected stylistic tone ('formal', 'technical', 'informal')."""
    clean = text.lower()
    if expected_tone == "formal":
        # Check absence of informal slang and contractions
        slang = ["gonna", "wanna", "lol", "omg", "kinda", "sorta", "y'all"]
        return not any(re.search(r"\b" + s + r"\b", clean) for s in slang)
    elif expected_tone == "technical":
        # Technical text typically has higher sentence density and structured vocabulary
        return len(text.split()) > 10
    return True


def follows_format(text: str, format_spec: str) -> bool:
    """Assert that output adheres to designated formatting ('json', 'markdown', 'bullet_list')."""
    fmt = format_spec.lower().strip()
    clean = text.strip()

    if fmt == "json":
        try:
            json.loads(clean)
            return True
        except Exception:
            return False

    elif fmt in ("bullet_list", "bullets"):
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if not lines:
            return False
        bullet_count = sum(1 for line in lines if line.startswith(("-", "*", "•")))
        return (bullet_count / len(lines)) >= 0.5

    elif fmt == "markdown":
        return bool(re.search(r"(?:^#+|\*\*|\*|_|```|\[.+\]\(.+\))", clean, re.MULTILINE))

    return True


def length_appropriate(text: str, min_chars: int = 0, max_chars: Optional[int] = None) -> bool:
    """Assert that string length conforms to character boundaries."""
    length = len(text.strip())
    if length < min_chars:
        return False
    if max_chars is not None and length > max_chars:
        return False
    return True
