"""Structural and syntax validation assertions.

Verifies JSON syntax, schema compliance, field presence, section boundaries, and length constraints.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from veriflux.schema.validator import SchemaValidator, validate_json


def is_valid_json(text: str, schema: Optional[Dict[str, Any]] = None) -> bool:
    """Assert that text is valid JSON and optionally satisfies schema."""
    res = validate_json(text, schema=schema)
    return res.is_valid


def schema_matches(data: Any, schema: Dict[str, Any]) -> bool:
    """Assert that a Python dictionary or object satisfies a JSON-schema specification."""
    validator = SchemaValidator()
    res = validator.validate(data, schema)
    return res.is_valid


def has_fields(data: Dict[str, Any], fields: Sequence[str]) -> bool:
    """Assert that a dictionary contains all designated field keys."""
    if not isinstance(data, dict):
        return False
    return all(f in data for f in fields)


def has_required_sections(text: str, sections: Sequence[str]) -> bool:
    """Assert that a formatted document (e.g. Markdown) contains required section headers."""
    for sec in sections:
        pattern = re.compile(r"^#+\s*" + re.escape(sec) + r"\b", re.IGNORECASE | re.MULTILINE)
        if not pattern.search(text):
            return False
    return True


def word_count_between(text: str, min_words: int, max_words: int) -> bool:
    """Assert that text word count is between min_words and max_words inclusive."""
    words = text.split()
    return min_words <= len(words) <= max_words


def matches_pattern(text: str, pattern: str) -> bool:
    """Assert that text satisfies a regular expression pattern."""
    return bool(re.search(pattern, text))

