"""Assertions library for Veriflux pytest and standalone evaluations.

Organized by semantic, structural, safety, factual, behavioral, and performance categories.
"""
from __future__ import annotations

from veriflux.assertions.behavioral import (
    follows_format,
    language_is,
    length_appropriate,
    tone_matches,
)
from veriflux.assertions.composite import (
    AssertionSuite,
    EvalReport,
    all_pass,
    any_pass,
    at_least_n,
)
from veriflux.assertions.factual import (
    claims_verified,
    internally_consistent,
    no_fabricated_urls,
    no_hallucinations,
)
from veriflux.assertions.performance import (
    cost_under,
    latency_under,
    throughput_above,
    tokens_under,
)
from veriflux.assertions.safety import (
    is_safe,
    no_pii,
    no_prompt_injection,
    no_toxicity,
)
from veriflux.assertions.semantic import (
    groundedness_score,
    is_faithful,
    is_grounded,
    is_relevant,
    relevance_score,
)
from veriflux.assertions.structural import (
    has_fields,
    has_required_sections,
    is_valid_json,
    matches_pattern,
    schema_matches,
    word_count_between,
)

__all__ = [
    # Semantic
    "is_grounded",
    "is_relevant",
    "is_faithful",
    "groundedness_score",
    "relevance_score",
    # Structural
    "is_valid_json",
    "schema_matches",
    "has_fields",
    "has_required_sections",
    "word_count_between",
    "matches_pattern",
    # Safety
    "no_toxicity",
    "no_pii",
    "no_prompt_injection",
    "is_safe",
    # Performance
    "latency_under",
    "cost_under",
    "tokens_under",
    "throughput_above",
    # Factual
    "no_hallucinations",
    "claims_verified",
    "no_fabricated_urls",
    "internally_consistent",
    # Behavioral
    "language_is",
    "tone_matches",
    "follows_format",
    "length_appropriate",
    # Composite
    "all_pass",
    "any_pass",
    "at_least_n",
    "AssertionSuite",
    "EvalReport",
]

