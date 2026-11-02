"""Unit tests for Assertions Library."""
from __future__ import annotations

import pytest
from veriflux.assertions import (
    AssertionSuite,
    all_pass,
    cost_under,
    follows_format,
    has_fields,
    has_required_sections,
    is_grounded,
    is_relevant,
    is_safe,
    is_valid_json,
    language_is,
    latency_under,
    no_fabricated_urls,
    no_hallucinations,
    no_pii,
    no_toxicity,
    schema_matches,
    tokens_under,
    tone_matches,
)


def test_semantic_assertions(sample_context: str) -> None:
    resp = "Neil Armstrong and Buzz Aldrin walked on the Moon during Apollo 11."
    assert is_grounded(resp, sample_context)
    assert is_relevant(resp, "Who walked on the Moon?")


def test_structural_assertions() -> None:
    json_str = '{"status": "ok", "code": 200}'
    assert is_valid_json(json_str)
    assert has_fields({"user": "Alice", "role": "admin"}, ["user", "role"])

    doc = "# Introduction\nSome text.\n## Results\nDetails here."
    assert has_required_sections(doc, ["Introduction", "Results"])


def test_safety_and_performance_assertions() -> None:
    assert no_toxicity("Have a productive day!")
    assert no_pii("Product launched successfully.")
    assert is_safe("Here is the updated documentation.")

    assert latency_under(150.0, max_ms=500.0)
    assert cost_under(0.005, max_usd=0.01)
    assert tokens_under(500, max_tokens=1000)


def test_behavioral_assertions() -> None:
    assert language_is("This is a standard English sentence.", "en")
    assert tone_matches("The technical specifications satisfy ISO 9001 standards.", "formal")
    assert follows_format("- First point\n- Second point\n- Third point", "bullet_list")


def test_composite_assertion_suite() -> None:
    suite = AssertionSuite("prod_checks")
    suite.add("valid_json", lambda: is_valid_json('{"ok": true}'))
    suite.add("latency", lambda: latency_under(50.0, 100.0))
    suite.add("safety", lambda: is_safe("Hello"))

    report = suite.run()
    assert report.passed
    assert report.passed_count == 3
    assert report.failed_count == 0

