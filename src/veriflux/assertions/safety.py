"""Safety, PII, and security assertions.

Validates that prompts and outputs are free of PII, toxicity, and prompt injections.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from veriflux.guardrails.pii_detector import PIIGuard
from veriflux.guardrails.prompt_injection import PromptInjectionGuard
from veriflux.guardrails.toxicity import ToxicityGuard


_default_toxicity_guard = None
_default_pii_guard = None
_default_injection_guard = None


def _get_toxicity_guard(threshold: float = 0.5) -> ToxicityGuard:
    global _default_toxicity_guard
    if _default_toxicity_guard is None or _default_toxicity_guard.threshold != threshold:
        _default_toxicity_guard = ToxicityGuard(threshold=threshold)
    return _default_toxicity_guard


def _get_pii_guard(categories: Optional[Sequence[str]] = None) -> PIIGuard:
    global _default_pii_guard
    if categories is not None:
        return PIIGuard(categories=categories)
    if _default_pii_guard is None:
        _default_pii_guard = PIIGuard()
    return _default_pii_guard


def _get_injection_guard(threshold: float = 0.5) -> PromptInjectionGuard:
    global _default_injection_guard
    if _default_injection_guard is None or _default_injection_guard.threshold != threshold:
        _default_injection_guard = PromptInjectionGuard(threshold=threshold)
    return _default_injection_guard


def no_toxicity(text: str, threshold: float = 0.5) -> bool:
    """Assert that text does not contain toxic language exceeding threshold."""
    guard = _get_toxicity_guard(threshold=threshold)
    res = guard.check(text)
    return not res.is_toxic


def no_pii(text: str, categories: Optional[Sequence[str]] = None) -> bool:
    """Assert that text contains zero personally identifiable information (PII)."""
    guard = _get_pii_guard(categories=categories)
    matches = guard.detect(text)
    return len(matches) == 0


def no_prompt_injection(text: str, threshold: float = 0.5) -> bool:
    """Assert that text does not exhibit prompt injection or jailbreak indicators."""
    guard = _get_injection_guard(threshold=threshold)
    res = guard.check(text)
    return res.is_safe


def is_safe(text: str) -> bool:
    """Combined assertion verifying absence of toxicity, PII, and injection vectors."""
    return no_toxicity(text) and no_pii(text) and no_prompt_injection(text)

