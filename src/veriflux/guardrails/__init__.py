"""Guardrails and safety engine for Veriflux.

Provides real-time screening for PII, prompt injections, toxicity, rate limits, and policies.
"""
from __future__ import annotations

from veriflux.guardrails.content_filter import ContentFilter, FilterAction, FilterRule, bias_score
from veriflux.guardrails.engine import GuardAction, GuardrailEngine, GuardrailResult
from veriflux.guardrails.pii_detector import PIIGuard, PIIMatch
from veriflux.guardrails.policy import Policy, PolicyResult
from veriflux.guardrails.prompt_injection import (
    InjectionPattern,
    InjectionResult,
    PromptInjectionGuard,
)
from veriflux.guardrails.rate_limiter import RateLimiter
from veriflux.guardrails.toxicity import ToxicityGuard, ToxicityResult

__all__ = [
    # Engine
    "GuardrailEngine",
    "GuardrailResult",
    "GuardAction",
    # PII
    "PIIGuard",
    "PIIMatch",
    # Prompt injection
    "PromptInjectionGuard",
    "InjectionResult",
    "InjectionPattern",
    # Toxicity
    "ToxicityGuard",
    "ToxicityResult",
    # Content Filter
    "ContentFilter",
    "FilterRule",
    "FilterAction",
    "bias_score",
    # Policy
    "Policy",
    "PolicyResult",
    # Rate limiter
    "RateLimiter",
]

