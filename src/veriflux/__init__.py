"""
Veriflux — The definitive AI/ML testing, evaluation, guardrail, and reliability library.

Built entirely from scratch. Zero dependency on existing AI/ML frameworks.
Every algorithm, every data structure, every scoring function — engineered from first principles.

    "Truth flows through every layer."

Usage:
    # One-liner assertions
    >>> import veriflux
    >>> veriflux.is_grounded("Paris is France's capital", context="The capital of France is Paris")
    True

    # Pytest integration
    >>> @veriflux.test(dataset="golden.jsonl")
    ... def test_qa(case):
    ...     assert veriflux.is_grounded(case.response, case.context)

    # Full pipeline
    >>> from veriflux import Pipeline, Stage, Gate
    >>> pipeline = Pipeline("production_eval")
"""

from __future__ import annotations

# ─── Version ────────────────────────────────────────────────────────────────
from veriflux.__version__ import __version__

# ─── Core Types ─────────────────────────────────────────────────────────────
from veriflux._types import (
    Claim,
    ClaimVerdict,
    EvalResult,
    EvalSuite,
    ProviderConfig,
    Severity,
    TestLayer,
    TokenUsage,
    TraceStep,
    Verdict,
    VerifluxConfig,
)

# ─── Exception Hierarchy 14────────────────────────────────────────────────────
from veriflux.exceptions import (
    ConfigurationError,
    HallucinationDetected,
    ProviderError,
    TokenBudgetExceeded,
    ValidationError,
    VerifluxError,
)

# ─── Pipeline Orchestration ─────────────────────────────────────────────────
from veriflux.cicd.gates import Gate
from veriflux.pipeline import Pipeline, PipelineResult, Stage

# ─── Top-Level Convenience API ──────────────────────────────────────────────
from veriflux._api import (
    configure,
    get_config,
    is_faithful,
    is_grounded,
    is_relevant,
    is_valid_json,
    no_hallucinations,
    no_pii,
    no_toxicity,
    test,
    token_count,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "EvalResult",
    "EvalSuite",
    "Severity",
    "TestLayer",
    "VerifluxConfig",
    "Claim",
    "ClaimVerdict",
    "Verdict",
    "TokenUsage",
    "TraceStep",
    "ProviderConfig",
    # Exceptions
    "VerifluxError",
    "ConfigurationError",
    "ProviderError",
    "ValidationError",
    "TokenBudgetExceeded",
    "HallucinationDetected",
    # Pipeline
    "Pipeline",
    "Stage",
    "Gate",
    "PipelineResult",
    # Top-level API
    "is_grounded",
    "is_relevant",
    "is_faithful",
    "is_valid_json",
    "no_pii",
    "no_toxicity",
    "no_hallucinations",
    "token_count",
    "test",
    "configure",
    "get_config",
]
