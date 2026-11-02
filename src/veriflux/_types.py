"""Core type definitions for Veriflux.

This module contains all foundational data structures, enumerations, and
type aliases used throughout the Veriflux library. Every dataclass uses
``slots=True`` for memory efficiency and ``frozen=True`` where immutability
is semantically appropriate.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    """Severity level for evaluation results and diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.CRITICAL: 3}
        return order[self] < order[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self == other or self.__lt__(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.CRITICAL: 3}
        return order[self] > order[other]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self == other or self.__gt__(other)


class TestLayer(enum.Enum):
    """Controls the depth / cost budget of a test run.

    * ``FAST``     – smoke tests, <1 s expected
    * ``STANDARD`` – default CI layer
    * ``DEEP``     – thorough, may hit external APIs
    * ``NIGHTLY``  – exhaustive, expensive
    """

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
    NIGHTLY = "nightly"


class ClaimType(enum.Enum):
    """Category of an atomic claim."""

    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    DEFINITIONAL = "definitional"
    OPINION = "opinion"
    OTHER = "other"


class Verdict(enum.Enum):
    """Result of verifying a single claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    PARTIALLY_SUPPORTED = "partially_supported"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class TokenUsage:
    """Tracks token consumption and estimated cost for a provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)

    def __add__(self, other: object) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    """Configuration for an external LLM / API provider."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Captures the outcome of a single evaluation metric.

    Attributes:
        passed: Whether the evaluation passed its threshold.
        score: Normalised score in [0, 1].
        metric_name: Identifier for the metric that produced this result.
        details: Arbitrary key-value details about the evaluation.
        severity: Severity level of a failure (or INFO on pass).
        timestamp: Unix timestamp when the result was created.
        duration_ms: Wall-clock time for the evaluation in milliseconds.
        metadata: Additional metadata (model name, run ID, etc.).
    """

    passed: bool
    score: float
    metric_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.INFO
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(slots=True)
class EvalSuite:
    """Collection of :class:`EvalResult` instances with summary statistics.

    Attributes:
        name: Human-readable suite name.
        results: Individual evaluation results.
        metadata: Suite-level metadata.
    """

    name: str
    results: List[EvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience properties ---------------------------------------------

    @property
    def total(self) -> int:
        """Total number of results."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Count of results that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Count of results that failed."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Fraction of results that passed (0.0 if empty)."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """Arithmetic mean of all scores (0.0 if empty)."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def min_score(self) -> float:
        """Minimum score across all results (0.0 if empty)."""
        if not self.results:
            return 0.0
        return min(r.score for r in self.results)

    @property
    def max_score(self) -> float:
        """Maximum score across all results (0.0 if empty)."""
        if not self.results:
            return 0.0
        return max(r.score for r in self.results)

    @property
    def total_duration_ms(self) -> float:
        """Sum of durations across all results."""
        return sum(r.duration_ms for r in self.results)

    @property
    def worst_severity(self) -> Severity:
        """Highest (worst) severity found among results."""
        if not self.results:
            return Severity.INFO
        return max(self.results, key=lambda r: r.severity).severity

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dictionary."""
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "worst_severity": self.worst_severity.value,
        }


@dataclass(slots=True)
class VerifluxConfig:
    """Global configuration container.

    Attributes:
        provider: Default provider configuration.
        default_thresholds: Mapping of metric-name → pass threshold.
        cache_enabled: Whether caching is active.
        cache_max_size: Maximum number of entries in the in-memory cache.
        cache_ttl: Time-to-live for cache entries in seconds.
        cache_directory: Directory for disk-based cache persistence.
        log_level: Logging level string (e.g. ``"INFO"``).
        log_format: ``"text"`` or ``"json"``.
        test_layer: Default test layer.
        max_workers: Thread pool size for parallel execution.
        token_budget: Maximum tokens per evaluation run (0 = unlimited).
        extra: Arbitrary additional settings.
    """

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    default_thresholds: Dict[str, float] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_max_size: int = 4096
    cache_ttl: float = 3600.0
    cache_directory: str = ".veriflux_cache"
    log_level: str = "INFO"
    log_format: str = "text"
    test_layer: TestLayer = TestLayer.STANDARD
    max_workers: int = 4
    token_budget: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Claim:
    """An atomic claim extracted from text.

    Attributes:
        text: The claim text.
        source_span: ``(start, end)`` character offsets into the source.
        claim_type: Category of the claim.
    """

    text: str
    source_span: Tuple[int, int] = (0, 0)
    claim_type: ClaimType = ClaimType.FACTUAL


@dataclass(slots=True, frozen=True)
class ClaimVerdict:
    """Result of verifying a single :class:`Claim`.

    Attributes:
        claim: The claim that was verified.
        verdict: Verification outcome.
        confidence: Confidence score in [0, 1].
        evidence: Supporting evidence or explanation.
    """

    claim: Claim
    verdict: Verdict = Verdict.UNVERIFIABLE
    confidence: float = 0.0
    evidence: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(slots=True)
class TraceStep:
    """A single step in an agent execution trace.

    Attributes:
        step_index: Zero-based ordinal within the trace.
        action: Description of the action taken (tool call, reasoning, etc.).
        input_data: Input to this step.
        output_data: Output produced by this step.
        timestamp: Unix timestamp of step execution.
        duration_ms: Duration in milliseconds.
        metadata: Additional key-value metadata.
        token_usage: Token consumption for this step (if applicable).
    """

    step_index: int
    action: str
    input_data: Any = None
    output_data: Any = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[TokenUsage] = None

    @property
    def agent(self) -> str:
        """Name of the executing agent if recorded."""
        return str(self.metadata.get("agent", ""))
