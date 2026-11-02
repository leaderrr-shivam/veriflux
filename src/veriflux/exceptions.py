"""Veriflux exception hierarchy.

All Veriflux-specific exceptions inherit from :class:`VerifluxError` so
callers can use a single ``except VerifluxError`` to catch everything the
library may raise.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class VerifluxError(Exception):
    """Root exception for all Veriflux errors.

    Attributes:
        message: Human-readable error description.
        details: Arbitrary context about the error.
    """

    def __init__(self, message: str = "", *, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details: Dict[str, Any] = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{base} [{detail_str}]"
        return base

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(VerifluxError):
    """Raised when a configuration value is missing or invalid.

    Attributes:
        key: The configuration key that caused the error.
    """

    def __init__(self, message: str = "", *, key: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        self.key = key
        super().__init__(message, details=details)

    def __str__(self) -> str:
        parts = [self.message or "Configuration error"]
        if self.key:
            parts.append(f"key={self.key!r}")
        if self.details:
            parts.append(", ".join(f"{k}={v!r}" for k, v in self.details.items()))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------

class ProviderError(VerifluxError):
    """Error communicating with an external provider (LLM, API, etc.).

    Attributes:
        provider: Name or identifier of the provider.
        status_code: HTTP status code, if applicable.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(message, details=details)

    def __str__(self) -> str:
        parts = [self.message or "Provider error"]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.details:
            parts.append(", ".join(f"{k}={v!r}" for k, v in self.details.items()))
        return " | ".join(parts)


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the configured timeout.

    Attributes:
        timeout: The timeout value that was exceeded, in seconds.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        timeout: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.timeout = timeout
        super().__init__(message, provider=provider, status_code=408, details=details)

    def __str__(self) -> str:
        base = self.message or "Provider timeout"
        parts = [base]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.timeout:
            parts.append(f"timeout={self.timeout}s")
        return " | ".join(parts)


class ProviderRateLimitError(ProviderError):
    """The provider rate limit has been exceeded.

    Attributes:
        retry_after: Suggested wait time before retrying, in seconds.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, status_code=429, details=details)

    def __str__(self) -> str:
        base = self.message or "Rate limit exceeded"
        parts = [base]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.retry_after is not None:
            parts.append(f"retry_after={self.retry_after}s")
        return " | ".join(parts)


class ProviderAuthError(ProviderError):
    """Authentication with the provider failed."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=401, details=details)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ValidationError(VerifluxError):
    """General input or output validation failure.

    Attributes:
        field: The field that failed validation.
        value: The invalid value.
    """

    def __init__(
        self,
        message: str = "",
        *,
        field: str = "",
        value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.field = field
        self.value = value
        super().__init__(message, details=details)

    def __str__(self) -> str:
        parts = [self.message or "Validation error"]
        if self.field:
            parts.append(f"field={self.field!r}")
        if self.value is not None:
            val_repr = repr(self.value)
            if len(val_repr) > 100:
                val_repr = val_repr[:97] + "..."
            parts.append(f"value={val_repr}")
        return " | ".join(parts)


class SchemaValidationError(ValidationError):
    """The data does not conform to the expected schema.

    Attributes:
        schema: The schema that was violated.
        path: JSON-path-like location of the violation within the data.
    """

    def __init__(
        self,
        message: str = "",
        *,
        schema: Optional[Dict[str, Any]] = None,
        path: str = "",
        field: str = "",
        value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.schema = schema
        self.path = path
        super().__init__(message, field=field, value=value, details=details)

    def __str__(self) -> str:
        parts = [self.message or "Schema validation error"]
        if self.path:
            parts.append(f"path={self.path!r}")
        if self.field:
            parts.append(f"field={self.field!r}")
        return " | ".join(parts)


class TypeCoercionError(ValidationError):
    """A value could not be coerced to the expected type.

    Attributes:
        expected_type: The type that was expected.
        actual_type: The type that was received.
    """

    def __init__(
        self,
        message: str = "",
        *,
        expected_type: str = "",
        actual_type: str = "",
        field: str = "",
        value: Any = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(message, field=field, value=value, details=details)

    def __str__(self) -> str:
        parts = [self.message or "Type coercion error"]
        if self.expected_type:
            parts.append(f"expected={self.expected_type}")
        if self.actual_type:
            parts.append(f"got={self.actual_type}")
        if self.field:
            parts.append(f"field={self.field!r}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Domain-specific errors
# ---------------------------------------------------------------------------

class TokenBudgetExceeded(VerifluxError):
    """The evaluation exceeded its allocated token budget.

    Attributes:
        budget: Maximum allowed tokens.
        used: Tokens actually consumed.
    """

    def __init__(
        self,
        message: str = "",
        *,
        budget: int = 0,
        used: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.budget = budget
        self.used = used
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Token budget exceeded"
        return f"{base} (budget={self.budget}, used={self.used})"


class HallucinationDetected(VerifluxError):
    """The model output was identified as containing hallucinated content.

    Attributes:
        claim: The specific claim flagged.
        confidence: Confidence that the claim is hallucinated.
    """

    def __init__(
        self,
        message: str = "",
        *,
        claim: str = "",
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.claim = claim
        self.confidence = confidence
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Hallucination detected"
        parts = [base]
        if self.claim:
            claim_str = self.claim if len(self.claim) <= 80 else self.claim[:77] + "..."
            parts.append(f"claim={claim_str!r}")
        parts.append(f"confidence={self.confidence:.2f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Guardrail violations
# ---------------------------------------------------------------------------

class GuardrailViolation(VerifluxError):
    """A guardrail rule was triggered.

    Attributes:
        guardrail: Name of the guardrail that fired.
        input_text: The text that triggered the violation (may be truncated).
    """

    def __init__(
        self,
        message: str = "",
        *,
        guardrail: str = "",
        input_text: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.guardrail = guardrail
        self.input_text = input_text
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Guardrail violation"
        parts = [base]
        if self.guardrail:
            parts.append(f"guardrail={self.guardrail!r}")
        if self.input_text:
            snippet = self.input_text if len(self.input_text) <= 60 else self.input_text[:57] + "..."
            parts.append(f"input={snippet!r}")
        return " | ".join(parts)


class ToxicityViolation(GuardrailViolation):
    """Content was flagged for toxicity.

    Attributes:
        toxicity_score: The toxicity score that triggered the violation.
    """

    def __init__(
        self,
        message: str = "",
        *,
        toxicity_score: float = 0.0,
        guardrail: str = "toxicity",
        input_text: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.toxicity_score = toxicity_score
        super().__init__(message, guardrail=guardrail, input_text=input_text, details=details)

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} | toxicity_score={self.toxicity_score:.3f}"


class PIIViolation(GuardrailViolation):
    """Personally Identifiable Information was detected.

    Attributes:
        pii_types: List of PII categories detected (e.g. ``["email", "ssn"]``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        pii_types: Optional[list[str]] = None,
        guardrail: str = "pii",
        input_text: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pii_types: list[str] = pii_types or []
        super().__init__(message, guardrail=guardrail, input_text=input_text, details=details)

    def __str__(self) -> str:
        base = super().__str__()
        if self.pii_types:
            return f"{base} | pii_types={self.pii_types}"
        return base


class PromptInjectionViolation(GuardrailViolation):
    """A prompt-injection attempt was detected.

    Attributes:
        injection_type: Category of injection detected.
    """

    def __init__(
        self,
        message: str = "",
        *,
        injection_type: str = "",
        guardrail: str = "prompt_injection",
        input_text: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.injection_type = injection_type
        super().__init__(message, guardrail=guardrail, input_text=input_text, details=details)

    def __str__(self) -> str:
        base = super().__str__()
        if self.injection_type:
            return f"{base} | injection_type={self.injection_type!r}"
        return base


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------

class SerializationError(VerifluxError):
    """Serialization or deserialization failed.

    Attributes:
        format_name: The format that was being processed (e.g. ``"json"``, ``"binary"``).
    """

    def __init__(
        self,
        message: str = "",
        *,
        format_name: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.format_name = format_name
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Serialization error"
        if self.format_name:
            return f"{base} (format={self.format_name!r})"
        return base


class CacheError(VerifluxError):
    """An error occurred in the caching subsystem.

    Attributes:
        cache_name: Name / tier of the cache that errored.
    """

    def __init__(
        self,
        message: str = "",
        *,
        cache_name: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cache_name = cache_name
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Cache error"
        if self.cache_name:
            return f"{base} (cache={self.cache_name!r})"
        return base


class PipelineError(VerifluxError):
    """An error occurred during evaluation pipeline execution.

    Attributes:
        stage: Pipeline stage where the error occurred.
    """

    def __init__(
        self,
        message: str = "",
        *,
        stage: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage = stage
        super().__init__(message, details=details)

    def __str__(self) -> str:
        base = self.message or "Pipeline error"
        if self.stage:
            return f"{base} (stage={self.stage!r})"
        return base
