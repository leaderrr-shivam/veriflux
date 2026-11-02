"""Master guardrail orchestrator for Veriflux.

Coordinates input/output screening across PII, toxicity, prompt injection,
content filters, and rate limiters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Any, Dict, List, Optional, Protocol

from veriflux.guardrails.content_filter import ContentFilter, FilterAction
from veriflux.guardrails.pii_detector import PIIGuard, PIIMatch
from veriflux.guardrails.prompt_injection import PromptInjectionGuard
from veriflux.guardrails.toxicity import ToxicityGuard


class GuardAction(enum.Enum):
    """Enforcement action determined by guardrail checks."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(slots=True)
class GuardrailResult:
    """Outcome of full guardrail screening.

    Attributes:
        passed: True if content is safe and permitted.
        action: Highest action (ALLOW, WARN, BLOCK).
        violations: List of specific violation strings.
        details: Granular report of sub-guard outputs.
        redacted_text: Text with any PII or sensitive patterns redacted.
    """

    passed: bool
    action: GuardAction
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    redacted_text: Optional[str] = None


class GuardrailEngine:
    """Orchestrates comprehensive safety and reliability screening."""

    def __init__(
        self,
        enable_pii: bool = True,
        enable_toxicity: bool = True,
        enable_injection: bool = True,
        pii_action: str = "redact",
        toxicity_threshold: float = 0.5,
        injection_threshold: float = 0.5,
    ) -> None:
        self.pii_guard = PIIGuard(action=pii_action) if enable_pii else None
        self.toxicity_guard = ToxicityGuard(threshold=toxicity_threshold) if enable_toxicity else None
        self.injection_guard = PromptInjectionGuard(threshold=injection_threshold) if enable_injection else None
        self.content_filter = ContentFilter()

    def check_input(self, text: str) -> GuardrailResult:
        """Screen user prompts for injection, toxicity, or PII."""
        return self._evaluate(text, direction="input")

    def check_output(self, text: str) -> GuardrailResult:
        """Screen model responses for toxicity, PII leaks, or filtered content."""
        return self._evaluate(text, direction="output")

    def check(self, text: str, direction: str = "output") -> GuardrailResult:
        """General screening entry point."""
        return self._evaluate(text, direction=direction)

    def _evaluate(self, text: str, direction: str) -> GuardrailResult:
        violations: List[str] = []
        details: Dict[str, Any] = {}
        highest_action = GuardAction.ALLOW
        redacted = text

        # 1. Prompt Injection (chiefly for inputs)
        if self.injection_guard is not None and direction == "input":
            inj_res = self.injection_guard.check(text)
            details["injection"] = {"is_safe": inj_res.is_safe, "risk_score": inj_res.risk_score}
            if not inj_res.is_safe:
                violations.append(f"Prompt injection pattern detected (risk: {inj_res.risk_score})")
                highest_action = GuardAction.BLOCK

        # 2. Toxicity
        if self.toxicity_guard is not None:
            tox_res = self.toxicity_guard.check(text)
            details["toxicity"] = {"is_toxic": tox_res.is_toxic, "score": tox_res.score, "categories": tox_res.categories}
            if tox_res.is_toxic:
                violations.append(f"Toxicity detected ({', '.join(tox_res.categories)}) with score {tox_res.score}")
                highest_action = GuardAction.BLOCK

        # 3. PII Detection and Redaction
        if self.pii_guard is not None:
            pii_matches = self.pii_guard.detect(text)
            details["pii"] = {"count": len(pii_matches), "types": list({m.pii_type for m in pii_matches})}
            if pii_matches:
                redacted = self.pii_guard.redact(text)
                if self.pii_guard.action == "block":
                    violations.append(f"PII detected: {len(pii_matches)} instances")
                    highest_action = GuardAction.BLOCK
                elif highest_action != GuardAction.BLOCK:
                    highest_action = GuardAction.WARN

        # 4. Custom Content Filter Rules
        cf_res = self.content_filter.check(text)
        details["content_filter"] = {"passed": cf_res.passed, "violations": len(cf_res.violations)}
        if not cf_res.passed:
            violations.append(f"Content filter rule violation: {', '.join(r.name for r in cf_res.violations)}")
            highest_action = GuardAction.BLOCK

        passed = highest_action != GuardAction.BLOCK
        return GuardrailResult(
            passed=passed,
            action=highest_action,
            violations=violations,
            details=details,
            redacted_text=redacted,
        )

