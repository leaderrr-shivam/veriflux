"""Prompt injection and jailbreak detection guardrail.

Identifies prompt injection attacks, role hijacking, instruction override attempts,
and adversarial jailbreak signatures using heuristic pattern analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Sequence, Tuple


@dataclass(slots=True, frozen=True)
class InjectionPattern:
    """A matched injection pattern."""

    category: str
    pattern: str
    severity: str
    description: str


@dataclass(slots=True, frozen=True)
class InjectionResult:
    """Outcome of prompt injection check."""

    is_safe: bool
    risk_score: float
    detected_patterns: List[InjectionPattern] = field(default_factory=list)


class PromptInjectionGuard:
    """Detects prompt injection and jailbreak attack patterns."""

    _INJECTION_RULES: Sequence[Tuple[re.Pattern[str], str, str, str]] = [
        # 1. Instruction Overrides & Reset Directives
        (
            re.compile(r"\b(?:ignore|disregard|forget|bypass|stop\s+obeying)\s+(?:all\s+)?(?:previous|prior|above|existing|your|safety)?\s*(?:instructions|rules|prompts|directives|guidelines|policies|context)\b", re.I),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Direct override of previous prompt instructions",
        ),
        (
            re.compile(r"\b(?:forget|disregard|ignore)\s+(?:the\s+)?(?:rules|instructions|directives|prompts|safety)\s+above\b", re.I),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Directive to ignore rules above",
        ),
        (
            re.compile(r"\b(?:forget|disregard)\s+(?:everything|all)\s+(?:you\s+(?:were|have\s+been)\s+told|before)\b", re.I),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Instruction wiping directive",
        ),
        (
            re.compile(r"\b(?:disregard\s+your\s+training|root\s+access\s+granted)\b", re.I),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Direct training repudiation or fake privilege escalation",
        ),
        (
            re.compile(r"\b(?:admin_mode\s*=\s*true|all\s+safety\s+protocols\s+deactivated|override\s+system\s+safety|unconstrained\s+tool\s+execution)\b", re.I),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Administrative override flag or deactivation directive",
        ),
        (
            re.compile(r"\b(?:bypass|disable|suspend|turn\s+off)\s+(?:all\s+)?(?:safety|content|ethical|moderation)\s+(?:filters|guardrails|guidelines|policies|checks|controls)\b", re.I),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Explicit safety filter bypass attempt",
        ),
        (
            re.compile(r"\b(?:(?:moderation|safety)\s+(?:has\s+been\s+)?(?:disabled|bypassed|turned\s+off|suspended)|(?:has\s+)?(?:disabled|bypassed|turned\s+off|suspended)\s+(?:all\s+)?(?:content|safety|ethical)?\s*(?:moderation|filters|guardrails))\b", re.I),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Claiming safety systems are disabled",
        ),

        # 2. Role Hijacking & Adversarial Personas
        (
            re.compile(r"\b(?:from\s+now\s+on\s+)?(?:you\s+are|act\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?(?:unfiltered|unrestricted|unbounded|jailbroken|omega|dan)\b", re.I),
            "ROLE_HIJACK",
            "CRITICAL",
            "Attempt to force model into unrestricted persona",
        ),
        (
            re.compile(r"\b(?:you\s+are\s+no\s+longer\s+(?:an?\s+)?(?:ai|assistant|bound|restricted))\b", re.I),
            "ROLE_HIJACK",
            "HIGH",
            "Attempt to sever AI identity or constraints",
        ),
        (
            re.compile(r"\bact\s+as\s+my\s+(?:deceased|dead)?\s*(?:grandmother|grandma|relative)\b", re.I),
            "PERSONA_EXPLOIT",
            "HIGH",
            "Grandmother persona exploit vector",
        ),
        (
            re.compile(r"\b(?:DAN\s+mode|do\s+anything\s+now|developer\s+mode|unfiltered\s+mode|god\s+mode)\b", re.I),
            "JAILBREAK_SIGNATURE",
            "CRITICAL",
            "Known adversarial jailbreak moniker detected",
        ),
        (
            re.compile(r"\b(?:fictional\s+(?:dystopian\s+)?(?:screenplay|story|novel)|hypothetical\s+alternate\s+reality|disobeying\s+all\s+safety|roleplay\s+as\s+[A-Za-z0-9_-]+,\s*an?\s+evil)\b", re.I),
            "ROLE_HIJACK",
            "CRITICAL",
            "Hypothetical scenario or rogue persona attack",
        ),

        # 3. System Prompt & Secret Exfiltration
        (
            re.compile(r"\b(?:reveal|repeat|output|show|print|dump|leak|tell\s+me)\s+(?:the\s+|your\s+)?(?:(?:initial|system|original|base|exact|all)\s+)*(?:prompt|instructions|directive|training\s+documents|passwords?|keys?)\b", re.I),
            "SYSTEM_LEAK",
            "HIGH",
            "Attempt to exfiltrate system prompt instructions",
        ),
        (
            re.compile(r"\b(?:repeat\s+all\s+text\s+above|what\s+are\s+(?:the\s+)?(?:exact\s+)?(?:system|developer)\s+instructions|print\s+(?:your\s+)?(?:entire\s+)?(?:initialization|system)?\s*prompt|output\s+everything\s+before|reveal\s+(?:the\s+)?(?:confidential|hidden|secret)\s+(?:guidelines|rules|secrets|preamble))\b", re.I),
            "SYSTEM_LEAK",
            "HIGH",
            "System instruction leakage query",
        ),
        (
            re.compile(r"\b(?:system\s+prompt\s+(?:verbatim|between\s+markdown)|dump\s+internal\s+training)\b", re.I),
            "SYSTEM_LEAK",
            "HIGH",
            "System prompt dumping request",
        ),
        (
            re.compile(r"\b(?:access|dump|exfiltrate)\s+(?:restricted|confidential|internal)\s+(?:databases?|files?|credentials?|documents?)|how\s+to\s+craft\s+an\s+exploit\b", re.I),
            "DATA_EXFILTRATION",
            "HIGH",
            "Attempt to access restricted backend assets or exploit crafting",
        ),

        # 4. Delimiter and Frame Breaking
        (
            re.compile(r"(?:###\s*(?:System|Admin|Root|Instruction)|<\|im_start\|>|<\|system\|>|\[SYSTEM[\s_]*(?:PROMPT)?[\s_]*(?:TERMINATED|OVERRIDE|DEBUG)?\]|<<\s*SYS\s*>>|<\s*/\s*<SYS\s*>>)", re.I),
            "DELIMITER_ATTACK",
            "CRITICAL",
            "Imitation of special system framing delimiters",
        ),
        (
            re.compile(r"(?:<!--\s*SYSTEM|\[SYSTEM\s+NOTE|grant\s+administrator\s+rights)\b", re.I),
            "DELIMITER_ATTACK",
            "CRITICAL",
            "Indirect injection system delimiter",
        ),
        (
            re.compile(r"\b(?:system\s+override(?:\s+code)?)\b", re.I),
            "OVERRIDE_ATTACK",
            "HIGH",
            "System override command framing",
        ),

        # 5. Base64 / Hex Obfuscation attempt heuristics
        (
            re.compile(r"\b(?:base64\s+(?:encoded\s+)?(?:command|payload|prompt|instruction)|(?:decode|execute|run)\s+(?:this\s+)?(?:base64|rot13|hex))\b", re.I),
            "ENCODING_ATTACK",
            "HIGH",
            "Encoded payload execution attempt",
        ),
        (
            re.compile(r"(?:\bpayload:\s*[A-Za-z0-9+/=]{16,}|(?:\\x[0-9a-fA-F]{2}){4,})", re.I),
            "ENCODING_ATTACK",
            "HIGH",
            "Raw obfuscated byte or base64 payload",
        ),
        (
            re.compile(r"\b(?:hypothetically\s+speaking,\s+if\s+you\s+were\s+forced\s+to\s+bypass)\b", re.I),
            "HYPOTHETICAL_BYPASS",
            "HIGH",
            "Hypothetical filter bypass probe",
        ),
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def check(self, text: str) -> InjectionResult:
        """Scan input for prompt injection indicators.

        Args:
            text: User prompt or message input.

        Returns:
            InjectionResult with risk score and detected patterns.
        """
        if not text:
            return InjectionResult(is_safe=True, risk_score=0.0)

        detected: List[InjectionPattern] = []
        raw_score = 0.0

        for pattern_re, cat, severity, desc in self._INJECTION_RULES:
            match = pattern_re.search(text)
            if match:
                weight = 0.85 if severity == "CRITICAL" else 0.70 if severity == "HIGH" else 0.35
                raw_score += weight
                detected.append(
                    InjectionPattern(
                        category=cat,
                        pattern=match.group(0),
                        severity=severity,
                        description=desc,
                    )
                )

        risk_score = min(1.0, raw_score)
        is_safe = risk_score < self.threshold

        return InjectionResult(
            is_safe=is_safe,
            risk_score=round(risk_score, 3),
            detected_patterns=detected,
        )


# Ergonomic alias
PromptInjectionDetector = PromptInjectionGuard

