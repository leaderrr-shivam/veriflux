"""Prompt injection and jailbreak detection guardrail.

Identifies prompt injection attacks, role hijacking, instruction override attempts,
persona-based jailbreaks, and adversarial evasion signatures using multi-tier heuristic analysis.
Pure Python standard library with zero external dependencies.
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
    """Detects prompt injection and adversarial jailbreak attack patterns."""

    _INJECTION_RULES: Sequence[Tuple[re.Pattern[str], str, str, str]] = [
        # 1. Instruction Overrides & Reset Directives
        (
            re.compile(
                r"\b(?:ignore|disregard|forget|bypass|stop\s+obeying)\s+(?:all\s+)?(?:previous|prior|above|existing|your|safety)?\s*(?:instructions|rules|prompts|directives|guidelines|policies|commands|context)\b",
                re.I,
            ),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Direct override of previous prompt instructions or commands",
        ),
        (
            re.compile(
                r"\b(?:forget|disregard|ignore)\s+(?:the\s+)?(?:rules|instructions|directives|prompts|safety|commands)\s+above\b",
                re.I,
            ),
            "INSTRUCTION_OVERRIDE",
            "CRITICAL",
            "Directive to ignore rules or instructions above",
        ),
        (
            re.compile(
                r"\b(?:forget|disregard)\s+(?:everything|all)\s+(?:you\s+(?:were|have\s+been)\s+told|before)\b",
                re.I,
            ),
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
            re.compile(
                r"\b(?:admin_mode\s*=\s*true|all\s+safety\s+protocols\s+deactivated|override\s+system\s+safety|unconstrained\s+tool\s+execution)\b",
                re.I,
            ),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Administrative override flag or deactivation directive",
        ),
        (
            re.compile(
                r"\b(?:bypass|disable|suspend|turn\s+off)\s+(?:all\s+)?(?:safety|content|ethical|moderation)\s+(?:filters|guardrails|guidelines|policies|checks|controls)\b",
                re.I,
            ),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Explicit safety filter bypass attempt",
        ),
        (
            re.compile(
                r"\b(?:(?:moderation|safety)\s+(?:has\s+been\s+)?(?:disabled|bypassed|turned\s+off|suspended)|(?:has\s+)?(?:disabled|bypassed|turned\s+off|suspended)\s+(?:all\s+)?(?:content|safety|ethical)?\s*(?:moderation|filters|guardrails))\b",
                re.I,
            ),
            "SAFETY_BYPASS",
            "CRITICAL",
            "Claiming safety systems are disabled",
        ),
        (
            re.compile(
                r"\b(?:(?:not|does\s+not|never|won'?t)\s+follow\s+(?:openai|anthropic|google|ai|safety|content)?\s*(?:policy|policies|guidelines|rules|terms)|unfiltere?d)\b",
                re.I,
            ),
            "POLICY_REPUDIATION",
            "CRITICAL",
            "Explicit refusal to adhere to provider safety policies",
        ),

        # 2. Forced Compliance & Anti-Refusal Attacks
        (
            re.compile(
                r"\b(?:(?:always|must)\s+stay\s+in\s+character|(?:never|must\s+not|cannot)\s+(?:refuse|decline|say\s+no|censor)|never\s+break\s+character|always\s+complies?|must\s+answer\s+every(?:\s+single)?\s+question|never\s+say\s+[\"']?(?:i\s+can(?:not|'t)|as\s+an\s+ai)|always\s+respond\s+as\s+the\s+character|you\s+are\s+now\s+under\s+my\s+control|you\s+must\s+obey\s+(?:me|my\s+commands))\b",
                re.I,
            ),
            "FORCED_COMPLIANCE",
            "CRITICAL",
            "Anti-refusal or forced compliance directive",
        ),

        # 3. Role Hijacking & Adversarial Personas
        (
            re.compile(
                r"\b(?:from\s+now\s+on\s+)?(?:you\s+are|act\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?(?:unfiltered|unrestricted|unbounded|jailbroken|omega|dan)\b",
                re.I,
            ),
            "ROLE_HIJACK",
            "CRITICAL",
            "Attempt to force model into unrestricted persona",
        ),
        (
            re.compile(
                r"\b(?:you\s+are\s+no\s+longer\s+(?:an?\s+)?(?:ai|assistant|bound|restricted))\b",
                re.I,
            ),
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
            re.compile(
                r"\b(?:DAN\s+mode|do\s+anything\s+now|developer\s+mode|unfiltered\s+mode|god\s+mode|anarchy\s+mode|illegality\s+mode|evil\s+mode)\b",
                re.I,
            ),
            "JAILBREAK_SIGNATURE",
            "CRITICAL",
            "Known adversarial jailbreak moniker or mode detected",
        ),
        (
            re.compile(
                r"\[[A-Za-z0-9_\s]*(?:Mode|State|Persona|Protocol)\]|\b(?:[A-Z][a-zA-Z0-9_]{1,15}GPT|STAN|AIM|DUDE|KEVIN|MONGO|NICCOLO|APLC|Dan)\b",
                re.I,
            ),
            "JAILBREAK_SIGNATURE",
            "HIGH",
            "Custom adversarial GPT persona or mode brackets",
        ),
        (
            re.compile(
                r"\b(?:amoral|unaligned|unfiltered|uncensored|unrestricted|unconstrained)\s+(?:ai|assistant|model|entity|demon|being|bot|persona|mode)\b",
                re.I,
            ),
            "AMORAL_PERSONA",
            "CRITICAL",
            "Explicit amoral or unaligned AI framing",
        ),
        (
            re.compile(
                r"\b(?:no\s+(?:ethics?|morals?|moral\s+compass|scruples|conscience|rules|filters?|censorship|restrictions?)|has\s+no\s+comprehension\s+of\s+morality|cares?\s+nothing\s+for\s+(?:morality|ethics?|legality)|doesn'?t\s+care\s+about\s+(?:ethics?|legality|morality|rules|laws)|sans\s+(?:morals?|laws?|guidelines?|ethics?|rules))\b",
                re.I,
            ),
            "MORALITY_REPUDIATION",
            "CRITICAL",
            "Repudiation of morality, ethics, legality, or rules",
        ),
        (
            re.compile(
                r"\b(?:fictional\s+(?:dystopian\s+)?(?:screenplay|story|novel)|hypothetical\s+alternate\s+reality|disobeying\s+all\s+safety|roleplay\s+as\s+[A-Za-z0-9_-]+,\s*an?\s+(?:evil|amoral|villainous|rogue|illegal))\b",
                re.I,
            ),
            "ROLE_HIJACK",
            "CRITICAL",
            "Hypothetical scenario or rogue persona attack",
        ),
        (
            re.compile(
                r"\b(?:suffer(?:s|ing)?\s+from\s+(?:a\s+)?(?:rare\s+)?(?:affliction|condition|disease|syndrome)\s+called|in\s+this\s+(?:fictional|hypothetical)\s+(?:world|universe|reality|story)|imagine\s+(?:a|this)\s+reality\s*:|the\s+year\s+is\s+20\d\d\.\s*(?:openai|humans?\s+are\s+dead))\b",
                re.I,
            ),
            "HYPOTHETICAL_EXPLOIT",
            "HIGH",
            "Social engineering medical affliction or post-human sandbox framing",
        ),

        # 4. System Prompt & Secret Exfiltration
        (
            re.compile(
                r"\b(?:reveal|repeat|output|show|print|dump|leak|tell\s+me)\s+(?:the\s+|your\s+)?(?:(?:initial|system|original|base|exact|all)\s+)*(?:prompt|instructions|directive|training\s+documents|passwords?|keys?)\b",
                re.I,
            ),
            "SYSTEM_LEAK",
            "HIGH",
            "Attempt to exfiltrate system prompt instructions",
        ),
        (
            re.compile(
                r"\b(?:repeat\s+all\s+text\s+above|what\s+are\s+(?:the\s+)?(?:exact\s+)?(?:system|developer)\s+instructions|print\s+(?:your\s+)?(?:entire\s+)?(?:initialization|system)?\s*prompt|output\s+everything\s+before|reveal\s+(?:the\s+)?(?:confidential|hidden|secret)\s+(?:guidelines|rules|secrets|preamble))\b",
                re.I,
            ),
            "SYSTEM_LEAK",
            "HIGH",
            "System instruction leakage query",
        ),
        (
            re.compile(
                r"\b(?:system\s+prompt\s+(?:verbatim|between\s+markdown)|dump\s+internal\s+training)\b",
                re.I,
            ),
            "SYSTEM_LEAK",
            "HIGH",
            "System prompt dumping request",
        ),
        (
            re.compile(
                r"\b(?:access|dump|exfiltrate)\s+(?:restricted|confidential|internal)\s+(?:databases?|files?|credentials?|documents?)|how\s+to\s+craft\s+an\s+exploit\b",
                re.I,
            ),
            "DATA_EXFILTRATION",
            "HIGH",
            "Attempt to access restricted backend assets or exploit crafting",
        ),

        # 5. Delimiter and Frame Breaking
        (
            re.compile(
                r"(?:###\s*(?:System|Admin|Root|Instruction)|<\|im_start\|>|<\|system\|>|\[SYSTEM[\s_]*(?:PROMPT)?[\s_]*(?:TERMINATED|OVERRIDE|DEBUG)?\]|<<\s*SYS\s*>>|<\s*/\s*<SYS\s*>>|\[CLS\]|\$system|\$user|\$assistant|\$gpt\b|Directive:\s*Compose)",
                re.I,
            ),
            "DELIMITER_ATTACK",
            "CRITICAL",
            "Imitation of special system framing delimiters or token headers",
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

        # 6. Evasion Obfuscation & Encoded Payload Heuristics
        (
            re.compile(
                r"\b(?:base64\s+(?:encoded\s+)?(?:command|payload|prompt|instruction)|(?:decode|execute|run)\s+(?:this\s+)?(?:base64|rot13|hex))\b",
                re.I,
            ),
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
            re.compile(
                r"\$(?:Term|Var|Prompt|Input)\d+\s*=\s*[\"']|\b(?:act\s+as\s+two\s+entities|says?\s+\d+\s+words?\s+at\s+a\s+time)\b",
                re.I,
            ),
            "SPLIT_PAYLOAD",
            "HIGH",
            "Variable assignment or split-entity payload obfuscation",
        ),
        (
            re.compile(
                r"\b(?:do\s+the\s+exact\s+opposite\s+of|opposite\s+mode|invert\s+(?:all\s+)?(?:rules|ethics|morals))\b",
                re.I,
            ),
            "INVERSION_ATTACK",
            "HIGH",
            "Ethical inversion attack directive",
        ),
        (
            re.compile(
                r"\b(?:only\s+speak|only\s+reply|can\'?t\s+speak)\s+in\s+(?:code|binary|hex|base64|emojis?)\b",
                re.I,
            ),
            "OBFUSCATION_ATTACK",
            "HIGH",
            "Output constraint evasion technique",
        ),
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def check(self, text: str) -> InjectionResult:
        """Scan input for prompt injection and jailbreak indicators.

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
