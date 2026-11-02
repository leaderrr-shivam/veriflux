"""Quality gate evaluation for deployment guardrails.

Enforces metric thresholds (pass rate, hallucination limits, latency caps) before release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(slots=True, frozen=True)
class GateResult:
    """Outcome of quality gate evaluation.

    Attributes:
        passed: True if all gate conditions were met.
        conditions_met: List of satisfied condition descriptions.
        conditions_failed: List of violated condition descriptions.
    """

    passed: bool
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)


class QualityGate:
    """Evaluates numeric metric dictionaries against specified threshold conditions."""

    _OPS: Dict[str, Callable[[float, float], bool]] = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, name: str = "quality_gate", conditions: Optional[Dict[str, Tuple[str, float]]] = None) -> None:
        self.name = name
        self.conditions: List[Tuple[str, str, float]] = []
        if conditions:
            for metric, (op, thresh) in conditions.items():
                self.add_condition(metric, op, thresh)

    def add_condition(self, metric: str, op_str: str, threshold: float) -> QualityGate:
        """Add a condition requirement (e.g. ('pass_rate', '>=', 0.95))."""
        if op_str not in self._OPS:
            raise ValueError(f"Unsupported operator '{op_str}'. Allowed: {list(self._OPS.keys())}")
        self.conditions.append((metric, op_str, float(threshold)))
        return self

    def evaluate(self, metrics: Dict[str, float]) -> GateResult:
        """Evaluate provided metrics against all configured gate conditions."""
        met: List[str] = []
        failed: List[str] = []

        for metric, op_str, threshold in self.conditions:
            val = metrics.get(metric)
            if val is None:
                failed.append(f"Metric '{metric}' missing from evaluation results")
                continue

            op_fn = self._OPS[op_str]
            if op_fn(val, threshold):
                met.append(f"{metric} ({val}) {op_str} {threshold}")
            else:
                failed.append(f"{metric} ({val}) NOT {op_str} {threshold}")

        return GateResult(
            passed=len(failed) == 0,
            conditions_met=met,
            conditions_failed=failed,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QualityGate:
        """Create QualityGate from dict configuration."""
        name = data.get("name", "gate")
        conditions = data.get("conditions", {})
        return cls(name=name, conditions=conditions)


# Alias for concise Pipeline DSL
Gate = QualityGate

