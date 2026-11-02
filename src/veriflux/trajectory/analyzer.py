"""Trajectory pattern and cost/latency attribution analysis.

Computes step distributions, identifies performance bottlenecks, attributes costs,
and exports execution histories for tabular analysis.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from veriflux._types import TraceStep
from veriflux.trajectory.tracer import Trace


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    """Consolidated trajectory diagnostics.

    Attributes:
        total_steps: Number of recorded steps.
        total_duration_ms: Total session duration.
        total_tokens: Total tokens consumed.
        total_cost: Total dollar cost.
        step_distribution: Frequency of each step action type.
        error_count: Number of failed steps.
        retry_count: Number of consecutive repeated action attempts.
    """

    total_steps: int
    total_duration_ms: float
    total_tokens: int
    total_cost: float
    step_distribution: Dict[str, int]
    error_count: int
    retry_count: int


class TrajectoryAnalyzer:
    """Analyzes execution traces to discover bottlenecks, anomalies, and cost drivers."""

    def analyze(self, trace: Trace) -> AnalysisResult:
        """Perform comprehensive trajectory analysis."""
        step_counts: Counter[str] = Counter()
        error_count = 0
        retry_count = 0
        last_action = ""

        for step in trace.steps:
            action_base = step.action.split(":")[0]
            step_counts[action_base] += 1

            if step.metadata.get("error") is not None:
                error_count += 1

            if step.action == last_action:
                retry_count += 1
            last_action = step.action

        return AnalysisResult(
            total_steps=trace.total_steps,
            total_duration_ms=round(trace.total_duration_ms, 2),
            total_tokens=trace.total_tokens,
            total_cost=round(trace.total_cost, 6),
            step_distribution=dict(step_counts),
            error_count=error_count,
            retry_count=retry_count,
        )

    def bottleneck_steps(self, trace: Trace, top_k: int = 5) -> List[TraceStep]:
        """Find the slowest execution steps by wall-clock latency."""
        return sorted(trace.steps, key=lambda s: s.duration_ms, reverse=True)[:top_k]

    def cost_attribution(self, trace: Trace) -> Dict[str, float]:
        """Compute dollar cost breakdown by step action category."""
        cost_by_action: Dict[str, float] = defaultdict(float)
        for s in trace.steps:
            if s.token_usage:
                cat = s.action.split(":")[0]
                cost_by_action[cat] += s.token_usage.estimated_cost
        return {k: round(v, 6) for k, v in cost_by_action.items()}

    def to_dataframe(self, trace: Trace) -> List[Dict[str, Any]]:
        """Export trace steps as a list of dictionaries suitable for Pandas/CSV."""
        rows: List[Dict[str, Any]] = []
        for s in trace.steps:
            rows.append({
                "step_index": s.step_index,
                "action": s.action,
                "duration_ms": s.duration_ms,
                "tokens": s.token_usage.total_tokens if s.token_usage else 0,
                "cost": s.token_usage.estimated_cost if s.token_usage else 0.0,
                "has_error": s.metadata.get("error") is not None,
            })
        return rows

