"""End-to-end evaluation pipeline orchestrator.

Coordinates multi-stage test executions, automated quality gates, and CI report rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from veriflux._types import EvalResult, EvalSuite, Severity
from veriflux.cicd.gates import QualityGate


class Stage:
    """An execution stage containing a suite of check callables."""

    def __init__(
        self,
        name: str,
        check_fn: Optional[Callable[[Any], Any]] = None,
        layer: str = "standard",
        checks: Optional[List[Callable[[Any], Any]]] = None,
    ) -> None:
        self.name = name
        self.layer = layer
        all_checks = list(checks or [])
        if check_fn is not None:
            all_checks.append(check_fn)
        self.checks = all_checks

    def add_check(self, check_fn: Callable[[Any], Any]) -> Stage:
        self.checks.append(check_fn)
        return self


@dataclass(slots=True)
class PipelineResult:
    """Outcome of running an evaluation pipeline across a test dataset.

    Attributes:
        pipeline_name: Name of evaluated pipeline.
        passed: True if all stages and quality gates passed.
        suite: Consolidated EvalSuite of all metric results.
        gate_results: Breakdown of quality gate evaluations.
        duration_ms: Total elapsed time in milliseconds.
    """

    pipeline_name: str
    passed: bool
    suite: EvalSuite
    gate_results: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 2),
            "summary": self.suite.summary(),
            "gates": self.gate_results,
            "results": [
                {
                    "metric": r.metric_name,
                    "passed": r.passed,
                    "score": r.score,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                }
                for r in self.suite.results
            ],
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_junit_xml(self, path: Optional[str] = None) -> str:
        from veriflux.cicd.reporters import JUnitReporter
        xml = JUnitReporter().generate(self.suite.results, suite_name=self.pipeline_name)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml)
        return xml

    def to_markdown(self) -> str:
        from veriflux.cicd.reporters import MarkdownReporter
        return MarkdownReporter().generate(self.suite.results)

    def to_github_pr_comment(self) -> str:
        status_emoji = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"## 🔱 Veriflux Quality Gate: {status_emoji}",
            f"**Pipeline:** `{self.pipeline_name}` | **Pass Rate:** {self.suite.pass_rate * 100:.1f}% ({self.suite.passed}/{self.suite.total})",
            "",
            self.to_markdown(),
        ]
        return "\n".join(lines)


class Pipeline:
    """Multi-stage evaluation pipeline manager."""

    def __init__(self, name: str = "veriflux_pipeline") -> None:
        self.name = name
        self.stages: List[Stage] = []
        self.gates: List[QualityGate] = []

    def add_stage(self, stage: Stage) -> Pipeline:
        """Register a stage in the execution graph."""
        self.stages.append(stage)
        return self

    def add_gate(self, gate: Union[QualityGate, Dict[str, Any]]) -> Pipeline:
        """Register a pass/fail quality gate."""
        if isinstance(gate, QualityGate):
            self.gates.append(gate)
        elif isinstance(gate, dict):
            self.gates.append(QualityGate.from_dict(gate))
        return self

    def execute(self, item: Any = None) -> PipelineResult:
        """Convenience method to execute all pipeline stages on a single item."""
        dataset = [item] if item is not None else None
        return self.evaluate(dataset=dataset)

    def evaluate(
        self,
        dataset: Optional[Sequence[Any]] = None,
        target_fn: Optional[Callable[[Any], Any]] = None,
    ) -> PipelineResult:
        """Execute all stages across the dataset."""
        start_t = time.perf_counter()
        suite = EvalSuite(name=self.name)
        items = list(dataset) if dataset else [None]

        for stage in self.stages:
            for item in items:
                subject = target_fn(item) if target_fn and item is not None else item
                for check in stage.checks:
                    c_start = time.perf_counter()
                    try:
                        res = check(subject)
                        c_dur = (time.perf_counter() - c_start) * 1000.0

                        if isinstance(res, EvalResult):
                            res.duration_ms = c_dur
                            suite.results.append(res)
                        elif isinstance(res, bool):
                            suite.results.append(
                                EvalResult(
                                    passed=res,
                                    score=1.0 if res else 0.0,
                                    metric_name=getattr(check, "__name__", stage.name),
                                    duration_ms=c_dur,
                                )
                            )
                        elif hasattr(res, "passed") and hasattr(res, "score"):
                            suite.results.append(
                                EvalResult(
                                    passed=bool(res.passed),
                                    score=float(res.score),
                                    metric_name=getattr(check, "__name__", stage.name),
                                    duration_ms=c_dur,
                                )
                            )
                    except Exception as exc:
                        c_dur = (time.perf_counter() - c_start) * 1000.0
                        suite.results.append(
                            EvalResult(
                                passed=False,
                                score=0.0,
                                metric_name=getattr(check, "__name__", stage.name),
                                severity=Severity.ERROR,
                                duration_ms=c_dur,
                                details={"error": str(exc)},
                            )
                        )

        # Evaluate quality gates
        metrics_dict: Dict[str, float] = {
            "pass_rate": suite.pass_rate,
            "mean_score": suite.mean_score,
            "failed_count": float(suite.failed),
        }
        gate_results: Dict[str, Any] = {}
        all_gates_passed = True

        for gate in self.gates:
            gres = gate.evaluate(metrics_dict)
            gate_results[gate.name] = {
                "passed": gres.passed,
                "conditions_met": gres.conditions_met,
                "conditions_failed": gres.conditions_failed,
            }
            if not gres.passed:
                all_gates_passed = False

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        overall_passed = (suite.failed == 0) and all_gates_passed

        return PipelineResult(
            pipeline_name=self.name,
            passed=overall_passed,
            suite=suite,
            gate_results=gate_results,
            duration_ms=duration_ms,
        )

