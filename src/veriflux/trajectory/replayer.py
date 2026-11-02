"""Deterministic trace replay and what-if counterfactual intervention.

Replays recorded agent traces while allowing mock injection, step skipping,
and fault injection to verify error recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from veriflux._types import TraceStep
from veriflux.trajectory.tracer import Trace


@dataclass(slots=True, frozen=True)
class StepDiff:
    """Discrepancy observed between original execution and replayed execution."""

    step_index: int
    original_action: str
    original_output: Any
    replayed_output: Any
    diverged: bool


@dataclass(slots=True)
class ReplayResult:
    """Outcome of a trace replay run.

    Attributes:
        success: True if replay completed without unhandled exceptions.
        steps_executed: Count of replayed steps.
        diffs: Discrepancies between historical and replayed values.
        error: Caught error if execution failed.
    """

    success: bool
    steps_executed: int
    diffs: List[StepDiff] = field(default_factory=list)
    error: Optional[str] = None


class Replayer:
    """Replays recorded agent traces deterministically with counterfactual overrides."""

    def __init__(self, trace: Union[Trace, List[TraceStep]]) -> None:
        if isinstance(trace, list):
            self.trace = Trace(name="replayed_trace", steps=trace)
        else:
            self.trace = trace
        self._mocks: Dict[int, Any] = {}
        self._skips: set[int] = set()
        self._injected_errors: Dict[int, Exception] = {}

    def replay(self) -> List[TraceStep]:
        """Iterate through the active trace steps for inspection."""
        return [s for idx, s in enumerate(self.trace.steps) if idx not in self._skips]

    def mock_step(self, step_index: int, result: Any) -> Replayer:
        """Override the output returned by a specific historical step."""
        self._mocks[step_index] = result
        return self

    def skip_step(self, step_index: int) -> Replayer:
        """Skip execution of a specific historical step."""
        self._skips.add(step_index)
        return self

    def inject_error(self, step_index: int, error: Exception) -> Replayer:
        """Inject a simulated exception at a specific historical step."""
        self._injected_errors[step_index] = error
        return self

    def run(self, handler_fn: Optional[Callable[[TraceStep], Any]] = None) -> ReplayResult:
        """Execute the replay pass across all recorded steps."""
        diffs: List[StepDiff] = []
        executed = 0

        for idx, step in enumerate(self.trace.steps):
            if idx in self._skips:
                continue

            if idx in self._injected_errors:
                err = self._injected_errors[idx]
                return ReplayResult(
                    success=False,
                    steps_executed=executed,
                    diffs=diffs,
                    error=f"Injected error at step {idx}: {err}",
                )

            if idx in self._mocks:
                replayed_out = self._mocks[idx]
            elif handler_fn is not None:
                try:
                    replayed_out = handler_fn(step)
                except Exception as exc:
                    return ReplayResult(
                        success=False,
                        steps_executed=executed,
                        diffs=diffs,
                        error=f"Replay handler failed at step {idx}: {exc}",
                    )
            else:
                replayed_out = step.output_data

            diverged = replayed_out != step.output_data
            diffs.append(
                StepDiff(
                    step_index=idx,
                    original_action=step.action,
                    original_output=step.output_data,
                    replayed_output=replayed_out,
                    diverged=diverged,
                )
            )
            executed += 1

        return ReplayResult(success=True, steps_executed=executed, diffs=diffs)


# Ergonomic alias
TrajectoryReplayer = Replayer

