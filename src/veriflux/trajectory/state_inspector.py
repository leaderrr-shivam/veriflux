"""Agent state diffing and memory mutation tracking.

Tracks mutable memory and tool side-effects across multi-step agent trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from veriflux.trajectory.tracer import Trace


@dataclass(slots=True, frozen=True)
class StateDiff:
    """An observed delta between two state dictionaries."""

    path: str
    old_value: Any
    new_value: Any
    diff_type: str  # 'added', 'removed', 'modified'


@dataclass(slots=True, frozen=True)
class MutationEvent:
    """A mutation to agent memory occurring at a specific execution step."""

    step_index: int
    path: str
    old_value: Any
    new_value: Any


class StateInspector:
    """Inspects state transitions and memory mutability across steps."""

    def diff(self, state_a: Dict[str, Any], state_b: Dict[str, Any], path: str = "$") -> List[StateDiff]:
        """Compute differences between two state snapshots."""
        diffs: List[StateDiff] = []
        all_keys = set(state_a.keys()) | set(state_b.keys())

        for k in sorted(all_keys):
            cur_path = f"{path}.{k}"
            if k in state_a and k not in state_b:
                diffs.append(StateDiff(cur_path, state_a[k], None, "removed"))
            elif k not in state_a and k in state_b:
                diffs.append(StateDiff(cur_path, None, state_b[k], "added"))
            else:
                val_a = state_a[k]
                val_b = state_b[k]
                if isinstance(val_a, dict) and isinstance(val_b, dict):
                    diffs.extend(self.diff(val_a, val_b, path=cur_path))
                elif val_a != val_b:
                    diffs.append(StateDiff(cur_path, val_a, val_b, "modified"))

        return diffs

    def track_mutations(self, trace: Trace) -> List[MutationEvent]:
        """Extract all state mutations captured across trace step metadata."""
        mutations: List[MutationEvent] = []
        last_state: Dict[str, Any] = {}

        for step in trace.steps:
            current_state = step.metadata.get("state")
            if current_state and isinstance(current_state, dict):
                step_diffs = self.diff(last_state, current_state)
                for d in step_diffs:
                    mutations.append(
                        MutationEvent(
                            step_index=step.step_index,
                            path=d.path,
                            old_value=d.old_value,
                            new_value=d.new_value,
                        )
                    )
                last_state = dict(current_state)

        return mutations

