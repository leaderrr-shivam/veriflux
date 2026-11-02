"""Infinite loop, repetitive action cycling, and thrashing detection.

Identifies when an agent enters repetitive tool loops or circular reasoning traps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from veriflux.trajectory.tracer import Trace


@dataclass(slots=True, frozen=True)
class LoopPattern:
    """A detected repeating execution cycle.

    Attributes:
        start_index: Step index where loop begins.
        cycle_length: Number of steps in the recurring cycle.
        repetitions: Number of times cycle was repeated consecutively.
        pattern: Sequence of step actions comprising the cycle.
    """

    start_index: int
    cycle_length: int
    repetitions: int
    pattern: List[str]


class LoopDetector:
    """Detects cycles and infinite loops in agent trajectories."""

    def __init__(
        self,
        max_cycle_length: int = 6,
        min_repetitions: int = 2,
        window_size: Optional[int] = None,
        max_repeats: Optional[int] = None,
    ) -> None:
        self.max_cycle_length = window_size if window_size is not None else max_cycle_length
        self.min_repetitions = max_repeats if max_repeats is not None else min_repetitions
        self._history: List[tuple[str, str]] = []

    def check_step(self, action: str, input_data: Any = None) -> bool:
        """Online check if the current step indicates a loop."""
        sig = (action, str(input_data))
        self._history.append(sig)

        if len(self._history) >= self.min_repetitions:
            recent = self._history[-self.min_repetitions:]
            if all(item == sig for item in recent):
                return True
        return False

    def detect(self, trace: Trace) -> List[LoopPattern]:
        """Scan a trace for consecutive repeating sub-sequences of actions."""
        actions = [s.action for s in trace.steps]
        n = len(actions)
        if n < self.min_repetitions * 2:
            return []

        loops: List[LoopPattern] = []
        i = 0
        while i < n:
            found_loop = False
            for cycle_len in range(1, min(self.max_cycle_length + 1, (n - i) // 2 + 1)):
                pattern = actions[i : i + cycle_len]
                reps = 1
                pos = i + cycle_len

                while pos + cycle_len <= n and actions[pos : pos + cycle_len] == pattern:
                    reps += 1
                    pos += cycle_len

                if reps >= self.min_repetitions:
                    loops.append(
                        LoopPattern(
                            start_index=i,
                            cycle_length=cycle_len,
                            repetitions=reps,
                            pattern=pattern,
                        )
                    )
                    i = pos
                    found_loop = True
                    break

            if not found_loop:
                i += 1

        return loops

