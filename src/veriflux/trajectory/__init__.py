"""Agent execution trajectory analyzer for Veriflux.

Provides zero-overhead execution tracing, deterministic replay, loop detection,
graph visualization, and state mutation inspection.
"""
from __future__ import annotations

from veriflux.trajectory.analyzer import AnalysisResult, TrajectoryAnalyzer
from veriflux.trajectory.graph import ExecutionGraph
from veriflux.trajectory.loop_detector import LoopDetector, LoopPattern
from veriflux.trajectory.recorder import TraceRecorder
from veriflux.trajectory.replayer import ReplayResult, Replayer, StepDiff
from veriflux.trajectory.state_inspector import MutationEvent, StateDiff, StateInspector
from veriflux.trajectory.tracer import Trace, Tracer, current_tracer

__all__ = [
    # Tracer
    "Tracer",
    "Trace",
    "current_tracer",
    # Recorder
    "TraceRecorder",
    # Replayer
    "Replayer",
    "ReplayResult",
    "StepDiff",
    # Analyzer
    "TrajectoryAnalyzer",
    "AnalysisResult",
    # Loop detector
    "LoopDetector",
    "LoopPattern",
    # Graph
    "ExecutionGraph",
    # State inspector
    "StateInspector",
    "StateDiff",
    "MutationEvent",
]

