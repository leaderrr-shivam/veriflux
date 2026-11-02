"""Unit tests for Agent Trajectory Analyzer."""
from __future__ import annotations

import pytest
from veriflux._types import TokenUsage
from veriflux.trajectory import (
    ExecutionGraph,
    LoopDetector,
    Replayer,
    StateInspector,
    Trace,
    TraceRecorder,
    Tracer,
    TrajectoryAnalyzer,
)


def test_tracer_and_analysis() -> None:
    with Tracer(name="booking_agent") as t:
        t.record_step("init", output_data="ready")
        t.record_llm_call("gpt-4o", "Book flight", "Flight options found", tokens=TokenUsage(100, 50, 150))
        t.record_tool_call("search_flights", {"dest": "JFK"}, [{"flight": "AA100"}], duration_ms=120.0)

    trace = t.trace
    assert trace.total_steps == 3
    assert trace.total_tokens == 150
    assert not trace.has_errors

    analyzer = TrajectoryAnalyzer()
    res = analyzer.analyze(trace)
    assert res.total_steps == 3
    assert "llm_call" in res.step_distribution
    assert "tool_call" in res.step_distribution


def test_replayer_counterfactual() -> None:
    trace = Trace(name="test_trace")
    with Tracer(name="test") as t:
        t.record_tool_call("api_lookup", {}, {"status": 200})
        t.record_step("render", output_data="Success page")

    replayer = Replayer(t.trace)
    replayer.mock_step(0, {"status": 500})
    result = replayer.run()

    assert result.success
    assert result.diffs[0].diverged is True
    assert result.diffs[0].replayed_output == {"status": 500}


def test_loop_detector() -> None:
    trace = Trace(name="cycling_agent")
    with Tracer(name="loop_test") as t:
        # Repeating pattern: search -> parse -> search -> parse
        t.record_step("tool:search")
        t.record_step("tool:parse")
        t.record_step("tool:search")
        t.record_step("tool:parse")

    detector = LoopDetector(min_repetitions=2)
    loops = detector.detect(t.trace)
    assert len(loops) > 0
    assert loops[0].cycle_length == 2
    assert loops[0].repetitions == 2


def test_graph_and_mermaid() -> None:
    with Tracer(name="graph_test") as t:
        t.record_step("step1")
        t.record_step("step2")

    graph = ExecutionGraph.from_trace(t.trace)
    mermaid = graph.to_mermaid()
    assert "flowchart TD" in mermaid
    assert "node0 --> node1" in mermaid

