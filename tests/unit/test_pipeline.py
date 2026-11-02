"""Unit tests for Pipeline Orchestrator and CI/CD Gates."""
from __future__ import annotations

import pytest
from veriflux.cicd.gates import Gate, QualityGate
from veriflux.pipeline import Pipeline, Stage


def test_pipeline_execution() -> None:
    pipeline = Pipeline("test_eval_pipeline")

    stage_fast = Stage(name="fast_structural_checks")
    stage_fast.add_check(lambda s: len(s) > 5)
    pipeline.add_stage(stage_fast)

    stage_safety = Stage(name="safety_checks")
    stage_safety.add_check(lambda s: "bad_word" not in s)
    pipeline.add_stage(stage_safety)

    gate = QualityGate("prod_gate")
    gate.add_condition("pass_rate", ">=", 0.99)
    pipeline.add_gate(gate)

    dataset = [
        "A valid generated response sentence for test item 1.",
        "A valid generated response sentence for test item 2.",
    ]

    res = pipeline.evaluate(dataset)
    assert res.passed is True
    assert res.suite.passed == 4
    assert res.suite.failed == 0

    xml = res.to_junit_xml()
    assert "<testsuite" in xml
    assert "test_eval_pipeline" in xml

    md = res.to_github_pr_comment()
    assert "PASSED" in md

