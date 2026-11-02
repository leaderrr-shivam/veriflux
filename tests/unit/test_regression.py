"""Unit tests for Regression Detection Engine."""
from __future__ import annotations

import os
import shutil
import pytest
from veriflux.regression import (
    BaselineManager,
    DriftDetector,
    MetricComparator,
    MetricStatus,
)


@pytest.fixture
def temp_baseline_dir(tmp_path: Any) -> str:
    d = str(tmp_path / "baselines")
    os.makedirs(d, exist_ok=True)
    return d


def test_baseline_manager(temp_baseline_dir: str) -> None:
    bm = BaselineManager(directory=temp_baseline_dir, default_tolerance=0.05)
    base_metrics = {"groundedness": 0.85, "relevance": 0.90, "latency_ms": 200.0}
    bm.save("v1.0", base_metrics)

    # Candidate with degraded groundedness and improved relevance
    cand_metrics = {"groundedness": 0.70, "relevance": 0.98, "latency_ms": 205.0}
    res = bm.compare("v1.0", cand_metrics)

    assert res.has_regressions is True
    assert any("groundedness" in reg for reg in res.regressions)
    assert any("relevance" in imp for imp in res.improvements)


def test_drift_detector() -> None:
    detector = DriftDetector(threshold=0.20)
    base_corpus = [
        "The server is running normally and accepting connections.",
        "Database latency is within standard operational thresholds.",
    ]
    same_distribution = [
        "The server is operational and responding to connections.",
        "Database performance metrics remain within nominal bounds.",
    ]
    drifted_corpus = [
        "Unicorns gallop across the rainbow meadow during midnight.",
        "The medieval castle was besieged by an army of stone gargoyles.",
    ]

    res_same = detector.detect(base_corpus, same_distribution)
    res_drifted = detector.detect(base_corpus, drifted_corpus)

    assert res_same.drift_score < res_drifted.drift_score
    assert res_drifted.has_drift is True
