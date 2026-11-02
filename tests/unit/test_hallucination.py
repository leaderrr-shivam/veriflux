"""Unit tests for Hallucination Detection Engine."""
from __future__ import annotations

import pytest
from veriflux.hallucination import (
    ConsistencyChecker,
    GroundednessChecker,
    HallucinationDetector,
    URLValidator,
)


def test_groundedness_checker(sample_context: str) -> None:
    checker = GroundednessChecker()
    grounded_resp = "Apollo 11 landed humans on the Moon in 1969, including Neil Armstrong."
    hallucinated_resp = "Apollo was launched by the Soviet Union in 1955 to colonize Mars."

    res_grounded = checker.check(grounded_resp, sample_context)
    res_hallucinated = checker.check(hallucinated_resp, sample_context)

    assert res_grounded.score > res_hallucinated.score
    assert res_grounded.score >= 0.7


def test_consistency_checker() -> None:
    checker = ConsistencyChecker()
    consistent_text = "The package was shipped on Monday. It arrived at the destination on Wednesday."
    contradictory_text = (
        "The project was completely cancelled in 2021. "
        "The project continued active development throughout 2022."
    )

    res_c = checker.check(consistent_text)
    res_bad = checker.check(contradictory_text)

    assert res_c.is_consistent
    assert not res_bad.is_consistent


def test_url_validator() -> None:
    validator = URLValidator()
    real_style = "Learn more at https://github.com/leaderrr-shivam/veriflux or https://python.org"
    fake_style = "Read the paper at https://arxiv.org/abs/9999.9999999 or doi:10.99999/fabricated.paper.2099"

    suspicious = validator.find_suspicious_urls(fake_style)
    assert len(suspicious) > 0


def test_hallucination_detector_pipeline(sample_context: str) -> None:
    detector = HallucinationDetector()
    grounded_output = "NASA landed the first humans on the Moon in 1969 with Apollo 11."
    res = detector.detect(grounded_output, sample_context)

    assert res.verdict == "grounded"
    assert res.score >= 0.7
    assert res.passed

