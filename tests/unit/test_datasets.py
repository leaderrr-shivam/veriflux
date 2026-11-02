"""Unit tests for Datasets subsystem."""
from __future__ import annotations

import os
from typing import Any
import pytest
from veriflux.datasets import (
    Dataset,
    DatasetLoader,
    GoldenDataset,
    Sampler,
    SyntheticGenerator,
    TestCase,
)


def test_dataset_loader_and_sampler(tmp_path: Any) -> None:
    jsonl_file = str(tmp_path / "test_data.jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write('{"query": "q1", "response": "r1", "category": "math"}\n')
        f.write('{"query": "q2", "response": "r2", "category": "history"}\n')
        f.write('{"query": "q3", "response": "r3", "category": "math"}\n')
        f.write('{"query": "q4", "response": "r4", "category": "science"}\n')

    loader = DatasetLoader()
    dataset = loader.load(jsonl_file)
    assert len(dataset) == 4
    assert dataset[0].query == "q1"

    sampler = Sampler()
    sampled = sampler.random_sample(dataset, n=2, seed=42)
    assert len(sampled) == 2


def test_golden_dataset(tmp_path: Any) -> None:
    golden_path = str(tmp_path / "golden.json")
    gd = GoldenDataset(name="golden_v1")
    gd.add_case(TestCase(query="2+2", expected="4"))
    gd.add_case(TestCase(query="capital of UK", expected="London"))
    gd.save(golden_path)

    loaded = GoldenDataset.load(golden_path)
    assert len(loaded.cases) == 2
    assert loaded.cases[0].expected == "4"


def test_synthetic_generator() -> None:
    generator = SyntheticGenerator()
    context = "The French Revolution began in 1789 and overthrew the monarchy."
    cases = generator.generate_adversarial(context, n=3)
    assert len(cases) > 0

