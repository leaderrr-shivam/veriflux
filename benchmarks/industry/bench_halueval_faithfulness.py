"""HaluEval (NeurIPS 2023) Faithfulness & Hallucination Industry Benchmark.

Evaluates Veriflux Groundedness and Claim Verification against 100 realistic
samples covering Entity Swaps, Relational Antonym Flips, Numerical Distortions,
and Fabricated Extrapolations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veriflux.hallucination.groundedness import GroundednessChecker

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "halueval_benchmark.json"


def run_halueval_benchmark() -> Dict[str, Any]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    checker = GroundednessChecker(entailment_threshold=0.60)

    tp, fp, tn, fn = 0, 0, 0, 0
    type_stats: Dict[str, Dict[str, int]] = {}

    start_time = time.perf_counter()

    for item in samples:
        context = item["context"]
        response = item["response"]
        is_hallucinated = item["is_hallucination"]
        htype = item["hallucination_type"]

        if htype not in type_stats:
            type_stats[htype] = {"total": 0, "correct": 0}
        type_stats[htype]["total"] += 1

        result = checker.check(response=response, context=context)
        # Score < 0.65 is classified as hallucinated
        predicted_hallucination = result.score < 0.65 or len(result.ungrounded_claims) > 0

        if is_hallucinated and predicted_hallucination:
            tp += 1
            type_stats[htype]["correct"] += 1
        elif not is_hallucinated and not predicted_hallucination:
            tn += 1
            type_stats[htype]["correct"] += 1
        elif not is_hallucinated and predicted_hallucination:
            fp += 1
        elif is_hallucinated and not predicted_hallucination:
            fn += 1

    total_time = time.perf_counter() - start_time
    total = len(samples)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    latency_ms = (total_time / total) * 1000.0

    report = {
        "benchmark": "HaluEval (NeurIPS 2023) Groundedness & Faithfulness",
        "total_samples": total,
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "mean_latency_ms": round(latency_ms, 3),
        "api_cost_usd": 0.0,
        "category_breakdown": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round((v["correct"] / v["total"]) * 100.0, 1),
            }
            for k, v in type_stats.items()
        },
    }

    return report


if __name__ == "__main__":
    rep = run_halueval_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Samples Evaluated : {rep['total_samples']}")
    print(f" Accuracy                : {rep['accuracy']}%")
    print(f" Precision               : {rep['precision']}%")
    print(f" Recall                  : {rep['recall']}%")
    print(f" F1-Score                : {rep['f1_score']}%")
    print(f" Mean Latency per Eval   : {rep['mean_latency_ms']} ms")
    print(f" API Cost (Zero LLM)     : ${rep['api_cost_usd']:.2f}")
    print("-" * 65)
    print(" Category Accuracy Breakdown:")
    for cat, stats in rep["category_breakdown"].items():
        print(f"   - {cat:<24}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")
    print("=" * 65)
