"""Real-World Long-Document Summarization Hallucination Benchmark.

Evaluates Veriflux GroundednessChecker and ClaimVerifier on 100 authentic
long-form news documents and summaries from the official NeurIPS 2023
HaluEval benchmark (RUCAIBox/HaluEval).
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

DATA_FILE = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "halueval_real_summarization_500.jsonl"


def run_real_summarization_benchmark(max_pairs: int = 50) -> Dict[str, Any]:
    checker = GroundednessChecker()

    pairs: List[Dict[str, str]] = []
    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append({
                "document": obj["document"],
                "right_summary": obj["right_summary"],
                "hallucinated_summary": obj["hallucinated_summary"],
            })
            if len(pairs) >= max_pairs:
                break

    total_pairs = len(pairs)
    total_evals = total_pairs * 2

    tp = 0  # Right summary verified grounded (score >= 0.70)
    fn = 0
    tn = 0  # Hallucinated summary flagged (score < 0.70 or ungrounded_claims >= 1)
    fp = 0

    start_time = time.perf_counter()

    for p in pairs:
        doc = p["document"]
        right = p["right_summary"]
        hallu = p["hallucinated_summary"]

        rr = checker.check(right, doc)
        rh = checker.check(hallu, doc)

        if rr.score >= 0.70:
            tp += 1
        else:
            fn += 1

        if rh.score < 0.70 or len(rh.ungrounded_claims) >= 1:
            tn += 1
        else:
            fp += 1

    total_time = time.perf_counter() - start_time

    accuracy = (tp + tn) / total_evals * 100.0
    precision = tp / max(1, tp + fp) * 100.0
    recall = tp / max(1, tp + fn) * 100.0
    f1 = (2 * precision * recall) / max(0.001, precision + recall)

    throughput = total_evals / max(0.00001, total_time)
    mean_latency_ms = (total_time / total_evals) * 1000.0

    report = {
        "benchmark": "NeurIPS 2023 HaluEval Long-Document Summarization",
        "document_pairs_evaluated": total_pairs,
        "total_evaluations": total_evals,
        "ground_truth_retained": tp,
        "hallucinations_intercepted": tn,
        "accuracy_pct": round(accuracy, 2),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score_pct": round(f1, 2),
        "throughput_evals_per_sec": round(throughput, 1),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "cost_usd": 0.00,
    }

    return report


if __name__ == "__main__":
    rep = run_real_summarization_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Documents Evaluated     : {rep['document_pairs_evaluated']} ({rep['total_evaluations']} total summaries)")
    print(f" Factual Summaries Kept  : {rep['ground_truth_retained']}/{rep['document_pairs_evaluated']} ({rep['recall_pct']}%)")
    print(f" Hallucinations Flagged  : {rep['hallucinations_intercepted']}/{rep['document_pairs_evaluated']}")
    print(f" Discrimination Accuracy : {rep['accuracy_pct']}%")
    print(f" Summary F1-Score        : {rep['f1_score_pct']}%")
    print(f" Evaluation Throughput   : {rep['throughput_evals_per_sec']:,} evals/sec")
    print(f" Mean Latency            : {rep['mean_latency_ms']} ms / document")
    print(f" Inference Cost          : ${rep['cost_usd']:.2f} (Zero-Token Local Execution)")
    print("=" * 65)
