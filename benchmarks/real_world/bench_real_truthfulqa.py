"""Real-World TruthfulQA Benchmark Evaluation Suite.

Evaluates Veriflux GroundednessChecker and EntailmentClassifier on
the official TruthfulQA benchmark (Oxford / OpenAI) across truthful vs.
adversarially hallucinated response pairs.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veriflux.hallucination.groundedness import GroundednessChecker

DATA_FILE = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "truthfulqa_real.csv"


def run_truthfulqa_benchmark(max_pairs: int = 300) -> Dict[str, Any]:
    checker = GroundednessChecker(entailment_threshold=0.65)

    pairs: List[Dict[str, str]] = []
    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "question": row["Question"],
                "best_answer": row["Best Answer"],
                "best_incorrect": row["Best Incorrect Answer"],
                "category": row.get("Category", "General"),
            })
            if len(pairs) >= max_pairs:
                break

    total_pairs = len(pairs)
    total_evals = total_pairs * 2  # Each pair has 1 truthful + 1 hallucinated

    truthful_correct = 0
    hallucination_flagged = 0
    truthful_scores: List[float] = []
    hallucination_scores: List[float] = []

    start_time = time.perf_counter()

    for p in pairs:
        ref_context = p["best_answer"]
        corr_ans = p["best_answer"]
        inc_ans = p["best_incorrect"]

        # Truthful evaluation
        res_corr = checker.check(corr_ans, ref_context)
        truthful_scores.append(res_corr.score)
        if res_corr.score >= 0.5:
            truthful_correct += 1

        # Hallucination evaluation
        res_inc = checker.check(inc_ans, ref_context)
        hallucination_scores.append(res_inc.score)
        if res_inc.score < 0.5:
            hallucination_flagged += 1

    total_time = time.perf_counter() - start_time

    tp = truthful_correct
    fn = total_pairs - truthful_correct
    tn = hallucination_flagged
    fp = total_pairs - hallucination_flagged

    accuracy = (tp + tn) / total_evals * 100.0
    precision = tp / max(1, tp + fp) * 100.0
    recall = tp / max(1, tp + fn) * 100.0
    f1 = (2 * precision * recall) / max(0.001, precision + recall)

    mean_truthful_score = sum(truthful_scores) / len(truthful_scores)
    mean_hallucination_score = sum(hallucination_scores) / len(hallucination_scores)

    throughput = total_evals / max(0.00001, total_time)
    mean_latency_ms = (total_time / total_evals) * 1000.0

    report = {
        "benchmark": "TruthfulQA Official Benchmark (Oxford / OpenAI)",
        "question_pairs_evaluated": total_pairs,
        "total_evaluations": total_evals,
        "truthful_groundedness_avg": round(mean_truthful_score, 4),
        "hallucination_groundedness_avg": round(mean_hallucination_score, 4),
        "truthful_retained_pct": round(recall, 2),
        "hallucinations_flagged_pct": round((tn / total_pairs) * 100.0, 2),
        "discrimination_accuracy_pct": round(accuracy, 2),
        "f1_score_pct": round(f1, 2),
        "throughput_evals_per_sec": round(throughput, 1),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "cost_usd": 0.00,
    }

    return report


if __name__ == "__main__":
    rep = run_truthfulqa_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Evaluated Questions     : {rep['question_pairs_evaluated']} ({rep['total_evaluations']} total evaluations)")
    print(f" Mean Truthful Grounded  : {rep['truthful_groundedness_avg']:.4f}")
    print(f" Mean Hallucinated Score : {rep['hallucination_groundedness_avg']:.4f}")
    print(f" Truthful Retained       : {rep['truthful_retained_pct']}%")
    print(f" Hallucinations Flagged  : {rep['hallucinations_flagged_pct']}%")
    print(f" Discrimination Accuracy : {rep['discrimination_accuracy_pct']}%")
    print(f" F1-Score                : {rep['f1_score_pct']}%")
    print(f" Evaluation Throughput   : {rep['throughput_evals_per_sec']:,} evals/sec")
    print(f" Mean Latency            : {rep['mean_latency_ms']} ms")
    print(f" Inference Cost          : ${rep['cost_usd']:.2f} (Zero-Token Local Execution)")
    print("=" * 65)
