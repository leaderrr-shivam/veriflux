"""Real-World HaluEval (NeurIPS 2023) QA Benchmark Evaluation.

Evaluates Veriflux GroundednessChecker against authentic downloaded samples
from the official RUCAIBox/HaluEval NeurIPS 2023 repository.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veriflux.hallucination.groundedness import GroundednessChecker

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "halueval_real_qa_500.jsonl"


def run_real_halueval(num_samples: int = 100) -> Dict[str, Any]:
    with open(DATASET_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(num_samples)]

    checker = GroundednessChecker(entailment_threshold=0.55)

    tp, fp, tn, fn = 0, 0, 0, 0
    total_evals = 0
    start_time = time.perf_counter()

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        data = json.loads(line)
        knowledge = data["knowledge"]
        right_ans = data["right_answer"]
        hallu_ans = data["hallucinated_answer"]

        # 1. Grounded check (right answer)
        res_g = checker.check(response=right_ans, context=knowledge)
        pred_hallu_g = res_g.score < 0.55 or len(res_g.ungrounded_claims) > 0
        total_evals += 1
        if pred_hallu_g:
            fp += 1
        else:
            tn += 1

        # 2. Hallucinated check
        res_h = checker.check(response=hallu_ans, context=knowledge)
        pred_hallu_h = res_h.score < 0.55 or len(res_h.ungrounded_claims) > 0
        total_evals += 1
        if pred_hallu_h:
            tp += 1
        else:
            fn += 1

    total_time = time.perf_counter() - start_time
    accuracy = (tp + tn) / total_evals if total_evals > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_latency_ms = (total_time / total_evals) * 1000.0

    report = {
        "benchmark": "HaluEval (NeurIPS 2023 Official Web Dataset - 100 Pairs)",
        "total_evaluations": total_evals,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "mean_latency_ms": round(mean_latency_ms, 3),
    }

    return report


if __name__ == "__main__":
    rep = run_real_halueval(100)
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Evaluations       : {rep['total_evaluations']}")
    print(f" Accuracy                : {rep['accuracy']}%")
    print(f" Precision               : {rep['precision']}%")
    print(f" Recall (Catch Rate)     : {rep['recall']}%")
    print(f" F1-Score                : {rep['f1_score']}%")
    print(f" Mean Latency per Eval   : {rep['mean_latency_ms']} ms")
    print(f" Confusion Matrix        : TP={rep['true_positives']}, FP={rep['false_positives']}, TN={rep['true_negatives']}, FN={rep['false_negatives']}")
    print("=" * 65)

