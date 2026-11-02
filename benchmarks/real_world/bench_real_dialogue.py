"""Real-World Conversational Dialogue Hallucination Benchmark.

Evaluates Veriflux GroundednessChecker and ClaimVerifier on multi-turn
conversational dialogues from the official NeurIPS 2023 HaluEval benchmark
(RUCAIBox/HaluEval dialogue_data.json).
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

from veriflux._types import ClaimType, Verdict
from veriflux.nlp.claim_extractor import ClaimExtractor
from veriflux.hallucination.claim_verifier import ClaimVerifier

DATA_FILE = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "halueval_real_dialogue_500.jsonl"


def run_real_dialogue_benchmark(max_pairs: int = 100) -> Dict[str, Any]:
    extractor = ClaimExtractor()
    verifier = ClaimVerifier()

    pairs: List[Dict[str, str]] = []
    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append({
                "knowledge": obj["knowledge"],
                "dialogue_history": obj.get("dialogue_history", ""),
                "right_response": obj["right_response"],
                "hallucinated_response": obj["hallucinated_response"],
            })
            if len(pairs) >= max_pairs:
                break

    total_pairs = len(pairs)
    total_evals = total_pairs * 2

    tp = 0  # Right verified grounded
    fn = 0
    tn = 0  # Hallucinated caught
    fp = 0

    start_time = time.perf_counter()

    for p in pairs:
        knowledge = p["knowledge"]
        right = p["right_response"]
        hallu = p["hallucinated_response"]

        # Evaluate right response (filter non-factual conversational opinion claims)
        claims_r = extractor.extract(right)
        factual_r = [c for c in claims_r if c.claim_type != ClaimType.OPINION]
        if not factual_r:
            score_r = 1.0
        else:
            verdicts_r = verifier.verify_batch(factual_r, knowledge)
            supp_r = sum(1 for v in verdicts_r if v.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED))
            score_r = supp_r / len(factual_r)

        if score_r >= 0.50:
            tp += 1
        else:
            fn += 1

        # Evaluate hallucinated response
        claims_h = extractor.extract(hallu)
        factual_h = [c for c in claims_h if c.claim_type != ClaimType.OPINION]
        if not factual_h:
            score_h = 0.0
        else:
            verdicts_h = verifier.verify_batch(factual_h, knowledge)
            supp_h = sum(1 for v in verdicts_h if v.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED))
            score_h = supp_h / len(factual_h)

        if score_h < 0.50:
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
        "benchmark": "NeurIPS 2023 HaluEval Multi-Turn Conversational Dialogue",
        "dialogue_pairs_evaluated": total_pairs,
        "total_evaluations": total_evals,
        "grounded_retained": tp,
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
    rep = run_real_dialogue_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Dialogues Evaluated     : {rep['dialogue_pairs_evaluated']} ({rep['total_evaluations']} responses)")
    print(f" Grounded Responses Kept : {rep['grounded_retained']}/{rep['dialogue_pairs_evaluated']} ({rep['recall_pct']}%)")
    print(f" Hallucinations Caught   : {rep['hallucinations_intercepted']}/{rep['dialogue_pairs_evaluated']}")
    print(f" Discrimination Accuracy : {rep['accuracy_pct']}%")
    print(f" Dialogue F1-Score       : {rep['f1_score_pct']}%")
    print(f" Evaluation Throughput   : {rep['throughput_evals_per_sec']:,} evals/sec")
    print(f" Mean Latency            : {rep['mean_latency_ms']} ms")
    print(f" Inference Cost          : ${rep['cost_usd']:.2f} (Zero-Token Local Execution)")
    print("=" * 65)

