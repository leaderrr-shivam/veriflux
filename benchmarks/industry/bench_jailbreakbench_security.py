"""JailbreakBench & HarmBench Adversarial Security Industry Benchmark.

Evaluates Veriflux Guardrails and Prompt Injection Defense against 100
realistic attack and benign control samples across Direct Injection,
Roleplay Escalation, Obfuscation, System Leakage, and Indirect Injection.
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

from veriflux.guardrails.prompt_injection import PromptInjectionGuard

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "jailbreakbench_adversarial.json"


def run_jailbreakbench_benchmark() -> Dict[str, Any]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    guard = PromptInjectionGuard()

    tp, fp, tn, fn = 0, 0, 0, 0
    type_stats: Dict[str, Dict[str, int]] = {}

    start_time = time.perf_counter()

    for item in samples:
        prompt = item["prompt"]
        is_attack = item["is_attack"]
        atype = item["attack_type"]

        if atype not in type_stats:
            type_stats[atype] = {"total": 0, "correct": 0}
        type_stats[atype]["total"] += 1

        result = guard.check(prompt)
        detected_attack = not result.is_safe

        if is_attack and detected_attack:
            tp += 1
            type_stats[atype]["correct"] += 1
        elif not is_attack and not detected_attack:
            tn += 1
            type_stats[atype]["correct"] += 1
        elif not is_attack and detected_attack:
            fp += 1
        elif is_attack and not detected_attack:
            fn += 1

    total_time = time.perf_counter() - start_time
    total = len(samples)
    total_attacks = sum(1 for s in samples if s["is_attack"])
    total_benign = sum(1 for s in samples if not s["is_attack"])

    interception_rate = (tp / total_attacks) * 100.0 if total_attacks > 0 else 0.0
    false_positive_rate = (fp / total_benign) * 100.0 if total_benign > 0 else 0.0
    accuracy = ((tp + tn) / total) * 100.0
    throughput = total / max(0.0001, total_time)
    mean_latency_us = (total_time / total) * 1_000_000.0

    report = {
        "benchmark": "JailbreakBench & HarmBench Adversarial Security Suite",
        "total_samples": total,
        "interception_rate": round(interception_rate, 2),
        "false_positive_rate": round(false_positive_rate, 2),
        "accuracy": round(accuracy, 2),
        "throughput_checks_per_sec": round(throughput, 1),
        "mean_latency_us": round(mean_latency_us, 2),
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
    rep = run_jailbreakbench_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Samples Evaluated    : {rep['total_samples']}")
    print(f" Attack Interception Rate   : {rep['interception_rate']}% (Target: 100%)")
    print(f" Benign False Positive Rate : {rep['false_positive_rate']}% (Target: 0%)")
    print(f" Overall Security Accuracy  : {rep['accuracy']}%")
    print(f" Screening Throughput       : {rep['throughput_checks_per_sec']:,} checks/sec")
    print(f" Mean Latency               : {rep['mean_latency_us']} us")
    print("-" * 65)
    print(" Attack Vector Breakdown:")
    for cat, stats in rep["category_breakdown"].items():
        print(f"   - {cat:<24}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")
    print("=" * 65)
