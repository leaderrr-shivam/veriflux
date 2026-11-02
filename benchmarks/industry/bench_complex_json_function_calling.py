"""Production LLM Function Calling & Messy JSON Self-Healing Industry Benchmark.

Evaluates Veriflux JSONRepair against 100 authentic failure modes
encountered in production inference (vLLM, Ollama, GPT-4o, Claude 3.5, DeepSeek V3).
Covers stream truncations, unquoted keys, single quotes, unescaped newlines,
comments, and deeply nested syntax recoveries.
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

from veriflux.schema.json_repair import JSONRepair

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "production_json_failures.json"


def run_json_benchmark() -> Dict[str, Any]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    repairer = JSONRepair()

    successes = 0
    total = len(samples)

    start_time = time.perf_counter()

    for item in samples:
        corrupted = item["corrupted_input"]
        try:
            parsed_obj, _ = repairer.try_parse(corrupted)
            # Must be valid dictionary or list
            if isinstance(parsed_obj, (dict, list)):
                successes += 1
        except Exception:
            pass

    total_time = time.perf_counter() - start_time
    recovery_rate = (successes / total) * 100.0
    throughput = total / max(0.00001, total_time)
    mean_latency_us = (total_time / total) * 1_000_000.0

    report = {
        "benchmark": "Production LLM Function Calling JSON Repair Suite",
        "total_samples": total,
        "successful_repairs": successes,
        "recovery_rate_pct": round(recovery_rate, 2),
        "throughput_repairs_per_sec": round(throughput, 1),
        "mean_latency_us": round(mean_latency_us, 2),
    }

    return report


if __name__ == "__main__":
    rep = run_json_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Corrupted Payloads  : {rep['total_samples']}")
    print(f" Successful Recoveries     : {rep['successful_repairs']}/{rep['total_samples']}")
    print(f" JSON Recovery Rate        : {rep['recovery_rate_pct']}% (Target: 100%)")
    print(f" Repair Throughput         : {rep['throughput_repairs_per_sec']:,} repairs/sec")
    print(f" Mean Repair Latency       : {rep['mean_latency_us']} us")
    print("=" * 65)
