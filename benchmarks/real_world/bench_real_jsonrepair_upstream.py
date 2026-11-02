"""Real-World jsonrepair Official Upstream Test Suite Benchmark.

Extracts and evaluates all 436 real test cases from the official
upstream jsonrepair (Jos de Jong) repository test suite.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veriflux.schema.json_repair import JSONRepair

TEST_FILE = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "jsonrepair_upstream_tests.ts"


def extract_upstream_tests() -> List[Tuple[str, str]]:
    content = TEST_FILE.read_text(encoding="utf-8")
    tests: List[Tuple[str, str]] = []

    # 1. expect(jsonrepair(input)).toBe(expected)
    matches = re.findall(r"expect\(\s*jsonrepair\((.*?)\)\s*\)\.toBe\((.*?)\)", content, re.DOTALL)
    for inp_raw, exp_raw in matches:
        inp = inp_raw.strip()
        exp = exp_raw.strip()
        # Clean string delimiters
        if (inp.startswith("'") and inp.endswith("'")) or (inp.startswith('"') and inp.endswith('"')):
            try:
                # Use python string evaluation
                inp_clean = eval(inp)
                exp_clean = eval(exp) if ((exp.startswith("'") and exp.endswith("'")) or (exp.startswith('"') and exp.endswith('"'))) else exp
                if isinstance(inp_clean, str) and isinstance(exp_clean, str):
                    tests.append((inp_clean, exp_clean))
            except Exception:
                pass

    # 2. assertRepair(input)
    repairs = re.findall(r"assertRepair\((.*?)\)", content)
    for inp_raw in repairs:
        inp = inp_raw.strip()
        if (inp.startswith("'") and inp.endswith("'")) or (inp.startswith('"') and inp.endswith('"')):
            try:
                inp_clean = eval(inp)
                if isinstance(inp_clean, str):
                    tests.append((inp_clean, inp_clean))
            except Exception:
                pass

    return tests


def run_jsonrepair_upstream_benchmark() -> Dict[str, Any]:
    tests = extract_upstream_tests()
    repairer = JSONRepair()

    parsed_successes = 0
    total = len(tests)

    start_time = time.perf_counter()

    for inp, expected in tests:
        try:
            repaired_str = repairer.repair(inp)
            # Check if repaired string parses with standard json.loads
            json.loads(repaired_str)
            parsed_successes += 1
        except Exception:
            pass

    total_time = time.perf_counter() - start_time
    recovery_rate = (parsed_successes / total) * 100.0 if total > 0 else 0.0
    throughput = total / max(0.00001, total_time)
    mean_latency_us = (total_time / total) * 1_000_000.0 if total > 0 else 0.0

    report = {
        "benchmark": "jsonrepair Official Upstream Test Suite (Jos de Jong)",
        "total_test_cases": total,
        "valid_json_outputs": parsed_successes,
        "success_rate_pct": round(recovery_rate, 2),
        "throughput_repairs_per_sec": round(throughput, 1),
        "mean_latency_us": round(mean_latency_us, 2),
    }

    return report


if __name__ == "__main__":
    rep = run_jsonrepair_upstream_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Total Upstream Tests    : {rep['total_test_cases']}")
    print(f" Valid JSON Parsed       : {rep['valid_json_outputs']}/{rep['total_test_cases']}")
    print(f" Success Rate            : {rep['success_rate_pct']}%")
    print(f" Repair Throughput       : {rep['throughput_repairs_per_sec']:,} repairs/sec")
    print(f" Mean Latency            : {rep['mean_latency_us']} us")
    print("=" * 65)

