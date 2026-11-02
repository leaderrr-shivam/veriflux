"""Real-World Microsoft Presidio PII Benchmark.

Evaluates Veriflux PIIGuard against 1,500 annotated multi-lingual PII samples
from the official Microsoft Presidio Research evaluation benchmark
(microsoft/presidio-research/data/synth_dataset_v2.json).
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

from veriflux.guardrails.pii_detector import PIIGuard

DATA_FILE = Path(__file__).resolve().parent.parent / "datasets" / "real_world" / "presidio_synth_pii_1500.json"

TYPE_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "IP_ADDRESS": "IPV4",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "STREET_ADDRESS": "ADDRESS",
    "ZIP_CODE": "ZIP_CODE",
}


def run_presidio_pii_benchmark(max_samples: int = 500) -> Dict[str, Any]:
    with open(DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
        dataset = json.load(f)

    samples = dataset[:max_samples]
    guard = PIIGuard()
    eval_entities = set(TYPE_MAP.keys())

    tp = 0
    gt_total = 0
    det_total = 0

    start_time = time.perf_counter()

    for item in samples:
        text = item["full_text"]
        gt_spans = [s for s in item.get("spans", []) if s.get("entity_type") in eval_entities]
        gt_total += len(gt_spans)

        detected = guard.detect(text)
        det_total += len(detected)

        for gt in gt_spans:
            gt_start = gt["start_position"]
            gt_end = gt["end_position"]
            matched = any(max(gt_start, d.start) < min(gt_end, d.end) for d in detected)
            if matched:
                tp += 1

    total_time = time.perf_counter() - start_time

    recall = (tp / max(1, gt_total)) * 100.0
    precision = (tp / max(1, det_total)) * 100.0
    f1 = (2 * precision * recall) / max(0.001, precision + recall)

    throughput = len(samples) / max(0.00001, total_time)
    mean_latency_us = (total_time / len(samples)) * 1_000_000.0

    report = {
        "benchmark": "Microsoft Presidio Official PII Benchmark",
        "samples_evaluated": len(samples),
        "ground_truth_entities": gt_total,
        "true_positives": tp,
        "total_detected": det_total,
        "recall_pct": round(recall, 2),
        "precision_pct": round(precision, 2),
        "f1_score_pct": round(f1, 2),
        "throughput_samples_per_sec": round(throughput, 1),
        "mean_latency_us": round(mean_latency_us, 2),
        "cost_usd": 0.00,
    }

    return report


if __name__ == "__main__":
    rep = run_presidio_pii_benchmark()
    print("=" * 65)
    print(f"  {rep['benchmark']}")
    print("=" * 65)
    print(f" Samples Evaluated       : {rep['samples_evaluated']}")
    print(f" Ground Truth Entities   : {rep['ground_truth_entities']}")
    print(f" Intercepted PII (TP)    : {rep['true_positives']}/{rep['ground_truth_entities']}")
    print(f" Detection Recall Rate   : {rep['recall_pct']}%")
    print(f" Redaction Precision     : {rep['precision_pct']}%")
    print(f" PII F1-Score            : {rep['f1_score_pct']}%")
    print(f" Screening Throughput    : {rep['throughput_samples_per_sec']:,} samples/sec")
    print(f" Mean Latency            : {rep['mean_latency_us']} us")
    print(f" Inference Cost          : ${rep['cost_usd']:.2f} (Zero External Dependencies)")
    print("=" * 65)

