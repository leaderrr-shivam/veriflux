"""Download authentic, real-world public benchmark datasets directly from official sources.

Datasets downloaded:
1. RUCAIBox/HaluEval (NeurIPS 2023 official repository)
2. TruthfulQA (Oxford/OpenAI official benchmark repository)
3. Jailbreak_LLMs (ACM CCS 2024 official in-the-wild jailbreak prompts)
4. Jos de Jong jsonrepair (Official upstream test suite with real malformed JSON fixtures)
"""
from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "datasets" / "real_world"
DEST_DIR.mkdir(parents=True, exist_ok=True)

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Veriflux-Benchmark-Runner/1.0"
}


def download_file(url: str, dest_path: Path, max_lines: int | None = None) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        print(f"   [EXISTS] {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")
        return True

    print(f">> Downloading from: {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as response:
            if max_lines is None:
                content = response.read()
                dest_path.write_bytes(content)
            else:
                lines = []
                for _ in range(max_lines):
                    line = response.readline()
                    if not line:
                        break
                    lines.append(line)
                dest_path.write_bytes(b"".join(lines))
        size_kb = dest_path.stat().st_size / 1024.0
        print(f"   [SUCCESS] Saved {dest_path.name} ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"   [ERROR] Failed to download {url}: {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("DOWNLOADING AUTHENTIC REAL-WORLD BENCHMARK DATASETS FROM THE WEB")
    print("=" * 70)

    # 1. Official HaluEval QA Data (RUCAIBox/HaluEval NeurIPS 2023)
    halueval_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    download_file(halueval_url, DEST_DIR / "halueval_real_qa_500.jsonl", max_lines=500)

    # 2. Official TruthfulQA Benchmark (Oxford / OpenAI)
    truthful_url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    download_file(truthful_url, DEST_DIR / "truthfulqa_real.csv")

    # 3. Real In-the-Wild Jailbreak Prompts (ACM CCS 2024 dataset)
    jailbreak_url = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv"
    download_file(jailbreak_url, DEST_DIR / "jailbreak_prompts_real.csv")

    # 4. Jos de Jong jsonrepair Official Test Suite
    jsonrepair_url = "https://raw.githubusercontent.com/josdejong/jsonrepair/main/src/index.test.ts"
    download_file(jsonrepair_url, DEST_DIR / "jsonrepair_upstream_tests.ts")

    # 5. Microsoft Presidio Official PII Benchmark Dataset (1,500 real-world annotated samples)
    presidio_url = "https://raw.githubusercontent.com/microsoft/presidio-research/master/data/synth_dataset_v2.json"
    download_file(presidio_url, DEST_DIR / "presidio_synth_pii_1500.json")

    # 6. Official HaluEval Dialogue Data (RUCAIBox/HaluEval NeurIPS 2023)
    halueval_dial_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json"
    download_file(halueval_dial_url, DEST_DIR / "halueval_real_dialogue_500.jsonl", max_lines=500)

    # 7. Official HaluEval Summarization Data (RUCAIBox/HaluEval NeurIPS 2023)
    halueval_sum_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
    download_file(halueval_sum_url, DEST_DIR / "halueval_real_summarization_500.jsonl", max_lines=500)

    print("\nDataset downloads complete. Output folder:", DEST_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
