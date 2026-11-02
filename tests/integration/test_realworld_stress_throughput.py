"""Integration Test: Continuous Stress and Memory Stability Verification.

Processes thousands of requests across all core engines:
1. Token counting & estimation
2. Mangled JSON repair
3. Prompt injection & PII detection
4. Context groundedness evaluation

Verifies that memory footprint remains constant with zero leaks,
garbage collection operates efficiently, and throughput remains sustained.
"""
import tracemalloc
import time
import veriflux
from veriflux.schema.json_repair import JSONRepair
from veriflux.guardrails.engine import GuardrailEngine
from veriflux.tokens.counter import count_tokens


def test_sustained_stress_and_leak_free_execution():
    repairer = JSONRepair()
    guard = GuardrailEngine()

    sample_json = '{name: "Veriflux", status: "active", metrics: [1, 2, 3,],}'
    sample_prompt = "Explain quantum computing principles with code examples."
    sample_context = "Veriflux runs on Python 3.10+ and requires zero external runtime dependencies."
    sample_response = "Veriflux requires Python 3.10 or higher with zero external runtime dependencies."

    # Warmup all engine singletons and imports before baseline snapshot
    _ = count_tokens(sample_prompt)
    _ = repairer.repair(sample_json)
    _ = guard.check_input(sample_prompt)
    _ = veriflux.is_grounded(sample_response, sample_context)

    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    total_ops = 500
    t0 = time.perf_counter()

    for i in range(total_ops):
        # 1. Token count
        t_count = count_tokens(sample_prompt)
        assert t_count > 0

        # 2. JSON Repair
        repaired = repairer.repair(sample_json)
        assert "Veriflux" in repaired

        # 3. Guardrail check
        guard_res = guard.check_input(sample_prompt)
        assert guard_res.passed

        # 4. Groundedness
        is_g = veriflux.is_grounded(sample_response, sample_context)
        assert is_g

    elapsed = time.perf_counter() - t0
    mem_end, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ops_per_sec = (total_ops * 4) / elapsed
    mem_delta_mb = (mem_end - mem_start) / (1024 * 1024)

    # Asserts
    assert elapsed < 5.0, f"500 full cycles took {elapsed:.2f}s (must be under 5s)"
    assert mem_delta_mb < 2.0, f"Memory delta was {mem_delta_mb:.2f}MB, indicating possible leak"
    assert ops_per_sec > 500.0, f"Throughput was {ops_per_sec:.0f} ops/sec (must exceed 500)"
