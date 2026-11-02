"""Standard-library test runner executing all Veriflux test suites without external dependencies."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
tests_dir = root_dir / "tests"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(tests_dir))


class Approx:
    """Mock for pytest.approx."""
    def __init__(self, expected: float, abs: float = 1e-4) -> None:
        self.expected = expected
        self.abs = abs

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return abs(float(other) - self.expected) <= self.abs
        return False

    def __repr__(self) -> str:
        return f"Approx({self.expected} ± {self.abs})"


# Install mock pytest in sys.modules so 'import pytest' works cleanly without network installation
class MockPytestModule:
    @staticmethod
    def approx(val: float, abs: float = 1e-4) -> Approx:
        return Approx(val, abs=abs)

    @staticmethod
    def fixture(fn=None):
        if fn:
            return fn
        return lambda f: f


sys.modules["pytest"] = MockPytestModule()  # type: ignore

SAMPLE_CONTEXT = (
    "The Apollo program was the third United States human spaceflight program carried out by NASA. "
    "First conceived in 1960 during the Eisenhower administration, Apollo succeeded in landing the "
    "first humans on the Moon in 1969 with Apollo 11. Neil Armstrong and Buzz Aldrin walked on the lunar surface."
)


def run_test_module(module_path: Path) -> tuple[int, int, list[str]]:
    rel_path = module_path.relative_to(root_dir)
    mod_name = module_path.stem
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        return 0, 1, [f"Failed to load spec for {rel_path}"]

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return 0, 1, [f"Error importing {rel_path}: {exc}\n{traceback.format_exc()}"]

    passed = 0
    failed = 0
    errors: list[str] = []

    # Find all test_* functions
    test_funcs = [
        (name, func)
        for name, func in inspect.getmembers(mod, inspect.isfunction)
        if name.startswith("test_")
    ]

    for name, func in test_funcs:
        sig = inspect.signature(func)
        kwargs = {}
        # Inject known fixtures
        if "sample_context" in sig.parameters:
            kwargs["sample_context"] = SAMPLE_CONTEXT
        if "tmp_path" in sig.parameters or "temp_baseline_dir" in sig.parameters:
            temp_dir = tempfile.mkdtemp(prefix="veriflux_test_")
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = Path(temp_dir)
            if "temp_baseline_dir" in sig.parameters:
                kwargs["temp_baseline_dir"] = temp_dir

        t_start = time.perf_counter()
        try:
            func(**kwargs)
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"  [PASS] {name} ({dur_ms:.1f}ms)")
            passed += 1
        except Exception as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"  [FAIL] {name} ({dur_ms:.1f}ms): {exc}")
            errors.append(f"{name}: {exc}\n{traceback.format_exc()}")
            failed += 1

    return passed, failed, errors


def main() -> int:
    unit_dir = tests_dir / "unit"
    integration_dir = tests_dir / "integration"
    
    test_files = sorted(unit_dir.glob("test_*.py"))
    if integration_dir.exists():
        test_files.extend(sorted(integration_dir.glob("test_*.py")))

    print("=" * 60)
    print("[Veriflux] TEST RUNNER (Zero-Dependency Execution)")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    all_errors = []
    start_time = time.perf_counter()

    for tf in test_files:
        print(f"\nRunning {tf.name}:")
        p, f, errs = run_test_module(tf)
        total_passed += p
        total_failed += f
        all_errors.extend(errs)

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("=" * 60)

    if total_failed > 0:
        print("\nFailures:")
        for err in all_errors:
            print("-" * 40)
            print(err)
        return 1

    print("\nALL VERIFLUX TESTS PASSED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
