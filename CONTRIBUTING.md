# Contributing to Veriflux

Thank you for your interest in contributing to **Veriflux**!  
We are building a legacy-grade, zero-dependency foundation for AI/ML reliability, evaluation, and guardrails.

---

## 🏛️ Core Principles for Contributions

1. **Zero External AI/ML Dependencies:**
   - The core library must run strictly on the Python Standard Library (3.10+).
   - No importing or wrapping third-party ML frameworks (no PyTorch, Transformers, LangChain, etc.) in core logic.
   - Every algorithm and scoring heuristic must be implemented from first principles.

2. **Deterministic & Blazing Fast:**
   - Optimize for low-latency, zero-copy buffers, and minimal memory allocations.
   - Algorithms should provide linear or sub-linear complexity where possible.

3. **Production-Grade Engineering:**
   - Every module must include `from __future__ import annotations`.
   - Comprehensive type hints throughout (strict mypy compliant).
   - Dataclasses must use `slots=True` and `frozen=True` where immutability is semantically sound.
   - Full test coverage for every added feature.

---

## 🚀 Development Workflow

### 1. Clone & Set Up

```bash
git clone https://github.com/veriflux/veriflux.git
cd veriflux
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### 2. Running Tests

```bash
python tests/run_all.py
```

### 3. Formatting & Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

---

## 📝 Pull Request Guidelines

- Ensure all 47+ unit tests pass without errors.
- Include unit tests for every newly introduced function or class.
- Update `CHANGELOG.md` with your additions or bug fixes.
- Keep pull requests focused on a single engine or architectural subsystem.

