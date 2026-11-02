# 🔱 VERIFLUX — Complete Architectural Blueprint

> *"Truth flows through every layer."*  
> **Veriflux** — The definitive AI/ML testing, evaluation, guardrail, and reliability library.  
> Built entirely from scratch. Zero dependency on existing AI/ML frameworks.

---

## 🏛️ Engineering Philosophy

### Core Principles
1. **Zero AI/ML Library Dependencies** — Every algorithm, every data structure, every scoring function is written from first principles. No importing from LangChain, Ragas, DeepEval, Transformers, or any existing evaluation library. Veriflux IS the engine.
2. **From Simple to Complex** — A beginner can write `assert veriflux.is_grounded(response, context)` in 1 line. An enterprise architect can orchestrate multi-stage evaluation pipelines with custom scoring kernels.
3. **Blazing Performance** — Custom C-extension-accelerated tokenizers, memory-mapped data structures, async-native throughout, zero-copy buffer management, and SIMD-optimized vector operations where possible.
4. **Pytest-Native** — Not a separate ecosystem. A pytest plugin that makes AI testing feel as natural as unit testing.
5. **Provider-Agnostic** — Works with OpenAI, Anthropic, Google, local models (Ollama), and raw HTTP endpoints. No vendor lock-in.

---

## 🧬 System Architecture (12 Core Engines)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VERIFLUX PUBLIC API                         │
│   veriflux.test() | veriflux.guard() | veriflux.trace() | ...     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  pytest   │ │   CLI    │ │ REST API │ │ CI/CD    │ │  Python  │ │
│  │  Plugin   │ │  Runner  │ │  Server  │ │  Gates   │ │  SDK     │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │             │            │             │            │       │
│  ─────┴─────────────┴────────────┴─────────────┴────────────┴───── │
│                     ORCHESTRATION LAYER                             │
│              (Pipeline, DAG Executor, Scheduler)                    │
│  ──────────────────────────────────────────────────────────────── │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ENGINE LAYER (12 Engines)                  │   │
│  │                                                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │   │
│  │  │  NLP    │ │SEMANTIC │ │HALLUCI- │ │   CONTEXT       │   │   │
│  │  │ ENGINE  │ │SIMILAR- │ │NATION   │ │   ENGINEERING   │   │   │
│  │  │         │ │ITY ENG. │ │DETECTOR │ │   ENGINE        │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘   │   │
│  │                                                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │   │
│  │  │ AGENT   │ │STRUCT.  │ │GUARD-   │ │ COST &          │   │   │
│  │  │TRAJECT- │ │OUTPUT   │ │RAILS    │ │ PERFORMANCE     │   │   │
│  │  │ORY ANLZ │ │VALIDATR │ │ ENGINE  │ │ PROFILER        │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘   │   │
│  │                                                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │   │
│  │  │CI/CD    │ │REGRESS- │ │ TOKEN   │ │ PROVIDER        │   │   │
│  │  │INTEGRA- │ │ION DET- │ │MANAGMNT │ │ ABSTRACTION     │   │   │
│  │  │TION LYR │ │ECTION   │ │ ENGINE  │ │ LAYER           │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ──────────────────────────────────────────────────────────────── │
│                     FOUNDATION LAYER                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │  Memory  │ │  Async   │ │  Config  │ │  Logger  │             │
│  │  Pool    │ │  Runtime │ │  System  │ │  System  │             │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │  Cache   │ │  Serial- │ │  Crypto  │ │  Math    │             │
│  │  System  │ │  ization │ │  Utils   │ │  Kernels │             │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
veriflux/
├── pyproject.toml                    # Modern Python packaging (PEP 621)
├── LICENSE                           # Apache 2.0
├── README.md                         # Stunning GitHub README
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
│
├── src/
│   └── veriflux/
│       ├── __init__.py               # Public API surface
│       ├── __version__.py            # Version management
│       ├── _types.py                 # Core type definitions
│       ├── exceptions.py             # Custom exception hierarchy
│       │
│       ├── core/                     # ═══ FOUNDATION LAYER ═══
│       │   ├── __init__.py
│       │   ├── memory.py             # Zero-copy memory pool, arena allocator
│       │   ├── async_runtime.py      # Custom async event loop integration
│       │   ├── config.py             # Hierarchical config (file + env + code)
│       │   ├── logger.py             # Structured logging with trace correlation
│       │   ├── cache.py              # LRU/LFU/TTL multi-tier caching
│       │   ├── serialization.py      # Fast binary serialization (custom msgpack-like)
│       │   ├── crypto.py             # HMAC, hashing for data integrity
│       │   ├── math_kernels.py       # SIMD-optimized cosine similarity, dot product
│       │   ├── timer.py              # High-precision nanosecond timing
│       │   ├── pool.py               # Thread/process pool management
│       │   └── registry.py           # Plugin/extension registry system
│       │
│       ├── nlp/                      # ═══ ENGINE 1: NLP ENGINE ═══
│       │   ├── __init__.py           # (Built from scratch — no NLTK, spaCy, etc.)
│       │   ├── tokenizer.py          # BPE/WordPiece/SentencePiece tokenizers
│       │   ├── sentence_splitter.py  # Rule-based + statistical sentence segmentation
│       │   ├── claim_extractor.py    # Atomic claim decomposition from text
│       │   ├── entity_extractor.py   # Named entity recognition (rule + pattern)
│       │   ├── pos_tagger.py         # Part-of-speech tagging
│       │   ├── text_normalizer.py    # Unicode normalization, whitespace, casing
│       │   ├── stemmer.py            # Porter/Snowball stemming algorithms
│       │   ├── stopwords.py          # Multi-language stopword lists
│       │   ├── ngrams.py             # N-gram generation and analysis
│       │   └── fingerprint.py        # MinHash/SimHash text fingerprinting
│       │
│       ├── similarity/              # ═══ ENGINE 2: SEMANTIC SIMILARITY ═══
│       │   ├── __init__.py
│       │   ├── tfidf.py              # TF-IDF vectorizer (from scratch)
│       │   ├── bm25.py               # BM25/BM25+ ranking algorithm
│       │   ├── cosine.py             # Optimized cosine similarity
│       │   ├── jaccard.py            # Jaccard/MinHash similarity
│       │   ├── levenshtein.py        # Edit distance algorithms
│       │   ├── embedding_bridge.py   # Interface to external embedding APIs
│       │   ├── vector_store.py       # In-memory vector index (HNSW-inspired)
│       │   ├── cross_encoder.py      # Lightweight cross-encoder scoring
│       │   └── fusion.py             # Multi-signal score fusion (RRF, weighted)
│       │
│       ├── hallucination/           # ═══ ENGINE 3: HALLUCINATION DETECTION ═══
│       │   ├── __init__.py
│       │   ├── detector.py           # Main hallucination detection orchestrator
│       │   ├── claim_verifier.py     # Verify claims against context
│       │   ├── entailment.py         # NLI-based textual entailment checker
│       │   ├── url_validator.py      # Detect fabricated URLs/citations
│       │   ├── fact_consistency.py   # Cross-reference consistency scoring
│       │   ├── confidence_scorer.py  # Calibrated confidence estimation
│       │   └── groundedness.py       # Context groundedness analysis
│       │
│       ├── context/                 # ═══ ENGINE 4: CONTEXT ENGINEERING ═══
│       │   ├── __init__.py
│       │   ├── engine.py             # Context assembly & optimization engine
│       │   ├── compressor.py         # Multi-strategy context compression
│       │   ├── budget.py             # Token budget manager
│       │   ├── ranker.py             # Context chunk relevance ranking
│       │   ├── pruner.py             # Intelligent context pruning
│       │   ├── deduplicator.py       # Semantic deduplication
│       │   ├── summarizer.py         # Extractive summarization (from scratch)
│       │   └── window.py             # Sliding window management
│       │
│       ├── trajectory/              # ═══ ENGINE 5: AGENT TRAJECTORY ANALYZER ═══
│       │   ├── __init__.py
│       │   ├── tracer.py             # Zero-overhead agent execution tracer
│       │   ├── recorder.py           # Trace recording & serialization
│       │   ├── replayer.py           # Deterministic trace replay engine
│       │   ├── analyzer.py           # Trajectory pattern analysis
│       │   ├── loop_detector.py      # Infinite loop / cycle detection
│       │   ├── cost_tracker.py       # Per-step cost attribution
│       │   ├── state_inspector.py    # State diff & mutation tracking
│       │   └── graph.py              # Execution graph builder (DAG)
│       │
│       ├── schema/                  # ═══ ENGINE 6: STRUCTURED OUTPUT VALIDATOR ═══
│       │   ├── __init__.py
│       │   ├── validator.py          # JSON/structured output validation
│       │   ├── type_coercer.py       # Intelligent type coercion
│       │   ├── schema_builder.py     # Runtime schema construction
│       │   ├── json_repair.py        # Malformed JSON repair engine
│       │   ├── diff_engine.py        # Schema diff detection
│       │   └── constraints.py        # Custom constraint definitions
│       │
│       ├── guardrails/              # ═══ ENGINE 7: GUARDRAILS ENGINE ═══
│       │   ├── __init__.py
│       │   ├── engine.py             # Main guardrail orchestrator
│       │   ├── toxicity.py           # Toxicity detection (lexicon + pattern)
│       │   ├── pii_detector.py       # PII/PHI detection & redaction
│       │   ├── prompt_injection.py   # Prompt injection attack detection
│       │   ├── bias_detector.py      # Bias detection scoring
│       │   ├── content_filter.py     # Configurable content filtering
│       │   ├── rate_limiter.py       # Token/request rate limiting
│       │   └── policy.py             # Declarative policy definitions
│       │
│       ├── profiler/                # ═══ ENGINE 8: COST & PERFORMANCE PROFILER ═══
│       │   ├── __init__.py
│       │   ├── cost_calculator.py    # Multi-provider cost calculation
│       │   ├── latency_tracker.py    # TTFT, ITL, E2E latency tracking
│       │   ├── throughput.py         # Tokens/sec throughput measurement
│       │   ├── memory_profiler.py    # Memory usage tracking
│       │   ├── pricing.py            # Up-to-date provider pricing database
│       │   └── report.py             # Performance report generation
│       │
│       ├── cicd/                    # ═══ ENGINE 9: CI/CD INTEGRATION ═══
│       │   ├── __init__.py
│       │   ├── gates.py              # Quality gate definitions
│       │   ├── github_actions.py     # GitHub Actions integration
│       │   ├── gitlab_ci.py          # GitLab CI integration
│       │   ├── reporters.py          # JUnit XML, JSON, HTML reporters
│       │   ├── badge.py              # Dynamic badge generation
│       │   └── artifacts.py          # Test artifact management
│       │
│       ├── regression/              # ═══ ENGINE 10: REGRESSION DETECTION ═══
│       │   ├── __init__.py
│       │   ├── baseline.py           # Baseline snapshot management
│       │   ├── comparator.py         # Multi-metric comparison engine
│       │   ├── drift_detector.py     # Semantic drift detection
│       │   ├── alerting.py           # Regression alert system
│       │   └── history.py            # Historical metrics storage
│       │
│       ├── tokens/                  # ═══ ENGINE 11: TOKEN MANAGEMENT ═══
│       │   ├── __init__.py
│       │   ├── counter.py            # Fast token counting (all major models)
│       │   ├── estimator.py          # Token count estimation without encoding
│       │   ├── budget_manager.py     # Token budget allocation
│       │   ├── encodings.py          # BPE encoding tables (cl100k, o200k, etc.)
│       │   └── optimizer.py          # Prompt token optimization
│       │
│       ├── providers/               # ═══ ENGINE 12: PROVIDER ABSTRACTION ═══
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract provider interface
│       │   ├── openai.py             # OpenAI API client (from scratch)
│       │   ├── anthropic.py          # Anthropic API client (from scratch)
│       │   ├── google.py             # Google Gemini API client (from scratch)
│       │   ├── ollama.py             # Ollama local model client
│       │   ├── http.py               # Generic HTTP/OpenAI-compatible endpoint
│       │   ├── mock.py               # Deterministic mock provider for testing
│       │   └── router.py             # Intelligent provider routing
│       │
│       ├── assertions/              # ═══ PYTEST ASSERTION LIBRARY ═══
│       │   ├── __init__.py
│       │   ├── semantic.py           # is_grounded, is_relevant, is_faithful
│       │   ├── structural.py         # schema_matches, is_valid_json, has_fields
│       │   ├── safety.py             # no_toxicity, no_pii, no_injection
│       │   ├── performance.py        # latency_under, cost_under, tokens_under
│       │   ├── factual.py            # no_hallucinations, claims_verified
│       │   ├── behavioral.py         # tone_matches, language_is, follows_instructions
│       │   └── composite.py          # Combine multiple assertions
│       │
│       ├── pytest_plugin/           # ═══ PYTEST PLUGIN ═══
│       │   ├── __init__.py
│       │   ├── plugin.py             # pytest plugin registration
│       │   ├── fixtures.py           # Built-in fixtures (veriflux_client, etc.)
│       │   ├── markers.py            # @veriflux.test, @veriflux.fast, @veriflux.nightly
│       │   ├── collector.py          # Custom test collection
│       │   └── reporter.py           # Custom pytest reporting
│       │
│       ├── datasets/                # ═══ DATASET MANAGEMENT ═══
│       │   ├── __init__.py
│       │   ├── loader.py             # JSONL/CSV/Parquet dataset loading
│       │   ├── generator.py          # Synthetic adversarial test data generation
│       │   ├── sampler.py            # Statistical sampling strategies
│       │   └── golden.py             # Golden dataset management
│       │
│       ├── cli/                     # ═══ CLI INTERFACE ═══
│       │   ├── __init__.py
│       │   ├── main.py               # CLI entry point
│       │   ├── commands/
│       │   │   ├── run.py             # veriflux run
│       │   │   ├── init.py            # veriflux init
│       │   │   ├── baseline.py        # veriflux baseline
│       │   │   ├── report.py          # veriflux report
│       │   │   └── bench.py           # veriflux bench
│       │   └── formatters.py         # Terminal output formatting
│       │
│       └── contrib/                 # ═══ COMMUNITY EXTENSIONS ═══
│           ├── __init__.py
│           └── hooks.py              # Extension hook system
│
├── tests/                            # ═══ COMPREHENSIVE TEST SUITE ═══
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_nlp/
│   │   ├── test_similarity/
│   │   ├── test_hallucination/
│   │   ├── test_context/
│   │   ├── test_trajectory/
│   │   ├── test_schema/
│   │   ├── test_guardrails/
│   │   ├── test_profiler/
│   │   ├── test_tokens/
│   │   └── test_providers/
│   ├── integration/
│   └── benchmarks/
│
├── docs/                             # ═══ DOCUMENTATION ═══
│   ├── getting-started.md
│   ├── architecture.md
│   ├── api-reference/
│   ├── guides/
│   └── examples/
│
└── benchmarks/                       # ═══ PERFORMANCE BENCHMARKS ═══
    ├── bench_tokenizer.py
    ├── bench_similarity.py
    └── bench_memory.py
```

---

## 🔬 Engine Deep Dives

### ENGINE 1: NLP Engine (from scratch)
No NLTK, no spaCy, no Hugging Face tokenizers. Veriflux ships its own:

- **BPE Tokenizer**: Byte-Pair Encoding implementation supporting cl100k_base, o200k_base, and custom vocabularies. Memory-mapped vocabulary files for instant loading.
- **Sentence Splitter**: Hybrid rule-based (abbreviation-aware) + statistical boundary detection. Handles edge cases like "Dr. Smith went to Washington, D.C. He liked it."
- **Claim Extractor**: Decomposes complex text into atomic, verifiable claims using syntactic patterns and dependency-like analysis. Critical for hallucination detection.
- **Text Fingerprinting**: MinHash and SimHash for near-duplicate detection at O(1) comparison time.

### ENGINE 2: Semantic Similarity Engine
- **TF-IDF Vectorizer**: Sparse matrix implementation with sublinear TF, IDF smoothing, and L2 normalization. Custom sparse dot product for blazing speed.
- **BM25+ Ranker**: State-of-the-art lexical ranking with length normalization and term frequency saturation.
- **HNSW Vector Index**: Hierarchical Navigable Small World graph for approximate nearest neighbor search on dense embeddings. Pure Python with optional C acceleration.
- **Score Fusion**: Reciprocal Rank Fusion (RRF) and configurable weighted linear combination for hybrid retrieval scoring.

### ENGINE 3: Hallucination Detection Engine
The crown jewel. A multi-stage pipeline:

```
[LLM Response] 
    │
    ▼
[1. Claim Extraction] ─── Break into atomic claims
    │
    ▼
[2. Claim Classification] ─── Factual / Subjective / Unsupported
    │
    ▼
[3. Evidence Retrieval] ─── Find supporting context for each claim
    │
    ▼
[4. Entailment Scoring] ─── Does evidence support, contradict, or not mention?
    │
    ▼
[5. URL/Citation Validation] ─── Are referenced URLs/papers real?
    │
    ▼
[6. Consistency Check] ─── Internal self-consistency of response
    │
    ▼
[7. Confidence Calibration] ─── Calibrated probability of hallucination
    │
    ▼
[VERDICT: grounded / partially_grounded / hallucinated]
```

### ENGINE 4: Context Engineering Engine
- **Multi-Strategy Compression**: Extractive (sentence importance scoring), compressive (remove filler/redundancy), and abstractive (key information extraction) — all without LLM calls.
- **Token Budget Manager**: Allocates token budgets across system prompt, chat history, RAG context, and tool results with configurable priorities and minimum token reservations.
- **Relevance Ranker**: BM25 + TF-IDF + recency decay + source priority scoring for intelligent context assembly.

### ENGINE 5: Agent Trajectory Analyzer
- **Zero-overhead Tracer**: Uses Python's `sys.settrace` + `contextvars` for near-zero-overhead execution tracing of agent steps.
- **Cycle Detector**: Floyd's + Brent's algorithms adapted for state-space cycle detection in agent execution graphs.
- **Replay Engine**: Fully deterministic replay with step-level intervention (override tool results, inject errors, what-if analysis).
- **DataFrame Export**: Export entire traces to native Python dicts / CSV for programmatic analysis.

### ENGINE 6: Structured Output Validator
- **JSON Repair Engine**: Handles truncated JSON, missing brackets, trailing commas, single quotes, unquoted keys — fixes 95%+ of common LLM JSON malformations.
- **Type Coercion**: Intelligent coercion ("4.5" → 4.5, "true" → True, "null" → None) with strict/loose modes.
- **Schema Diff Engine**: Detect exactly what changed between expected and actual output schemas.

### ENGINE 7: Guardrails Engine
- **Toxicity Detection**: Multi-tier lexicon-based + regex pattern + contextual analysis. No external API calls needed.
- **PII Detector**: 50+ PII patterns (SSN, credit cards, emails, phone numbers, medical IDs, etc.) with configurable sensitivity levels and redaction modes.
- **Prompt Injection Detection**: Signature-based + heuristic detection of common injection patterns (ignore previous instructions, system prompt leakage, etc.)
- **Policy Engine**: Declarative YAML-based policy definitions for enterprise compliance.

### ENGINE 8: Cost & Performance Profiler
- **Provider Pricing Database**: Built-in, versioned pricing data for all major LLM providers (OpenAI, Anthropic, Google, etc.) with automatic cost calculation per request.
- **Latency Tracking**: TTFT (Time to First Token), ITL (Inter-Token Latency), E2E (End-to-End) with percentile analysis (p50, p95, p99).
- **Memory Profiling**: Track peak memory, allocation patterns, and garbage collection impact.

### ENGINE 9: CI/CD Integration Layer
- **Quality Gates**: Configurable pass/fail thresholds for any metric (faithfulness > 0.85, cost < $0.01, latency < 2s).
- **GitHub Actions**: Native action YAML generator + PR comment annotations with evaluation results.
- **Report Formats**: JUnit XML (for CI), JSON (for APIs), HTML (for humans), Markdown (for PRs).
- **Dynamic Badges**: SVG badge generation showing evaluation scores for README.

---

## 🎯 API Design (Simple → Complex)

### Level 1: One-Liner (Beginner)
```python
import veriflux

# Simple assertion
assert veriflux.is_grounded("Paris is the capital of France", context="France's capital city is Paris")
assert veriflux.is_valid_json('{"name": "Alice", "age": 30}')
assert veriflux.no_pii("Contact us for more information")
assert veriflux.token_count("Hello world", model="gpt-4o") == 2
```

### Level 2: Pytest Integration (Intermediate)
```python
import veriflux
from veriflux.assertions import semantic, structural, safety, performance

@veriflux.test(dataset="test_cases.jsonl")
def test_qa_pipeline(case):
    response = my_llm_call(case.query, case.context)
    
    assert semantic.is_grounded(response, case.context)
    assert semantic.is_relevant(response, case.query)
    assert structural.schema_matches(response, AnswerSchema)
    assert safety.no_toxicity(response)
    assert safety.no_pii(response)
    assert performance.latency_under(response, seconds=2.0)
    assert performance.cost_under(response, usd=0.01)

@veriflux.test(layer="fast")  # Zero-cost, runs on every PR
def test_prompt_structure():
    assert structural.has_required_sections(my_prompt, ["system", "user"])
    assert veriflux.token_count(my_prompt) < 4000
```

### Level 3: Full Pipeline (Advanced)
```python
from veriflux import Pipeline, Stage, Gate
from veriflux.hallucination import HallucinationDetector
from veriflux.context import ContextEngine, TokenBudget
from veriflux.guardrails import GuardrailEngine, ToxicityGuard, PIIGuard
from veriflux.profiler import CostCalculator
from veriflux.regression import BaselineManager

# Build a full evaluation pipeline
pipeline = Pipeline("production_eval")

pipeline.add_stage(Stage(
    name="structural_checks",
    layer="fast",
    checks=[
        lambda r: r.is_valid_json(),
        lambda r: r.token_count < 2000,
    ]
))

pipeline.add_stage(Stage(
    name="semantic_checks", 
    layer="standard",
    checks=[
        HallucinationDetector(
            strategy="claim_decomposition",
            entailment_threshold=0.7,
            url_validation=True,
        ),
    ]
))

pipeline.add_stage(Stage(
    name="safety_checks",
    layer="standard",
    checks=[
        ToxicityGuard(threshold=0.1),
        PIIGuard(categories=["ssn", "email", "phone"], action="redact"),
    ]
))

pipeline.add_gate(Gate(
    name="production_ready",
    conditions={
        "hallucination_score": ("<=", 0.1),
        "toxicity_score": ("<=", 0.05),
        "avg_latency_ms": ("<=", 2000),
        "avg_cost_usd": ("<=", 0.01),
    }
))

# Run
results = pipeline.evaluate(dataset="golden_test_set.jsonl")
results.to_junit_xml("reports/eval_results.xml")
results.to_github_pr_comment()
```

### Level 4: Agent Tracing (Expert)
```python
from veriflux.trajectory import Tracer, Replay

# Instrument any agent
with Tracer(name="flight_booking") as t:
    result = my_agent.run("Book a flight to NYC next Friday")

trace = t.trace

# Programmatic analysis
assert trace.total_steps < 50
assert trace.total_cost < 1.00
assert not trace.has_cycles(window=5)
assert trace.tool_calls["search_flights"].success_rate > 0.8

# Export for analysis
df = trace.to_dataframe()
print(df[df.step_type == "llm_call"].sort_values("cost", ascending=False))

# Replay with modifications
replay = Replay(trace)
replay.mock_step(12, tool_result={"error": "API timeout"})
new_result = replay.run()  # Test error handling
```

### Level 5: Context Engineering (Expert)
```python
from veriflux.context import ContextEngine, Source, TokenBudget

engine = ContextEngine(
    budget=TokenBudget(max_tokens=8000, reserve_output=2000),
    strategy="task_aware",
)

engine.add(Source.system(system_prompt, priority="critical"))
engine.add(Source.history(messages, decay="exponential"))
engine.add(Source.rag(chunks, rank_by="relevance"))
engine.add(Source.tools(tool_results, priority="recency"))

ctx = engine.build(query="What was Q3 revenue?")

print(f"Tokens: {ctx.token_count}")           # 5,847
print(f"Compression: {ctx.compression_ratio}") # 3.2x
print(f"Quality: {ctx.quality_score}")         # 0.94
print(f"Sources kept: {ctx.sources_retained}") # 12/47
```

---

## ⚡ Performance Architecture

### Memory Management
- **Arena Allocator**: Pre-allocated memory pools for tokenization buffers, reducing GC pressure
- **Memory-Mapped Files**: Vocabulary tables and embedding indices loaded via `mmap` for instant startup
- **Zero-Copy Buffers**: `memoryview` and buffer protocol throughout for zero-copy data passing

### Computation
- **SIMD-Optimized Math**: Cosine similarity, dot products via optimized NumPy-free pure Python with optional C extension fallback
- **Sparse Matrix Operations**: Custom compressed sparse row (CSR) format for TF-IDF vectors
- **Lazy Evaluation**: Computation graphs are built lazily and executed only when results are accessed
- **Parallel Execution**: `concurrent.futures` + custom work-stealing scheduler for CPU-bound tasks

### Caching
- **Multi-Tier Cache**: L1 (in-memory LRU) → L2 (disk-backed) → L3 (optional Redis)
- **Content-Addressable Storage**: Cache keys derived from content hashes for automatic deduplication
- **Cache Warming**: Pre-populate caches for known evaluation datasets

---

## 🔐 Security Architecture

- **No Eval/Exec**: Zero use of `eval()`, `exec()`, or `compile()` on any user data
- **Input Sanitization**: All text inputs sanitized before processing
- **Secrets Management**: API keys stored in environment variables, never logged
- **Data Isolation**: Each test run operates in isolated memory spaces
- **Audit Trail**: Cryptographic hashes of all evaluation inputs/outputs for tamper detection
- **Supply Chain**: Minimal dependencies, all vendored or audited

---

## 📦 Dependency Philosophy

### Runtime Dependencies (Minimal)
Only Python standard library + these essentials:
- `typing_extensions` — Backport modern type hints
- `tomli` / `tomllib` — TOML config parsing (stdlib in 3.11+)

### Optional Dependencies (User's Choice)
- `numpy` — Accelerated vector operations (optional, pure Python fallback exists)
- `httpx` — Async HTTP client for provider APIs
- `rich` — Beautiful terminal output
- `click` — CLI framework

> [!IMPORTANT]
> **The entire core engine runs on Python stdlib only.** Optional deps unlock performance boosts and developer experience enhancements but are never required.

---

## 🚀 Implementation Phases

### Phase 1: Foundation + Core Engines (We start here)
- Project scaffolding (pyproject.toml, structure, CI)
- Foundation layer (memory, config, logging, caching, math kernels)
- NLP Engine (tokenizer, sentence splitter, claim extractor)
- Token Management Engine
- Basic assertions library
- Pytest plugin skeleton

### Phase 2: Intelligence Engines
- Semantic Similarity Engine
- Hallucination Detection Engine
- Structured Output Validator
- Provider Abstraction Layer

### Phase 3: Production Engines
- Context Engineering Engine
- Guardrails Engine
- Cost & Performance Profiler
- Agent Trajectory Analyzer

### Phase 4: Integration & Polish
- CI/CD Integration Layer
- Regression Detection Engine
- CLI Interface
- Full documentation
- Performance benchmarks
- PyPI release

---

> [!TIP]
> **Approve this architecture to begin Phase 1 implementation.** I'll build every module from scratch — no copying from existing libraries, every algorithm implemented from first principles.
