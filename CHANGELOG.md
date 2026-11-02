# Changelog

All notable changes to **Veriflux** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-02

### 🔱 Initial Release — Built Entirely From Scratch

#### Added
- **Core Architecture:**
  - Zero-runtime-dependency foundation operating strictly on the Python Standard Library (3.10+).
  - High-precision timing kernel (`time.perf_counter_ns`), memory tracking, and arena allocation pools.
  - Hierarchical configuration manager supporting TOML files, environment variables (`VERIFLUX_*`), and runtime overrides.
  - Multi-tier caching infrastructure (LRU, LFU, Content-Addressable, and Tiered).
  - Pure Python SIMD-optimized mathematical kernels (cosine similarity, sparse dot product, softmax, sigmoid).

- **NLP Engine:**
  - BPE (Byte-Pair Encoding) subword tokenizer with full merge training algorithm from scratch.
  - WordPiece subword tokenizer with greedy longest-match.
  - Sentence splitter handling 100+ abbreviations, initials, honorifics, decimals, and URLs.
  - Complete 5-step Porter Stemming algorithm.
  - Atomic claim decomposition with clause-level syntactic splitting and claim classification.
  - Rule-based named entity recognition (NER) across 8 entity types.
  - Text fingerprinting with MinHash and SimHash.

- **Similarity Engine:**
  - Sparse TF-IDF vectorizer with sublinear TF scaling and smoothed IDF.
  - State-of-the-art BM25+ lexical ranking algorithm.
  - In-memory VectorStore with HNSW-inspired proximity graph search.
  - Edit distance suite: Levenshtein, Damerau-Levenshtein, and atomic edit operation backtracker.
  - Score fusion algorithms: Reciprocal Rank Fusion (RRF), CombMNZ, and weighted linear combination.

- **Hallucination Detection Engine:**
  - End-to-end multi-stage hallucination detection pipeline.
  - Textual entailment checker evaluating lexical overlap, entity overlap, and negation asymmetry.
  - Factual consistency checker detecting internal self-contradictions and semantic conflicts.
  - URL and citation validator flagging fabricated DOIs, arXiv IDs, and malformed links.
  - Groundedness checker measuring context coverage, precision, and density.

- **Structured Output & Schema Engine:**
  - Fault-tolerant JSON self-healing engine repairing truncated JSON, missing brackets, unquoted keys, single quotes, and Python literals.
  - Comprehensive JSON-schema validator with path-indexed error reporting.
  - Type coercer supporting strict and permissive coercion modes.
  - Fluent `SchemaBuilder` with dataclass and example-based inference.
  - Deep schema diff comparison engine.

- **Guardrails & Safety Engine:**
  - 50+ PII pattern detectors with Luhn checksum validation for credit cards and multi-mode redaction.
  - Adversarial prompt injection and jailbreak detector.
  - Toxicity, threat, and profanity screening guards.
  - Token bucket rate limiter.
  - Declarative policy governance definitions.

- **Agent Trajectory Analyzer:**
  - Zero-overhead execution tracing via ContextVar propagation across async boundaries.
  - Infinite loop and cyclic behavior detector.
  - Deterministic trace replayer with what-if counterfactual intervention.
  - DAG execution graph builder with export to GitHub Mermaid flowcharts.

- **Context Engineering Engine:**
  - Dynamic multi-stream context assembler under strict token budgets.
  - Extractive and compressive context compressor.
  - MinHash-based passage deduplication.
  - BM25 chunk relevance ranker and sliding context window manager.

- **Cost & Performance Profiler:**
  - Multi-provider cost attribution supporting OpenAI, Anthropic, Google Gemini, and open models.
  - Latency distribution tracker computing p50, p95, p99 percentiles.
  - Throughput meter measuring tokens per second (TPS) and requests per minute (RPM).
  - RAM memory allocation profiler.

- **CI/CD & Pytest Integration:**
  - Pytest plugin registering `@veriflux.test` decorators, fixtures, and layer markers.
  - JUnit XML, structured JSON, styled HTML, and GitHub PR comment markdown reporters.
  - Dynamic standalone SVG status badge generator.
  - Automated quality gate thresholds.

- **Datasets & CLI:**
  - Dataset loader for JSONL, CSV, and JSON files.
  - Synthetic adversarial test data generator.
  - Golden dataset manager.
  - Standalone `veriflux` CLI application.

