# Implementation Plan: OEIS Learn Neuro-Symbolic Synthesis

**Branch**: `001-oeis-neurosymbolic-synthesis` | **Date**: 2026-08-30 | **Spec**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md)

**Input**: Feature specification from [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md)

---

## Summary

`oeis-learn` is a Neuro-Symbolic AI system designed to learn continuous mathematical representations, perform automated theorem discovery, and synthesize exact generating algorithms for integer sequences from the Online Encyclopedia of Integer Sequences (OEIS).

The technical approach integrates:
1. A **Tri-Stream Continuous Neural Encoder** ($S_1$ continuous signed log-magnitude, $S_2$ 100-moduli Fourier phase spectrum, and $S_3$ finite differences + $p$-adic valuations) unified via Hierarchical Two-Stage FiLM Fusion under strict FP32 precision.
2. A **Grammar-Guided Autoregressive Transformer Decoder** utilizing `llguidance` / `XGrammar-2` with Environment-Indexed Grammars ($\mathcal{G}_{\Gamma_t}$) to enforce lexical scope validity, stack depth constraints, and No-Ghost Soundness.
3. A **High-Throughput Sandboxed WASM Execution Engine** embedded via a native Rust PyO3 extension (`oeis_wasm_evaluator`) with Wasmtime Cranelift JIT, 10,000 instruction fuel budgets, 16 MiB linear memory limits, and Rayon multi-core parallelism releasing the Python GIL.
4. An **Execution-Guided Credit Assignment GRPO (EGCA-GRPO)** reinforcement learning optimizer with asymmetric prompt weighting under binary outcome rewards ($\pm 1$).
5. A **5-Stage Taxonomic Curriculum Pipeline** with automated graduation gates ($C(S_k) \ge 0.85$, $\min(\hat{\rho}_x) \ge 0.50$), $N+K$ ($K=100$) extrapolation testing, and Minimum Description Length ($M_{\text{MDL}} \le 1.2$) anti-memorization verification.
6. A **Self-Supervised Latent Discovery Pipeline** using non-contrastive VICReg, GPU-accelerated cuML UMAP/HDBSCAN manifold clustering, high-precision ($>500$ digits) `mpmath` sampling, PSLQ integer relation detection, and SymPy symbolic proof execution.

---

## Technical Context

**Language/Version**: Python 3.11+, Rust 2021 Edition (1.75+)

**Primary Dependencies**:
- *Deep Learning & Inference:* PyTorch 2.3+ (strict FP32), `llguidance` / `XGrammar-2`
- *Native WASM Execution & Interop:* `pyo3` (0.20+), `wasmtime` (20.0+), `wat` (1.0+), `rayon` (1.8+), `maturin` (1.4+)
- *Data & Storage:* `duckdb` (0.10+), `sqlite3`, `pyarrow`
- *Mathematics & Symbolic Verification:* `mpmath` (1.3+), `sympy` (1.12+), `cuml` (RAPIDS 24.04+)
- *Testing & Benchmarking:* `pytest`, `pytest-benchmark`, `cargo test`

**Storage**: Local DuckDB / SQLite database (`data/oeis_learn.duckdb`) storing parsed OEIS sequence records, metadata tags, curriculum staging indices, and benchmark logs.

**Testing**: `pytest` for Python unit/integration tests; `cargo test` for native Rust WASM evaluator crate; `pytest-benchmark` for throughput validation.

**Target Platform**: Linux (x86_64), Ubuntu 22.04 LTS / 24.04 LTS with NVIDIA CUDA 12.0+

**Project Type**: Hybrid Python / Rust Neuro-Symbolic ML Framework & CLI Tool

**Performance Goals**:
- $\ge 500$ WASM module evaluations/second across 8 CPU threads via PyO3/Rayon bridge.
- Sub-$100\,\mu\text{s}$ per-token grammar masking evaluation latency during autoregressive decoding.
- Trapping 100% of infinite loops within 10,000 fuel units in $<1\,\text{ms}$ per candidate.
- $\ge 80\%$ program synthesis pass rate on Curriculum Stage 1 (polynomials) on local Tier 1 hardware.

**Constraints**:
- *Tier 1 Local Workstation Baseline:* 4 Cores / 8 Threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- *VRAM Footprint:* GPU memory usage $\le 4\,\text{GB}$ via scaled hidden dimensions ($d=256$ or $d=384$) and micro-batching (4–8).
- *Strict Precision:* No FP16/BF16 mixed precision in encoder layers.
- *Sandbox Limits:* 10,000 fuel units and 16 MiB linear memory per execution.

**Scale/Scope**:
- *Tier 1 Baseline:* 10,000 to 25,000 sequences across Curriculum Stages 1 & 2.
- *Tier 2 Scale-Up:* Full 390,000+ OEIS database with $d=768$ across multi-GPU clusters.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / Gate | Requirement | Status | Compliance Verification |
| :--- | :--- | :--- | :--- |
| **Principle I: Exact Representation & FP32** | Tri-stream continuous encoding ($S_1$ magnitude, $S_2$ 100-moduli Fourier, $S_3$ differences + $p$-adics) with Hierarchical FiLM in strict FP32. | **PASSED** | Specified in data model and research; AMP explicitly disabled; FP32 enforcement configured in encoder module. |
| **Principle II: Provably Sound WAT Synthesis** | Dynamic Earley trie grammar masking (`llguidance`) with Environment-Indexed scope tracking ($\mathcal{G}_{\Gamma_t}$). | **PASSED** | Formal EBNF contract defined in `wat-grammar.ebnf`; Environment-Indexed tracker designed in `data-model.md`. |
| **Principle III: Sandboxed Deterministic Execution** | In-memory compilation, 10,000 fuel cap, 16 MiB linear memory, GIL-free PyO3/Rayon execution. | **PASSED** | Native Rust crate `oeis_wasm_evaluator` designed with `wat::parse_str`, Cranelift fuel metering, and Rayon parallel worker pools. |
| **Principle IV: Workstation Feasibility** | Tier 1 baseline ($d=256/384$, batch 4–8, 8-thread CPU offload) on 4-core Xeon / 4GB Quadro M2000M. | **PASSED** | Configuration profiles `configs/train_tier1.yaml` explicitly cap $d=256/384$, micro-batches 4–8, and offload all WASM evals to CPU. |
| **Principle V: Curriculum & Anti-Memorization** | 5-stage curriculum with $C(S_k) \ge 0.85$, $N+K$ ($K=100$) extrapolation testing, and $M_{\text{MDL}} \le 1.2$. | **PASSED** | Curriculum scheduler and anti-memorization verifiers defined in `src/oeis_learn/curriculum/` and validated in `data-model.md`. |
| **Principle VI: Credit Assignment & Discovery** | EGCA-GRPO with asymmetric weighting, VICReg self-supervision, PSLQ integer relations ($<10^{-50}$ drop), SymPy proofs. | **PASSED** | Tracing sandbox for credit assignment designed in `src/oeis_learn/rl/`; VICReg loss and PSLQ engine formalized in `src/oeis_learn/discovery/`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-oeis-neurosymbolic-synthesis/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan (this document)
├── research.md          # Technical decisions and architectural research
├── data-model.md        # Domain entities, schemas, and state transitions
├── quickstart.md        # Setup instructions and runnable validation scenarios
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── contracts/
    ├── oeis-wasm-evaluator.ffi.json # Native PyO3 Rust FFI contract
    ├── cli-interface.contract.json  # CLI interface schema
    ├── wat-grammar.ebnf             # Dynamic Environment-Indexed WAT Grammar
    └── database-schema.sql          # DuckDB / SQLite local storage schema
```

### Source Code (repository root)

```text
crates/
└── oeis_wasm_evaluator/             # Native Rust PyO3 execution engine
    ├── Cargo.toml                   # Rust package manifest (pyo3, wasmtime, wat, rayon)
    └── src/
        ├── lib.rs                   # PyO3 module bindings & evaluate_wat_batch export
        ├── engine.rs                # Wasmtime engine configuration & fuel injection
        └── sandbox.rs               # In-memory parsing, execution trap handler & term generator

src/
└── oeis_learn/                      # Main Python package
    ├── __init__.py
    ├── data/                        # Ingestion & Dataset pipeline
    │   ├── __init__.py
    │   ├── ingest.py                # oeisdata & joeis parser and DuckDB loader
    │   ├── dataset.py               # Sequence dataset provider & batch collation
    │   └── transforms.py            # Algebraic transformation pair generator for SSL
    ├── encoder/                     # Tri-Stream Continuous Neural Encoder (FP32)
    │   ├── __init__.py
    │   ├── magnitude_stream.py      # S1 signed logarithmic continuous scalar projection
    │   ├── modulo_stream.py         # S2 100-moduli trigonometric Fourier phase embeddings
    │   ├── difference_stream.py     # S3 finite differences & p-adic valuation ordinals
    │   ├── film_fusion.py           # Hierarchical Two-Stage FiLM fusion block
    │   └── tri_stream_encoder.py    # Unified Bidirectional Transformer encoder backbone
    ├── decoder/                     # Grammar-Guided Autoregressive Generator
    │   ├── __init__.py
    │   ├── wat_decoder.py           # Transformer decoder conditioning on latent Z
    │   ├── grammar_masker.py        # llguidance / XGrammar-2 dynamic Earley trie masker
    │   └── environment_tracker.py   # Lexical scope (Vars_t) and stack depth (Types_t) tracker
    ├── sandbox/                     # Python WASM Sandbox Wrapper & Execution Manager
    │   ├── __init__.py
    │   ├── runner.py                # Wrapper invoking oeis_wasm_evaluator or wasmtime-py
    │   └── tracer.py                # Execution trace recorder for credit assignment
    ├── rl/                          # Reinforcement Learning Pipeline
    │   ├── __init__.py
    │   ├── egca_grpo.py             # Execution-Guided Credit Assignment GRPO trainer
    │   ├── prompt_weighting.py      # Asymmetric prompt weighting for hard curriculum prompts
    │   └── reward.py                # Strict binary outcome reward (+1 / -1) calculator
    ├── curriculum/                  # 5-Stage Taxonomy-Aligned Curriculum Engine
    │   ├── __init__.py
    │   ├── scheduler.py             # Competence scoring C(Sk) & automated graduation gates
    │   ├── sampler.py               # Dynamic mixture prompt sampling (70/20/10)
    │   ├── extrapolation.py         # Extrapolation Horizon (N+K, K=100) verifier
    │   └── mdl_verifier.py          # Minimum Description Length (M_MDL <= 1.2) complexity test
    ├── discovery/                   # Self-Supervised Discovery & Theorem Prover
    │   ├── __init__.py
    │   ├── vicreg_loss.py           # Non-contrastive Variance-Invariance-Covariance loss
    │   ├── manifold.py              # GPU cuML UMAP & HDBSCAN density clustering
    │   ├── vector_search.py         # HNSW vector arithmetic candidate search
    │   ├── pslq_solver.py           # mpmath high-precision sampling & PSLQ relation finder
    │   └── symbolic_prover.py       # SymPy / SageMath algebraic proof execution
    └── cli/                         # Command-Line Interfaces
        ├── __init__.py
        └── main.py                  # CLI entrypoint (ingest, train, synthesize, discover)

configs/
├── train_tier1.yaml                 # Local workstation baseline config (d=256/384, bs=4-8, 8 threads)
└── train_tier2.yaml                 # Cluster scale-up config (d=768, multi-GPU)

tests/
├── unit/
│   ├── test_tri_stream_encoder.py   # Encoder numerical stability & FP32 precision tests
│   ├── test_film_fusion.py          # Hierarchical FiLM modulation unit tests
│   ├── test_grammar_masker.py       # Dynamic Earley trie masking & environment tracking tests
│   ├── test_wasm_sandbox.py         # 10,000 fuel limit, memory ceiling, and trap tests
│   ├── test_curriculum_gates.py     # Competence scoring & extrapolation verification tests
│   └── test_vicreg_loss.py          # VICReg variance/invariance/covariance loss tests
├── integration/
│   ├── test_data_ingestion.py       # End-to-end DuckDB ingestion from raw sequence records
│   ├── test_batch_throughput.py     # Multi-threaded Rayon throughput benchmark (>500 evals/sec)
│   ├── test_egca_training_step.py   # EGCA gradient localization and asymmetric weighting step
│   └── test_discovery_pipeline.py   # Latent vector arithmetic -> PSLQ -> SymPy proof flow
└── contract/
    ├── test_ffi_contract.py         # Native PyO3 bridge contract test
    ├── test_cli_contract.py         # CLI arguments and output format contract test
    └── test_wat_grammar_contract.py # EBNF grammar completeness and parser test
```

**Structure Decision**: The selected hybrid Python/Rust workspace cleanly isolates performance-critical WebAssembly compilation, fuel metering, and multi-core batch execution in a native Rust PyO3 crate (`crates/oeis_wasm_evaluator`), while leveraging Python for PyTorch neural modeling, curriculum scheduling, and symbolic computer algebra (`src/oeis_learn/`).

---

## Complexity Tracking

| Architectural Component | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| **Native Rust PyO3 Crate (`oeis_wasm_evaluator`)** | High-throughput reinforcement learning loops evaluate batches of 1,000+ candidate programs. Python's GIL causes severe lock contention during batch WASM execution. | Python `ThreadPoolExecutor` suffers from GIL contention during `wasmtime.Store` allocations (~450ms/1000 modules); `ProcessPoolExecutor` incurs heavy IPC serialization overhead (~180ms/1000 modules). Rust + Rayon evaluates 1,000 modules in ~15ms with 100% CPU utilization. |
| **Tri-Stream Continuous Encoder ($S_1, S_2, S_3$)** | OEIS integer sequence values span from $-10^6$ to $>10^{30}$. Standard tokenizers drop numbers as OOV or explode sequence length, while single-scalar continuous encodings are squashed by LayerNorm. | BPE/WordPiece tokenization fails on large integers ($>10,000$); single-scalar xVal cannot represent modular arithmetic congruences or local difference dynamics. |
| **Environment-Indexed Grammar ($\mathcal{G}_{\Gamma_t}$)** | Standard context-free grammars only ensure syntactic validity (balanced parens), allowing models to emit invalid variable indices or violate stack depths. | Without lexical scope tracking, synthesized programs fail in the sandbox due to unbound variable references, wasting exploration capacity. |
| **EGCA-GRPO with Asymmetric Weighting** | Standard GRPO yields zero gradient updates when all completions in a group fail on hard prompts, and uniform credit assignment penalizes valid boilerplate code. | Vanilla GRPO suffers zero-advantage collapse; PPO requires an auxiliary critic network that doubles VRAM consumption on 4GB workstation hardware. |

