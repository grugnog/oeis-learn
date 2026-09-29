# Implementation Plan: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine

**Branch**: `004-decoupled-grounding-and-symple-engine` | **Date**: 2026-09-02 | **Spec**: [specs/004-decoupled-grounding-and-symple-engine/spec.md](specs/004-decoupled-grounding-and-symple-engine/spec.md)

**Input**: Feature specification from [specs/004-decoupled-grounding-and-symple-engine/spec.md](specs/004-decoupled-grounding-and-symple-engine/spec.md)

---

## Summary

`oeis-learn` Phase 4 resolves the four fundamental bottlenecks identified during the 60-epoch Run 005 benchmark: the constant grounding gap, Abstract Syntax Tree (AST) dead-code bloat, 65-step task dilution, and unnormalized latent space Euclidean scaling.

The technical approach integrates:
1. **Decoupled Symbolic-Numeric Grounding & Solvers:** Decouples computational topology synthesis from continuous parameter guessing. The decoder vocabulary and grammar masker admit generic placeholders `i64.const_?`. When skeletons are sampled, exact Hermite Normal Form (HNF) integer row reduction resolves linear systems in $<1\,\text{ms}$, while Z3 SMT (`QF_BV`) solves non-linear modulo/bitwise expressions within a $250\,\text{ms}$ timeout. Policy gradients are backpropagated through placeholder skeleton trajectories using the grounded reward ($R=1.0$), while grounded programs are archived in the Elite Demonstration Buffer.
2. **Compiler-in-the-Loop Canonicalization & Parsimony-Regularized RLVR:** Integrates online `wasm-opt` optimization passes (`--vacuum`, `--dce`, `--remove-unused-locals`) in the execution sandbox ($<1.5\,\text{ms}$). Computes Syntactic Waste Ratio $\rho_{\text{waste}}$ with a hard $30\%$ cutoff threshold ($\tau_{\text{thresh}} = 0.30$), continuous dense log-distance returns $R_{\text{dense}}$, Covariant Parsimony Pressure (CPP), and Lexicographical Group Advantage Ranking ($R_{\text{exec}} \succ -|P_{\text{opt}}|$). Regularizes exploration via Partitioned Semantic Policy Entropy and stack-depth temperature scaling.
3. **Tri-Stream Encoder v2 & Linear Invariants:** Replaces non-linear FiLM modulation with direct vector concatenation and FP32 bidirectional self-attention. Computes normalized Newton forward difference quotients ($D^{(k)} = \Delta^k y / k!$), orthogonal Prime Fourier Embeddings (PFE) across 16 odd prime fields, and prepends learnable summary tokens ($\mathbf{z}_{\text{affine}}, \mathbf{z}_{\text{geom}}$) with auxiliary regression heads ($\lambda_{\text{aux}} = 0.10$).
4. **SYMPLE Multi-Task Bandit Curriculum & Anti-Forgetting Replay:** Replaces static stage gating with an EXP3.S non-stationary bandit task scheduler targeting the Zone of Proximal Development ($r_i = \hat{p}_i(1-\hat{p}_i) + |\Delta C_i| + 2\max(0, -\Delta C_i)$). Dynamically scales rollout group sizes ($G_i \in [8, 16]$) via Ada-G to guarantee $P(\text{Hit} \ge 1) \ge 0.50$ on frontier tasks. Prevents catastrophic forgetting via an Elite Demonstration Buffer (EDB) storing the top-4 shortest canonical ASTs per sequence and sampling $B_{\text{replay}} = 2$ dormant sequences for joint SFT consistency loss $\mathcal{L}_{\text{total}}$.
5. **$L_2$-Normalized Manifold & Automated Theorem Discovery:** Performs $L_2$ normalization ($\hat{z} = z / \|z\|_2$) on all sequence representations, restoring scale invariance for `VectorRelationSearcher` ($\varepsilon = 0.8$) and high-precision ($>500$ digits) PSLQ integer relation proving with automated SymPy symbolic proof export.

---

## Technical Context

**Language/Version**: Python 3.11+, Rust 2021 Edition (1.75+)

**Primary Dependencies**:
- *Deep Learning & Optimization:* PyTorch 2.3+ (strict FP32), `llguidance` / `XGrammar-2`
- *Solvers & Theorem Proving:* `z3-solver` (4.12+), `mpmath` (1.3+), `sympy` (1.12+), `scipy` (1.11+), `numpy` (1.26+)
- *Native WASM Execution & Interop:* `pyo3` (0.20+), `wasmtime` (20.0+), `wat` (1.0+), `rayon` (1.8+), `maturin` (1.4+), `binaryen-rs` / `wasm-opt` (116+)
- *Data & Storage:* `duckdb` (0.10+), `sqlite3`, `pyarrow`
- *Testing & Benchmarking:* `pytest`, `pytest-benchmark`, `cargo test`

**Storage**: Local DuckDB / SQLite databases (`data/oeis_learn.duckdb`), synthetic demonstration JSON datasets (`data/sft_demonstrations.json`), and model checkpoints (`checkpoints/*.pt`).

**Testing**: `pytest` for unit/contract/integration tests and progressive 4-tier validation; `cargo test` for the native Rust `oeis_wasm_evaluator` crate.

**Target Platform**: Linux (x86_64), Ubuntu 22.04 LTS / 24.04 LTS with NVIDIA CUDA 12.0+

**Project Type**: Hybrid Python / Rust Neuro-Symbolic Machine Learning Framework & CLI Engine

**Performance Goals**:
- Linear Diophantine solver latency $< 1.0\,\text{ms}$ per candidate with $100\%$ exact recovery.
- SMT solver fallback latency $< 250\,\text{ms}$ on non-linear modulo/bitwise skeletons.
- Compiler canonicalization pass (`wasm-opt`) latency $< 1.5\,\text{ms}$ per candidate.
- Sustained batch evaluation throughput $> 5,000$ WASM module evaluations/sec across 8 CPU threads.
- Advantage Collapse Rate bounded at $\text{ACR} \le 0.05$ across 60 production epochs.
- Curriculum Stage 1 rolling competence $C(S_1) \ge 0.85$ graduation.
- Zero-error generalization on $K=100$ extrapolation terms for graduated algorithms.

**Constraints**:
- *Tier 1 Workstation Baseline:* 4 CPU Cores / 8 Threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- *Strict Single Precision:* All neural forward and backward operations execute in `torch.float32` (no mixed precision / AMP).
- *GPU Memory Ceiling:* Peak VRAM usage $\le 3.5\,\text{GB}$ via micro-batching ($B_{\text{active}}=2, B_{\text{replay}}=2$) and sequence chunking ($L_{\text{chunk}}=256$).
- *Sandbox Limits:* 10,000 instruction fuel limit and 16 MiB linear memory per execution instance.

**Scale/Scope**:
- *Benchmark Cohort:* 524 OEIS sequence catalog spanning polynomials, recurrences, and modular families.
- *Production Run 006:* 60 Epochs, 24,000 optimization steps, 96,000 candidate evaluations.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / Gate | Requirement | Status | Compliance Verification |
| :--- | :--- | :--- | :--- |
| **Principle I: Exact Representation & FP32** | Multi-axis encoding ($S_1$ magnitude, $S_2$ prime Fourier, $S_3$ Newton quotients, summary tokens) in strict FP32 precision. | **PASSED** | Specified in spec and research; non-linear FiLM replaced with direct concatenation and self-attention; AMP disabled; all operations execute in `torch.float32`. |
| **Principle II: Provably Sound WAT Synthesis** | Dynamic Earley grammar masking (`llguidance`) with Environment-Indexed scope tracking and placeholder terminal `i64.const_?`. | **PASSED** | Formal EBNF contract defined in [contracts/wat-grammar.ebnf](contracts/wat-grammar.ebnf); No-Ghost Soundness and stack depth tracking enforced by structural state machine. |
| **Principle III: Sandboxed Deterministic Execution** | In-memory compilation, 10,000 fuel cap, 16 MiB linear memory, GIL-free PyO3/Rayon execution with online `wasm-opt` DCE pass. | **PASSED** | Native Rust crate `oeis_wasm_evaluator` provides Cranelift fuel metering, linear memory isolation, and multi-core batch throughput ($>5,500$ evals/sec). |
| **Principle IV: Workstation Feasibility** | Tier 1 baseline ($d=256$, active batch $B=2$, 8-thread CPU offload) on 4-core Xeon / 4GB Quadro M2000M. | **PASSED** | Configuration profiles in `train_tier1.yaml` enforce $d=256$, $B_{\text{active}}=2$, dynamic group sizing ($G \in [8, 16]$), and 100% CPU offload for WASM evals and DCE. |
| **Principle V: Curriculum & Anti-Memorization** | 5-stage taxonomy with EXP3.S bandit, $C(S_1) \ge 0.85$, $N+K$ ($K=100$) extrapolation testing, and $M_{\text{MDL}} \le 1.20$. | **PASSED** | EXP3.S non-stationary bandit, Ada-G allocator, extrapolation horizon verifier ($K=100$), and Minimum Description Length checks ($M_{\text{MDL}} \le 1.20$) integrated into training pipeline. |
| **Principle VI: Credit Assignment & Discovery** | Decoupled GRPO skeleton gradients, EDB SFT replay, CPP parsimony, $L_2$-normalized manifold, PSLQ integer relations, SymPy proofs. | **PASSED** | Decoupled Diophantine/SMT solver, EDB vulnerability replay, CPP parsimony, $L_2$-normalized vector search, and PSLQ theorem verification designed in `data-model.md` and `research.md`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/004-decoupled-grounding-and-symple-engine/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan (this document)
├── research.md          # Technical decisions and architectural research
├── data-model.md        # Domain entities, schemas, and state transitions
├── quickstart.md        # Setup instructions and runnable validation scenarios
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── contracts/
    ├── cli-interface.contract.json      # Expanded CLI interface schema
    ├── constant-solver.contract.json    # Diophantine & SMT solver interface schema
    ├── symple-config.schema.json        # Phase 4 hyperparameters schema
    └── wat-grammar.ebnf                 # WAT Grammar with i64.const_? placeholders
```

### Source Code (repository root)

```text
crates/
└── oeis_wasm_evaluator/                 # Native Rust PyO3 execution engine
    ├── Cargo.toml                       # Rust package manifest (pyo3, wasmtime, wat, rayon)
    └── src/
        ├── lib.rs                       # PyO3 module bindings & evaluate_wat_batch export
        ├── engine.rs                    # Wasmtime engine configuration & fuel injection
        └── sandbox.rs                   # In-memory parsing, execution trap handler & DCE optimizer

src/
└── oeis_learn/                          # Main Python package
    ├── __init__.py
    ├── curriculum/                      # Curriculum & Task Scheduling
    │   ├── __init__.py
    │   ├── taxonomy.py                  # 5-stage taxonomy definition
    │   ├── scheduler.py                 # Multi-stage curriculum scheduler
    │   ├── symple_bandit.py             # EXP3.S Bandit Scheduler & Ada-G Allocator
    │   ├── sampler.py                   # Sequence task sampler
    │   ├── extrapolation.py             # K=100 extrapolation verifier
    │   └── mdl_verifier.py              # Minimum Description Length ratio verifier
    ├── data/                            # Ingestion & Dataset pipeline
    │   ├── __init__.py
    │   ├── models.py                    # Domain entities (ASTSkeleton, SolverResult, etc.)
    │   ├── ingest.py                    # oeisdata & joeis parser and DuckDB loader
    │   ├── dataset.py                   # Sequence dataset provider & batch collation
    │   ├── synthetic_generator.py       # Procedural generator with randomized affine sweeps
    │   └── transforms.py                # Algebraic transformation pair generator for SSL
    ├── decoder/                         # WAT Transformer Decoder & Solvers
    │   ├── __init__.py
    │   ├── wat_decoder.py               # Autoregressive transformer decoder (FP32)
    │   ├── grammar_masker.py            # Dynamic grammar logit masker (Earley trie)
    │   ├── wat_grammar.py               # EBNF grammar rules & token definitions
    │   ├── environment_tracker.py       # Variable scope & stack depth tracker
    │   └── constant_solver.py           # Hermite Diophantine & Z3 SMT constant solver
    ├── discovery/                       # Self-Supervised Discovery & Theorem Prover
    │   ├── __init__.py
    │   ├── manifold.py                  # UMAP/HDBSCAN manifold reduction & clustering
    │   ├── vector_search.py             # L2-normalized vector relation searcher
    │   ├── pslq_solver.py               # Arbitrary-precision mpmath PSLQ relation solver
    │   ├── symbolic_prover.py           # SymPy automated symbolic proof generator
    │   └── vicreg_loss.py               # Kernel VICReg non-contrastive loss
    ├── encoder/                         # Tri-Stream Continuous Neural Encoder (FP32)
    │   ├── __init__.py
    │   ├── tri_stream_encoder.py        # Tri-Stream Encoder v2 with summary tokens & self-attention
    │   ├── magnitude_stream.py          # S1 signed logarithmic continuous scalar projection
    │   ├── modulo_stream.py             # S2 16-prime orthogonal Fourier embeddings (PFE)
    │   ├── difference_stream.py         # S3 normalized Newton forward difference quotients
    │   └── heads.py                     # Summary token linear regression heads (slope, ratio)
    ├── rl/                              # Reinforcement Learning & Optimization
    │   ├── __init__.py
    │   ├── trainer.py                   # SYMPLE unified execution trainer loop
    │   ├── egca_grpo.py                 # GRPO with partitioned semantic entropy & dynamic temp
    │   ├── reward.py                    # Dense log-distance, CPP penalty & lexicographic advantages
    │   ├── elite_buffer.py              # Elite Demonstration Buffer with dormancy replay
    │   ├── prompt_weighting.py          # Asymmetric prompt weighting & PBRS
    │   ├── sft_trainer.py               # Supervised Fine-Tuning warmup trainer
    │   ├── progressive.py               # 4-tier progressive pre-flight validation harness
    │   └── telemetry.py                 # Real-time metrics logger (ACR, waste, entropy)
    ├── sandbox/                         # Execution Sandbox & Attribution Tracer
    │   ├── __init__.py
    │   ├── runner.py                    # High-throughput Rayon WASM execution runner & DCE pass
    │   ├── tracer.py                    # Execution trace divergence localizer
    │   └── fallback_runner.py           # Pure-Python wasmtime fallback runner
    └── tracking/                        # Artifact Management & Run Directory
        ├── __init__.py
        └── run_manager.py               # Run directory lifecycle & checkpoint archiver

tests/
├── contract/                            # Schema and API contract validation tests
├── integration/                         # Multi-module pipeline integration tests
└── unit/                                # Subsystem unit tests
    ├── test_constant_solver.py          # Diophantine & SMT solver unit tests
    ├── test_parsimony_rlvr.py           # Compiler DCE, waste ratio & CPP reward tests
    ├── test_encoder_v2.py               # Newton differences, PFE & summary token tests
    └── test_symple_curriculum.py        # EXP3.S bandit, Ada-G & EDB dormancy replay tests
```

**Structure Decision**: Monolithic hybrid Python/Rust package rooted under `src/oeis_learn/` with native Rust PyO3 crate under `crates/oeis_wasm_evaluator/`. All experimental runs and artifacts are preserved under `runs/`.

---

## Complexity Tracking

> **Constitution Check**: No constitutional principles were violated. The design adheres strictly to FP32 precision, dynamic grammar masking, fuel-bounded sandboxed execution, Tier 1 workstation constraints, curriculum progression, and deterministic credit assignment.

