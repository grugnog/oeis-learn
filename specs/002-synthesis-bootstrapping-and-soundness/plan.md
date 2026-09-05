# Implementation Plan: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization

**Branch**: `002-synthesis-bootstrapping-and-soundness` | **Date**: 2026-08-31 | **Spec**: [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md)

**Input**: Feature specification from [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md)

---

## Summary

`oeis-learn` Phase 2 resolves the fundamental bottlenecks identified during the 18.65-hour end-to-end benchmark: context-sensitive grammar soundness gaps, zero-advantage exploration collapse in GRPO, reward sparsity, and high diagnostic feedback latency.

The technical approach integrates:
1. **Dynamic Environment-Indexed Grammar Decoding ($\mathcal{G}_{\Gamma_t}$):** A dual-layer masking pipeline coupling static token tries ($M_{\text{CFG}}$) with a zero-allocation dynamic state machine ($M_{\text{Context}}$) tracking structural phases ($\Phi_t$), lexical symbol tables ($\Gamma_t$), operand type stacks ($\Sigma_t$), and control block nesting ($H_t$). Guarantees 100% compilable WebAssembly bytecode and strict No-Ghost Soundness under sub-$100\,\mu\text{s}$ per-token latency.
2. **Synthetic Demonstration Generator & Supervised Warmup (SFT):** Forward synthesis of 5,000 diverse sequence-program pairs across polynomial, linear recurrence, and modular algorithmic families, followed by teacher-forced cross-entropy SFT pretraining to establish initial syntactic and semantic fluency ($\ge 80\%$ Stage 1 pass rate).
3. **Multi-Tiered Reward Shaping & Supervised GRPO (S-GRPO):** Composite reward shaping (compiler validity $R_{\text{comp}}$, prefix length $R_{\text{prefix}}$, continuous distance $R_{\text{dist}}$) annealed via a cosine schedule toward strict binary rewards ($+1/-1$). Conditional Ground-Truth Trajectory Injection (CGI) and AVSPO eliminate zero-advantage collapse on difficult batches ($\text{ACR} \le 0.15$).
4. **5-Tier Progressive Validation Hierarchy & Telemetry:** Pre-flight verification suite (Tier 0 deterministic unit checks $\to$ Tier 1 oracle fitting $\to$ Tier 2 single-prompt RL $\to$ Tier 3 micro-cohort tuning $\to$ Tier 4 full run) executing in $<45\text{ minutes}$ with real-time diagnostic telemetry (policy entropy $\mathcal{H}$, reward variance $\sigma_R^2$, ACR, and compiler trap rate).
5. **Self-Supervised Latent Manifold Structuring:** Kernel VICReg in RKHS combined with explicit additive homomorphism loss ($\mathcal{L}_{\text{add}}$) and shift equivariance ($\mathcal{L}_{\text{shift}}$) over algebraic operator pairs, preventing dimensional collapse and generating high-yield candidates for PSLQ integer relation searches.

---

## Technical Context

**Language/Version**: Python 3.11+, Rust 2021 Edition (1.75+)

**Primary Dependencies**:
- *Deep Learning & Optimization:* PyTorch 2.3+ (strict FP32), `llguidance` / `XGrammar-2`
- *Native WASM Execution & Interop:* `pyo3` (0.20+), `wasmtime` (20.0+), `wat` (1.0+), `rayon` (1.8+), `maturin` (1.4+)
- *Data & Storage:* `duckdb` (0.10+), `sqlite3`, `pyarrow`
- *Mathematics & Symbolic Proving:* `mpmath` (1.3+), `sympy` (1.12+), `cuml` (RAPIDS 24.04+)
- *Testing & Benchmarking:* `pytest`, `pytest-benchmark`, `cargo test`

**Storage**: Local DuckDB / SQLite database (`data/oeis_learn.duckdb`), synthetic demonstration JSON datasets (`data/sft_demonstrations.json`), and model checkpoint files (`checkpoints/*.pt`).

**Testing**: `pytest` for Python unit/integration tests and progressive pre-flight validation; `cargo test` for native Rust WASM evaluator crate.

**Target Platform**: Linux (x86_64), Ubuntu 22.04 LTS / 24.04 LTS with NVIDIA CUDA 12.0+

**Project Type**: Hybrid Python / Rust Neuro-Symbolic ML Framework & CLI Tool

**Performance Goals**:
- $\ge 500$ WASM module evaluations/second across 8 CPU threads via PyO3/Rayon bridge.
- Sub-$100\,\mu\text{s}$ per-token grammar masking evaluation latency (targeting $5\text{--}20\,\mu\text{s}$ median).
- $\ge 80\%$ program synthesis pass rate on Curriculum Stage 1 (polynomials) after SFT warmup.
- $< 45\text{ minutes}$ total pre-flight execution time for Tiers 0 through 3.
- Bounded Advantage Collapse Rate ($\text{ACR} \le 0.15$) during online RL exploration.

**Constraints**:
- *Tier 1 Workstation Baseline:* 4 CPU Cores / 8 Threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- *Strict Single Precision:* All neural operations execute in `torch.float32` without mixed precision.
- *GPU Memory Ceiling:* Peak VRAM usage $\le 3.5\,\text{GB}$ via mini-batch sequence chunking ($L_{\text{chunk}} = 256$) and micro-batching ($B=1\text{--}4$).
- *Sandbox Limits:* 10,000 instruction fuel limit and 16 MiB linear memory per execution instance.

**Scale/Scope**:
- *Tier 1 Baseline:* 5,000 synthetic SFT pairs, 500+ multi-stage sequence tasks, Curriculum Stages 1 & 2.
- *Tier 2 Scale-Up:* Full 390,000+ OEIS database with $d=768$ across multi-GPU clusters (deferred until Stage 2 graduation).

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / Gate | Requirement | Status | Compliance Verification |
| :--- | :--- | :--- | :--- |
| **Principle I: Exact Representation & FP32** | Tri-stream continuous encoding ($S_1$ magnitude, $S_2$ 100-moduli Fourier, $S_3$ differences + $p$-adics) with Hierarchical FiLM in strict FP32. | **PASSED** | Specified in spec and research; AMP explicitly disabled; all forward/backward passes run in strict `torch.float32`. |
| **Principle II: Provably Sound WAT Synthesis** | Dynamic Earley trie grammar masking (`llguidance`) with Environment-Indexed scope tracking ($\mathcal{G}_{\Gamma_t}$). | **PASSED** | Formal EBNF contract defined in `wat-grammar.ebnf`; dual-layer structural phase state machine ($\Phi_t$), symbol table ($\Gamma_t$), and operand stack ($\Sigma_t$) designed in `data-model.md`. |
| **Principle III: Sandboxed Deterministic Execution** | In-memory compilation, 10,000 fuel cap, 16 MiB linear memory, GIL-free PyO3/Rayon execution. | **PASSED** | Native Rust crate `oeis_wasm_evaluator` provides Cranelift fuel metering, linear memory isolation, and multi-core batch throughput ($>5,500$ evals/sec). |
| **Principle IV: Workstation Feasibility** | Tier 1 baseline ($d=256/384$, batch 1–4, 8-thread CPU offload) on 4-core Xeon / 4GB Quadro M2000M. | **PASSED** | Configuration profiles `configs/train_tier1.yaml` enforce $d=256/384$, sequence chunking ($L_{\text{chunk}}=256$), and 100% CPU offload for WASM evals. |
| **Principle V: Curriculum & Anti-Memorization** | 5-stage curriculum with $C(S_k) \ge 0.85$, $N+K$ ($K=100$) extrapolation testing, and $M_{\text{MDL}} \le 1.2$. | **PASSED** | Automated curriculum gates, extrapolation horizon verifier, and Minimum Description Length checks integrated into training pipeline. |
| **Principle VI: Credit Assignment & Discovery** | EGCA-GRPO with asymmetric weighting, S-GRPO trajectory injection, VICReg homomorphism loss, PSLQ integer relations, SymPy proofs. | **PASSED** | Trace-grounded credit assignment, CGI injection, additive homomorphism loss $\mathcal{L}_{\text{add}}$, and PSLQ theorem verification designed in `data-model.md` and `research.md`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/002-synthesis-bootstrapping-and-soundness/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan (this document)
├── research.md          # Technical decisions and architectural research
├── data-model.md        # Domain entities, schemas, and state transitions
├── quickstart.md        # Setup instructions and runnable validation scenarios
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── contracts/
    ├── wat-grammar.ebnf                 # Dynamic Environment-Indexed WAT Grammar
    ├── sft-dataset.schema.json          # Synthetic demonstration dataset schema
    ├── progressive-test-harness.contract.json # 5-Tier pre-flight validation contract
    └── cli-interface.contract.json      # Expanded CLI interface schema
```

### Source Code (repository root)

```text
crates/
└── oeis_wasm_evaluator/                 # Native Rust PyO3 execution engine
    ├── Cargo.toml                       # Rust package manifest (pyo3, wasmtime, wat, rayon)
    └── src/
        ├── lib.rs                       # PyO3 module bindings & evaluate_wat_batch export
        ├── engine.rs                    # Wasmtime engine configuration & fuel injection
        └── sandbox.rs                   # In-memory parsing, execution trap handler & term generator

src/
└── oeis_learn/                          # Main Python package
    ├── __init__.py
    ├── data/                            # Ingestion & Dataset pipeline
    │   ├── __init__.py
    │   ├── ingest.py                    # oeisdata & joeis parser and DuckDB loader
    │   ├── dataset.py                   # Sequence dataset provider & batch collation
    │   ├── synthetic_generator.py       # Forward synthetic sequence-program pair generator
    │   └── transforms.py                # Algebraic transformation pair generator for SSL
    ├── encoder/                         # Tri-Stream Continuous Neural Encoder (FP32)
    │   ├── __init__.py
    │   ├── magnitude_stream.py          # S1 signed logarithmic continuous scalar projection
    │   ├── modulo_stream.py             # S2 100-moduli trigonometric Fourier phase embeddings
    │   ├── difference_stream.py         # S3 finite differences & p-adic valuation ordinals
    │   ├── film_fusion.py               # Hierarchical Two-Stage FiLM fusion block
    │   └── tri_stream_encoder.py        # Unified Bidirectional Transformer encoder backbone
    ├── decoder/                         # Grammar-Guided Autoregressive Generator
    │   ├── __init__.py
    │   ├── wat_grammar.py               # Vocabulary tokens, opcodes, and type signatures
    │   ├── wat_decoder.py               # Transformer decoder conditioning on latent Z
    │   ├── grammar_masker.py            # Dynamic Earley trie logit masker (llguidance bridge)
    │   └── environment_tracker.py       # Structural phase (Phi_t), scope (Vars_t), and stack (Sigma_t) tracker
    ├── sandbox/                         # Python WASM Sandbox Wrapper & Execution Manager
    │   ├── __init__.py
    │   ├── runner.py                    # Wrapper invoking oeis_wasm_evaluator
    │   └── tracer.py                    # Execution trace recorder for credit assignment
    ├── rl/                              # Reinforcement Learning Pipeline
    │   ├── __init__.py
    │   ├── sft_trainer.py               # Supervised Fine-Tuning teacher-forcing trainer
    │   ├── elite_buffer.py              # Elite seed demonstration replay buffer (D_elite)
    │   ├── egca_grpo.py                 # Execution-Guided Credit Assignment GRPO trainer
    │   ├── prompt_weighting.py          # S-GRPO trajectory injection & asymmetric prompt weighting
    │   ├── reward.py                    # Multi-tiered composite & verifiable binary reward calculator
    │   ├── telemetry.py                 # Real-time diagnostic telemetry logger (entropy, ACR, traps)
    │   └── trainer.py                   # Unified curriculum RL optimization manager
    ├── curriculum/                      # 5-Stage Taxonomy-Aligned Curriculum Engine
    │   ├── __init__.py
    │   ├── scheduler.py                 # Competence scoring C(Sk) & automated graduation gates
    │   ├── sampler.py                   # Dynamic mixture prompt sampling (70/20/10)
    │   ├── extrapolation.py             # Extrapolation Horizon (N+K, K=100) verifier
    │   └── mdl_verifier.py              # Minimum Description Length (M_MDL <= 1.2) complexity test
    ├── discovery/                       # Self-Supervised Discovery & Theorem Prover
    │   ├── __init__.py
    │   ├── vicreg_loss.py               # Non-contrastive VICReg with additive homomorphism loss
    │   ├── manifold.py                  # GPU cuML UMAP & HDBSCAN density clustering
    │   ├── vector_search.py             # HNSW vector arithmetic candidate search
    │   ├── pslq_solver.py               # mpmath high-precision sampling & PSLQ relation finder
    │   └── symbolic_prover.py           # SymPy / SageMath algebraic proof execution
    └── cli/                             # Command-Line Interfaces
        ├── __init__.py
        ├── reporting.py                 # Markdown and JSON report generator
        └── main.py                      # CLI entrypoint (generate-sft, warmup-sft, test-progressive, train, synthesize, discover)

scripts/
├── run_progressive_validation.py        # 5-tier pre-flight validation harness runner
└── run_long_e2e_benchmark.py            # Long-running multi-stage training benchmark

configs/
├── train_tier1.yaml                     # Local workstation baseline config (d=256/384, bs=1-4, 8 threads)
└── train_tier2.yaml                     # Cluster scale-up config (d=768, multi-GPU)

tests/
├── unit/
│   ├── test_tri_stream_encoder.py       # Encoder numerical stability & FP32 precision tests
│   ├── test_film_fusion.py              # Hierarchical FiLM modulation unit tests
│   ├── test_grammar_masker.py           # Dynamic Earley trie masking & environment tracking tests
│   ├── test_wasm_sandbox.py             # 10,000 fuel limit, memory ceiling, and trap tests
│   ├── test_curriculum_gates.py         # Competence scoring & extrapolation verification tests
│   └── test_vicreg_loss.py              # VICReg & additive homomorphism loss tests
├── integration/
│   ├── test_tier1_oracle_fitting.py     # Tier 1 single canonical solution fitting test
│   ├── test_tier2_single_prompt_rl.py   # Tier 2 single-prompt RL convergence test
│   ├── test_tier3_micro_cohort.py       # Tier 3 synthetic micro-cohort progression test
│   ├── test_sft_warmup.py               # Synthetic demonstration pretraining test
│   ├── test_egca_training_step.py       # EGCA gradient localization and CGI step
│   └── test_discovery_pipeline.py       # Latent vector arithmetic -> PSLQ -> SymPy proof flow
└── contract/
    ├── test_ffi_contract.py             # Native PyO3 bridge contract test
    ├── test_cli_contract.py             # CLI arguments and output format contract test
    └── test_wat_grammar_contract.py     # EBNF grammar completeness and parser test
```

**Structure Decision**: The architecture organizes functionality into modular, decoupled packages within `src/oeis_learn/`, maintaining strict division between neural modeling, grammar constraint state machines, execution sandboxing, RL optimization, and symbolic proving. Pre-flight testing is encapsulated in a dedicated 5-tier progressive test suite (`scripts/run_progressive_validation.py` and `tests/integration/test_tier*.py`).

---

## Complexity Tracking

| Component / Pattern | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| **Dual-Layer Dynamic Masking ($\Phi_t, \Gamma_t, \Sigma_t$)** | WebAssembly validation requires strict function header structures, in-scope lexical variables, and operand stack height/type soundness. | Static CFGs (Earley/DFA) allow missing headers and unbound variables, causing 100% execution parser traps during unconstrained exploration. |
| **Forward Synthetic Demonstration Generator & SFT Warmup** | Reinforcement learning with sparse binary rewards ($\pm 1.0$) from random Gaussian weights suffers zero-advantage collapse ($\text{ACR} = 1.0$). | Pure RL requires exponential sample complexity on stack bytecodes; SFT pretraining establishes baseline syntactic and arithmetic fluency in $<5$ epochs. |
| **S-GRPO with CGI Trajectory Injection** | When all rollouts in a group fail on hard tasks, standard GRPO advantage normalization yields $\hat{A}_i = 0$, halting gradient flow. | PPO requires an auxiliary value network that doubles VRAM consumption on 4GB workstation hardware; vanilla GRPO stalls on hard tasks without reference trajectory injection. |
| **5-Tier Progressive Test Hierarchy** | Catching configuration, grammar, or gradient flow bugs during 10-to-20-hour runs introduces unacceptable diagnostic feedback latency. | Monolithic end-to-end integration tests waste compute and obscure root causes; the progressive hierarchy isolates failures in $<5\,\text{s}$ (Tier 0) to $<45\,\text{m}$ (Tier 3). |
| **Additive Homomorphism Loss in VICReg ($\mathcal{L}_{\text{add}}$)** | Unconstrained RL or naive SSL causes continuous latent embeddings to collapse into low-rank subspaces where vector arithmetic ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$) fails. | Contrastive InfoNCE suffers from class collisions (forcing mathematically equivalent expressions apart); standard Euclidean VICReg does not constrain linear algebraic operations. |

