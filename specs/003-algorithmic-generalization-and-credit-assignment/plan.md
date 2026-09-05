# Implementation Plan: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment

**Branch**: `003-algorithmic-generalization-and-credit-assignment` | **Date**: 2026-09-01 | **Spec**: [specs/003-algorithmic-generalization-and-credit-assignment/spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md)

**Input**: Feature specification from [specs/003-algorithmic-generalization-and-credit-assignment/spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md)

---

## Summary

`oeis-learn` Phase 3 eliminates degenerate constant shortcut collapse, enforces input parameter sensitivity ($I(n; P(n)) > 0$), anchors policy representations through supervised demonstration co-training, and localizes gradient updates onto causal bytecode error spans via Execution-Grounded Credit Assignment (EGCA).

The technical approach integrates:
1. **Non-Degenerate Reward Design & Input Sensitivity:** Output variance gating ($\mathbb{Var}_n[P(n)]$), input parameter sensitivity checks ($\mathcal{S}_{\text{input}}(P) = \sum |P(n+1) - P(n)|$), and a batch-level cross-input mutual information proxy ($R_{\text{MI}}$) that zeros out surrogate rewards and penalizes static constants.
2. **Demonstration Co-Training & Policy Regularization:** Online loss blending combining group-relative policy gradients with teacher-forced SFT cross-entropy loss over the elite replay buffer ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta_{\text{SFT}} \mathcal{L}_{\text{SFT}}$ with default $\beta_{\text{SFT}} = 0.20$), bounded by unbiased Schulman per-token reference model KL divergence ($\beta_{\text{KL}} = 0.05$) and entropy regularization ($\alpha_{\text{ent}} = 0.01$).
3. **Fine-Grained Execution-Grounded Credit Assignment (EGCA):** Priority-gated failure classification (`SYNTAX`, `CONSTRAINT`, `LOGIC`, `CORRECT`), sequence-to-token divergence mapping ($k^* \to T_{k^*}$), downstream token zero-masking ($t > \max T_{k^*}$), and total advantage mass conservation ($\sum a_{i,t} = A_i$).
4. **Potential-Based Reward Shaping (PBRS) & Lexicase Selection:** Telescoping potential differences ($\gamma \Phi(s') - \Phi(s)$) over AST structural phases and variable bindings ($\phi_{\text{bind}}$) guaranteeing policy invariance relative to $R_{\text{exact}}$, coupled with down-sampled lexicase rollout evaluation over randomized test cases.
5. **Exact Padding Attention Masks & VRAM Chunking:** Strict `tgt_key_padding_mask` enforcement in Transformer attention layers and mini-chunk logit projections ($L_{\text{chunk}} = 256$), capping GPU memory to $<3.5\,\text{GB}$ VRAM in strict FP32 precision.
6. **Generalization Extrapolation ($K=100$) & Automated Theorem Proving:** Anti-memorization MDL ratio verification ($M_{\text{MDL}} \le 1.20$), 100-term out-of-distribution evaluation ($n \in [20, 119]$), and additive homomorphism regularized Kernel VICReg discovery verified by PSLQ ($<10^{-50}$ drop) and SymPy symbolic proofs.

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

**Testing**: `pytest` for Python unit/integration tests and progressive validation; `cargo test` for native Rust WASM evaluator crate.

**Target Platform**: Linux (x86_64), Ubuntu 22.04 LTS / 24.04 LTS with NVIDIA CUDA 12.0+

**Project Type**: Hybrid Python / Rust Neuro-Symbolic ML Framework & CLI Tool

**Performance Goals**:
- $\ge 500$ WASM module evaluations/second across 8 CPU threads via PyO3/Rayon bridge.
- Sub-$100\,\mu\text{s}$ per-token grammar masking evaluation latency.
- Non-triviality parameter usage rate $\ge 95\%$ on non-constant tasks.
- Credit attribution localization concentrating $\ge 90\%$ gradient mass on causal error spans.
- Curriculum Stage 1 rolling competence $C(S_1) \ge 0.80$ under Tier 1 workstation constraints.
- 100% exact match on $K=100$ extrapolation terms for graduated algorithms.

**Constraints**:
- *Tier 1 Workstation Baseline:* 4 CPU Cores / 8 Threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- *Strict Single Precision:* All neural operations execute in `torch.float32` without mixed precision (AMP).
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
| **Principle II: Provably Sound WAT Synthesis** | Dynamic Earley trie grammar masking (`llguidance`) with Environment-Indexed scope tracking ($\mathcal{G}_{\Gamma_t}$). | **PASSED** | Formal EBNF contract defined in `wat-grammar.ebnf`; parameter `$n` bindings and No-Ghost Soundness enforced by structural state machine. |
| **Principle III: Sandboxed Deterministic Execution** | In-memory compilation, 10,000 fuel cap, 16 MiB linear memory, GIL-free PyO3/Rayon execution. | **PASSED** | Native Rust crate `oeis_wasm_evaluator` provides Cranelift fuel metering, linear memory isolation, and multi-core batch throughput ($>5,500$ evals/sec). |
| **Principle IV: Workstation Feasibility** | Tier 1 baseline ($d=256/384$, batch 1–4, 8-thread CPU offload) on 4-core Xeon / 4GB Quadro M2000M. | **PASSED** | Configuration profiles `configs/train_tier1.yaml` enforce $d=256/384$, sequence chunking ($L_{\text{chunk}}=256$), and 100% CPU offload for WASM evals. |
| **Principle V: Curriculum & Anti-Memorization** | 5-stage curriculum with $C(S_k) \ge 0.85$, $N+K$ ($K=100$) extrapolation testing, and $M_{\text{MDL}} \le 1.20$. | **PASSED** | Automated curriculum gates, extrapolation horizon verifier ($K=100$), and Minimum Description Length checks ($M_{\text{MDL}} \le 1.20$) integrated into training pipeline. |
| **Principle VI: Credit Assignment & Discovery** | EGCA-GRPO with downstream zero-masking, SFT co-training, PBRS, Kernel VICReg homomorphism loss, PSLQ integer relations, SymPy proofs. | **PASSED** | Fine-grained EGCA credit localization, SFT co-training loss, PBRS potential differences, and PSLQ theorem verification designed in `data-model.md` and `research.md`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/003-algorithmic-generalization-and-credit-assignment/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan (this document)
├── research.md          # Technical decisions and architectural research
├── data-model.md        # Domain entities, schemas, and state transitions
├── quickstart.md        # Setup instructions and runnable validation scenarios
├── checklists/
│   └── requirements.md  # Specification quality checklist
└── contracts/
    ├── cli-interface.contract.json      # Expanded CLI interface schema
    ├── credit-attribution.schema.json   # Localized EGCA credit assignment schema
    ├── regularization-config.schema.json# Anti-shortcut & co-training configuration schema
    └── wat-grammar.ebnf                 # Dynamic Environment-Indexed WAT Grammar
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
    │   ├── models.py                    # Domain entities (NonTrivialityEvaluation, CoTrainingBatch, etc.)
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
    │   ├── wat_decoder.py               # Transformer decoder with exact tgt_key_padding_mask
    │   ├── grammar_masker.py            # Dynamic Earley trie logit masker (llguidance bridge)
    │   └── environment_tracker.py       # Structural phase (Phi_t), scope (Vars_t), and stack (Sigma_t) tracker
    ├── sandbox/                         # Python WASM Sandbox Wrapper & Execution Manager
    │   ├── __init__.py
    │   ├── runner.py                    # Wrapper invoking oeis_wasm_evaluator
    │   └── tracer.py                    # Execution trace analyzer with causal span & coverage mapping
    ├── rl/                              # Reinforcement Learning Pipeline
    │   ├── __init__.py
    │   ├── sft_trainer.py               # Supervised Fine-Tuning teacher-forcing trainer
    │   ├── elite_buffer.py              # Elite seed demonstration replay buffer (D_elite)
    │   ├── egca_grpo.py                 # EGCA-GRPO trainer with downstream zero-masking & KL penalty
    │   ├── prompt_weighting.py          # S-GRPO trajectory injection, AVSPO & lexicase filtering
    │   ├── reward.py                    # Non-triviality gating, output variance, R_MI proxy & PBRS shaping
    │   ├── telemetry.py                 # Real-time diagnostic telemetry logger (entropy, ACR, traps)
    │   └── trainer.py                   # Unified curriculum RL optimization manager with SFT co-training
    ├── curriculum/                      # 5-Stage Taxonomy-Aligned Curriculum Engine
    │   ├── __init__.py
    │   ├── scheduler.py                 # Competence scoring C(Sk) & automated graduation gates
    │   ├── sampler.py                   # Dynamic mixture prompt sampling with down-sampled lexicase
    │   ├── extrapolation.py             # Extrapolation Horizon (N+K, K=100) verifier
    │   └── mdl_verifier.py              # Minimum Description Length (M_MDL <= 1.20) complexity test
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
        └── main.py                      # CLI entrypoint (train, test-progressive, synthesize, discover)

configs/
├── train_tier1.yaml                     # Tier 1 configuration with Phase 3 co-training & PBRS hyperparams
└── train_tier2.yaml                     # Cluster scale-up configuration

tests/
├── unit/
│   ├── test_tri_stream_encoder.py       # Encoder numerical stability & FP32 precision tests
│   ├── test_film_fusion.py              # Hierarchical FiLM modulation unit tests
│   ├── test_grammar_masker.py           # Dynamic Earley trie masking & environment tracking tests
│   ├── test_wasm_sandbox.py             # 10,000 fuel limit, memory ceiling, and trap tests
│   ├── test_curriculum_gates.py         # Competence scoring & extrapolation verification tests
│   ├── test_vicreg_loss.py              # VICReg & additive homomorphism loss tests
│   ├── test_reward_evaluator.py         # Non-triviality gating, variance, R_MI & PBRS unit tests
│   ├── test_egca_credit_assignment.py   # Downstream zero-masking and advantage conservation tests
│   └── test_prompt_weighting.py         # Lexicase rollout filtering and CGI injection tests
├── integration/
│   ├── test_co_training_step.py         # Mixed SFT + RL co-training loss & KL penalty step
│   ├── test_tier1_oracle_fitting.py     # Tier 1 single canonical solution fitting test
│   ├── test_tier2_single_prompt_rl.py   # Tier 2 single-prompt RL convergence test
│   ├── test_tier3_micro_cohort.py       # Tier 3 synthetic micro-cohort progression test
│   └── test_discovery_pipeline.py       # Latent vector arithmetic -> PSLQ -> SymPy proof flow
└── contract/
    ├── test_ffi_contract.py             # Native PyO3 bridge contract test
    ├── test_cli_contract.py             # CLI arguments and output format contract test
    ├── test_progressive_harness_contract.py # Progressive test harness contract test
    └── test_wat_grammar_contract.py     # EBNF grammar completeness and parser test
```

**Structure Decision**: The hybrid Python/Rust structure established in Phases 1 and 2 is extended with dedicated modules for anti-shortcut reward regularization, downstream zero-masking credit assignment in `src/oeis_learn/rl/` and `src/oeis_learn/sandbox/tracer.py`, and demonstration co-training in `src/oeis_learn/rl/trainer.py`.

---

## Complexity Tracking

| Component / Pattern | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| **Non-Triviality Gating & Output Variance Check** | Dense surrogate distance rewards allow static constants ($P(n) = C$) to achieve safe partial credit without runtime traps, collapsing the policy gradient. | Pure binary rewards without gating suffer extreme sample inefficiency and zero-advantage exploration deserts. |
| **SFT Demonstration Co-Training ($\beta_{\text{SFT}} = 0.20$)** | Standalone RL with sparse rewards rapidly forgets structured loop constructs learned during SFT warmup (policy drift). | Increasing SFT warmup epochs alone does not prevent RL policy drift after 10+ epochs of online exploration. |
| **Downstream Token Zero-Masking in EGCA** | Stack bytecode errors cascade; applying uniform negative advantage to an entire failed program penalizes valid function headers and loop setups. | Full sequence advantage broadcasting destroys structural fluency; parametric PRM critics suffer from value drift on OOD bytecodes. |
| **Potential-Based Reward Shaping (PBRS)** | Heuristic distance metrics alter the optimal policy landscape, making static constants the global optimum under shaped rewards. | Unshaped rewards fail to guide intermediate exploration; PBRS mathematically guarantees policy invariance relative to $R_{\text{exact}}$. |
| **Decoder Attention Padding Masking (`tgt_key_padding_mask`)** | Variable-length batched sequences allow attention and backpropagation gradients to leak across padded token positions (`PAD_ID`). | Truncating all sequences to the minimum batch length loses critical control-flow suffix tokens in complex programs. |
