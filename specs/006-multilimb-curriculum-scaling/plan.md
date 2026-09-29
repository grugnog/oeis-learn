# Implementation Plan: 4 × i64 Multi-Limb Migration, Corpus-Informed Curriculum Scaling, & Warm-Started Training Plan

**Branch**: `006-multilimb-curriculum-scaling` | **Date**: 2026-09-06 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from [spec.md](spec.md)

## Summary

This feature resolves the mathematical hardware truncation boundary that currently causes exact algebraic recurrences (Fibonacci $F_{93}$, Lucas $L_{91}$, Powers of 2 $2^{63}$, Pell $P_{38}$) to fail 100-term extrapolation due to 64-bit integer overflow. Grounded in our empirical census across 399,005 OEIS sequences, the system migrates to a fixed $4 \times i64$ (255-bit signed quad-limb, `i256x4_v1`) representation covering 99.51% of the entire OEIS.

The architecture adopts a **Hybrid Macro-Synthesized Architecture**: the autoregressive decoder emits compact high-level macros (`i256.add`, `i256.sub`, `i256.mul_scalar`, `i256.const`, etc.) lowered to a static, pre-verified value-stack WebAssembly preamble ($0$ heap memory, $10{,}000$ instruction fuel cap, keeping program ASTs under 40 tokens to preserve RL credit assignment). To solve undetermined coefficients without SMT bit-blasting timeouts, a decoupled three-tier symbolic grounding pipeline is introduced (Mersenne-61 fast modular filter $<0.5\text{ ms} \to$ Dixon 1-step Diophantine solver $<1.0\text{ ms} \to$ Z3 tactical QF_NIA $<240\text{ ms}$).

Continual learning executes via **Strategy C (Progressive SFT Bridge Transfer)** from the Run 010 Epoch 50 checkpoint: convex hull vocabulary expansion $\to$ 1,500-step embedding warmup $\to$ 5,000-step joint SFT bridge with discriminative learning rates $\to$ Transition Gate validation (ACR $\le 15\%$, grammar validity $\ge 98.5\%$, pass rate $\ge 75\%$) $\to$ reference policy re-anchoring $\to$ production Run 011 GRPO reinforcement learning protected by an Elite Demonstration Buffer and preflight qualification on 6 landmark canaries.

## Technical Context

**Language/Version**: Python 3.11+; Rust 2021 edition

**Primary Dependencies**: PyTorch 2.2+ (strict FP32 precision); NumPy; DuckDB; PyArrow; PyYAML; SymPy 1.12+; mpmath 1.3+; Z3 Solver (`z3-solver`); Wasmtime 20+, `wat`, Rayon, and PyO3 in the native evaluator; `jsonschema` for contract validation

**Storage**: Immutable versioned JSON manifests, schema definitions, and qualification reports under `reports/`; DuckDB databases (`data/oeis_corpus.duckdb`, `data/oeis_learn.duckdb`); versioned PyTorch checkpoints (`.pt`)

**Testing**: pytest unit, contract, and integration test suites; cargo test for native evaluator behavior; automated preflight canary benchmark suite on 6 landmark sequences

**Target Platform**: Linux x86_64 Tier 1 workstation with 4 CPU cores / 8 threads, up to 64 GB host RAM, and up to 4 GB GPU VRAM; generated programs execute in sandboxed WebAssembly

**Project Type**: Hybrid Python/Rust neuro-symbolic research library and execution engine

**Performance Goals**:

- 100% exact 100-term extrapolation across all 6 landmark canaries (`A000217`, `A000290`, `A000079`, `A000045`, `A000032`, `A000129`);
- Sustained multi-limb execution throughput $\ge 500$ module evaluations per second across 8 CPU worker threads;
- Tier 1 Mersenne-61 filter rejection latency $< 0.5\text{ ms}$ on invalid candidate skeletons;
- Tier 2 Dixon Diophantine coefficient resolution latency $< 1.0\text{ ms}$ on linear recurrence skeletons;
- Candidate program token length $\le 40$ tokens across Stages 1–3, preserving policy gradient credit assignment;
- Strategy C SFT bridge achieving Advantage Collapse Rate $\le 15\%$, grammar assembly validity $\ge 98.5\%$, and candidate pass rate $\ge 75\%$ before RL launch;
- 0 linear memory allocation traps and strict adherence to the 10,000 instruction fuel limit.

**Constraints**:

- All neural forward and backward passes execute in strict FP32 within the Tier 1 4 GB VRAM limit;
- WebAssembly sandboxed execution enforces zero linear memory (`max_memories(0)`) and 10,000 instruction fuel cap;
- SMT bit-blasting (`QF_BV`) for 256-bit recurrence systems is strictly prohibited;
- No direct RL warm-start without SFT bridge (preventing 100% advantage collapse);
- No cold-start from random weights (preserving 10+ hours of mathematical induction learning);
- Candidate programs must satisfy Minimum Description Length ratio $M_{\text{MDL}} \le 1.20$.

**Scale/Scope**:

- 399,005 sequences in full OEIS corpus, 74,986 `jOEIS` classes;
- 20,000 verified multi-limb synthetic demonstrations ($\mathcal{D}_{\text{SFT}}$);
- 150–200 sequences across Stages 1–4 in production Run 011;
- 6 landmark canary sequences with 120 verified terms (20 observed + 100 unseen);
- 5,000 SFT bridge steps + 60 epochs $\times$ 100 steps production GRPO.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / Gate | Plan Requirement | Status | Evidence & Enforcement |
| :--- | :--- | :---: | :--- |
| **I. Exact Multi-Axis & FP32** | Preserve Tri-Stream Encoder representations in strict FP32; eliminate 64-bit integer truncation via exact 255-bit two's complement decoding. | **PASS** | Encoder weights from Run 010 preserved in FP32; 4-limb value stack outputs decoded into exact Python arbitrary-precision integers without float approximations. |
| **II. Sound Grammar-Guided WAT** | Generate only through environment-indexed Earley grammar masking; lower high-level macros to verified preamble; guarantee 100% syntactically valid WASM. | **PASS** | Macro grammar enforces stack typing and lexical scopes; lowered WAT pre-checked via `wat::parse_str`; 0 syntax/environment errors. |
| **III. Deterministic Bounded Sandbox** | Native PyO3/Rayon execution; reset at most 10,000 fuel per invocation; enforce zero linear memory (`max_memories(0)`); classify every runtime trap. | **PASS** | Verified value-stack preamble uses 0 linear memory; native sandbox consumes 55–271 fuel per operation; terminates in $< 1\,\mu\text{s}$ per term. |
| **IV. Workstation-First Feasibility** | Bounded compute, micro-batches (4–8), model size within Tier 1 profile (4 CPU cores, 64 GB RAM, 4 GB VRAM). | **PASS** | Macro abstraction keeps decoder compact; SFT warmup takes ~20 min; joint bridge takes ~1.2 hr; production GRPO runs in ~8 hr on Tier 1 hardware. |
| **V. Curriculum & Anti-Memorization** | Require exact 20+100 verification on canaries; MDL ratio $\le 1.20$; 4-stage macro-scaffolding; flexible extrapolation margin. | **PASS** | Scaffolding templates mirror `jOEIS` taxonomy; canaries tested on 120 terms; general corpus tested on $\min(100, N_{\text{avail}}-N_{\text{obs}})$ with $\ge 15$ unseen margin. |
| **VI. Credit Assignment & Discovery** | Preserve EGCA-GRPO with compact ASTs ($\le 40$ tokens); prevent zero-advantage collapse via Elite Demonstration Buffer injection ($\sigma_{\mathcal{R}} > 0$). | **PASS** | Macro tokens preserve policy gradient signal ($\gamma^{25} \gg \gamma^{350}$); EDB seeds canonical trajectories for all evaluation prompts. |
| **TDD & Subsystem Gates** | Comprehensive unit, contract, and integration test fixtures; preflight canary benchmark suite before production launch. | **PASS** | Contracts defined in `contracts/`; test plans and quickstart scenarios defined in `quickstart.md`. |

No constitutional violation requires an exception or amendment.

## Project Structure

### Documentation (this feature)

```text
specs/006-multilimb-curriculum-scaling/
├── spec.md                          # Feature specification with 38 requirements
├── plan.md                          # This implementation plan
├── research.md                      # Phase 0 research & architectural decisions
├── data-model.md                    # Phase 1 data entities and relationships
├── quickstart.md                    # Phase 1 runnable validation scenarios
├── checklists/
│   └── requirements.md              # Requirements quality checklist (16/16 passed)
└── contracts/
    ├── multi-limb-preamble.wat      # Verified value-stack WebAssembly arithmetic library
    ├── macro-instruction-set.md     # Macro instruction specification & lowering rules
    ├── symbolic-grounding.schema.json # Three-tier grounding pipeline contract
    ├── warmstart-transfer-config.schema.json # Strategy C transfer & Transition Gate schema
    └── canary-benchmark.schema.json # 6 landmark canaries evaluation schema
```

### Source Code (repository root)

```text
crates/
└── oeis_wasm_evaluator/
    └── src/
        ├── engine.rs                # Multi-value Wasmtime engine configuration
        ├── lib.rs                   # PyO3 bindings for multi-limb execution
        └── sandbox.rs               # Quad-limb value-stack execution & fuel bounding

src/oeis_learn/
├── cli/
│   ├── evaluate_canaries.py         # Canary preflight qualification runner
│   ├── train_run011.py              # Production Run 011 training launcher
│   └── train_warmstart.py           # Strategy C progressive SFT bridge CLI
├── curriculum/
│   ├── macro_templates.py           # Stages 1-4 macro-scaffolding templates
│   └── scheduler.py                 # EXP3.S task bandit & adaptive sampling
├── data/
│   ├── models.py                    # Pydantic data models for multi-limb & grounding
│   └── synthetic_generator.py       # 20k multi-limb synthetic demonstration generator
├── decoder/
│   ├── constant_solver.py           # Decoupled 3-tier solver (M61, Dixon, Z3 QF_NIA)
│   ├── wat_grammar.py               # Macro tokens, vocabulary expansion, & grammar masking
│   └── transformer_decoder.py       # Logit soft-capping & unembedding calibration
├── rl/
│   ├── elite_buffer.py              # Elite Demonstration Buffer (EDB) & rollout injection
│   └── grpo_trainer.py              # EGCA-GRPO with re-anchored reference policy
└── sandbox/
    ├── preamble.wat                 # Inlined verified multi-limb math preamble
    ├── lowering.py                  # Macro lowering engine (i256.* -> preamble calls)
    └── runner.py                    # Multi-limb WASM runner with exact integer decoding

tests/
├── contract/
│   ├── test_macro_instruction_lowering.py
│   ├── test_preamble_contract.py
│   └── test_symbolic_grounding_contract.py
├── integration/
│   ├── test_canary_qualification.py
│   ├── test_symbolic_grounding_pipeline.py
│   └── test_transition_gate.py
└── unit/
    ├── test_dixon_diophantine_solver.py
    ├── test_mersenne61_filter.py
    ├── test_multi_limb_arithmetic.py
    └── test_vocabulary_surgery.py
```

**Structure Decision**: Hybrid Python/Rust modular layout. WebAssembly execution and multi-limb value-stack extraction are handled in native Rust (`crates/oeis_wasm_evaluator/`) with zero-copy PyO3 bindings. Macro grammar, vocabulary surgery, three-tier constant solving, curriculum scaffolding, and EGCA-GRPO training orchestration reside in `src/oeis_learn/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| *None* | *N/A* | All design choices fully comply with constitutional principles I through VI. |
