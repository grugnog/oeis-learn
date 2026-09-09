# Quickstart Validation Guide: 4 × i64 Multi-Limb Migration & Curriculum Scaling

**Feature Branch**: `006-multilimb-curriculum-scaling`  
**Date**: 2026-09-06  
**Status**: Completed Phase 1 Design

This guide provides step-by-step commands to validate the multi-limb execution engine, decoupled symbolic grounding pipeline, warm-started continual learning bridge, and canary preflight qualification.

---

## 1. Prerequisites & Environment Setup

Verify the Python virtual environment and native Rust WASM evaluator:

```bash
# Verify active environment and dependencies
.venv/bin/python --version
cargo --version

# Verify native evaluator build
cargo test -p oeis_wasm_evaluator
```

---

## 2. Validation Scenario A: Multi-Limb Execution & Truncation-Free Arithmetic

Validate that the $4 \times i64$ value-stack preamble executes correctly and reconstructs exact signed integers beyond 64-bit bounds:

```bash
# Run unit and contract tests for multi-limb arithmetic and macro lowering
pytest tests/unit/test_multi_limb_arithmetic.py -v
pytest tests/contract/test_macro_instruction_lowering.py -v
```

**Expected Outcome**:
- Fibonacci $F_{93} = 12{,}200{,}160{,}415{,}121{,}876{,}738$ and $F_{100} = 354{,}224{,}848{,}179{,}261{,}915{,}075$ evaluate exactly without wrapping to negative numbers.
- Consumes zero linear memory (`max_memories(0)`).
- Consumes $< 1{,}500$ fuel units for 100 terms.

---

## 3. Validation Scenario B: Decoupled Symbolic Grounding Pipeline

Validate the three-tier grounding pipeline (Mersenne-61 fast filter, Dixon Diophantine solver, Z3 QF_NIA):

```bash
# Run tests for Mersenne-61 modular rank filter and Dixon bounded solver
pytest tests/unit/test_mersenne61_filter.py -v
pytest tests/unit/test_dixon_diophantine_solver.py -v
pytest tests/integration/test_symbolic_grounding_pipeline.py -v
```

**Expected Outcome**:
- Inconsistent and underdetermined skeletons rejected in $< 0.5\text{ ms}$ with status `INCONSISTENT` or `UNDERDETERMINED`.
- Linear recurrence skeletons grounded to exact integer coefficients in $< 1.0\text{ ms}$.
- Zero SMT bit-blasting (`QF_BV`) timeouts.

---

## 4. Validation Scenario C: Strategy C Progressive SFT Warmstart Transfer

Validate the vocabulary surgery, convex hull projection, and Transition Gate validation:

```bash
# Run vocabulary surgery test on Run 010 checkpoint
pytest tests/unit/test_vocabulary_surgery.py -v

# Run SFT bridge training smoke check (100 steps)
.venv/bin/python -m oeis_learn.cli.train_warmstart \
  --base-checkpoint checkpoints/model_epoch_050.v2.pt \
  --stage warmup \
  --steps 100 \
  --dry-run

# Evaluate Transition Gate criteria
pytest tests/integration/test_transition_gate.py -v
```

**Expected Outcome**:
- Embedding rows for `$a0`..`$t3`, `i256.add`, `i256.sub`, `i256.mul_scalar` initialized within the convex hull of semantic ancestors.
- Unembedding head norms match pre-existing vocabulary mean norm.
- Logit soft-capping active ($C_{\text{cap}} = 30.0$).
- Transition Gate checks: Advantage Collapse Rate $\le 15\%$, Grammar Validity $\ge 98.5\%$, Pass Rate $\ge 75\%$.

---

## 5. Validation Scenario D: Canary Preflight Verification (6 Landmark Sequences)

Evaluate the 6 landmark canaries across 20 observed terms and 100 unseen terms:

```bash
# Run canary preflight qualification
.venv/bin/python -m oeis_learn.cli.evaluate_canaries \
  --checkpoint checkpoints/model_epoch_050.v2.pt \
  --profile i256x4_v1 \
  --output reports/canary_qualification_report.json
```

**Canary Target Sequences**:
1. `A000217`: Triangular numbers ($n(n+1)/2$)
2. `A000290`: Squares ($n^2$)
3. `A000079`: Powers of 2 ($2^n$, overflows 64-bit at $n=63$)
4. `A000045`: Fibonacci numbers ($F_n$, overflows 64-bit at $n=93$)
5. `A000032`: Lucas numbers ($L_n$, overflows 64-bit at $n=91$)
6. `A000129`: Pell numbers ($P_n$, overflows 64-bit at $n=38$)

**Expected Outcome**:
- All 6 canaries achieve `EXTRAPOLATING_SUCCESS` over 100 unseen terms.
- 0 integer truncation or overflow failures.

---

## 6. Validation Scenario E: Production Run 011 Launch Verification

Launch production Run 011 in workstation tier 1 mode:

```bash
# Launch Run 011 with phased curriculum and EDB anti-collapse protection
.venv/bin/python -m oeis_learn.cli.train_run011 \
  --config configs/train_run011.yaml \
  --device cuda
```

**Expected Outcome**:
- 150–200 sequences across Stages 1 through 4.
- Advantage variance $\sigma_{\mathcal{R}} > 0$ across 100% of batches via Elite Demonstration Buffer injection.
- Telemetry logged to `runs/011_multilimb_256bit_production/`.
