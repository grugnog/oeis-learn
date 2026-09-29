# Quickstart & Validation Guide: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine

**Feature**: [specs/004-decoupled-grounding-and-symple-engine/spec.md](specs/004-decoupled-grounding-and-symple-engine/spec.md)  
**Branch**: `004-decoupled-grounding-and-symple-engine`  
**Date**: 2026-09-02

---

## 1. Prerequisites & Environment Setup

Ensure the Python virtual environment and native Rust WASM evaluator extensions are compiled and accessible:

```bash
# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Build native Rust WASM evaluator with Binaryen optimizations
cd crates/oeis_wasm_evaluator
cargo build --release
maturin develop --release
cd ../..

# 3. Verify external solver dependencies
python -c "import z3, mpmath, sympy; print('Dependencies verified successfully!')"
```

---

## 2. Validation Scenarios

### Scenario 1: Decoupled Constant Solver Validation (<2 seconds)
Validates that both the exact Hermite Normal Form (HNF) Diophantine solver and the Z3 SMT fallback resolve integer constants from AST skeletons:

```bash
# Run unit tests for linear Diophantine and non-linear SMT constant solving
pytest tests/unit/test_constant_solver.py -v
```

**Expected Outcome**:
- `test_linear_affine_grounding`: Recovers $c_0 = 2, c_1 = 5$ for $a(n) = 5n + 2$ in $<1.0\,\text{ms}$.
- `test_quadratic_diophantine`: Recovers quadratic parameters for triangular numbers in $<1.5\,\text{ms}$.
- `test_nonlinear_smt_modulo`: Solves modulo constants via Z3 within $250\,\text{ms}$.

---

### Scenario 2: Online Compiler Canonicalization & Waste Ratio (<3 seconds)
Validates that `wasm-opt` dead code elimination (DCE) strips redundant push-drop loops and computes syntactic waste $\rho_{\text{waste}}$:

```bash
# Run parsimony and dead code elimination unit tests
pytest tests/unit/test_parsimony_rlvr.py -v
```

**Expected Outcome**:
- Redundant `local.get ... drop` instructions are stripped in $<1.5\,\text{ms}$.
- Programs with $\rho_{\text{waste}} > 0.30$ receive zero validity reward.
- Lexicographic advantage assigns positive advantage to compact programs over padded equivalents.

---

### Scenario 3: Tri-Stream Encoder v2 & Summary Tokens (<3 seconds)
Validates normalized Newton forward differences ($D^{(k)}$), Prime Fourier Embeddings (PFE), and summary token regressions:

```bash
# Run Encoder v2 unit tests
pytest tests/unit/test_encoder_v2.py -v
```

**Expected Outcome**:
- Newton difference quotients for quadratic sequences yield constant second differences $D^{(2)} = \text{const}$.
- Prime Fourier embeddings produce 32-dim orthogonal representations across 16 odd prime fields.
- Auxiliary regression heads predict sequence slope $\hat{m}$ with $<1\%$ error.

---

### Scenario 4: SYMPLE Multi-Task Bandit & EDB Dormancy Replay (<5 seconds)
Validates EXP3.S bandit task selection, Ada-G dynamic group allocation ($G_i \in [8, 16]$), and EDB dormancy replay:

```bash
# Run SYMPLE curriculum and replay unit tests
pytest tests/unit/test_symple_curriculum.py -v
```

**Expected Outcome**:
- Frontier tasks with competence $\hat{p} \approx 0.1$ receive deep rollout groups ($G \ge 12$).
- Virtual sample injection restores non-zero negative advantages ($\hat{A}^- = -1/\sqrt{G}$) when all rollouts fail.
- Dormancy sampling selects sequences unvisited for the longest duration.

---

### Scenario 5: Full 4-Tier Pre-Flight Progressive Validation (<30 seconds)
Executes the unified pre-flight test hierarchy across Tier 0 (sandbox/parser), Tier 1 (SFT fitting), Tier 2 (single-prompt RL), and Tier 3 (micro-cohort SYMPLE):

```bash
python scripts/run_progressive_validation.py --max-tier 3
```

**Expected Outcome**:
- All tiers pass in $<30\text{ seconds}$.
- Telemetry verifies $\text{ACR} \le 0.05$, peak VRAM $<3.5\,\text{GB}$, and compiler trap rate $0.0\%$.

---

### Scenario 6: Production Run 006 Launch & Telemetry Monitoring
Launches the full 60-epoch production run with Phase 4 architecture:

```bash
python -m oeis_learn.cli.main train \
  --config configs/train_tier1.yaml \
  --tier 1 \
  --epochs 60 \
  --run-name 006_phase4_decoupled_symple
```

**Key Telemetry Signals to Monitor**:
- `acr_rate <= 0.05` (Advantage Collapse bounded)
- `syntactic_waste_ratio < 0.05` (DCE eliminates padding)
- `stage_1_competence >= 0.85` (Graduation achieved)
- `discovered_theorems >= 1` (PSLQ theorem discovery verified)
