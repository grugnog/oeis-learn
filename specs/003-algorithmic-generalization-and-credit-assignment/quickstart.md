# Quickstart & Validation Guide: Inductive Algorithmic Generalization & Credit Assignment

**Feature**: [specs/003-algorithmic-generalization-and-credit-assignment/spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md)  
**Branch**: `003-algorithmic-generalization-and-credit-assignment`  
**Date**: 2026-09-01

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **OS:** Linux (Ubuntu 22.04 LTS or newer)
- **CPU:** 4+ Cores / 8+ Threads (e.g., Intel Xeon E3-1505M v5 @ 2.80GHz)
- **RAM:** 64 GB DDR4
- **GPU:** NVIDIA Quadro M2000M (4 GB VRAM) with CUDA 12.0+ support
- **Toolchains:** Python 3.11+, Rust 1.75+ (Cargo), DuckDB CLI

### Environment Activation

```bash
# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Verify native Rust WASM evaluator is compiled & functional
python3 -c "import oeis_wasm_evaluator; print('WASM Evaluator loaded successfully!')"
```

---

## 2. End-to-End Validation Scenarios

### Scenario 1: Pre-Flight Progressive Validation Suite ($< 5\text{ minutes}$)
Verifies deterministic unit checks, single-solution SFT likelihood fitting, single-prompt RL convergence, and micro-cohort curriculum advancement before launching long training jobs.

```bash
python -m oeis_learn.cli.main test-progressive --max-tier 3
```

**Expected Outcome:**
- All Tiers 0 through 3 pass within $<5\text{ minutes}$.
- Reference perplexity $\text{PPL}_{\text{ref}} \le 1.25$ on canonical solutions.
- Advantage Collapse Rate $\text{ACR} \le 0.10$.

---

### Scenario 2: Non-Triviality & Input Parameter Sensitivity Validation
Verifies that candidate programs emitting static constant sequences for dynamic targets receive zero surrogate reward and a static penalty.

```bash
pytest tests/unit/test_reward_evaluator.py -v -k "test_non_triviality"
```

**Expected Outcome:**
- Constant program candidates ($P(n) = C, \forall n$) on non-constant sequence targets yield $R_{\text{dist}} = 0, R_{\text{prefix}} = 0$ and non-triviality penalty.
- Programs referencing `$n` with empirical sensitivity $\mathcal{S}_{\text{input}}(P) > 0$ receive positive reward allocation.

---

### Scenario 3: Fine-Grained EGCA Credit Localization & Downstream Masking
Verifies that when a program correctly computes terms $n \in [0, 5]$ but fails at $n=6$, token advantages for $t > \max T_{k^*}$ are zero-masked and gradient mass is concentrated on the causal error span.

```bash
pytest tests/unit/test_egca_credit_assignment.py -v
```

**Expected Outcome:**
- Causal error span $T_{k^*}$ correctly localized.
- $\ge 90\%$ of gradient magnitude concentrated on $T_{k^*}$ with exact total advantage conservation $\sum a_{i,t} = A_i$.

---

### Scenario 4: Demonstration Co-Training & Policy Regularization
Verifies mixed SFT + RL co-training loss computation, Schulman KL divergence estimation against $\pi_{\text{ref}}$, and decoder padding attention masking.

```bash
pytest tests/integration/test_co_training_step.py -v
```

**Expected Outcome:**
- Total loss evaluates $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta_{\text{SFT}}\mathcal{L}_{\text{SFT}} + \beta_{\text{KL}}\mathbb{D}_{\text{KL}}$.
- Reference token perplexity remains $\text{PPL}_{\text{ref}} \le 1.30$.
- Zero gradient flow across padded positions (`PAD_ID`).

---

### Scenario 5: Potential-Based Reward Shaping & Lexicase Policy Optimization Run
Execute online RL training under Tier 1 workstation constraints, integrating non-triviality gating, PBRS potentials, down-sampled lexicase selection, and SFT co-training.

```bash
python -m oeis_learn.cli.main train \
    --config configs/train_tier1.yaml \
    --sft-checkpoint checkpoints/sft_warmup_best.pt \
    --tier 1 \
    --curriculum-stage 1 \
    --beta-sft 0.20 \
    --beta-kl 0.05 \
    --enable-pbrs \
    --enable-lexicase
```

**Expected Outcome:**
- Stage 1 competence $C(S_1)$ exceeds $0.80$ within 30 epochs.
- Policy entropy maintains $\mathcal{H}(\pi_\theta) \ge 1.50$.
- Advantage Collapse Rate maintains $\text{ACR} \le 0.10$.

---

### Scenario 6: Generalization Extrapolation ($K=100$) & Automated Theorem Discovery
Synthesize candidate programs for test sequences, verify 100% exact match across 100 unseen future terms ($n \in [20, 119]$) with $M_{\text{MDL}} \le 1.20$, and discover algebraic relations via PSLQ ($<10^{-50}$ drop) and SymPy proofs.

```bash
# 1. Synthesize and verify extrapolation for Powers of 2 (A000079)
python -m oeis_learn.cli.main synthesize \
    --oeis-id A000079 \
    --checkpoint checkpoints/model_epoch_030.pt \
    --extrapolate 100 \
    --mdl-max 1.20

# 2. Run automated discovery and prove novel relations
python -m oeis_learn.cli.main discover \
    --checkpoint checkpoints/vicreg_latent.pt \
    --max-candidates 50 \
    --precision-digits 500 \
    --output-proofs reports/discovered_proofs_phase3.md
```

**Expected Outcome:**
- Extrapolation horizon test passes 100/100 unseen future terms ($K=100$).
- Minimum Description Length satisfies $M_{\text{MDL}} \le 1.20$.
- Discovered theorems verified by PSLQ ($<10^{-50}$ drop) and exported with formal SymPy symbolic proofs.
