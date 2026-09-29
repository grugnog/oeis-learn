# Quickstart & Validation Guide: Synthesis Bootstrapping & Progressive Optimization

**Feature**: [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md)  
**Branch**: `002-synthesis-bootstrapping-and-soundness`  
**Date**: 2026-08-31

---

## 1. Prerequisites & Environment Verification

### System Requirements
- **OS:** Linux (Ubuntu 22.04 LTS or newer)
- **CPU:** 4+ Cores / 8+ Threads (e.g., Intel Xeon E3-1505M v5)
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

## 2. Progressive Validation Hierarchy (Pre-Flight Suite)

Execute the 5-tier progressive validation hierarchy to verify compilation soundness, gradient flow, and RL exploration before launching multi-hour training.

### Step 1: Tier 0 Deterministic Unit & Sandbox Validation ($<5\text{ seconds}$)
Verifies in-memory compilation, 10,000 instruction fuel traps, linear memory bounding, and static grammar boundaries without neural passes.

```bash
pytest tests/unit/test_wasm_sandbox.py tests/unit/test_grammar_masker.py -v
```

**Expected Gate Threshold:**
- 100% of infinite loop programs trapped in $<1\,\text{ms}$ with `OUT_OF_FUEL`.
- Zero host process crashes, memory leaks, or unhandled exceptions.

---

### Step 2: Tier 1 Oracle Solution Fitting & SFT Likelihood ($<2\text{ minutes}$)
Verifies that the encoder-decoder policy network contains sufficient capacity and correct gradient flow to overfit canonical reference programs.

```bash
pytest tests/integration/test_tier1_oracle_fitting.py -v
```

**Expected Gate Threshold:**
- Reference token perplexity $\text{PPL}_{\text{ref}} < 1.25$ achieved within 20 optimization steps.
- Cross-entropy loss decays monotonically below $0.20$.

---

### Step 3: Tier 2 Single-Prompt RL Policy Convergence ($<10\text{ minutes}$)
Verifies GRPO policy gradient updates, advantage variance, and sampling dynamics on an isolated benchmark sequence (Triangular Numbers A000217).

```bash
pytest tests/integration/test_tier2_single_prompt_rl.py -v
```

**Expected Gate Threshold:**
- 100% rollout pass rate achieved within 15 gradient iterations ($G=4$).
- Policy probability mass shifts decisively toward valid generating bytecode.

---

### Step 4: Tier 3 Synthetic Micro-Cohort Curriculum Progression ($<45\text{ minutes}$)
Verifies rolling competence scoring $C(S_k)$, automated graduation gates, and hyperparameter stability across 10–20 synthetic tasks.

```bash
pytest tests/integration/test_tier3_micro_cohort.py -v
```

**Expected Gate Threshold:**
- Rolling competence reaches $C(S_1) \ge 0.85$ and triggers automated promotion to Stage 2.
- Advantage Collapse Rate remains bounded at $\text{ACR} \le 0.15$.

---

## 3. End-to-End Bootstrapped Training & Synthesis Workflow

### Step 5: Generate Synthetic Demonstration Dataset
Generate 5,000 diverse forward-executed program-sequence pairs covering linear, quadratic, cubic, recurrence, and modular algorithmic families.

```bash
python -m oeis_learn.cli.main generate-sft \
    --output-path data/sft_demonstrations.json \
    --num-samples 5000 \
    --seed 42
```

---

### Step 6: Supervised Fine-Tuning (SFT) Warmup
Pretrain the Transformer policy on synthetic demonstrations using teacher-forced cross-entropy loss to establish initial syntactic fluency.

```bash
python -m oeis_learn.cli.main warmup-sft \
    --dataset-path data/sft_demonstrations.json \
    --output-checkpoint checkpoints/sft_warmup_best.pt \
    --epochs 5 \
    --lr 0.0005
```

**Expected Outcome:**
- SFT loss converges to $<0.50$.
- Greedy synthesis on Stage 1 polynomial tasks achieves $>80\%$ functional compilation and sequence accuracy.

---

### Step 7: Online S-GRPO Policy Optimization with Trajectory Injection
Launch online reinforcement learning initialized from the warmed-up SFT policy, utilizing Conditional Ground-Truth Trajectory Injection (CGI) and cosine-decayed reward shaping.

```bash
python -m oeis_learn.cli.main train \
    --config configs/train_tier1.yaml \
    --sft-checkpoint checkpoints/sft_warmup_best.pt \
    --tier 1 \
    --curriculum-stage 1 \
    --enable-cgi
```

**Expected Outcome:**
- Rolling competence score $C(S_1)$ exceeds $0.80$ within 25 epochs.
- Real-time diagnostic telemetry maintains $\text{ACR} \le 0.15$ and policy entropy $\mathcal{H} \ge 1.0$.

---

### Step 8: Synthesis & Anti-Memorization Extrapolation Verification
Synthesize a candidate algorithm for a target sequence and verify generalization across 100 unseen future terms.

```bash
# Synthesize program for Triangular Numbers (A000217)
python -m oeis_learn.cli.main synthesize \
    --oeis-id A000217 \
    --checkpoint checkpoints/model_epoch_025.pt \
    --extrapolate 100
```

**Expected Outcome:**
- Synthesized WAT program compiles in-memory with 0 syntax or scoping errors.
- Extrapolation horizon test passes 100/100 unseen terms ($K=100$).
- Minimum Description Length ratio satisfies $M_{\text{MDL}} \le 1.20$.
