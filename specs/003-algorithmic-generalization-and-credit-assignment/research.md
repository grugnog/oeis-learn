# Research & Technical Decisions: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment

**Feature**: [specs/003-algorithmic-generalization-and-credit-assignment/spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md)  
**Branch**: `003-algorithmic-generalization-and-credit-assignment`  
**Date**: 2026-09-01

---

## 1. Anti-Shortcut Non-Triviality Gating, Output Variance & Cross-Input Mutual Information

### Problem Statement & Root Cause
In Run 002 (Phase 2), 100% of generated programs compiled without parser traps, but the reinforcement learning policy collapsed into emitting degenerate static constants (e.g., `i64.const 16` returning `[16, 16, 16...]` for $2^n$).
Root causes:
1. **Surrogate Metric Exploitation:** Heuristic distance rewards awarded safe partial credit for constants near sequence values, while loops carried higher execution variance.
2. **Signal-to-Noise Ratio (SNR) Collapse:** When all rollouts in a group output identical constants, within-group reward variance $\mathbb{Var}_g[R]$ vanishes ($\|g_{\text{task}}\| \to 0$), leaving the regularization gradient $g_{\text{reg}}$ to contract the policy into an input-agnostic prior ($I(n; P(n)) \to 0$).

### Technical Decisions

1. **Empirical Output Variance & Input Sensitivity Gating:**
   - Compute empirical variance $\mathbb{Var}_n[P(n)] = \frac{1}{N}\sum (P(n) - \mu_P)^2$ across sequence terms $n \in [0, N-1]$.
   - Compute empirical input sensitivity $\mathcal{S}_{\text{input}}(P) = \sum_{n=0}^{N-2} |P(n+1) - P(n)|$.
   - **Non-Triviality Gate:** If the target sequence has non-zero variance ($\mathbb{Var}_n[y_n] > 0$) but the candidate program has $\mathbb{Var}_n[P(n)] < 10^{-6}$ or $\mathcal{S}_{\text{input}}(P) == 0$, all surrogate rewards are strictly zeroed ($R_{\text{dist}} = 0, R_{\text{prefix}} = 0$) and a static penalty $R_{\text{non\_trivial}} = -0.5$ is applied.

2. **Batch-Level Cross-Input Mutual Information Proxy ($R_{\text{MI}}$):**
   - For a minibatch of $B$ tasks, evaluate the normalized cosine similarity $\mathbf{S}_{i,j}$ of executed output embedding vectors between distinct tasks $i \ne j$.
   - Compute $R_{\text{MI}}(P_i) = -\log \left( \frac{1}{B-1} \sum_{j \ne i} \exp\left( \frac{\mathbf{S}_{i,j}}{\tau} \right) \right)$.
   - Outputs that remain identical across different sequence prompts are penalized heavily, ensuring $I(n; P(n)) > 0$.

3. **Active Parameter `$n$` Binding Check:**
   - Inspect AST tokens; if the candidate code contains zero occurrences of `local.get $n` or parameter transformations, the program is penalized as unparameterized.

### Alternatives Considered
- **Pure Terminal Binary Rewards Without Gating:** Rejected; leads to high sample inefficiency and early exploration starvation on complex recurrence tasks.
- **Unconstrained Output Range Penalty:** Rejected; fails on sequences with small ranges (e.g., binary sequences $0, 1, 0, 1$) which are non-trivial despite small dynamic ranges.

---

## 2. Demonstration Co-Training & Unbiased Schulman KL Regularization

### Problem Statement & Root Cause
Standalone reinforcement learning initialized from an SFT checkpoint rapidly suffers from policy drift and catastrophic forgetting. Over 20–30 epochs of online exploration, the policy drifts away from structured control loops (`loop`, `block`, `br_if`, multi-variable accumulators) acquired during SFT warmup toward short static templates.

### Technical Decisions

1. **Hybrid SFT + RL Co-Training Objective:**
   - Blend online policy gradient loss with supervised cross-entropy loss over canonical demonstrations from the elite replay buffer $\mathcal{D}_{\text{elite}}$:
     $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) + \beta_{\text{SFT}} \mathcal{L}_{\text{SFT}}(\theta)$$
   - Default mixing coefficient $\beta_{\text{SFT}} = 0.20$.
   - Ensures continuous supervised gradient flow on complex algorithmic structures throughout RL training.

2. **Unbiased Per-Token Schulman KL Penalty:**
   - Maintain a frozen reference model $\pi_{\text{ref}}$ (the warmed-up SFT baseline).
   - Penalize policy divergence using Schulman's sample-based estimator:
     $$\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \approx \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - \log \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - 1$$
   - Added to the objective with coefficient $\beta_{\text{KL}} = 0.05$.
   - Strictly non-negative and variance-reduced without requiring an auxiliary critic network.

3. **Entropy Regularization Bonus:**
   - Maintain exploratory token entropy via $\alpha_{\text{ent}} \mathcal{H}(\pi_\theta)$ with $\alpha_{\text{ent}} = 0.01$, keeping token entropy $\mathcal{H} \ge 1.50$.

### Alternatives Considered
- **Full Sequence KL Divergence:** Rejected due to excessive compute overhead on autoregressive sequences.
- **Pure SFT Warmup Without Co-Training:** Rejected; empirical evidence showed that policies drift within 15 epochs without continuous co-training.

---

## 3. Fine-Grained Execution-Grounded Credit Assignment (EGCA) & Coverage Tracing

### Problem Statement & Root Cause
In stack-based WebAssembly bytecode, a single corrupted token late in the sequence breaks all downstream stack operations. Broadcasting uniform sequence returns $\hat{A}_i$ across all tokens heavily penalizes valid module headers, parameter signatures, and initial variable initializations alongside the single faulty operator.

### Technical Decisions

1. **Deterministic Failure Gating:**
   - Rollouts are classified into four priority gates:
     $$m(y) \in \{\text{SYNTAX}, \text{CONSTRAINT}, \text{LOGIC}, \text{CORRECT}\}$$

2. **Divergence Localization & Token Span Mapping:**
   - For `LOGIC` failures on test case $d$, identify the earliest sequence step $k^*$ where candidate output $P(n) \ne y_n$.
   - Map execution step $k^*$ to the causal bytecode token span $T_{k^*}$ using AST instruction boundaries.

3. **Downstream Token Zero-Masking & Advantage Conservation:**
   - For `LOGIC` and `SYNTAX` failures, tokens generated *after* the causal error span ($t > \max T_{k^*}$) receive zero advantage ($a_{i,t} = 0$).
   - Tokens preceding the error ($t < \min T_{k^*}$) are protected from negative updates.
   - Total advantage is strictly conserved: $\sum_{t=1}^T a_{i,t} = A_i$, concentrating 100% of gradient mass onto the localized causal error window:
     $$a_{i,t} = \frac{A_i}{|T_{k^*}|} \mathbf{1}[t \in T_{k^*}]$$

4. **Runtime Instruction Coverage Tracing:**
   - Track basic block execution coverage during sandbox evaluation.
   - Mask out unexecuted basic block tokens ($M_t = 0$) from policy gradient updates (FGO principle).

### Alternatives Considered
- **Uniform Sequence Advantage Broadcasting:** Rejected; causes severe credit smear and suppresses exploration of valid loop headers.
- **Parametric Process Reward Model (PRM Critic):** Rejected due to value estimation drift on out-of-distribution code and doubling VRAM consumption.

---

## 4. Potential-Based Reward Shaping (PBRS) & Down-Sampled Lexicase Selection

### Problem Statement & Root Cause
Heuristic distance metrics violate policy invariance, creating artificial global optima corresponding to average static constants. Batch-averaged distances favor compromise generalists over per-input specialists.

### Technical Decisions

1. **Potential-Based Reward Shaping (PBRS):**
   - Auxiliary dense rewards are formulated strictly as potential differences $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ over AST completion states:
     $$\Phi(s) = \phi_{\text{comp}}(s) + \phi_{\text{bind}}(s) + \Phi_{\text{terminal}}(P)$$
   - $\phi_{\text{comp}}(s) > 0$: Allocated upon reaching valid structural phases.
   - $\phi_{\text{bind}}(s) > 0$: Allocated when AST binds `$n` within a control loop or arithmetic operation.
   - Telescoping property: $\sum F = \gamma^T \Phi(s_T) - \Phi(s_0)$, mathematically guaranteeing policy invariance relative to $R_{\text{exact}}$.

2. **Down-Sampled Lexicase Rollout Selection:**
   - For rollout group evaluation, evaluate candidate programs sequentially on randomized individual test cases $n_r \in \{0, \dots, N-1\}$.
   - Only candidates that are elite on the current test case survive to the next filtering step.
   - Eliminates static constants immediately, as they fail on non-zero domain points.

3. **Competence-Driven Cosine Annealing:**
   - Anneal surrogate potential scale dynamically based on rolling task competence $C(S_k)$:
     $$w_{\text{surr}}(t) = w_0 \cdot \cos\left( \frac{\pi \cdot C(S_k)}{2} \right)$$
   - Transitions smoothly to pure verifiable binary evaluation on stage mastery.

### Alternatives Considered
- **Scalar Batch-Averaged MSE Fitness:** Rejected; notoriously prone to converging on compromise constant solutions in Genetic Programming.
- **Constant Static Shaping Weights:** Rejected; risk of reward hacking in late training epochs.

---

## 5. Transformer Decoder Attention Padding Masking & Mini-Chunk Projection

### Problem Statement & Root Cause
Variable-length program sequences in batched training suffer from gradient leakage across padded positions (`PAD_ID`) when attention masks are not strictly enforced. Furthermore, projecting hidden states to vocabulary logits over full sequences can cause transient VRAM spikes on 4GB hardware.

### Technical Decisions

1. **Exact Key Padding Masking (`tgt_key_padding_mask`):**
   - Construct boolean padding masks `(tgt_tokens == PAD_ID)` in `WatTransformerDecoder` forward pass and pass directly to `nn.TransformerDecoder`.
   - Ensures padded positions contribute zero attention weight and receive zero backpropagation gradients.

2. **Mini-Chunk Logit Projection ($L_{\text{chunk}} = 256$):**
   - Slice hidden tensor states into chunks of length $L_{\text{chunk}} = 256$ before applying `lm_head`, capping vocabulary projection VRAM to $<0.15\,\text{GB}$.

---

## 6. Generalization Extrapolation ($K=100$) & Automated Theorem Proving

### Technical Decisions

1. **Extrapolation Horizon Verification ($N+K$ with $N=20, K=100$):**
   - Evaluate synthesized algorithms on $n \in [20, 119]$.
   - Require $G_{\text{ext}} = 1.0$ (100% exact integer match) for stage graduation.

2. **Minimum Description Length (MDL) Complexity Bound:**
   - Ratio $M_{\text{MDL}} = \frac{|P|_{\text{bytes}}}{C(A_N)} \le 1.20$ relative to sequence Lempel-Ziv complexity.

3. **Homomorphism-Regularized Latent Discovery:**
   - Continuous representations regularized by Kernel VICReg + Additive Homomorphism Loss ($\mathcal{L}_{\text{add}}$).
   - High-precision ($>500$ digits) `mpmath` sampling + PSLQ integer relation searches ($<10^{-50}$ drop) + SymPy automated symbolic proofs.

---

## 7. Technology Stack & Component Mapping

| Subsystem | Primary Technology / Module | File Location in Repo | Role in Phase 3 |
| :--- | :--- | :--- | :--- |
| **Non-Triviality & Gating** | PyTorch / NumPy | `src/oeis_learn/rl/reward.py` | Output variance $\mathbb{Var}_n[P(n)]$, input sensitivity $\mathcal{S}_{\text{input}}$, $R_{\text{MI}}$ proxy |
| **Demonstration Co-Training** | PyTorch (Strict FP32) | `src/oeis_learn/rl/egca_grpo.py`, `trainer.py` | Loss blending $\mathcal{L}_{\text{RL}} + \beta_{\text{SFT}}\mathcal{L}_{\text{SFT}}$, Schulman KL penalty |
| **Execution Attribution** | Execution AST Tracer | `src/oeis_learn/sandbox/tracer.py` | Divergence step $k^* \to T_{k^*}$, downstream zero-masking, coverage |
| **PBRS & Lexicase** | Potential Engine / Sampler | `src/oeis_learn/rl/prompt_weighting.py`, `sampler.py` | $\gamma \Phi(s') - \Phi(s)$ potential differences, randomized per-case filtering |
| **Decoder Masking** | PyTorch Transformer | `src/oeis_learn/decoder/wat_decoder.py` | `tgt_key_padding_mask`, mini-chunk logit projections |
| **Extrapolation & Discovery** | `mpmath`, SymPy, Rust WASM | `src/oeis_learn/curriculum/extrapolation.py`, `discovery/` | $K=100$ horizon verifier, PSLQ ($<10^{-50}$ drop), SymPy proofs |
