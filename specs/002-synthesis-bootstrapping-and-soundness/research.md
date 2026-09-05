# Research & Technical Decisions: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization

**Feature**: [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md)  
**Branch**: `002-synthesis-bootstrapping-and-soundness`  
**Date**: 2026-08-31

---

## 1. Context-Sensitive Autoregressive Decoding & Dynamic State Tracking

### Problem Statement & Root Cause
During the initial 18.65-hour end-to-end benchmark on Tier 1 hardware, the policy decoder achieved a 0.0% execution pass rate across all 100 epochs. Analysis of sampled tokens revealed that the context-free grammar engine (`EnvironmentTracker` in `src/oeis_learn/decoder/environment_tracker.py`) allowed arbitrary continuations inside function definitions. The decoder frequently skipped mandatory function export, parameter, and result headers—emitting instructions like `i32.add` directly after `(func (`—or generated dangling/unbound variable references (`local.get $ghost_var`) and stack underflow states (`i64.add` with $\vert\Sigma_t\vert < 2$). These malformed outputs triggered immediate `PARSE_ERROR` or `EXECUTION_TRAP` failures in `oeis_wasm_evaluator`.

### Technical Decisions

1. **Dual-Layer Masking Pipeline:**
   Decouple static syntactic checks from context-sensitive dynamic semantic checks:
   $$M_{\text{Final}} = M_{\text{CFG}} \land M_{\text{Context}}$$
   - **Layer 1 ($M_{\text{CFG}}$):** Byte-level token trie ensuring parenthetical balance and valid WAT lexical tokens via `llguidance` / precomputed tries.
   - **Layer 2 ($M_{\text{Context}}$):** Zero-allocation dynamic state machine tracking the 4-tuple $S_t = \langle \Phi_t, \Gamma_t, \Sigma_t, H_t \rangle$.

2. **Structural Phase State Machine ($\Phi_t$):**
   Enforce strict sequencing of module and function declaration blocks before any body instructions are allowed:
   $$\Phi_t \in \{\text{MODULE\_HEADER}, \text{FUNC\_HEADER}, \text{EXPORT\_DECL}, \text{PARAM\_SEQUENCE}, \text{RESULT\_SEQUENCE}, \text{LOCAL\_SEQUENCE}, \text{BODY}, \text{MODULE\_END}\}$$
   - When entering `FUNC_HEADER`, the only legal continuation is `(export "compute")`.
   - In `PARAM_SEQUENCE`, the mandatory parameter `(param $n i32)` must be declared.
   - In `RESULT_SEQUENCE`, the mandatory return type `(result i64)` must be declared.
   - Only after transition to `LOCAL_SEQUENCE` or `BODY` are `(local ...)` allocations and executable opcodes enabled.

3. **Lexical Scope Tracking ($\Gamma_t$) & No-Ghost Soundness:**
   - Maintain $\Gamma_t = \langle \Gamma_{\text{param}}, \Gamma_{\text{local}} \rangle$.
   - When parsing `(param $name type)` or `(local $name type)`, insert `$name \mapsto \text{type}$` into $\Gamma_t$.
   - When emitting `local.get`, `local.set`, or `local.tee`, dynamic tightening operator $\tau_{\Gamma_t}$ restricts candidate identifier tokens strictly to $\text{dom}(\Gamma_t)$. Emitting undeclared symbols is assigned mask value 0, guaranteeing No-Ghost Soundness by construction.

4. **Operand Stack ($\Sigma_t$) & Control Block ($H_t$) Soundness:**
   - Maintain a pushdown value stack $\Sigma_t \in \tau^*$ where $\tau \in \{\text{i32}, \text{i64}\}$.
   - Every opcode $op$ with stack effect $[\tau_{\text{in}}^1 \dots \tau_{\text{in}}^m] \to [\tau_{\text{out}}^1 \dots \tau_{\text{out}}^n]$ is valid if and only if $\vert\Sigma_t\vert \ge m$ and the top $m$ elements match $[\tau_{\text{in}}^1 \dots \tau_{\text{in}}^m]$.
   - Maintain control stack $H_t \in (\{\text{block}, \text{loop}, \text{if}\} \times \tau^* \times \mathbb{N})^*$. Branch instructions `br k` and `br_if k` are valid only if $k \le \vert H_t\vert$.
   - On function closing parenthesis `)`, require $\Sigma_t == [\text{i64}]$.

5. **Sub-$100\,\mu\text{s}$ Latency Optimization:**
   - Bit-parallel type mask indexing: Maintain precomputed bitmasks for unary and binary arithmetic signatures.
   - Contiguous flat arrays: Store $\Gamma_t$, $\Sigma_t$, and $H_t$ using fixed-size buffers allocated once at decoder initialization, eliminating heap allocations during autoregressive decoding.

### Alternatives Considered
- **Unconstrained S-Expression Generation with Retries:** Rejected; exponential rejection overhead under high temperature sampling.
- **Pure CFG Earley Parsing (XGrammar/Outlines):** Rejected; CFGs cannot enforce lexical scope or stack depth, causing semantic validation traps.
- **AST-to-WASM Lowering (ASDL/SKI combinators):** Rejected; combinator expressions inflate sequence length by $5\times\text{--}20\times$, exceeding decoder context limits.

---

## 2. Supervised Fine-Tuning (SFT) & Demonstration Bootstrapping

### Problem Statement & Root Cause
In cold-start reinforcement learning with verifiable binary rewards ($\pm 1.0$), initialized policy weights generate random token sequences. Because computing a correct recurrence relation from scratch via random stack bytecodes has probability $P(R=+1) \approx 0$, all rollout groups in GRPO fail ($R_i = -1, \forall i$). This causes the Advantage Collapse Rate to hit $\text{ACR} = 1.0$, rendering policy gradients $\nabla_\theta J(\theta) = 0$ and trapping the model in permanent exploration starvation.

### Technical Decisions

1. **Synthetic Forward Demonstration Generator:**
   Generate a forward dataset $\mathcal{D}_{\text{SFT}} = \{(Y_i, P_i)\}_{i=1}^{M}$ ($M = 5,000\text{--}10,000$ pairs) by sampling well-typed programs $P_i \sim \text{Grammar}(\text{WAT})$ from a template-driven domain-specific generator covering:
   - *Polynomial Family (Degree 0–3):* Constant ($c$), Linear ($a\cdot n + b$), Quadratic ($a\cdot n^2 + b\cdot n + c$), Triangular ($n(n+1)/2$), Cubic ($a\cdot n^3 + \dots$).
   - *Linear Recurrence Family (Order 1–3):* Geometric ($a\cdot r^n$), Fibonacci-like ($a_{n} = a_{n-1} + a_{n-2}$), Tribonacci-like, with varying initial conditions.
   - *Modular & Periodic Family:* Alternating sequences ($(-1)^n$), cyclic residues ($n \bmod m$), periodic wave patterns.
   - *Factorial & Exponential Bounds:* Accumulator loops with 64-bit overflow guards.

2. **Supervised Fine-Tuning (SFT) Warmup Objective:**
   Train the Transformer encoder-decoder via Maximum Likelihood Estimation (MLE) on teacher-forced synthetic demonstrations:
   $$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{\vert P\vert} \log \pi_\theta(p_t \mid p_{<t}, Y)$$
   - Optimizer: AdamW, learning rate $5\times 10^{-4}$ with cosine decay to $5\times 10^{-5}$, weight decay $0.01$.
   - Target convergence: $\mathcal{L}_{\text{SFT}} < 0.50$ and reference perplexity $\text{PPL}_{\text{ref}} < 1.25$ within 5 epochs.

3. **Elite Seed Demonstration Buffer ($\mathcal{D}_{\text{elite}}$):**
   - Populate $\mathcal{D}_{\text{elite}}$ with canonical solutions for all Stage 1 and Stage 2 sequences discovered during synthetic generation or symbolic search.
   - Store entries as `(oeis_id, terms, wat_code, byte_size, lz_complexity)`.
   - Make $\mathcal{D}_{\text{elite}}$ queryable by prompt during online RL exploration.

### Alternatives Considered
- **Pure Online RL from Scratch:** Rejected; 18-hour benchmark proved 0% pass rate due to zero-advantage collapse.
- **Genetic Program Enumeration (PushGP):** Rejected as primary engine due to heavy CPU overhead for long sequences; used selectively to populate $\mathcal{D}_{\text{elite}}$.

---

## 3. Multi-Tiered Reward Shaping, S-GRPO & Trajectory Injection

### Problem Statement & Root Cause
In standard GRPO, group advantages are normalized as $\hat{A}_i = \frac{r_i - \mu_{\mathbf{r}}}{\sigma_{\mathbf{r}} + \epsilon}$. When all completions in a group fail on hard tasks, $\sigma_{\mathbf{r}} = 0$, yielding $\hat{A}_i = 0$. Furthermore, monolithic binary outcome rewards assign the same $-1.0$ penalty to a candidate that computes 19 out of 20 terms correctly as to one that generates unparseable garbage, eliminating smooth gradient guidance.

### Technical Decisions

1. **Multi-Tiered Composite Reward Function:**
   During early curriculum stages, compute a shaped reward $R_{\text{composite}}(P, Y) \in [-1.0, 1.0]$:
   $$R_{\text{composite}}(P, Y) = w_{\text{comp}} R_{\text{comp}}(P) + w_{\text{prefix}} R_{\text{prefix}}(P, Y) + w_{\text{dist}} R_{\text{dist}}(P, Y) + R_{\text{exact}}(P, Y)$$
   - **Compiler Validation Reward ($R_{\text{comp}}$):** $+0.2$ if WAT compiles to WASM without traps; $-0.5$ if compilation fails.
   - **Prefix Match Length ($R_{\text{prefix}}$):** $R_{\text{prefix}}(P, Y) = \frac{1}{N} \max \{ k \le N \mid P(n) = y_n, \forall n < k \}$.
   - **Normalized Numerical Distance ($R_{\text{dist}}$):** $R_{\text{dist}}(P, Y) = 1.0 - \frac{1}{N} \sum_{n=0}^{N-1} \tanh(0.1 \cdot \vert P(n) - y_n\vert)$.
   - **Exact Outcome Reward ($R_{\text{exact}}$):** $+1.0$ if $P(n) = y_n, \forall n \in [0, N-1]$; $0.0$ otherwise.

2. **Cosine Annealing toward Pure RLVR:**
   Anneal dense surrogate weights over training steps $s \in [0, S]$:
   $$w(s) = \cos\left( \frac{\pi \cdot s}{2S} \right)$$
   As $s \to S$, $R_{\text{composite}} \to R_{\text{exact}} \in \{-1.0, +1.0\}$, preventing reward hacking and preserving strict formal verification.

3. **Supervised Group Relative Policy Optimization (S-GRPO) & Trajectory Injection:**
   - For prompt $q$, sample $G=4\text{--}8$ rollouts $\{y_1, \dots, y_G\}$.
   - If $\sigma_{\mathbf{r}} == 0$ and all $r_i \le 0$:
     - Fetch reference trajectory $y_{\text{gt}}^*$ from $\mathcal{D}_{\text{elite}}$ (if available) and inject it into the group with $r_{\text{gt}} = +1.0$.
     - Recompute group mean and standard deviation over $G+1$ completions, yielding $\hat{A}_{\text{gt}} > 0$ and $\hat{A}_{\text{gen}} < 0$.
     - If no reference exists, engage Adaptive Virtual Sample Policy Optimization (AVSPO), inserting a virtual anchor advantage.

4. **Advantage Collapse Rate (ACR) Monitoring:**
   - Track $\text{ACR} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\sigma_{\mathbf{r}_b}^2 == 0)$ over a sliding window of 20 batches.
   - If $\text{ACR} \ge 0.30$, trigger trajectory injection and boost sampling temperature from $T=0.4$ to $T=0.7$.

5. **Execution-Grounded Credit Assignment (EGCA):**
   - Trace candidate runtime state against target terms $Y$.
   - Identify the exact token index $t_{\text{div}}$ where execution diverged ($P(n) \ne y_n$).
   - Set token advantage masks $M_t = 0.0$ for $t < t_{\text{div}}$ and $M_t = 1.0$ for $t \ge t_{\text{div}}$, concentrating policy updates onto erroneous instruction spans.

---

## 4. Progressive Micro-Benchmarking Hierarchy & Diagnostic Telemetry

### Problem Statement & Root Cause
Relying on full 10-to-20-hour training runs to discover that the grammar engine had a bug or that GRPO had collapsed produced unacceptable development latency. A hierarchical testing framework is required to catch failure modes in seconds or minutes.

### Technical Decisions

1. **5-Tier Progressive Testing Hierarchy:**

| Tier | Latency Budget | Scope & Objective | Success Threshold |
| :--- | :--- | :--- | :--- |
| **Tier 0** | $< 5\text{ seconds}$ | Deterministic unit checks: in-memory WASM compilation, 10,000 fuel traps, linear memory ceiling, static grammar token bounds. Zero neural forward/backward passes. | 100% trap rate on infinite loops, 0 sandbox escapes, $<1\,\text{ms}$ trap latency. |
| **Tier 1** | $< 2\text{ minutes}$ | Oracle reference solution fitting: Single-sequence supervised fine-tuning on canonical programs. Validates gradient flow, tokenization alignment, and decoder capacity. | $\text{PPL}_{\text{ref}} < 1.25$ within 20 optimization steps. |
| **Tier 2** | $< 10\text{ minutes}$ | Single-prompt RL convergence: GRPO optimization on an isolated prompt (e.g., Triangular Numbers A000217) with $G=4$. Validates rollout generation, advantage scaling, and ratio clipping. | $100\%$ pass rate achieved within 15 gradient iterations. |
| **Tier 3** | $< 45\text{ minutes}$ | Synthetic micro-cohort curriculum progression: 10–20 synthetic tasks across difficulty tiers. Validates rolling competence $C(S_k)$, graduation gates, and hyperparameter stability. | Automated promotion from Stage 1 to Stage 2 based on $C(S_1) \ge 0.85$. |
| **Tier 4** | $2\text{--}4\text{ hours}$ | Full dataset multi-epoch curriculum training: 500+ sequences across all active stages. | Monotonic pass rate scaling, $C(S_1) \ge 0.80$, zero regression. |

2. **Real-Time Diagnostic Telemetry Signals:**
   - **Policy Entropy $\mathcal{H}(\pi_\theta)$:** Healthy range $1.20 \le \mathcal{H} \le 3.50$. Early warning: $\mathcal{H} < 0.20$ or $>70\%$ drop in $\le 5$ steps (mode collapse).
   - **Group Reward Variance $\sigma_R^2$:** Healthy range $\sigma_R^2 > 0.05$. Early warning: $\sigma_R^2 = 0.0$ consistently (exploration stall).
   - **Compiler Trap Rate $P_{\text{trap}}$:** Healthy range $<15\%$ at step 10, decaying to $0\%$. Early warning: $>60\%$ after 5 steps.
   - **Average Prefix Match Length $\bar{L}_{\text{prefix}}$:** Must increase monotonically.
   - **Oracle Perplexity $\text{PPL}_{\text{ref}}$:** Target $1.05 \le \text{PPL}_{\text{ref}} \le 1.30$.

---

## 5. Self-Supervised Latent Manifold Structuring & Algebraic Homomorphism

### Problem Statement & Root Cause
When the Tri-Stream Encoder is trained solely via end-to-end RL without representation regularization, the continuous latent space $Z \in \mathbb{R}^d$ collapses into low-rank subspaces. This prevents vector arithmetic ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$) from identifying mathematical identities for PSLQ theorem proving.

### Technical Decisions

1. **Non-Contrastive Kernel VICReg in RKHS:**
   Optimize latent representations $Z, Z'$ over positive algebraic transformation pairs:
   $$\mathcal{L}_{\text{VICReg}} = \lambda s(Z, Z') + \mu [v(Z) + v(Z')] + \nu [c(Z) + c(Z')]$$
   - Invariance $s(Z, Z') = \frac{1}{B} \sum \Vert z_i - z_i'\Vert_2^2$.
   - Variance $v(Z) = \frac{1}{d} \sum \max(0, \gamma - \sqrt{\text{Var}(z^j) + \epsilon})$ with $\gamma = 1.0$.
   - Covariance $c(Z) = \frac{1}{d} \sum_{j \ne k} [C(Z)]_{jk}^2$ to decorrelate feature dimensions.

2. **Explicit Additive Homomorphism Loss ($\mathcal{L}_{\text{add}}$):**
   For sequence pairs $A = (a_n), B = (b_n)$ and their sum $C = A + B = (a_n + b_n)$:
   $$\mathcal{L}_{\text{add}} = \frac{1}{B} \sum_{i=1}^B \left\Vert f(A_i + B_i) - \left( f(A_i) + f(B_i) \right) \right\Vert_2^2$$
   Enforces that vector addition in latent space reflects termwise addition in sequence space.

3. **Shift Equivariance Loss ($\mathcal{L}_{\text{shift}}$):**
   Constrain sequence shift operations $T_1 A = (a_{n+1})$ to correspond to a continuous linear operator $M_{\text{shift}} \in \mathbb{R}^{d \times d}$:
   $$\mathcal{L}_{\text{shift}} = \frac{1}{B} \sum_{i=1}^B \left\Vert f(T_1 A_i) - M_{\text{shift}} f(A_i) \right\Vert_2^2$$

4. **Discovery Pipeline Coupling:**
   - Latent candidates retrieved via $\|\vec{v}_A + \vec{v}_B - \vec{v}_C\|_2 < \epsilon_{\text{geom}}$.
   - High-precision ($>500$ digits) numerical evaluation of generating functions via `mpmath`.
   - PSLQ lattice reduction algorithm with confidence ratio drop $< 10^{-50}$.
   - SymPy symbolic proof execution exporting machine-verified markdown reports.

---

## 6. Tier 1 Workstation Memory & Execution Budget

### Resource Constraints & Division of Labor
- **CPU Offloading:** 100% of WebAssembly module parsing, Cranelift JIT compilation, and fuel metering is executed in Rust via `oeis_wasm_evaluator` using Rayon across 8 CPU threads, fully releasing the Python GIL.
- **GPU Scope:** Strict FP32 precision (`torch.float32`) for neural encoder/decoder forward and backward passes.
- **Logit Memory Chunking:** Mini-chunk projection ($L_{\text{chunk}} = 256$) bounds peak VRAM to $< 0.15\,\text{GB}$ for vocabulary projections.
- **Micro-Batching:** $B = 1\text{--}4$ on GPU with 8 gradient accumulation steps (effective batch size 32) holds peak GPU memory under $3.2\,\text{GB}$ VRAM, well within the 4 GB hardware ceiling.

---

## 7. Technology Stack & Component Mapping

| Subsystem | Primary Library / Technology | Component in Repository | Purpose |
| :--- | :--- | :--- | :--- |
| **Dynamic Grammar Masking** | `llguidance` / Pure Rust Trie | `oeis_learn.decoder.grammar_masker`, `environment_tracker` | Sub-$100\,\mu\text{s}$ state-machine logit masking and No-Ghost scoping |
| **Demonstration Generation** | Python / Native WASM | `oeis_learn.data.synthetic_generator`, `oeis_learn.rl.sft_trainer` | Synthetic forward $(Y, P)$ dataset creation and SFT warmup |
| **RL Optimization** | PyTorch 2.3+ (Strict FP32) | `oeis_learn.rl.egca_grpo`, `prompt_weighting`, `reward` | S-GRPO, CGI trajectory injection, ACR telemetry, composite shaping |
| **Progressive Testing** | `pytest`, custom CLI | `scripts.run_progressive_validation`, `tests.unit`, `tests.integration` | 5-tier pre-flight verification hierarchy (Tiers 0–4) |
| **Latent Discovery** | PyTorch / `mpmath` / SymPy | `oeis_learn.discovery.vicreg_loss`, `pslq_solver`, `symbolic_prover` | Homomorphism-regularized VICReg, PSLQ relation search, SymPy proofs |
| **WASM Sandbox** | `wasmtime` / PyO3 / Rayon | `crates/oeis_wasm_evaluator`, `oeis_learn.sandbox` | 10,000 fuel limit, 16 MiB linear memory, 5,500+ evals/sec |
