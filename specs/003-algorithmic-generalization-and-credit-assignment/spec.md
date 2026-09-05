# Feature Specification: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment

**Feature Branch**: `003-algorithmic-generalization-and-credit-assignment`  
**Created**: 2026-09-01  
**Status**: Draft  
**Input**: User description: "Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment to resolve degenerate constant shortcut collapse, enforce input parameter sensitivity, anchor policy distributions via demonstration co-training, and isolate causal bytecode error spans via execution-grounded credit assignment."  
**Prerequisites**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md), [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md)

---

## 1. Executive Context & Problem Statement

The evolution of `oeis-learn` across Phases 1 and 2 established a deterministic, high-throughput neuro-symbolic framework:
- **Phase 1 Infrastructure**: Delivered a native Rust Cranelift WebAssembly execution engine evaluating $>5,300$ modules/sec across 8 CPU threads, a Tri-Stream Continuous Neural Encoder operating in strict FP32 precision across values from $-10^6$ to $10^{30}$, and a formal PSLQ symbolic theorem prover.
- **Phase 2 Bootstrapping & Soundness**: Resolved context-free grammar gaps and cold-start exploration deserts. The dual-layer environment tracker ($\Phi_t, \Gamma_t, \Sigma_t, H_t$) achieved **100% compilation soundness** (0% `PARSE_ERROR` or missing entrypoint traps), while Supervised Fine-Tuning (SFT) warmup and S-GRPO Conditional Ground-Truth Trajectory Injection (CGI) eliminated zero-advantage collapse ($\text{ACR} = 0.0$ on pre-flight checks).

### The Next Empirical Barrier: Degenerate Constant Shortcut Collapse
While 100% of generated programs in Phase 2 compiled and executed safely within the 10,000-fuel sandbox, multi-hour policy gradient optimization revealed that the policy gradient collapsed into an unintended local optimum: **degenerate constant shortcut collapse**.

```
Candidate synthesized for A000079 (Powers of 2: 1, 2, 4, 8, 16, 32...):
(module (func (export "compute") (param $n i32) (result i64) nop nop nop nop i64.const 16))
Output: [16, 16, 16, 16, 16...] | Status: SUCCESS | Extrap Passed: False
```

### Root-Cause Analysis from Literature & Telemetry
1. **Surrogate Metric Exploitation**: Dense reward shaping awarded $+0.2$ for compilation ($R_{\text{comp}}$) and partial distance scores ($R_{\text{dist}}$) for outputs that landed near sequence elements. Emitting a single constant minimized token sequence length and variance, incurring zero risk of runtime traps while accumulating safe partial rewards.
2. **Signal-to-Noise Ratio (SNR) Collapse**: When all rollouts in a group generate identical static constants, within-group reward variance $\mathbb{Var}_g[R]$ vanishes. The task gradient $\|g_{\text{task}}\|$ decays to zero while the regularization gradient $g_{\text{reg}}$ (KL penalty or entropy) contracts the policy into an input-agnostic prior, driving cross-input mutual information $I(n; P(n)) \to 0$.
3. **Policy Drift & SFT Forgetting**: Without continuous demonstration co-training, policy gradients drifted rapidly away from the complex control loops (multi-variable accumulators, `loop`, `br_if`) learned during SFT warmup.
4. **Credit Smear in Stack Bytecodes**: When multi-instruction candidates failed on extended terms, uniform group advantage broadcasting penalized valid variable initializations and structural prefixes alongside faulty arithmetic operators.

This specification defines **Phase 3: Inductive Algorithmic Generalization, Anti-Shortcut Regularization & Fine-Grained Credit Assignment** to force the policy from safe static approximations into synthesizing parameterized, extrapolating algorithms.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 — Non-Degenerate Inductive Synthesis & Input Sensitivity (Priority: P1) 🎯 MVP

As a neuro-symbolic synthesizer, I want policy optimization to condition rewards on input-parameter sensitivity and cross-input mutual information, so that candidate programs that ignore the input parameter `$n` or emit static constant sequences receive zero surrogate reward and cannot serve as optimization attractors.

**Why this priority**: Eliminating constant shortcuts is the primary prerequisite for learning parameterized loops and achieving non-zero extrapolation accuracy on unseen terms.

**Independent Test**: Evaluate the model on 100 non-constant sequence tasks; verify that $\ge 95\%$ of generated programs contain active bindings to `$n`, produce non-zero empirical variance $\mathbb{Var}_n[P(n)] > 0$, and achieve non-zero input sensitivity $\mathcal{S}_{\text{input}}(P) > 0$.

**Acceptance Scenarios**:
1. **Given** a generated program that outputs a constant sequence ($P(n) = C, \forall n$) for a non-constant sequence task, **When** evaluating reward, **Then** the non-triviality gate zeros out all surrogate rewards ($R_{\text{dist}} = 0, R_{\text{prefix}} = 0$) and assigns a static negative penalty.
2. **Given** a batch of distinct sequence tasks, **When** calculating batch rewards, **Then** a cross-input mutual information proxy $R_{\text{MI}}$ penalizes outputs that are identical across different sequence prompts ($I(n; P(n)) \approx 0$).
3. **Given** a candidate program utilizing `$n` in an arithmetic expression or loop condition, **When** executed across domain indices $n \in [0, 19]$, **Then** the output spectrum exhibits dynamic range matching the task profile.

---

### User Story 2 — Demonstration Co-Training & Anchor Loss Regularization (Priority: P1)

As an ML training pipeline, I want reinforcement learning policy updates to be co-trained with an auxiliary Supervised Fine-Tuning (SFT) demonstration loss and bounded by reference model KL divergence, so that the policy retains inductive loop templates and syntactic idioms acquired during warmup without suffering from policy drift or catastrophic forgetting.

**Why this priority**: Standalone RL with sparse rewards rapidly drifts into trivial syntactic shortcuts. Demonstration co-training anchors the policy in valid control-flow structures throughout optimization.

**Independent Test**: Train the policy for 30 epochs under mixed SFT+RL optimization; verify that the policy's capacity to generate valid `loop`, `block`, and `br_if` constructs does not degrade relative to the initial SFT baseline (token entropy $\mathcal{H}(\pi_\theta) \ge 1.50$, reference perplexity $\text{PPL}_{\text{ref}} \le 1.30$).

**Acceptance Scenarios**:
1. **Given** an online RL training batch, **When** computing gradients, **Then** the loss function combines the group-relative policy gradient with a blended SFT loss evaluated over the elite replay buffer: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta_{\text{SFT}} \mathcal{L}_{\text{SFT}}$.
2. **Given** policy updates on exploratory rollouts, **When** measuring token divergence, **Then** an unbiased per-token Schulman KL penalty $\beta_{\text{KL}} \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ prevents the policy distribution from collapsing into low-entropy static templates.
3. **Given** the Transformer decoder attention layers, **When** processing variable-length programs, **Then** exact padding attention masks (`tgt_key_padding_mask`) prevent gradient diffusion across padded positions.

---

### User Story 3 — Fine-Grained Execution-Grounded Credit Assignment (EGCA) (Priority: P2)

As a reinforcement learning optimizer, I want execution trace divergence to isolate the exact instruction token where candidate execution state deviated from expected sequence values, zero-masking advantages for all subsequent downstream tokens, so that valid module headers, parameter definitions, and partially correct loops are not penalized by credit smear.

**Why this priority**: In low-level stack bytecode, a single corrupted token late in the sequence breaks all downstream operations. Uniform sequence-level penalties destroy structural fluency.

**Independent Test**: Evaluate gradient updates on candidate programs that correctly compute initial terms $n \in [0, 5]$ but fail at $n = 6$; verify that token advantages for the prefix ($t < t_{\text{div}}$) and suffix ($t > t_{\text{div}}$) are masked out, concentrating $\ge 90\%$ of gradient mass on the causal error span.

**Acceptance Scenarios**:
1. **Given** a generated program that fails logical verification on test case $d$, **When** tracing execution against a canonical reference, **Then** the system locates the earliest divergence step $k^*$ and maps it to the causal bytecode token span $T_{k^*}$.
2. **Given** the localized token span $T_{k^*}$, **When** constructing token advantage vectors, **Then** all tokens generated after the causal error ($t > \max T_{k^*}$) receive zero advantage ($a_{i,t} = 0$), preserving downstream boilerplate from negative reinforcement.
3. **Given** a program failing compilation or type validation, **When** compiling in-memory, **Then** compiler diagnostic spans $T_{\text{err}}$ are extracted to concentrate negative gradients exclusively on the unparseable tokens.

---

### User Story 4 — Potential-Based Reward Shaping (PBRS) & Down-Sampled Lexicase Selection (Priority: P2)

As a training scheduler, I want dense surrogate rewards to follow potential-based state differences ($\gamma \Phi(s') - \Phi(s)$) over AST completion states and test-case evaluations to use down-sampled lexicase selection, so that intermediate rewards preserve policy invariance and reward per-input specialists rather than compromise constant generalists.

**Why this priority**: Heuristic distance metrics create artificial global optima that favor average static constants. Potential-Based Reward Shaping mathematically guarantees that the optimal policy under shaped rewards is identical to the optimal policy under exact verifiable rewards.

**Independent Test**: Run 20 iterations comparing PBRS against raw heuristic distance rewards; verify that PBRS eliminates constant shortcut attractors while accelerating convergence toward exact sequence matching.

**Acceptance Scenarios**:
1. **Given** an intermediate partial AST state during decoding, **When** computing shaped rewards, **Then** the shaping signal telescopes over generation steps ($\sum F = \gamma^T \Phi(s_T) - \Phi(s_0)$), ensuring policy invariance relative to $R_{\text{exact}}$.
2. **Given** a batch of test cases for a sequence, **When** performing rollout selection, **Then** down-sampled lexicase filtering evaluates candidate completions against individual randomized test cases sequentially, eliminating static constants that fail on non-zero inputs.
3. **Given** a candidate program achieving partial sequence match, **When** moving from early to late curriculum epochs, **Then** surrogate potentials decay smoothly via an adaptive schedule driven by rolling task competence.

---

### User Story 5 — Generalization Extrapolation ($K=100$) & Automated Theorem Discovery (Priority: P3)

As a mathematical discovery pipeline, I want synthesized candidate algorithms to be verified across an extended extrapolation horizon ($N+K$ terms with $N=20, K=100$) and Minimum Description Length bounds ($M_{\text{MDL}} \le 1.20$), with latent representations queried via vector arithmetic and verified via PSLQ integer relation detection and SymPy symbolic proofs.

**Why this priority**: Differentiating between memorized lookup polynomials and true generating algorithms requires rigorous extrapolation testing, while continuous representations enable automated mathematical theorem discovery.

**Independent Test**: Evaluate graduated candidate programs on $K=100$ future terms; verify that $100\%$ of passing algorithms compute exact integer values across the full horizon and generate verified symbolic proofs.

**Acceptance Scenarios**:
1. **Given** a candidate program passing training terms ($n \in [0, 19]$), **When** evaluated on unseen future terms ($n \in [20, 119]$), **Then** the extrapolation verifier requires 100% exact match ($G_{\text{ext}} = 1.0$) to authorize stage graduation.
2. **Given** a synthesized program, **When** measuring compressed WebAssembly binary size against sequence Lempel-Ziv complexity, **Then** programs with $M_{\text{MDL}} > 1.20$ are rejected as bloated memorizations.
3. **Given** continuous sequence embeddings regularized by additive homomorphism loss $\mathcal{L}_{\text{add}}$, **When** queried with vector arithmetic triples $(\vec{v}_A + \vec{v}_B \approx \vec{v}_C)$, **Then** high-precision ($>500$ digits) PSLQ searches recover exact integer relations ($<10^{-50}$ confidence drop) proved by SymPy.

---

### Edge Cases

- **Astronomical Values in Accumulator Loops**: When synthesizing exponential or factorial sequences (e.g., $2^n, n!$), intermediate 64-bit integer calculations may overflow. The execution sandbox traps integer overflow deterministically, assigning structured trap diagnostics without crashing the host runner.
- **Dead Branch Tokens in Complex Control Flow**: When programs contain unreachable basic blocks (e.g., instructions following an unconditional `br` or `return`), coverage-based fine-grained attribution masks out unexecuted tokens from the policy gradient update.
- **Degenerate Periodic Residues vs. True Constants**: For modular sequences with low period (e.g., alternating $0, 1, 0, 1\dots$), the non-triviality guard verifies that output variance is computed across the full evaluation window ($N=20$), correctly distinguishing non-trivial cyclic patterns from static constants ($C, C, C\dots$).
- **VRAM Memory Limits on 4GB Hardware**: Peak GPU memory is strictly capped at $<3.5\,\text{GB}$ VRAM by applying mini-chunk logit projections ($L_{\text{chunk}} = 256$), micro-batching ($B=1\text{--}4$), gradient accumulation ($N_{\text{accum}} = 8$), and offloading all WebAssembly executions to 8 CPU threads via Rayon.

---

## 3. Requirements *(mandatory)*

### Functional Requirements

#### Non-Degenerate Reward Design & Input Sensitivity
- **FR-001**: System MUST compute empirical output variance $\mathbb{Var}_n[P(n)]$ across evaluated sequence terms $n \in [0, N-1]$.
- **FR-002**: System MUST enforce a non-triviality reward gate: if a target sequence has non-zero variance ($\mathbb{Var}_n[y_n] > 0$) but the synthesized program produces near-zero variance ($\mathbb{Var}_n[P(n)] < 10^{-6}$), all surrogate rewards ($R_{\text{dist}}, R_{\text{prefix}}$) MUST be zeroed out and a static penalty applied.
- **FR-003**: System MUST compute a batch-level cross-input mutual information proxy ($R_{\text{MI}}$) penalizing policies whose output representations remain identical across distinct sequence tasks ($I(n; P(n)) \approx 0$).
- **FR-004**: System MUST verify empirical input parameter sensitivity $\mathcal{S}_{\text{input}}(P) = \sum_{n=0}^{N-2} |P(n+1) - P(n)|$, gating out positive reward allocation if sensitivity is zero for dynamic sequence targets.

#### Demonstration Co-Training & Policy Regularization
- **FR-005**: System MUST support hybrid co-training during online reinforcement learning, blending supervised cross-entropy loss over elite reference demonstrations with the policy gradient loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \beta_{\text{SFT}} \mathcal{L}_{\text{SFT}}$ (with default $\beta_{\text{SFT}} = 0.20$).
- **FR-006**: System MUST compute an unbiased per-token Schulman KL divergence penalty relative to the pre-warmed reference model $\pi_{\text{ref}}$, bounding policy drift away from functional loop templates.
- **FR-007**: System MUST apply exact target key padding masks (`tgt_key_padding_mask`) in the Transformer decoder during both SFT warmup and RL training to prevent gradient diffusion across padded token positions.
- **FR-008**: System MUST maintain an entropy regularization bonus ($\alpha_{\text{ent}} \mathcal{H}(\pi_\theta)$ with default $\alpha_{\text{ent}} = 0.01$) to maintain exploratory token entropy $\mathcal{H} \ge 1.50$.

#### Execution-Grounded Credit Assignment (EGCA)
- **FR-009**: System MUST classify rollout failures into deterministic priority gates: `SYNTAX`, `CONSTRAINT`, `LOGIC`, and `CORRECT`.
- **FR-010**: For `LOGIC` failures, system MUST trace candidate execution states against expected sequence values to locate the earliest divergence step $k^*$.
- **FR-011**: System MUST map execution divergence step $k^*$ to the causal bytecode token span $T_{k^*}$, zero-masking advantages for all subsequent tokens ($t > \max T_{k^*}$).
- **FR-012**: System MUST preserve total advantage scale $\sum_{t=1}^T a_{i,t} = A_i$, concentrating the full sequence return strictly onto the causal error window.
- **FR-013**: System MUST track runtime instruction coverage, masking unexecuted branch tokens from receiving spurious policy updates.

#### Potential-Based Reward Shaping & Selection
- **FR-014**: System MUST formulate dense auxiliary rewards strictly as potential differences $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ over AST states to preserve policy invariance.
- **FR-015**: System MUST assign positive potential $\phi_{\text{bind}}(s) > 0$ when an AST explicitly binds the input parameter `$n` within a loop or arithmetic expression.
- **FR-016**: System MUST support down-sampled lexicase selection during rollout evaluation, assessing candidate completions against randomized individual test cases rather than scalar batch-averaged distances.
- **FR-017**: System MUST anneal dense surrogate potentials dynamically based on rolling task competence $C(S_k)$ via a cosine schedule.

#### Extrapolation & Theorem Proving
- **FR-018**: System MUST evaluate candidate WebAssembly algorithms across an extended extrapolation horizon of $K=100$ unseen future terms ($n \in [20, 119]$), requiring 100% exact match ($G_{\text{ext}} = 1.0$) for curriculum graduation.
- **FR-019**: System MUST calculate Minimum Description Length ratios ($M_{\text{MDL}} = \frac{|P|_{\text{bytes}}}{C(A_N)} \le 1.20$) to reject lookup tables and memorized Lagrange polynomials.
- **FR-020**: System MUST extract continuous latent sequence representations regularized by non-contrastive Kernel VICReg and additive homomorphism penalties ($\mathcal{L}_{\text{add}}$).
- **FR-021**: System MUST perform high-precision ($>500$ digits) `mpmath` sampling on candidate relation triples $(\vec{v}_A + \vec{v}_B \approx \vec{v}_C)$ and verify integer relations via the PSLQ algorithm ($<10^{-50}$ confidence drop).
- **FR-022**: System MUST submit verified relations to SymPy to generate formal, machine-verified mathematical proofs.

#### Hardware-Constrained Execution
- **FR-023**: System MUST execute all neural forward and backward passes in strict single precision (`torch.float32`) within $\le 3.5\,\text{GB}$ GPU VRAM on Tier 1 hardware (4 CPU cores / 8 threads, 4GB GPU VRAM).
- **FR-024**: System MUST offload 100% of WebAssembly module compilation, fuel metering (10,000 instruction cap), and batch evaluation to 8 CPU threads via the native Rust `oeis_wasm_evaluator` extension.

---

### Key Entities

- **NonTrivialityEvaluation**: Represents the validation breakdown for candidate program outputs, capturing output variance $\mathbb{Var}_n[P(n)]$, input sensitivity $\mathcal{S}_{\text{input}}(P)$, mutual information proxy $R_{\text{MI}}$, and the gating flag authorizing reward allocation.
- **CoTrainingBatch**: Encapsulates an online training batch pairing RL exploratory rollouts with teacher-forced SFT demonstration sequences and reference policy log probabilities for KL divergence estimation.
- **FineGrainedAttributionSpan**: Stores localized credit assignment spans, mapping failure modes (`SYNTAX`, `CONSTRAINT`, `LOGIC`), divergence step $k^*$, and token range $T_{k^*}$ with binary masking flags for prefix, causal window, and suffix tokens.
- **PotentialState**: Tracks potential-based shaping variables ($\Phi(s)$, $\phi_{\text{comp}}$, $\phi_{\text{bind}}$) across incremental AST decoding steps to enforce policy invariance.
- **ExtrapolationBenchmarkResult**: Captures verification metrics over $N+K$ terms ($N=20, K=100$), compiled byte size, sequence Lempel-Ziv complexity, Minimum Description Length ratio $M_{\text{MDL}}$, and graduation eligibility.
- **DiscoveredIdentityRecord**: Formal representation of an uncovered algebraic relation, including vector Euclidean distance, arbitrary-precision PSLQ integer certificate ($<10^{-50}$ drop), SymPy proof status, and Markdown proof export.

---

## 4. Success Criteria *(mandatory)*

### Measurable Outcomes

| Metric ID | Target Metric | Required Threshold | Verification Method |
| :--- | :--- | :--- | :--- |
| **SC-001** | **Compilation Soundness Rate** | **$100.0\%$** valid WebAssembly binaries | 1,000 random samples at $T=1.0$ yield 0 `PARSE_ERROR` or scoping traps. |
| **SC-002** | **Non-Triviality Rate** | **$\ge 95.0\%$** parameter usage | Percentage of generated programs referencing `$n` with non-zero output variance $\mathbb{Var}_n[P(n)] > 0$. |
| **SC-003** | **SFT-RL Retention Rate** | **$\text{PPL}_{\text{ref}} \le 1.30$** | Reference token perplexity on canonical solutions after 30 epochs of RL exploration. |
| **SC-004** | **Credit Attribution Precision** | **$\ge 90.0\%$** localized gradient mass | Percentage of policy gradient magnitude concentrated on the causal divergence token window $T_{k^*}$. |
| **SC-005** | **Curriculum Stage 1 Competence** | **$C(S_1) \ge 0.80$** | Weighted rolling pass-rate across 25 Stage 1 polynomial tasks on Tier 1 hardware. |
| **SC-006** | **Extrapolation Generalization** | **$100.0\%$** match on $K=100$ terms | Extrapolation horizon test ($n \in [20, 119]$) on all graduated candidate algorithms. |
| **SC-007** | **Anti-Memorization Ratio** | **$M_{\text{MDL}} \le 1.20$** | Ratio of compiled WebAssembly byte size to sequence Lempel-Ziv complexity. |
| **SC-008** | **Advantage Collapse Rate (ACR)** | **$\text{ACR} \le 0.10$** | Rolling fraction of zero-variance rollout groups during online exploration. |
| **SC-009** | **Mathematical Discovery Yield** | **$\ge 5$ verified theorems** | Number of novel algebraic identities proved by PSLQ ($<10^{-50}$ drop) and SymPy. |
| **SC-010** | **Pre-Flight Execution Duration** | **$< 5\text{ minutes}$** for Tiers 0–3 | Full pre-flight validation hierarchy runtime before multi-hour training authorization. |

---

## 5. Architectural & System Boundaries

### In Scope
- Implementing non-triviality reward gating, output variance penalties $\mathbb{Var}_n[P(n)]$, and mutual information proxy $R_{\text{MI}}$ in [src/oeis_learn/rl/reward.py](src/oeis_learn/rl/reward.py).
- Implementing SFT co-training loss blending ($\mathcal{L}_{\text{RL}} + \beta_{\text{SFT}} \mathcal{L}_{\text{SFT}}$) and Schulman KL divergence penalties in [src/oeis_learn/rl/egca_grpo.py](src/oeis_learn/rl/egca_grpo.py) and [src/oeis_learn/rl/trainer.py](src/oeis_learn/rl/trainer.py).
- Enhancing token padding attention masks (`tgt_key_padding_mask`) in [src/oeis_learn/decoder/wat_decoder.py](src/oeis_learn/decoder/wat_decoder.py) and [src/oeis_learn/rl/sft_trainer.py](src/oeis_learn/rl/sft_trainer.py).
- Enhancing Execution-Grounded Credit Assignment (EGCA) with downstream token zero-masking and basic block coverage tracing in [src/oeis_learn/sandbox/tracer.py](src/oeis_learn/sandbox/tracer.py) and [src/oeis_learn/rl/egca_grpo.py](src/oeis_learn/rl/egca_grpo.py).
- Implementing Potential-Based Reward Shaping (PBRS) and down-sampled lexicase rollout evaluation in [src/oeis_learn/rl/prompt_weighting.py](src/oeis_learn/rl/prompt_weighting.py) and [src/oeis_learn/curriculum/sampler.py](src/oeis_learn/curriculum/sampler.py).
- Benchmarking the improved architecture in an autonomous overnight run tracked under `runs/003_phase3_inductive_generalization/`.

### Out of Scope / Deferred
- Distributed multi-node multi-GPU cluster scaling ($d=768$, full 390,000+ catalog) — deferred until local Tier 1 Stage 1 and Stage 2 graduation gates are passed.
- Differentiable interpretation of full WebAssembly specifications — soft stack relaxations remain deferred due to $\mathcal{O}(T \cdot |\mathcal{O}| \cdot S^2)$ memory scaling overhead; discrete execution-guided decoding remains the primary engine.
- Dynamic heap memory growth (`memory.grow`) for Stages 1–3 — restricted to scalar stack operations and local variable buffers.

---

## 6. Assumptions

- **Hardware Profile (Tier 1 Baseline)**: Workstation with 4 CPU cores / 8 threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, and an NVIDIA GPU with 4 GB VRAM (Quadro M2000M).
- **GPU Precision & Memory Management**: Strict single precision (`torch.float32`) without Automatic Mixed Precision (AMP). Peak VRAM is capped at $< 3.5\,\text{GB}$ via micro-batching ($B=1\text{--}4$) and sequence chunking ($L_{\text{chunk}} = 256$).
- **WASM Execution Sandbox**: 100% of WebAssembly parsing, Cranelift JIT compilation, and fuel-metered execution (10,000 instruction budget, 16 MiB linear memory) is offloaded to 8 CPU threads via the native Rust `oeis_wasm_evaluator` extension, releasing the Python GIL.
- **Experiment Tracking**: All artifacts (configs, metadata, checkpoints, logs, telemetry, synthesis results, discovered theorems) are automatically archived in structured numbered directories (`runs/003_...`).
