# Feature Specification: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization

**Feature Branch**: `002-synthesis-bootstrapping-and-soundness`  
**Created**: 2026-08-31  
**Status**: Draft  
**Prerequisites**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md), [reports/long_e2e_summary.md](reports/long_e2e_summary.md), [reports/tier1_baseline_report.md](reports/tier1_baseline_report.md)

---

## 1. Executive Context & Problem Statement

Phase 1 established a high-throughput, deterministic execution and perception infrastructure for `oeis-learn`:
- **Native WASM Evaluator:** Evaluated $>96,000$ modules in-memory with Rayon multi-threading and Cranelift fuel metering ($>5,500$ evals/sec, zero host crashes).
- **Tri-Stream Continuous Encoder:** Achieved 100% numerical stability in strict FP32 across astronomical dynamic ranges ($-10^6$ to $10^{30}$).
- **Theorem Prover:** Successfully proved real mathematical identities via high-precision ($>500$ digits) PSLQ integer relation searches and SymPy symbolic proofs.

However, an 18.65-hour end-to-end training benchmark revealed three fundamental bottlenecks preventing autonomous reinforcement learning from synthesizing working programs from cold start:

1. **Context-Sensitive Grammar Soundness Gaps:** The context-free grammar engine allowed generation to omit mandatory function signature declarations (e.g., `(param $n i32) (result i64)`), producing structurally malformed WASM strings that caused immediate `PARSE_ERROR` traps.
2. **Cold-Start Exploration Desert & Zero-Advantage Collapse:** Initializing policy optimization from purely random weights under strict binary rewards ($\pm 1.0$) resulted in a 0% pass rate across all rollouts. In Group Relative Policy Optimization (GRPO), when all $G$ completions fail, advantage variance collapses to zero ($\sigma_{\mathbf{r}} = 0$), starving the policy of positive learning gradients.
3. **Reward Sparsity & High Diagnostic Latency:** All-or-nothing binary outcome evaluation penalizes near-correct solutions identically to complete noise, providing no gradient surface for parameter updates. Furthermore, relying on multi-hour training runs to diagnose configuration or grammar issues creates excessive feedback latency.

This specification defines **Phase 2: Synthesis Bootstrapping, Semantic Soundness & Progressive Optimization**, establishing the functional and operational requirements to achieve robust program synthesis on integer sequences under Tier 1 workstation hardware constraints.

---

## 2. User Scenarios & Testing

### User Story 1 — Structurally & Semantically Sound Environment-Indexed Decoding (Priority: P1)

As a neuro-symbolic synthesizer, I want autoregressive token decoding to enforce mandatory function signature structures, lexical scoping rules, and stack depth/type constraints at sub-$100\,\mu\text{s}$ latency, so that 100% of generated program candidates assemble into compilable WebAssembly binaries without syntax errors, missing parameter declarations, or unbound variable traps.

**Why this priority**: Without semantic compilation soundness, exploratory rollouts fail at the parser stage before execution begins, wasting computational budget and starving RL optimization of execution feedback.

**Independent Test**: Generate 1,000 candidate programs under unconstrained temperature sampling ($T=1.0$); verify that 100% of generated programs compile in-memory without `PARSE_ERROR`, `MISSING_ENTRYPOINT`, or unbound identifier traps.

**Acceptance Scenarios**:
1. **Given** an active decoding state at function initialization, **When** generating the function header, **Then** the grammar engine mandates the exact sequence `(export "compute") (param $n i32) (result i64)` before allowing any instruction body tokens.
2. **Given** variable access instructions (`local.get`, `local.set`, `local.tee`), **When** selecting variable identifiers, **Then** logit masks restrict options strictly to declared, in-scope identifiers ($\text{Vars}_t$), guaranteeing No-Ghost Soundness.
3. **Given** stack-consuming instructions (e.g., binary arithmetic `i64.add`, branches `br_if`), **When** generating tokens, **Then** the constraint engine verifies operand stack depth ($\vert\Sigma_t\vert \ge m$) and type compatibility before enabling the opcode.
4. **Given** the function closing token `)`, **When** completing generation, **Then** the decoder enforces that the final stack state precisely matches the declared function result signature (`[i64]`).

---

### User Story 2 — Demonstration Bootstrapping & Supervised Warmup (Priority: P1)

As an ML training pipeline, I want the policy decoder to be initialized via Supervised Fine-Tuning (SFT) / Imitation Learning on forward-generated synthetic and canonical reference programs, so that the policy acquires basic WebAssembly syntax idioms and arithmetic templates before initiating reinforcement learning exploration.

**Why this priority**: Pure RL from random weights suffers from exponential exploration complexity on stack-based bytecode. SFT demonstration warmup provides an inductive anchor that ensures non-zero baseline synthesis pass rates.

**Independent Test**: Train the Transformer decoder on a synthetic dataset of 5,000 forward-generated sequence-program pairs for 5 epochs; verify that greedy synthesis ($T=0.0$) produces $>80\%$ compilable and mathematically valid programs on Stage 1 polynomial tasks.

**Acceptance Scenarios**:
1. **Given** a synthetic grammar generator, **When** generating training pairs, **Then** the system samples diverse, valid WebAssembly programs covering linear, polynomial, exponential, modular, and loop primitives paired with their executed $N$-term integer sequences.
2. **Given** the encoder-decoder architecture, **When** running supervised pretraining with cross-entropy teacher forcing, **Then** token prediction loss converges to $<0.50$ without gradient anomalies.
3. **Given** held-out Stage 1 evaluation prompts, **When** generating programs via greedy decoding, **Then** the warmed-up policy achieves $\ge 80\%$ functional compilation and sequence accuracy on basic arithmetic progressions.

---

### User Story 3 — Multi-Tiered Dense-to-Sparse Reward Shaping & S-GRPO Exploration (Priority: P2)

As a reinforcement learning optimizer, I want policy gradient optimization to utilize multi-tiered reward signals (compilation validity, prefix match distance, execution trace divergence attribution) and reference trajectory injection (S-GRPO / AVSPO), so that the policy receives continuous learning gradients during early exploration and smoothly transitions to strict verifiable binary rewards without zero-advantage collapse.

**Why this priority**: Strict binary rewards ($\pm 1.0$) produce flat, uninformative optimization landscapes on difficult sequences. Multi-tiered reward shaping and ground-truth injection prevent exploration starvation while maintaining asymptotic rigor.

**Independent Test**: Run 20 training steps across a batch containing hard prompts where generated completions fail; verify that Advantage Collapse Rate remains bounded ($\text{ACR} \le 0.15$), non-zero gradient updates are applied, and pass rate scales monotonically.

**Acceptance Scenarios**:
1. **Given** a generated candidate program, **When** evaluating reward during early curriculum stages, **Then** the reward function computes a dense composite score incorporating compilation success, normalized output prefix length, and numerical proximity.
2. **Given** a rollout group where all $G$ sampled completions fail ($R_i \le 0$), **When** computing group advantages, **Then** the system engages Conditional Ground-Truth Trajectory Injection (CGI) or virtual anchor advantages to prevent zero-advantage collapse ($\sigma_{\mathbf{r}} > 0$).
3. **Given** advancing training epochs, **When** the policy competence improves, **Then** dense auxiliary shaping terms are annealed via a cosine schedule toward strict binary outcome rewards ($+1.0 / -1.0$) on extended evaluation horizons ($K=100$).

---

### User Story 4 — Progressive Micro-Benchmarking & Hyperparameter Tuning Protocol (Priority: P2)

As a developer and system operator, I want a multi-tier progressive testing suite (Tier 0 deterministic unit checks $\to$ Tier 1 SFT fitting $\to$ Tier 2 single-prompt RL convergence $\to$ Tier 3 micro-cohort tuning $\to$ Tier 4 full curriculum runs) with real-time diagnostic telemetry, so that bugs and hyperparameter misconfigurations are identified in seconds or minutes rather than after multi-hour failures.

**Why this priority**: Long-running 10+ hour jobs create unacceptable diagnostic feedback loops. A progressive test hierarchy guarantees that compute is allocated only to verified, healthy configurations.

**Independent Test**: Execute the progressive testing harness across Tiers 0, 1, 2, and 3 in $<45\text{ minutes}$; verify that all diagnostic telemetry metrics (entropy, advantage variance, compiler trap rate, oracle perplexity) fall within healthy operational bounds before authorizing Tier 4 execution.

**Acceptance Scenarios**:
1. **Given** Tier 0 static validation, **When** evaluated, **Then** the test verifies deterministic sandbox trapping, linear memory isolation, and grammar token boundaries in $<5\text{ seconds}$.
2. **Given** Tier 1 SFT alignment, **When** fitting a canonical reference solution, **Then** token perplexity drops below $1.25$ within 20 optimization steps in $<2\text{ minutes}$.
3. **Given** Tier 2 single-prompt RL training, **When** optimizing on an isolated sequence (e.g., Triangular numbers), **Then** policy pass rate reaches $100\%$ within 15 gradient iterations in $<10\text{ minutes}$.
4. **Given** real-time training telemetry, **When** monitoring the run, **Then** the harness tracks policy entropy $\mathcal{H}(\pi_\theta)$, group reward variance $\sigma_{\mathbf{r}}^2$, Advantage Collapse Rate (ACR), and compiler trap rate, halting execution if divergence thresholds are breached.

---

### User Story 5 — Self-Supervised Latent Manifold Structuring & Algebraic Homomorphism (Priority: P3)

As a mathematical discovery pipeline, I want continuous sequence representations to be regularized via self-supervised non-contrastive objectives (Kernel VICReg / Distribution Matching) and algebraic homomorphism constraints across operator pairs, so that latent space geometry reflects formal mathematical operations and yields high-quality candidate relations for PSLQ theorem proving.

**Why this priority**: When sequence encoders are trained solely on downstream synthesis without structural regularization, the latent space collapses into low-dimensional subspaces with poor family clustering, preventing vector arithmetic ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$) from discovering algebraic identities.

**Independent Test**: Extract embeddings for a 100-sequence mathematical benchmark; verify that latent space exhibits high rank dispersion ($\text{RDR} > 0.80$), forms distinct topological clusters under UMAP/HDBSCAN ($\ge 5$ families), and yields candidate triples verified by PSLQ ($<10^{-50}$ drop) and SymPy proofs.

**Acceptance Scenarios**:
1. **Given** algebraic sequence transformation pairs (partial sums $\Sigma$, finite differences $\Delta$, binomial transforms $\mathcal{B}$, shift operators $T_k$), **When** pretraining the encoder, **Then** non-contrastive VICReg loss prevents dimensional collapse without suffering from class-collision penalties.
2. **Given** sequence pairs $A, B$ and their sum $A+B$, **When** mapping to latent vectors, **Then** an additive homomorphism loss enforces $\Vert f(A+B) - (f(A) + f(B))\Vert_2 < \epsilon$.
3. **Given** the structured continuous manifold, **When** queried with vector arithmetic, **Then** candidate triples pass high-precision ($>500$ digits) PSLQ integer relation searches and produce machine-verified symbolic proofs.

---

## 3. Functional Requirements

### Structural Grammar & Decoding Soundness
- **FR-001**: System MUST enforce mandatory structural sequencing in the decoder, requiring `(module (func (export "compute") (param $n i32) (result i64) ...))` before instruction body generation.
- **FR-002**: System MUST maintain dynamic symbol tables ($\text{Vars}_t$) during autoregressive decoding, restricting variable indices in `local.get`, `local.set`, and `local.tee` exclusively to in-scope declared identifiers (No-Ghost Soundness).
- **FR-003**: System MUST track operand stack depth ($\vert\Sigma_t\vert$) and value types during decoding, masking out instructions that violate stack height or operand type requirements.
- **FR-004**: System MUST evaluate grammar logit masks under a strict latency budget not exceeding $100\,\mu\text{s}$ per token (targeting $5\text{--}20\,\mu\text{s}$ median latency).
- **FR-005**: System MUST ensure that 100% of generated programs compile in-memory without syntax errors, missing entrypoints, or stack underflow traps.

### Synthesis Bootstrapping & Demonstration Warmup
- **FR-006**: System MUST generate synthetic dataset pairs $(Y, P)$ mapping $N$-term integer sequences $Y$ to valid WebAssembly Text programs $P$ across polynomial, recurrence, and modular algorithmic families.
- **FR-007**: System MUST provide a Supervised Fine-Tuning (SFT) training mode optimizing cross-entropy loss over canonical programs before reinforcement learning initialization.
- **FR-008**: System MUST support populating and querying an Elite Seed Demonstration Buffer ($\mathcal{D}_{\text{elite}}$) for reference trajectory retrieval during policy optimization.

### Reinforcement Learning & Reward Structuring
- **FR-009**: System MUST support composite reward shaping during early curriculum stages combining compiler validation ($R_{\text{comp}}$), normalized output prefix match length ($R_{\text{prefix}}$), and continuous numerical distance ($R_{\text{dist}}$).
- **FR-010**: System MUST support Conditional Ground-Truth Trajectory Injection (S-GRPO / CGI) or virtual anchor rewards when all completions in a rollout group fail ($R_i \le 0$).
- **FR-011**: System MUST track real-time Advantage Collapse Rate ($\text{ACR} = \frac{1}{B} \sum \mathbb{I}(\sigma_{\mathbf{r}}^2 = 0)$) and trigger exploration recovery mechanisms when $\text{ACR} \ge 0.30$.
- **FR-012**: System MUST support cosine annealing of dense shaping terms, transitioning asymptotically to exact binary outcome rewards ($+1.0 / -1.0$) on extended evaluation horizons ($K=100$).
- **FR-013**: System MUST trace candidate execution paths to localize policy gradient updates to the earliest instruction tokens associated with state divergence (Execution-Grounded Credit Assignment).

### Progressive Testing & Diagnostic Telemetry
- **FR-014**: System MUST implement a 5-tier progressive test harness:
  - Tier 0: Deterministic sandbox and static grammar validation ($<5\,\text{s}$).
  - Tier 1: Oracle reference solution fitting and perplexity verification ($<2\,\text{m}$).
  - Tier 2: Single-prompt policy gradient convergence ($<10\,\text{m}$).
  - Tier 3: Synthetic micro-cohort curriculum progression and hyperparameter grid ($<45\,\text{m}$).
  - Tier 4: Full dataset scaling and multi-stage curriculum run ($2\text{--}4\,\text{h}$).
- **FR-015**: System MUST log real-time diagnostic telemetry during training, including policy entropy $\mathcal{H}(\pi_\theta)$, group reward variance $\sigma_{\mathbf{r}}^2$, compiler trap rate, average prefix match length, and reference token perplexity.
- **FR-016**: System MUST operate within Tier 1 workstation memory constraints ($\le 4\,\text{GB}$ GPU VRAM) using sequence chunking, FP32 precision, and CPU offloading for WASM evaluation.

### Self-Supervised Manifold & Discovery
- **FR-017**: System MUST train continuous sequence representations using non-contrastive regularization (VICReg / Kernel VICReg) over algebraic transformation pairs ($\Sigma, \Delta, \mathcal{B}, T_k$).
- **FR-018**: System MUST apply explicit additive homomorphism loss penalties ($\mathcal{L}_{\text{add}} = \Vert f(A+B) - (f(A) + f(B))\Vert_2^2$) during representation pretraining.
- **FR-019**: System MUST perform manifold reduction (UMAP/PaCMAP) and density clustering (HDBSCAN) to discover unannotated sequence families without dimensional collapse.
- **FR-020**: System MUST verify candidate relation triples via arbitrary-precision ($>500$ digits) PSLQ integer relation searches and generate SymPy symbolic proofs.

---

## 4. Success Criteria & Measurable Outcomes

| Metric ID | Target Metric | Required Threshold | Verification Method |
| :--- | :--- | :--- | :--- |
| **SC-001** | **Compilation Soundness Rate** | **$100.0\%$** compilable WASM binaries | 1,000 random token generations at $T=1.0$ yield 0 `PARSE_ERROR` or `MISSING_ENTRYPOINT`. |
| **SC-002** | **Grammar Masking Latency** | **$< 100\,\mu\text{s}$** per token | Average per-token logit mask generation time across 10,000 steps. |
| **SC-003** | **SFT Synthesis Pass Rate** | **$\ge 80.0\%$** on Stage 1 primitives | Greedy synthesis on held-out linear, quadratic, and modular sequence prompts after SFT. |
| **SC-004** | **Single-Prompt RL Convergence** | **$100.0\%$** pass rate in $\le 15$ steps | Tier 2 single-prompt overfitting test on Triangular Numbers (A000217). |
| **SC-005** | **Advantage Collapse Rate (ACR)** | **$\text{ACR} \le 0.15$** during RL exploration | Rolling fraction of zero-variance rollout groups during Stage 1 training. |
| **SC-006** | **Tier 1 Stage 1 Competence** | **$C(S_1) \ge 0.80$** after RL training | Weighted rolling competence score across 25 Stage 1 sequence tasks. |
| **SC-007** | **Extrapolation & Anti-Memorization** | **$100\%$** exact match on $K=100$ terms with **$M_{\text{MDL}} \le 1.2$** | Generalization verifier on all graduated Stage 1 and Stage 2 candidate algorithms. |
| **SC-008** | **Latent Rank Dispersion Ratio** | **$\text{RDR} \ge 0.80$** ($\ge 5$ distinct clusters) | Covariance rank and HDBSCAN clustering on 100 real sequence embeddings. |
| **SC-009** | **Progressive Test Duration** | **$< 45\text{ minutes}$** for Tiers 0–3 | End-to-end execution of pre-flight validation harness before Tier 4 runs. |

---

## 5. Architectural & System Boundaries

### In Scope
- Enhancing `EnvironmentTracker` and `GrammarMasker` in [src/oeis_learn/decoder/](src/oeis_learn/decoder/) with mandatory header state machines, stack type trackers, and No-Ghost identifier slots.
- Building a synthetic demonstration generator and SFT training pipeline in [src/oeis_learn/rl/](src/oeis_learn/rl/) and [src/oeis_learn/decoder/](src/oeis_learn/decoder/).
- Implementing composite reward shaping (compilation + prefix match + distance) and S-GRPO trajectory injection in [src/oeis_learn/rl/](src/oeis_learn/rl/).
- Building the 5-tier progressive test harness in [tests/](tests/) and [scripts/](scripts/).
- Integrating algebraic homomorphism penalties into VICReg pretraining in [src/oeis_learn/discovery/](src/oeis_learn/discovery/).

### Out of Scope / Deferred
- Distributed multi-node cluster scaling (Tier 2 $d=768$, multi-GPU DDP) — deferred until local Tier 1 Stage 1 & Stage 2 graduation gates pass.
- Non-WebAssembly target compilation (e.g., C++/Rust source generation) — WebAssembly stack bytecode remains the sole sandboxed execution target.
- Dynamic heap memory allocation in WASM (`memory.grow`) for Stages 1–3 — restricted to scalar stack operations and local variable buffers.

---

## 6. Assumptions & Hardware Constraints

- **Hardware Profile (Tier 1 Baseline):** 4 CPU Cores / 8 Threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- **GPU Precision & Memory:** All neural operations execute in strict single precision (`torch.float32`). VRAM usage is capped at $< 3.5\,\text{GB}$ via micro-batching ($B=1\text{--}4$) and sequence chunking.
- **Execution Offloading:** 100% of WebAssembly compilation, fuel metering, and execution is offloaded to CPU worker threads via the native Rust `oeis_wasm_evaluator` extension.
- **Deterministic Data Access:** Real sequence records are loaded from local caches (`data/sample_oeis/`) with reproducible fallback generation for synthetic benchmarks.
