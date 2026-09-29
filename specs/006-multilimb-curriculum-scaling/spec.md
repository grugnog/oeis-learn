# Feature Specification: 4 × i64 Multi-Limb Migration, Corpus-Informed Curriculum Scaling, & Warm-Started Training Plan

**Feature Branch**: `006-multilimb-curriculum-scaling`

**Created**: 2026-09-06

**Status**: Draft

**Input**: User description: "4 × i64 Multi-Limb Migration, Corpus-Informed Curriculum Scaling, & Warm-Started Training Plan"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Limb Precision Execution and Truncation-Free Synthesis (Priority: P1)

As a research operator, I want the program synthesis and evaluation pipeline to execute in 255-bit signed multi-limb precision using compact value-stack macro operations, so that linear recurrences, geometric progressions, and holonomic loops synthesize and extrapolate correctly without encountering 64-bit integer overflow truncation.

**Why this priority**: Prior training runs demonstrated strong structural competence across Stage 1 polynomials, Stage 2 linear recurrences, and Stage 3 holonomic loops. However, key benchmark canaries (Fibonacci past $F_{92}$, Lucas past $L_{90}$, Powers of 2 past $2^{62}$, and Pell past $P_{37}$) immediately fail 100-term extrapolation solely due to hardware scalar overflow. An empirical census of 399,005 OEIS sequences confirms that a $4 \times i64$ (255-bit signed) representation covers 99.51% of the entire OEIS and 99.56% of computable implementations, resolving the mathematical bottleneck.

**Independent Test**: Evaluate candidate execution on Fibonacci (`A000045`), Lucas (`A000032`), Powers of 2 (`A000079`), and Pell (`A000129`) through 100 terms; verify that multi-limb execution produces exact mathematical integers without sign flipping, bit wrapping, or runtime crashes.

**Acceptance Scenarios**:

1. **Given** a sequence whose terms exceed $2^{63}-1$, **When** a candidate multi-limb program is executed in the isolated sandbox, **Then** all calculations are performed across four 64-bit limbs and reconstructed into exact arbitrary-precision signed integers matching the mathematical target.
2. **Given** a candidate program generated with high-level macro operations (`i256.add`, `i256.sub`, `i256.mul_scalar`, `i256.const`), **When** lowered for sandboxed execution, **Then** the candidate links against a static, pre-verified value-stack arithmetic preamble without allocating linear heap memory or introducing dynamic pointers.
3. **Given** generated multi-limb programs for order-2 linear recurrences, **When** sequence token length is measured, **Then** the structural body remains under 40 tokens, preserving reinforcement learning credit assignment.
4. **Given** sequences with negative integer values or signed subtractions, **When** quad-limb operations execute, **Then** two's-complement arithmetic produces exact negative values matching ground truth.
5. **Given** an invalid or runaway candidate program, **When** execution exceeds the non-negotiable 10,000 instruction fuel limit or triggers an arithmetic fault, **Then** execution terminates deterministically without destabilizing or crashing the host process.

---

### User Story 2 - High-Throughput Decoupled Symbolic Grounding (Priority: P1)

As a synthesis researcher, I want candidate program skeletons with undetermined constant placeholders to be resolved via fast modular rank filtering and bounded Diophantine solving, so that coefficient grounding finishes in under a millisecond rather than timing out in SMT bit-blasting.

**Why this priority**: Attempting to bit-blast 256-bit recurrence systems into SAT clauses causes an 88.6% timeout rate (>250ms) because 256-bit multipliers create millions of CNF clauses across carry-save trees. A decoupled three-tier solver (fast modular rank filter $\to$ exact Diophantine solver $\to$ bounded non-linear tactical solver) achieves sub-millisecond resolution while rejecting invalid skeletons instantly.

**Independent Test**: Feed 1,000 multi-limb candidate skeletons with constant placeholders into the grounding pipeline; verify that inconsistent skeletons are rejected in under 0.5 ms, linear recurrence skeletons are resolved in under 1 ms, and valid coefficients match ground-truth integers.

**Acceptance Scenarios**:

1. **Given** an invalid or underdetermined candidate skeleton, **When** evaluated by the Tier 1 modular filter, **Then** it is rejected in under 0.5 ms with an explicit failure certificate and assigned a negative grounding penalty.
2. **Given** a consistent linear recurrence skeleton, **When** passed to the Tier 2 bounded Diophantine solver, **Then** unknown integer coefficients within $[-1000, 1000]$ are reconstructed and verified against exact integer terms in under 1 ms.
3. **Given** a non-linear candidate skeleton, **When** dispatched to the Tier 3 constraint solver, **Then** multi-precision integer assertions solve or time out cleanly within a bounded 240 ms tactical budget.
4. **Given** grounded coefficients from any solver tier, **When** inserted into the candidate skeleton and executed against observed terms, **Then** outputs match all observed sequence terms with 100% exactness.

---

### User Story 3 - Progressive Warm-Started Continual Learning (Priority: P2)

As a machine learning engineer, I want to warm-start the multi-limb policy from the trained Run 010 checkpoint through progressive vocabulary surgery and supervised fine-tuning (SFT) bridge transfer, so that the mathematical induction representations of the encoder are preserved without suffering policy advantage collapse.

**Why this priority**: Cold-start training discards hours of representation learning in the Tri-Stream Encoder, incurring a 3.5×–4.5× compute penalty. Conversely, direct RL warm-start causes 100% advantage collapse because the scalar policy cannot emit new multi-limb syntax. Progressive SFT transfer bridges the syntax shift while preserving representation priors.

**Independent Test**: Perform vocabulary surgery and embedding warmup on the Run 010 checkpoint, run the joint SFT bridge phase, and verify that the transition gate achieves $\le 15\%$ advantage collapse rate, $\ge 98.5\%$ syntactic validity, and $\ge 75\%$ pass rate before initiating reinforcement learning.

**Acceptance Scenarios**:

1. **Given** a trained scalar checkpoint, **When** the vocabulary is expanded with multi-limb syntax tokens, **Then** new token embeddings are initialized via convex hull projection from semantic ancestors and unembedding weights are norm-calibrated.
2. **Given** newly expanded embedding parameters, **When** Phase 2 embedding warmup executes for 1,500 steps, **Then** encoder and decoder backbone parameters remain frozen while new token representations align.
3. **Given** warmed embeddings, **When** Phase 3 joint SFT executes for 5,000 steps with discriminative learning rates, **Then** decoder backbone and top encoder layers adapt to multi-limb demonstration trajectories.
4. **Given** completion of SFT bridge training, **When** the transition gate is evaluated, **Then** the model must satisfy advantage collapse rate $\le 15\%$, grammar validity $\ge 98.5\%$, and candidate pass rate $\ge 75\%$ before reinforcement learning is authorized.
5. **Given** transition gate satisfaction, **When** RL policy optimization begins, **Then** the reference policy is re-anchored to the bridge model, the task bandit is recalibrated, and the elite demonstration buffer is populated with verified 4-limb programs to prevent zero-advantage collapse.

---

### User Story 4 - Corpus-Informed Multi-Stage Curriculum Scaling (Priority: P2)

As a curriculum designer, I want the training and evaluation framework to support macro-scaffolding templates for Stages 1 through 4, negative numbers, and flexible extrapolation horizons, so that the model can learn and synthesize diverse algebraic, recurrence, holonomic, and number-theoretic sequences.

**Why this priority**: The empirical OEIS census reveals that 10.2% of sequences contain negative terms, recurrences span orders $k=1..4$, holonomic factorials require loop-index accumulation, and number-theoretic sequences require nested divisor loops. Enforcing rigid 100-term checks on sequences with fewer published terms creates false failure signals.

**Independent Test**: Train and evaluate across a curated cohort of 150–200 sequences spanning Stage 1 (polynomials), Stage 2 (linear recurrences), Stage 3 (holonomic factorials), and Stage 4 (divisor loops); verify correct template application, signed arithmetic handling, and flexible extrapolation assessment.

**Acceptance Scenarios**:

1. **Given** Stage 1 polynomial sequences, **When** candidates are generated, **Then** single-variable closed-form arithmetic is produced without control-flow loops.
2. **Given** Stage 2 linear recurrences of order $k \in \{1, 2, 3, 4\}$, **When** candidates are generated, **Then** multi-limb sliding window registers update state correctly across iterations.
3. **Given** Stage 3 holonomic sequences, **When** candidates are generated, **Then** loop indices are converted and multiplied into multi-limb accumulators to compute factorials and D-finite terms.
4. **Given** Stage 4 multiplicative or combinatorial sequences, **When** candidates are generated, **Then** nested control loops evaluate divisibility within bounded execution fuel.
5. **Given** sequences with negative or alternating values, **When** candidates are generated and verified, **Then** signed multi-limb operations produce correct sign patterns.
6. **Given** an evaluation sequence with $N_{\text{avail}} < 120$ published terms, **When** extrapolation is evaluated, **Then** the unseen evaluation horizon adapts to $\min(100, N_{\text{avail}} - N_{\text{obs}})$ with a mandatory unseen margin of at least $\max(15, 0.4 \times N_{\text{avail}})$.

---

### User Story 5 - Canary Verification and Production Run Launch (Priority: P3)

As a research lead, I want an automated benchmark qualification harness evaluating all 6 landmark canaries over 100 unseen terms without overflow, so that production Run 011 can be launched with confidence and verified against quantifiable milestones.

**Why this priority**: Production training runs consume 8–10 hours of workstation compute. Passing all landmark canaries across polynomial, factorial, and recurrence classes is the definitive empirical proof of multi-limb correctness.

**Independent Test**: Execute automated preflight verification on the 6 landmark canaries (`A000217`, `A000290`, `A000079`, `A000045`, `A000032`, `A000129`) using the warm-started multi-limb checkpoint; confirm 100% exact 100-term extrapolation.

**Acceptance Scenarios**:

1. **Given** the 6 landmark canaries, **When** evaluated against 100 unseen terms, **Then** all 6 achieve exact match status without integer truncation, arithmetic overflow, or runtime traps.
2. **Given** production Run 011 configuration, **When** initialized, **Then** it loads the warm-started SFT checkpoint, enforces multi-limb grammar constraints, and logs per-task competence and extrapolation telemetry.
3. **Given** a failed prompt group during GRPO training, **When** all sampled candidates fail, **Then** the elite demonstration buffer injects a canonical 4-limb trajectory to maintain non-zero policy gradient variance.

### Edge Cases

- Super-exponential sequences whose terms exceed 255 signed bits ($|a(n)| \ge 2^{255}$) within 100 terms.
- Sequences with fewer than 20 available published terms in the OEIS corpus.
- Sequences containing alternating zeros or sparse integer patterns where modular rank filtering might find degenerate ranks.
- Skeletons with cyclic dependencies or underdetermined systems where multiple coefficient sets fit observed terms.
- High-degree nested loops in Stage 4 triggering fuel exhaustion before term completion.
- Non-monotonic loss or sudden spike in advantage-collapse rate during joint SFT bridge training.
- Logit divergence or extreme probability mass shifts on newly introduced tokens during initial RL steps.
- Sequences with zero-padded prefixes or offset start indices ($n_0 \neq 0$).

## Requirements *(mandatory)*

### Functional Requirements

#### Multi-Limb Representation and Execution Engine

- **FR-001**: The system MUST support a $4 \times i64$ (255-bit signed) multi-limb integer representation across code generation, sandboxed execution, and term verification.
- **FR-002**: Generated candidate programs MUST use high-level multi-limb macro instructions (`i256.add`, `i256.sub`, `i256.mul_scalar`, `i256.const`) alongside structural WebAssembly operations (`loop`, `br_if`, `local.get`, `local.set`).
- **FR-003**: The execution engine MUST link candidate programs against a static, pre-compiled WebAssembly preamble providing pure value-stack implementations of multi-limb addition, subtraction, widening multiplication, and scalar multiplication.
- **FR-004**: Execution of candidate modules MUST consume zero linear memory (`max_memories(0)`) and MUST NOT allocate dynamic heap memory or use host-guest pointers.
- **FR-005**: The execution sandbox MUST strictly enforce an instruction fuel ceiling capped at 10,000 instructions per execution.
- **FR-006**: The sandbox MUST return 4 64-bit integer limbs on the value stack in little-endian order and reconstruct the exact signed arbitrary-precision integer in the host environment.
- **FR-007**: Generated multi-limb programs for order-2 linear recurrences MUST NOT exceed 40 tokens in length, preserving policy gradient credit assignment.
- **FR-008**: The execution engine MUST support signed subtraction, signed scalar multiplication, and signed division/remainder operations without sign-mask distortion.

#### Decoupled Symbolic Grounding Pipeline

- **FR-009**: Candidate program skeletons containing unknown constant placeholders MUST be processed through a decoupled three-tier symbolic grounding pipeline before full sandboxed verification.
- **FR-010**: Tier 1 MUST implement a fast modular filter projecting 256-bit terms into a finite field via limb folding and performing Gaussian elimination to verify rank and consistency in under 0.5 ms.
- **FR-011**: Skeletons classified as inconsistent or underdetermined by the Tier 1 filter MUST be rejected immediately and assigned a negative grounding reward penalty.
- **FR-012**: Tier 2 MUST implement a bounded Diophantine solver for linear recurrence skeletons that reconstructs integer coefficients in $[-1000, 1000]$ and certifies exactness over $\mathbb{Z}$ in under 1.0 ms.
- **FR-013**: Tier 3 MUST implement a bounded tactical non-linear constraint solver that constrains multi-precision integer unknowns within $[-1000, 1000]$ and enforces a hard tactical timeout of 240 ms.
- **FR-014**: Every grounded candidate program MUST retain lineage linking its raw skeleton, resolved coefficients, solver tier, solving duration, and verification outcome.
- **FR-015**: SMT bit-blasting (`QF_BV`) of multi-limb recurrence systems MUST be prohibited in production synthesis and training rollouts.

#### Warm-Started Continual Learning and Vocabulary Transfer

- **FR-016**: The training pipeline MUST support warm-starting from pre-trained scalar checkpoints (specifically Run 010 Epoch 50) using Strategy C (Progressive SFT Bridge Transfer).
- **FR-017**: Vocabulary expansion MUST perform convex hull semantic projection to initialize input embeddings for new multi-limb tokens from their scalar semantic ancestors.
- **FR-018**: Unembedding linear head weights for new multi-limb tokens MUST be calibrated to match the mean norm of pre-existing vocabulary tokens.
- **FR-019**: Decoder logit computation MUST apply hyperbolic tangent logit soft-capping with a threshold parameter $C_{\text{cap}} = 30.0$ to prevent gradient explosion and probability mass monopolization during vocabulary expansion.
- **FR-020**: Phase 2 embedding warmup MUST train only the newly allocated embedding and head rows on multi-limb demonstrations for 1,500 steps while freezing the Tri-Stream Encoder and Transformer Decoder backbones.
- **FR-021**: Phase 3 joint SFT bridge MUST train the decoder backbone and top 2 encoder layers for 5,000 steps using discriminative learning rates ($10^{-5}$ for encoder, $5 \times 10^{-5}$ for decoder, $10^{-4}$ for new heads).
- **FR-022**: The system MUST enforce a pre-RL Transition Gate requiring advantage collapse rate $\le 15\%$, grammar validity $\ge 98.5\%$, and candidate pass rate $\ge 75\%$ on held-out multi-limb data before authorizing reinforcement learning.
- **FR-023**: Upon satisfying the Transition Gate, Phase 4 MUST re-anchor the RL reference policy to the bridge model, initialize KL penalty $\beta_{\text{KL}} = 0.00$ with scheduled annealing to $0.04$, and recalibrate task bandit weights with an exploration floor $\gamma_{\text{floor}} = 0.25$.

#### Corpus-Informed Curriculum and Macro-Scaffolding

- **FR-024**: The grammar and synthesis engine MUST support macro-scaffolding templates for Stage 1 (closed-form polynomials), Stage 2 (bounded linear recurrences of order $k \in \{1, 2, 3, 4\}$), Stage 3 (holonomic factorials with loop-index multiplication), and Stage 4 (nested combinatorial/divisor loops).
- **FR-025**: The training dataset generator MUST produce at least 20,000 verified multi-limb demonstrations balanced across elementary arithmetic, Stage 1 polynomials, Stage 2 recurrences, and Stage 3 holonomic sequences.
- **FR-026**: All synthetic training demonstrations MUST be verified to achieve 100% execution pass rate on the multi-limb execution engine prior to training.
- **FR-027**: The Elite Demonstration Buffer (EDB) MUST be seeded with canonical 4-limb implementations for all Stage 1, Stage 2, and Stage 3 evaluation sequences.
- **FR-028**: When all candidate rollouts in a GRPO prompt group fail, the training engine MUST inject a canonical 4-limb trajectory from the elite buffer into the group to maintain non-zero reward variance ($\sigma_{\mathcal{R}} > 0$).
- **FR-029**: The synthesis grammar masker MUST permit signed subtraction, signed division/remainder, and bitwise negation to support sequences with negative terms and alternating signs.
- **FR-030**: For benchmark canaries, extrapolation MUST evaluate 120 total terms (20 observed terms + 100 unseen terms) against verified extended b-files.
- **FR-031**: For general corpus sequences with fewer than 120 published terms, extrapolation MUST evaluate over $\min(100, N_{\text{avail}} - N_{\text{obs}})$ terms, enforcing an unseen evaluation margin of at least $\max(15, 0.4 \times N_{\text{avail}})$.

#### Production Qualification and Canary Verification

- **FR-032**: The system MUST provide an automated preflight qualification suite verifying exact 100-term extrapolation on the 6 landmark canaries (`A000217`, `A000290`, `A000079`, `A000045`, `A000032`, `A000129`).
- **FR-033**: A production run MUST be blocked if any landmark canary fails exact extrapolation or encounters integer overflow.
- **FR-034**: Telemetry during multi-limb training and evaluation MUST track advantage collapse rate, unique candidate ratio, modular filter rejection rate, Diophantine solving rate, runtime trap rate by cause, and per-stage competence.
- **FR-035**: Candidate programs MUST satisfy the Minimum Description Length (MDL) ratio constraint $M_{\text{MDL}} \le 1.20$ relative to target sequence complexity to prevent lookup table memorization.
- **FR-036**: Synthesized programs MUST execute deterministically, producing bit-identical limb values across repeated runs with identical inputs.
- **FR-037**: Multi-limb execution throughput MUST sustain at least 500 module evaluations per second across 8 CPU worker threads on Tier 1 hardware.
- **FR-038**: The production training configuration for Run 011 MUST specify checkpoint lineage, phased training durations, task curriculum distributions, and canary evaluation intervals.

### Key Entities

- **Multi-Limb State Register**: A 4-limb tuple ($[l_0, l_1, l_2, l_3]$) of 64-bit unsigned words representing a 255-bit signed integer in two's-complement little-endian layout.
- **Macro-Synthesized Program**: A compact WebAssembly candidate program composed of high-level multi-limb arithmetic macros and structural control flow, linked against a static value-stack preamble.
- **Static Arithmetic Preamble**: A pre-compiled, verified WebAssembly library containing pure value-stack multi-limb arithmetic functions ($i256\_add$, $i256\_sub$, $mul64\_wide$, $i256\_mul\_scalar$) with zero linear memory usage.
- **Symbolic Candidate Skeleton**: A program abstract syntax tree containing structural control flow and placeholder tokens for undetermined numerical coefficients.
- **Modular Filter Certificate**: A mathematical certificate produced by finite-field rank analysis indicating whether a candidate skeleton is consistent, inconsistent, or underdetermined.
- **Progressive Transfer Checkpoint**: A versioned model state representing one phase of continual learning (vocabulary expansion, embedding warmup, joint SFT bridge, or RL policy).
- **Curriculum Macro-Template**: A structural code scaffold defining register layouts and control flow patterns for a specific class of OEIS sequences (Stages 1 through 4).
- **Canary Benchmark Cohort**: A frozen collection of 6 landmark OEIS sequences (`A000217`, `A000290`, `A000079`, `A000045`, `A000032`, `A000129`) with 120 verified terms used for truncation and regression testing.
- **Elite Demonstration Buffer**: An in-memory replay cache of verified, high-reward multi-limb trajectories used to prevent advantage collapse during reinforcement learning.

### Scope Boundaries

**In scope**:

- Implementing the $4 \times i64$ (255-bit signed) multi-limb value representation and macro abstraction in the grammar and execution engine.
- Integrating the pre-compiled, verified value-stack arithmetic preamble into the sandboxed runtime.
- Implementing the three-tier symbolic grounding pipeline (Mersenne-61 fast modular filter, Dixon Diophantine solver, Z3 tactical QF_NIA solver).
- Executing Strategy C continual learning (vocabulary surgery, convex hull projection, embedding warmup, joint SFT bridge, reference policy re-anchoring).
- Generating 20,000 verified multi-limb synthetic demonstrations and seeding the Elite Demonstration Buffer.
- Expanding curriculum scaffolding and grammar masking to support Stages 1–4, negative numbers, and flexible extrapolation horizons.
- Launching and monitoring production Run 011 with preflight qualification on 6 landmark canaries.

**Out of scope**:

- Dynamic heap memory allocation, arbitrary-precision pointer-based BigInt implementations, or unconstrained linear memory growth.
- Bit-blasting SAT solving (`QF_BV`) for 256-bit recurrence systems.
- Cold-start training from random weights (Strategy A) or direct RL warm-start without SFT bridge (Strategy B).
- Modifying the core Tri-Stream Encoder architecture or introducing floating-point shortcuts.
- High-compute cluster scaling (Tier 2); all phases must operate within Tier 1 workstation bounds (4 CPU cores, 64 GB RAM, 4 GB VRAM).
- Unrestricted recursion, mutual recursion, or non-terminating loop structures.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 6 out of 6 landmark canaries (`A000217`, `A000290`, `A000079`, `A000045`, `A000032`, `A000129`) achieve 100% exact match across all 20 observed and 100 unseen terms without arithmetic overflow or runtime errors.
- **SC-002**: Multi-limb program execution sustains an average throughput of at least 500 candidate module evaluations per second across 8 CPU worker threads.
- **SC-003**: The Tier 1 modular consistency filter rejects 100% of invalid candidate skeletons with an average latency under 0.5 milliseconds per skeleton.
- **SC-004**: The Tier 2 exact Diophantine solver reconstructs bounded integer coefficients for solvable linear recurrence skeletons in under 1.0 millisecond with zero false positives.
- **SC-005**: Generated multi-limb programs maintain a median token length of 40 tokens or fewer across Stage 1, Stage 2, and Stage 3 sequence tasks.
- **SC-006**: The progressive SFT bridge transfer achieves an advantage collapse rate of at most 15%, grammar assembly validity of at least 98.5%, and candidate pass rate of at least 75% on held-out multi-limb demonstrations before RL launch.
- **SC-007**: Synthesized candidate programs execute with zero linear memory allocation traps and strictly respect the 10,000 instruction fuel limit.
- **SC-008**: Rolling task competence reaches at least 0.90 on Curriculum Stage 1 and Stage 2 evaluation sequences within 20 RL epochs of production Run 011.
- **SC-009**: The elite demonstration buffer maintains non-zero advantage variance across 100% of training prompt batches, completely eliminating zero-advantage policy collapse.
- **SC-010**: The synthesis pipeline handles 100% of valid test sequences containing negative integers or alternating signs without runtime exceptions or sign-mask distortion.

## Assumptions

- The pre-trained model checkpoint from Run 010 (Epoch 50, `model_epoch_050.v2.pt`) is available and verified to possess strong mathematical induction representations across Stage 1, Stage 2, and Stage 3.
- A 255-bit signed integer range ($[-2^{255}, 2^{255}-1]$) is sufficient for 99.51% of all sequences in the OEIS corpus and 100% of the project's target benchmark canaries through 100 terms.
- Synthetic multi-limb demonstrations generated programmatically provide adequate training signal to align newly expanded vocabulary rows during embedding warmup and bridge training.
- Workstation hardware remains bounded to Tier 1 constraints (4 CPU cores / 8 threads, 64 GB DDR4 RAM, 4 GB VRAM), with neural passes executed on GPU and multi-limb sandboxed execution offloaded to CPU threads.
- Verification requires exact integer equality across all evaluated terms; floating-point approximations or bounded absolute errors do not count as synthesis success.
- The project constitution remains authoritative regarding strict FP32 precision in the neural encoder, zero linear memory in the WASM sandbox, and anti-memorization Minimum Description Length limits.
- No branch is created by this specification command because no pre-specification branch hook is configured.
