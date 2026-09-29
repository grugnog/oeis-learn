# Feature Specification: OEIS Learn Neuro-Symbolic Synthesis

**Feature Branch**: `001-oeis-neurosymbolic-synthesis`

**Created**: 2026-08-30

**Status**: Draft

**Input**: User description: "oeis-learn is a Neuro-Symbolic AI system designed to learn representation spaces, perform automated mathematical discovery, and synthesize exact generating algorithms for integer sequences from the Online Encyclopedia of Integer Sequences (OEIS) using joeis and oeisdata."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Exact Multi-Axis Integer Ingestion & Representation (Priority: P1)

As a mathematical researcher or machine learning engineer, I want the system to ingest diverse OEIS integer sequence records and represent extreme numerical values along distinct orthogonal axes (magnitude, modular congruences, finite differences, and $p$-adic valuations) without precision loss or out-of-vocabulary truncation, so that sequence data can be processed by neural perception backbones with full numerical stability.

**Why this priority**: Without accurate, non-truncated representation of unbounded integer sequences, downstream model components cannot perceive arithmetic invariants or general recurrence patterns.

**Independent Test**: Ingest a benchmark suite of 1,000 sequence entries spanning dynamic ranges from $-10^6$ to $10^{30}$; verify that all terms are encoded into continuous representations without throwing numerical errors or generating undefined/NaN states.

**Acceptance Scenarios**:

1. **Given** an integer sequence record with large values ($> 10^{15}$), **When** the ingestion and multi-axis encoding pipeline processes the sequence, **Then** all values are decomposed into magnitude, residue spectrum, and difference features without out-of-vocabulary token drop or precision loss.
2. **Given** a sequence governed by modular periodicity (e.g., Fibonacci numbers modulo $m$), **When** passed through the representation pipeline, **Then** cyclic modular phase embeddings retain continuous periodic alignment across 100 base moduli ($m \in \{2, \dots, 101\}$).
3. **Given** negative, zero, and positive integers across varying steps, **When** computing finite differences and $p$-adic valuations for small primes ($p \le 13$), **Then** the local dynamic stream generates valid bounded ordinals and logarithmic step features.

---

### User Story 2 - Deterministic Sandboxed Program Execution & Fuel Bounding (Priority: P1)

As an evaluation engine or RL training pipeline, I want synthesized candidate program routines to be compiled in-memory and evaluated inside a deterministic, resource-bounded sandbox with strict instruction fuel limits and isolated linear memory, so that non-terminating loops, memory attacks, or runtime exceptions are trapped safely without impacting host process stability or throughput.

**Why this priority**: Autoregressively generated code frequently contains infinite loops or invalid operations. A deterministic sandbox with instruction-level fuel metering is essential to solve the halting problem and prevent system crashes during batch evaluation.

**Independent Test**: Submit a batch of candidate programs containing intentional infinite loops, divisions by zero, and valid sequence generators; verify that execution terminates within 10,000 fuel units in $<1\,\text{ms}$ per trapped candidate, returning structured execution status codes while valid programs return exact sequence outputs.

**Acceptance Scenarios**:

1. **Given** a candidate program that enters an infinite loop, **When** executed in the sandbox, **Then** the runtime halts execution precisely upon exhausting the 10,000 instruction fuel budget and returns an out-of-fuel status code without hanging or crashing the host process.
2. **Given** a valid sequence-generating program, **When** evaluated for sequence indices $n = 0 \dots N-1$, **Then** the sandbox returns the computed integer list and the exact count of consumed fuel units.
3. **Given** a candidate program attempting excessive memory allocation, **When** memory requests exceed the 16 MiB ceiling, **Then** the sandbox intercepts the allocation trap and returns an execution trap status code.

---

### User Story 3 - Syntactically and Semantically Sound Grammar-Guided Synthesis (Priority: P2)

As a neuro-symbolic synthesizer, I want autoregressive token generation to be constrained at every decoding step by dynamic grammar parsing and lexical scope tracking, so that 100% of generated program candidates adhere strictly to valid S-expression syntax and reference only declared variables and valid stack states.

**Why this priority**: Unconstrained language models waste significant sample capacity proposing malformed programs with syntax errors or uninitialized variables. Grammar guidance guarantees valid compilation and eliminates trivial execution failures.

**Independent Test**: Generate 1,000 candidate programs under grammar-constrained decoding; verify that 100% of generated outputs successfully compile in-memory without syntax errors or unbound variable reference traps.

**Acceptance Scenarios**:

1. **Given** an active decoding state at step $t$, **When** predicting the next token, **Then** the grammar engine produces a validation mask restricting the candidate vocabulary exclusively to syntactically valid continuations under sub-$100\,\mu\text{s}$ per-token latency.
2. **Given** a generated function declaring a set of local variables, **When** generating variable access instructions, **Then** environment-indexed grammar rules restrict emitted identifiers exclusively to declared, in-scope variable indices.
3. **Given** an incomplete or unbalanced S-expression, **When** reaching maximum generation length, **Then** the grammar constraint prevents premature termination until all opened parenthetical control blocks are structurally closed.

---

### User Story 4 - Taxonomy-Aligned Curriculum Progression & Generalization Verification (Priority: P2)

As a training scheduler, I want sequence learning tasks to be organized into a 5-stage progressive curriculum aligned with mathematical taxonomy (polynomials $\rightarrow$ linear recurrences $\rightarrow$ holonomic recurrences $\rightarrow$ combinatorial/number-theoretic algorithms $\rightarrow$ search/graph algorithms) with automated graduation gates and anti-memorization verification, so that the model learns general algorithmic rules rather than memorizing lookup tables.

**Why this priority**: Integer sequences present steep difficulty cliffs. Curriculum progression prevents exploration collapse, while extrapolation and complexity tests guarantee that synthesized algorithms generalize beyond training context terms.

**Independent Test**: Evaluate synthesized programs on training terms ($N=20$) and out-of-distribution extrapolation terms ($N+K$, $K=100$) alongside Minimum Description Length (MDL) ratio checks; verify that only candidates passing both checks qualify for curriculum graduation scoring.

**Acceptance Scenarios**:

1. **Given** a training cohort in Curriculum Stage $k$, **When** the rolling task competence score reaches $C(S_k) \ge 0.85$, coverage equilibrium $\min(\hat{\rho}_x) \ge 0.50$, and epoch variance stabilizes, **Then** the scheduler triggers automated graduation to Stage $k+1$.
2. **Given** a synthesized program matching the first 20 terms of a sequence, **When** evaluated on the subsequent 100 unseen terms, **Then** the generalization verifier confirms 100% exact match across all 100 extrapolated terms.
3. **Given** a candidate program using an over-fitted lookup table or Lagrange interpolation polynomial, **When** assessed against the sequence's Kolmogorov complexity proxy, **Then** the Minimum Description Length verifier flags $M_{\text{MDL}} > 1.2$ and rejects the candidate as memorized.

---

### User Story 5 - Localized Credit Assignment Reinforcement Learning (Priority: P3)

As a reinforcement learning engine, I want policy gradient optimization under exact binary reward ($+1/-1$) to trace execution paths and isolate the exact instruction token where sequence output deviates from the ground-truth sequence, so that gradient updates are concentrated on localized error windows rather than uniformly penalizing entire valid programs.

**Why this priority**: Sparse binary rewards and coarse sequence-level credit assignment lead to optimization collapse on hard curriculum prompts where most attempts fail. Localized credit assignment and asymmetric prompt weighting provide informative learning signals.

**Independent Test**: Execute policy optimization rollouts across batches where success rates are near zero; verify that non-zero gradient updates are assigned to execution divergence tokens and that policy entropy remains stable across training iterations.

**Acceptance Scenarios**:

1. **Given** a rollout group where all candidate completions produce incorrect sequence terms, **When** computing policy advantages, **Then** asymmetric prompt weighting applies negative updates to the specific execution divergence window rather than yielding zero gradients.
2. **Given** a generated program that computes correct terms for $n = 0 \dots 5$ but fails at $n = 6$, **When** execution-guided credit assignment traces the module, **Then** the loss function localizes parameter updates to the instruction tokens active during the $n=6$ state divergence.

---

### User Story 6 - Latent Space Geometry & Automated Mathematical Relation Discovery (Priority: P3)

As a mathematical discovery pipeline, I want sequence embeddings to be organized in a continuous latent space trained via non-contrastive self-supervision, clustered to detect uncatalogued sequence families, and queried via vector arithmetic coupled with high-precision integer relation detection and symbolic verification, so that latent mathematical relationships are formulated into machine-verified theorems.

**Why this priority**: Mapping discrete sequences to continuous geometric spaces enables unsupervised discovery of deep mathematical connections across distinct mathematical domains without class collision penalties.

**Independent Test**: Query latent space vector triples $(\vec{v}_A, \vec{v}_B, \vec{v}_C)$ satisfying linear vector closeness; verify that candidates pass through arbitrary-precision evaluation ($>500$ digits), PSLQ integer relation filtering, and symbolic theorem proving before acceptance.

**Acceptance Scenarios**:

1. **Given** sequence pairs linked by exact algebraic operators (partial sums, first differences, binomial transforms, shift operators), **When** optimizing latent embeddings under non-contrastive regularization (VICReg), **Then** representations maintain feature variance and decorrelation without representation collapse.
2. **Given** high-dimensional latent sequence embeddings, **When** analyzed via non-linear manifold reduction and density-based clustering, **Then** unannotated sequences are mapped into structural families while isolated mathematical primitives are flagged as anomalies.
3. **Given** candidate relation tuples identified via vector arithmetic ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$), **When** passed to high-precision numerical sampling and the PSLQ integer relation algorithm, **Then** true identities produce sharp confidence ratio drops ($< 10^{-50}$) and generate verified symbolic proofs.

---

### Edge Cases

- **Astronomical Values & Factorial Growth**: How does the system handle sequence terms exceeding standard 64-bit integer limits ($> 10^{30}$)? The multi-axis representation uses continuous signed logarithmic scaling and $p$-adic modular factorizations to preserve magnitude and divisibility properties without integer overflow errors.
- **Degenerate & Constant Sequences**: How does the system handle sequences with identical repeating elements (e.g., $1, 1, 1\dots$) or alternating signs? The difference stream generates zero-difference signals while the Fourier modular spectrum captures periodicity, preventing division-by-zero or undefined logarithm calculations.
- **Infinite Loops in Generated Code**: How does the system handle non-terminating backward jumps or recursive calls in synthesized programs? The execution sandbox enforces an immutable 10,000-fuel limit and returns an out-of-fuel termination code within $1\,\text{ms}$.
- **Uninitialized Memory & Stack Underflows**: How does the system handle programs with malformed stack operations? Dynamic environment-indexed grammars prevent emitting instructions that underflow stack depth or reference undeclared registers, and sandbox runtime traps isolate any execution anomalies.
- **Hardware Memory Saturation on Local Baseline**: How does the system maintain throughput on Tier 1 hardware (4 cores, 4 GB GPU VRAM)? The architecture enforces scaled model dimensions ($d=256/384$), strict GPU micro-batches of 4–8, and full delegation of WASM execution to 8 CPU threads.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest OEIS integer sequence records, metadata tags, and reference generation definitions into a structured local index.
- **FR-002**: System MUST encode integer sequence terms using a multi-axis architecture comprising a signed continuous magnitude stream, a 100-moduli Fourier spectrum stream ($m \in \{2, \dots, 101\}$), and a local difference and $p$-adic valuation stream ($p \le 13$).
- **FR-003**: System MUST unify the three representation streams using hierarchical two-stage feature-wise linear modulation (FiLM) where modular features modulate magnitude, and dynamic difference features modulate the combined state.
- **FR-004**: System MUST perform all neural encoder forward passes, backward passes, and intermediate state operations in strict FP32 precision to prevent phase underflow and gradient cancellation.
- **FR-005**: System MUST decode latent sequence representations into WebAssembly Text (WAT) S-expression programs autoregressively.
- **FR-006**: System MUST enforce dynamic grammar masking over byte-level tries during decoding with per-token masking evaluation latency not exceeding $100\,\mu\text{s}$.
- **FR-007**: System MUST track lexical variable declarations and stack depth during decoding via environment-indexed grammar rules, ensuring zero unbound identifier references and zero stack type violations (No-Ghost Soundness).
- **FR-008**: System MUST compile generated WAT programs into WebAssembly binaries in-memory without invoking disk I/O.
- **FR-009**: System MUST execute candidate WebAssembly programs within an isolated sandbox environment with a deterministic fuel budget capped at 10,000 instructions per execution.
- **FR-010**: System MUST enforce a linear memory allocation ceiling of 16 MiB per sandbox execution instance.
- **FR-011**: System MUST execute batch program evaluations across host CPU cores in parallel, fully released from the host interpreter's Global Interpreter Lock (GIL).
- **FR-012**: System MUST schedule learning tasks across a 5-stage taxonomy-aligned curriculum (Stage 1: Polynomials/Primitives $\rightarrow$ Stage 2: Linear Recurrences $\rightarrow$ Stage 3: Holonomic Sequences $\rightarrow$ Stage 4: Combinatorics/Number Theory $\rightarrow$ Stage 5: Search/Graph Invariants).
- **FR-013**: System MUST trigger curriculum stage graduation only when the rolling task competence score satisfies $C(S_k) \ge 0.85$, coverage equilibrium satisfies $\min(\hat{\rho}_x) \ge 0.50$, and consecutive epoch variance is bounded.
- **FR-014**: System MUST verify synthesized programs against anti-memorization criteria using Extrapolation Horizon Testing across 100 unseen terms ($N+K$ evaluation) and Minimum Description Length ratio bounds ($M_{\text{MDL}} \le 1.2$).
- **FR-015**: System MUST optimize code-generating policies using Execution-Guided Credit Assignment (EGCA) with asymmetric prompt weighting under strict binary outcome rewards ($+1/-1$).
- **FR-016**: System MUST trace execution state trajectories to localize gradient updates to the specific instruction tokens where sequence generation deviates from ground truth.
- **FR-017**: System MUST train continuous latent sequence representations using non-contrastive self-supervised regularization (VICReg: Variance, Invariance, Covariance) over algebraic transformation pairs (partial sums, first differences, binomial transforms, shift operators).
- **FR-018**: System MUST perform high-dimensional manifold reduction and hierarchical density clustering on latent sequence representations to discover unannotated sequence families and flag anomalies.
- **FR-019**: System MUST verify candidate latent mathematical relations ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$) through arbitrary-precision numerical sampling ($>500$ digits) and PSLQ integer relation detection.
- **FR-020**: System MUST pass verified integer relations to symbolic computer algebra systems to generate formal mathematical proofs.
- **FR-021**: System MUST operate within Tier 1 local workstation hardware constraints (4 CPU cores / 8 threads, 64 GB RAM, 4 GB GPU VRAM) using scaled hidden dimensions ($d=256$ or $d=384$) during baseline prototyping before scaling to multi-GPU clusters.

### Key Entities *(include if feature involves data)*

- **Sequence Record**: Represents an OEIS sequence entry, containing unique sequence identifier (A-number), initial integer terms, descriptive name, mathematical keywords/tags, and reference generation specifications.
- **Multi-Axis Sequence Embedding**: Continuous vector representation combining signed log-magnitude scalar projections, 100-moduli Fourier phase vectors, finite difference logs, and $p$-adic valuation ordinals.
- **Synthesized Program Candidate**: Abstract syntax and source text (WAT S-expression) representing a sequence generating algorithm, including parameter definitions, local variable allocations, loop constructs, and arithmetic instructions.
- **Execution Trace & Result**: Outcome of sandboxed program execution, containing execution status code (`SUCCESS`, `OUT_OF_FUEL`, `PARSE_ERROR`, `EXECUTION_TRAP`), consumed instruction fuel count, generated integer terms, and step-by-step state divergence markers.
- **Curriculum State**: Tracking record for curriculum progression, maintaining active stage identifier ($S_1 \dots S_5$), rolling prompt pass-rates ($\hat{\rho}_x$), aggregate stage competence score $C(S_k)$, and prompt sampling mixture ratios.
- **Latent Discovery Relation**: Conjectured algebraic relationship between sequence representations in latent space, associated numerical verification vectors, PSLQ integer relation certificates, and symbolic proof scripts.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The multi-axis sequence encoder processes 1,000 benchmark OEIS sequences spanning values from $-10^6$ to $10^{30}$ in FP32 precision with 0 NaN, infinite, or numerical underflow errors.
- **SC-002**: 100% of candidate WAT programs synthesized under dynamic grammar masking assemble into valid WebAssembly binaries without syntax or lexical scoping errors.
- **SC-003**: Sandboxed execution terminates 100% of non-terminating infinite loops within 10,000 instruction fuel units in $<1\,\text{ms}$ per candidate without crashing or leaking memory.
- **SC-004**: Multi-threaded execution bridge achieves a sustained batch evaluation throughput exceeding 500 WASM module evaluations per second across 8 CPU threads on Tier 1 hardware.
- **SC-005**: The synthesis engine achieves $\ge 80\%$ program synthesis pass rate on Curriculum Stage 1 (primitives and polynomials) on the Tier 1 workstation baseline.
- **SC-006**: 100% of graduated candidate programs achieve zero-error extrapolation across 100 unseen future sequence terms ($N=20, K=100$) and maintain a Minimum Description Length ratio $M_{\text{MDL}} \le 1.2$.
- **SC-007**: Grammar masking engine maintains an average per-token masking calculation latency under $100\,\mu\text{s}$ during continuous autoregressive generation.
- **SC-008**: Mathematical discovery pipeline verifies candidate integer relations with PSLQ confidence drops exceeding $10^{-50}$ and produces valid symbolic proofs for verified identities.

## Assumptions

- **Local Workstation Hardware**: Prototyping and Tier 1 MVP validation are hosted on a machine with 4 CPU cores / 8 threads, 64 GB DDR4 RAM, and an NVIDIA GPU with 4 GB VRAM (e.g., Intel Xeon E3-1505M v5 + Quadro M2000M).
- **Data Availability**: Sequence definitions, metadata tags, and reference implementations are accessible locally from ingested `oeisdata` and `joeis` datasets.
- **Deterministic Execution**: WebAssembly execution without host-system imports provides complete determinism and isolation for mathematical integer computations.
- **Strict FP32 Arithmetic**: Hardware GPU execution natively supports continuous single-precision floating-point operations with sufficient throughput for model training.
- **Two-Tier Scaling Path**: Multi-GPU cluster scale-up ($d=768$, complete 390,000+ sequence catalog) is deferred until local Tier 1 validation satisfies Stage 1 & Stage 2 graduation gates.
