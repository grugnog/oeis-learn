# Feature Specification: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine

**Feature Branch**: `004-decoupled-grounding-and-symple-engine`  
**Created**: 2026-09-02  
**Status**: Draft  
**Input**: User description: "Phase 4 Specification: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine to resolve constant grounding gaps, AST dead-code bloat, task dilution, and latent space scaling issues."  
**Prerequisites**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md), [specs/002-synthesis-bootstrapping-and-soundness/spec.md](specs/002-synthesis-bootstrapping-and-soundness/spec.md), [specs/003-algorithmic-generalization-and-credit-assignment/spec.md](specs/003-algorithmic-generalization-and-credit-assignment/spec.md)

---

## 1. Executive Summary & Historical Context

`OEIS-Learn` is an autonomous neuro-symbolic program synthesis engine that translates few-shot integer sequences $Y = [y_0, y_1, \dots, y_{19}]$ from the On-Line Encyclopedia of Integer Sequences (OEIS) into compact, verifiable, and extrapolatable WebAssembly Text (WAT) programs $P$ such that $\forall n \in \mathbb{N}_0, P(n) = y_n$.

### 1.1 Optimization Trajectory & Run 005 Diagnostics
Over five iterations, the core execution stability, compilation soundness, and basic synthesis foundations have been established:
- **Run 001 (Cold-Start)**: Unconstrained autoregressive generation resulted in 99.2% compiler syntax traps and zero policy gradient flow.
- **Run 002 (Grammar Masking)**: Dynamic grammar masking eliminated compiler traps (0% syntax errors), but optimization succumbed to static constant shortcuts ($P(n) = c$).
- **Run 003 (SFT Demonstration Warmup & CGI)**: Trajectory injection and SFT warmup broke constant shortcuts, but suffered from Advantage Collapse Rate ($\text{ACR} = 1.0$) due to SFT prompt mismatch.
- **Run 004 / 005 (Production 60-Epoch Run: 9.96 Hours, 24,000 Steps, 96,000 Programs)**: Fixed SFT conditioning, bounded stack curvature, smooth loss convergence ($2.15 \to 0.1195$), zero compiler panics, bounded $\text{ACR} \le 0.05$, and solved modular periodic and real OEIS sequences (A000012, A000027).

### 1.2 The Four Phase 4 Bottlenecks
Despite structural stability, Run 005 revealed four fundamental root causes stalling Stage 1 graduation ($C(S_1) \ge 0.85$):
1. **Constant Grounding Gap**: Softmax cross-attention partitions hidden space into Voronoi cells; linear attention cannot compute finite-difference slopes $\frac{\Delta y}{\Delta n}$ across continuous manifolds. The model synthesizes correct topological ASTs (e.g. `n * C_1 + C_2`) but defaults slope constants to unity ($C_1 = 1$).
2. **AST Padding & Dead-Code Bloat**: The grammar masker strictly demands a balanced stack at function return ($\Sigma_t = 1$); when the policy is uncertain of mathematical logic, it emits balanced no-op dead code (`local.get ... drop`) to preserve stack balance and collect $+0.1$ validity rewards without risking compiler traps.
3. **Task Dilution & Replay Gap**: Uniform sampling across 524 sequences in the active pool visits each prompt only once per 65 gradient steps. Parameter drift erases fragile discoveries, while shallow rollout groups ($G=4$) yield $<4\%$ hit probabilities on difficult frontier tasks.
4. **Latent Space Euclidean Scale Mismatch**: Raw sequence embeddings with $L_2$ norm $\|z\| \approx 10.0$ were queried with fixed Euclidean radius $\varepsilon = 0.8$, yielding 0 discovered PSLQ relations despite structured representations.

This specification defines **Phase 4: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine** to decouple structural synthesis from parameter solving, eliminate syntactic padding, optimize task exploration, and restore theorem discovery.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 — Decoupled Symbolic-Numeric Grounding & Diophantine/SMT Solvers (Priority: P1) 🎯 MVP

As a neuro-symbolic synthesizer, I want the autoregressive decoder to emit abstract program skeletons with generic constant placeholders (`i64.const_?`) and dispatch numerical coefficient binding to deterministic Diophantine and SMT solvers, so that the policy focuses entirely on discovering computational AST topology while exact constants are resolved instantaneously from execution traces.

**Why this priority**: Resolves the primary bottleneck where models discover correct algorithmic structures (multiplications, polynomial terms, offsets) but fail verification due to categorical literal mispredictions.

**Independent Test**: Pass a batch of synthesized program skeletons with 1 to 4 placeholders across linear, affine, quadratic, and modular sequence targets; verify that the solver pipeline computes exact integer constants in $<2\,\text{ms}$ for linear traces and $<250\,\text{ms}$ for non-linear traces, producing 100% ground-truth matching programs.

**Acceptance Scenarios**:
1. **Given** a generated program skeleton containing linear constant placeholders (e.g., $P_{\mathbf{C}}(n) = c_0 + c_1 f_1(n) + \dots + c_k f_k(n)$), **When** evaluated against target terms $Y$, **Then** the exact Diophantine solver constructs system matrix $A \in \mathbb{Z}^{20 \times (k+1)}$ via sandbox basis execution and solves $A \mathbf{C} = Y$ via Hermite Normal Form (HNF) row reduction in $<1\,\text{ms}$.
2. **Given** a generated program skeleton containing placeholders inside non-linear operations (`i64.rem_u`, `i64.shl`, `br_if`), **When** the linear solver fails or detects non-linearity, **Then** the system lowers the skeleton to a QF_BV SMT-LIB2 formula and resolves constants via Z3 within a $250\,\text{ms}$ timeout.
3. **Given** solved concrete integer constants, **When** the solver completes, **Then** the constants are spliced directly into the WAT code, assembled into an executable binary, and awarded positive execution reward ($R_{\text{exec}} = 1.0$), while the on-policy policy gradient $\mathcal{L}_{\text{GRPO}}$ is backpropagated through the emitted placeholder token sequence and the concrete grounded program is ingested into the Elite Demonstration Buffer.
4. **Given** a program skeleton with no mathematically viable integer constant solution, **When** solvers return unsatisfiable/no-solution, **Then** the skeleton is marked unsolvable without crashing the training worker.

---

### User Story 2 — Anti-Padding Parsimony Regularization & Compiler-in-the-Loop RLVR (Priority: P1) 🎯 MVP

As a reinforcement learning optimizer, I want all generated candidate programs to undergo online compiler dead-code elimination (DCE), continuous log-distance reward evaluation, Covariant Parsimony Pressure (CPP), and lexicographic group ranking, so that the policy cannot exploit grammar constraints via dead-code padding and is strictly incentivized to synthesize minimal, compact algorithms.

**Why this priority**: Eliminates the dead-code attractor modes where policies emit redundant `local.get ... drop` loops to capture $+0.1$ validity rewards, preventing policy entropy collapse on meaningful mathematical primitives.

**Independent Test**: Generate a batch of padded programs containing redundant stack pairs and dead variable writes; verify that the optimizing compiler pass strips all dead operations in $<1.5\,\text{ms}$, computes exact syntactic waste ratio $\rho_{\text{waste}}$, and assigns negative ordinal advantages to bloated candidates relative to compact equivalents.

**Acceptance Scenarios**:
1. **Given** a candidate WebAssembly program, **When** verified in the sandbox, **Then** the runtime passes bytecode through `wasm-opt -O3 --vacuum --dce --remove-unused-locals`, returning optimized binary $B_{\text{opt}}$, disassembled canonical text $P_{\text{opt}}$, and syntactic waste ratio $\rho_{\text{waste}}(P) = \frac{|P| - |P_{\text{opt}}|}{|P|}$.
2. **Given** rollout completions across a prompt group, **When** computing partial rewards, **Then** the reward engine evaluates dense continuous log-distance $R_{\text{dense}}(P, Y) = \frac{1}{20} \sum_{n=0}^{19} \frac{1}{1 + \log_{10}(|P(n) - y_n| + 1)}$ combined with group Covariant Parsimony Pressure $R_{\text{CPP}} = R_{\text{dense}} - \max(0, -c_k)(\ell(P) - \ell_{\min}) - \lambda_{\text{waste}} \rho_{\text{waste}}(P)$, and zeros out all validity rewards if $\rho_{\text{waste}}(P) > 0.30$.
3. **Given** group rollouts $\{P_1, \dots, P_G\}$, **When** evaluating advantages, **Then** rollouts are sorted lexicographically ($R_{\text{exec}} \succ -|P_{\text{opt}}|$) and mapped to normalized ordinal advantages $\hat{A}_i^{\text{lex}} \in [-1, 1]$.
4. **Given** autoregressive token generation, **When** computing policy entropy and sampling temperatures, **Then** the model applies Partitioned Semantic Entropy (positive on arithmetic tokens $\mathcal{A}_{\text{sem}}$, penalized on structural tokens $\mathcal{A}_{\text{struct}}$) and scales temperature by stack depth $T(s_t) = T_{\text{base}} \cdot (1 - 0.6 \frac{\Sigma_t}{\Sigma_{\max}})$.

---

### User Story 3 — SYMPLE Bandit Curriculum & Elite Demonstration Replay (Priority: P2)

As an automated curriculum manager and training coordinator, I want task scheduling to use an EXP3.S non-stationary bandit focusing on the Zone of Proximal Development (ZPD), dynamic group sizing (Ada-G), and an Elite Demonstration Buffer (EDB) with vulnerability-weighted SFT replay, so that exploration compute is concentrated on frontier tasks while mastered sequences are preserved against catastrophic parameter drift.

**Why this priority**: Eliminates the 65-step task dilution gap and shallow rollout starvation ($G=4$) that prevented Run 005 from graduating Stage 1.

**Independent Test**: Simulate 100 training steps over a 524-sequence pool; verify that EXP3.S concentrates $\ge 70\%$ of active sampling on frontier tasks ($0.05 \le \hat{p}_i \le 0.50$), Ada-G allocates deep groups ($G_i \in [8, 16]$) ensuring $P(\text{Hit} \ge 1) \ge 0.50$, and EDB replays dormant sequences to maintain $100\%$ retention on previously solved tasks.

**Acceptance Scenarios**:
1. **Given** the full pool of $K=524$ sequences, **When** sampling active training prompts, **Then** the EXP3.S bandit updates task probabilities using binomial dispersion feedback $r_{i,t} = \hat{p}_i(1 - \hat{p}_i) + |\Delta C_i| + 2.0 \max(0, -\Delta C_i)$ and selects $B_{\text{active}} = 2$ frontier prompts per step.
2. **Given** selected frontier prompts with estimated competence $\hat{p}_i$, **When** allocating rollout compute, **Then** Ada-G dynamically computes group sizes $G_i = \text{clip}(\lceil \frac{\ln(0.5)}{\ln(1 - \max(\hat{p}_i, 0.02))} \rceil, 8, 16)$ under a fixed active rollout budget ($M_{\text{active}} = 24\text{--}32$).
3. **Given** an active sequence with historical solutions in $\mathcal{D}_{\text{elite}}$ where all $G_i$ rollouts fail ($k_i = 0$), **When** computing group statistics, **Then** Virtual Sample Injection introduces a synthetic positive return ($r=1.0$) to restore contrastive negative advantages ($\hat{A}^- = -1/\sqrt{G}$) and prevent zero-gradient collapse.
4. **Given** any candidate rollout achieving functional verification ($R_{\text{exec}} = 1.0$), **When** verified, **Then** its canonical AST is ingested into the Elite Demonstration Buffer $\mathcal{D}_{\text{elite}}$, maintaining the top-4 shortest solutions per sequence.
5. **Given** each training optimization step, **When** computing parameter updates, **Then** the system samples $B_{\text{replay}} = 2$ dormant sequences from $\mathcal{D}_{\text{elite}}$ prioritized by elapsed dormancy $\Delta t_{\text{dormant}} = t_{\text{current}} - t_{\text{last\_visit}}$ and co-optimizes joint loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}}(\mathcal{D}_{\text{active}}) + 0.50 \mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{replay}}) + 0.10 \mathcal{L}_{\text{aux}}(\mathbf{z}) - 0.02 \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$.

---

### User Story 4 — Tri-Stream Encoder v2 & Linear Invariant Representation (Priority: P2)

As a neural perception backbone, I want the Tri-Stream Encoder to compute normalized Newton forward difference quotients ($D^{(k)} = \Delta^k y / k!$), orthogonal Prime Fourier Embeddings (PFE) across 16 odd prime fields, and prepend global summary tokens ($\mathbf{z}_{\text{affine}}, \mathbf{z}_{\text{geom}}$) using direct concatenation and self-attention, so that sequence slopes, polynomial curvatures, and periodicities are linearly accessible to cross-attention queries without non-linear FiLM distortion.

**Why this priority**: Direct linear representation of sequence derivatives and modular invariants allows cross-attention heads to isolate polynomial slopes and geometric multipliers in a single query-key dot product.

**Independent Test**: Pass polynomial, geometric, and periodic sequences through the updated encoder; verify that intermediate representation vectors linearly correlate ($R^2 > 0.99$) with true polynomial degrees and slopes, and summary token auxiliary heads predict linear slopes $\hat{m}$ with $<1\%$ relative error.

**Acceptance Scenarios**:
1. **Given** an input sequence $Y$, **When** computing difference features ($S_3$), **Then** the encoder evaluates normalized Newton quotients $D^{(1)}_i = y_{i+1} - y_i$, $D^{(2)}_i = \frac{y_{i+2} - 2y_{i+1} + y_i}{2}$, and $D^{(3)}_i = \frac{\Delta^3 y_i}{6}$ in strict FP32 precision.
2. **Given** sequence modular harmonics ($S_2$), **When** embedding modular features, **Then** the encoder projects terms onto 16 orthogonal prime fields ($p \in \{3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59\}$) via sine/cosine pairs.
3. **Given** the continuous encoder sequence, **When** constructing the input matrix, **Then** the encoder prepends learnable summary tokens $\mathbf{z}_{\text{affine}}$ and $\mathbf{z}_{\text{geom}}$ supervised via auxiliary regression targets ($\frac{\text{Cov}(n, Y)}{\text{Var}(n)}$ and $\text{median}(y_{i+1}/y_i)$).
4. **Given** the multi-stream representations, **When** unifying features, **Then** the encoder combines $S_1, S_2, S_3$ via direct linear concatenation followed by bidirectional self-attention, replacing non-linear FiLM modulation.
5. **Given** synthetic demonstration dataset generation, **When** sampling forward training examples, **Then** the procedural generator applies randomized affine scaling ($\tilde{Y} = \alpha Y + \beta$, $\alpha \sim \pm 10^{\mathcal{U}(0, 5)}$) across AST skeletons to prevent identity idiom memorization ($C=1$).

---

### User Story 5 — Normalized Latent Manifold & PSLQ Theorem Discovery (Priority: P3)

As a mathematical discovery pipeline, I want sequence latent embeddings $z_i \in \mathbb{R}^{256}$ to be $L_2$-normalized prior to nearest-neighbor search and PSLQ relation query execution, so that vector arithmetic and integer relation detection operate on scale-invariant directional manifolds and reliably uncover machine-verified mathematical theorems.

**Why this priority**: Fixes the Euclidean scale mismatch in the benchmark pipeline where unnormalized vectors ($\|z\| \approx 10.0$) yielded 0 relation discoveries when queried with fixed search radii.

**Independent Test**: Run the automated discovery pipeline on 524 normalized sequence representations; verify that `VectorRelationSearcher` identifies candidate triples with cosine proximity and that PSLQ integer relation searches recover $\ge 1$ formally verified theorems proved by SymPy.

**Acceptance Scenarios**:
1. **Given** extracted continuous sequence embeddings $Z \in \mathbb{R}^{K \times 256}$, **When** constructing nearest-neighbor indices, **Then** all embedding vectors are $L_2$-normalized: $\hat{z}_i = \frac{z_i}{\|z_i\|_2 + 10^{-8}}$.
2. **Given** normalized representations $\hat{z}$, **When** querying vector triples $(\hat{z}_A + \hat{z}_B \approx \hat{z}_C)$ with $\varepsilon_{\text{dist}} = 0.8$, **Then** the searcher identifies candidate algebraic relations without distance-scale distortion.
3. **Given** candidate relation tuples, **When** evaluated via high-precision ($>500$ digits) `mpmath` and PSLQ, **Then** valid relations produce sharp confidence drops ($<10^{-50}$) and generate verified SymPy symbolic proofs.

---

### Edge Cases

- **Unsolvable AST Skeletons**: When a generated program skeleton is structurally incapable of matching target sequence $Y$ under any integer parameter assignment (e.g. non-divisible modulo constraints or inconsistent linear equations), the linear and SMT solvers return `None` within their timeout, and the candidate receives negative baseline reward without interrupting batch execution.
- **Underdetermined Linear Systems ($k > \text{rank}(A)$)**: When multiple integer solutions exist for a linear skeleton, the Diophantine solver chooses the minimal $L_1$-norm solution $\mathbf{C}^* = \arg\min \|\mathbf{C}\|_1$ to promote parsimonious constants.
- **SMT Solver Timeout / Non-Linear Deadlock**: When Z3 encounters complex non-linear bitvector operations exceeding $250\,\text{ms}$, the execution sandbox halts the solver thread, logs a timeout status, and falls back to ungrounded execution.
- **Empty Elite Demonstration Buffer at Cold Start**: When training begins and $\mathcal{D}_{\text{elite}}$ contains no verified programs for dormant tasks, vulnerability replay gracefully falls back to synthetic forward SFT demonstrations until self-discovered programs populate the buffer.
- **Degenerate Single-Token Modules in Compiler Pass**: If a candidate program consists of empty or malformed tokens, `wasm-opt` handles syntax failures safely, returning a compilation trap status without leaking memory or crashing the host worker.

---

## 3. Requirements *(mandatory)*

### Functional Requirements

#### Decoupled Symbolic-Numeric Grounding & Solvers
- **FR-001**: System MUST support the generic constant placeholder token `i64.const_?` in the decoder vocabulary and dynamic `GrammarMasker`.
- **FR-002**: System MUST parse candidate WebAssembly Text programs to extract AST skeletons containing $k \in [1, 4]$ constant placeholders and classify placeholder linearity in the execution trace.
- **FR-003**: For linear/affine placeholders ($P_{\mathbf{C}}(n) = c_0 + \sum_{j=1}^k c_j f_j(n)$), system MUST evaluate basis functions in the native sandbox for $n \in [0, 19]$ and solve $A \mathbf{C} = Y$ via exact Hermite Normal Form (HNF) integer row reduction in $<1\,\text{ms}$.
- **FR-004**: For non-linear placeholders (e.g., inside `i64.rem_u`, `i64.shl`, `br_if`), system MUST lower the program skeleton to an SMT-LIB2 formula under Quantifier-Free BitVectors (`QF_BV`) and solve for constants via Z3 with a strict $250\,\text{ms}$ timeout.
- **FR-005**: System MUST splice concrete integer solutions $\mathbf{C}^*$ back into the WAT skeleton, assemble the executable binary, and execute the grounded program in the sandbox.
- **FR-006**: System MUST decouple gradient attribution from demonstration storage: the on-policy GRPO surrogate loss $\mathcal{L}_{\text{GRPO}}$ MUST be evaluated and backpropagated through the emitted AST skeleton token sequence (containing `i64.const_?` placeholders) using the grounded program's execution reward ($R_{\text{exec}} = 1.0$), while the grounded program (with concrete integer constants spliced in) MUST be ingested into the Elite Demonstration Buffer ($\mathcal{D}_{\text{elite}}$) to supervise off-policy SFT replay $\mathcal{L}_{\text{SFT}}$.

#### Anti-Padding Regularization & Compiler-in-the-Loop RLVR
- **FR-007**: System MUST pass all compiled WebAssembly binaries through an optimizing compiler pass (`wasm-opt -O3 --vacuum --dce --remove-unused-locals`), returning canonical optimized binary $B_{\text{opt}}$ and disassembled text $P_{\text{opt}}$.
- **FR-008**: System MUST compute the Syntactic Waste Ratio $\rho_{\text{waste}}(P) = \frac{|P|_{\text{tokens}} - |P_{\text{opt}}|_{\text{tokens}}}{|P|_{\text{tokens}}}$ and enforce a hard waste cutoff threshold $\tau_{\text{thresh}} = 0.30$:
  $$R_{\text{validity}}(P) = \begin{cases} 0.1 \cdot \exp(-\kappa \cdot \rho_{\text{waste}}(P)) & \text{if } \rho_{\text{waste}}(P) \le 0.30 \\ 0.0 & \text{if } \rho_{\text{waste}}(P) > 0.30 \end{cases}$$
  and penalize waste in CPP via $\lambda_{\text{waste}} \cdot \rho_{\text{waste}}(P)$ (with default $\lambda_{\text{waste}} = 0.20, \kappa = 2.0$). If $\rho_{\text{waste}}(P) > 0.30$, all validity rewards MUST be zeroed out.
- **FR-009**: System MUST evaluate the continuous dense log-distance return: $R_{\text{dense}}(P, Y) = \frac{1}{20} \sum_{n=0}^{19} \frac{1}{1 + \log_{10}(|P(n) - y_n| + 1)}$.
- **FR-010**: System MUST compute Covariant Parsimony Pressure (CPP) across group rollouts: $c_k = \frac{\operatorname{Cov}_{i \in [1, G]}(\ell(P_i), R_{\text{dense}}(P_i))}{\operatorname{Var}_{i \in [1, G]}(\ell(P_i)) + \epsilon}$, penalizing length when covariance is negative: $R_{\text{CPP}} = R_{\text{dense}} - \max(0, -c_k)(\ell(P) - \ell_{\min}) - \lambda_{\text{waste}} \rho_{\text{waste}}(P)$.
- **FR-011**: System MUST compute ordinal group advantages using Lexicographical Ranking ($R_{\text{exec}} \succ -|P_{\text{opt}}|$), mapping candidate ranks to normalized advantages $\hat{A}_i^{\text{lex}} = \frac{2(\operatorname{rank}(P_i) - 1)}{G - 1} - 1 \in [-1, 1]$.
- **FR-012**: System MUST apply Partitioned Semantic Entropy regularization: $\mathcal{L}_{\text{ent}}(\theta) = \alpha_{\text{sem}} \frac{\mathcal{H}_{\text{sem}}(\pi \mid s_t)}{\log |\mathcal{A}_{\text{sem}}(s_t)| + \epsilon} - \beta_{\text{pen}} \max(0, P(\mathcal{A}_{\text{struct}} \mid s_t) - 0.15)$ (with default $\alpha_{\text{sem}} = 0.02, \beta_{\text{pen}} = 0.05$).
- **FR-013**: System MUST dynamically scale sampling temperature by stack height: $T(s_t) = T_{\text{base}} \cdot (1 - 0.6 \frac{\Sigma_t}{\Sigma_{\max}})$.

#### Tri-Stream Encoder v2 & Linear Invariant Representation
- **FR-014**: System MUST compute normalized Newton forward difference quotients in stream $S_3$: $D^{(1)}_i = y_{i+1} - y_i$, $D^{(2)}_i = \frac{y_{i+2} - 2y_{i+1} + y_i}{2}$, $D^{(3)}_i = \frac{\Delta^3 y_i}{6}$.
- **FR-015**: System MUST compute orthogonal Prime Fourier Embeddings (PFE) in stream $S_2$ across the first 16 odd prime fields ($p \in \{3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59\}$).
- **FR-016**: System MUST prepend two learnable global summary tokens ($\mathbf{z}_{\text{affine}}, \mathbf{z}_{\text{geom}}$) to encoder input representations, supervised via auxiliary MSE regression heads against $\frac{\text{Cov}(n, Y)}{\text{Var}(n)}$ and $\text{median}(y_{i+1}/y_i)$.
- **FR-017**: System MUST unify representation streams via direct linear concatenation and bidirectional self-attention in strict FP32 precision.
- **FR-017b**: System MUST apply randomized affine scaling ($\tilde{Y} = \alpha Y + \beta$, with $\alpha \sim \pm 10^{\mathcal{U}(0, 5)}, \beta \sim \mathcal{U}(-10^5, 10^5)$) to generated AST skeletons during forward synthetic SFT dataset generation, breaking structural idiom memorization and forcing cross-attention queries to dynamically bind slope and scale features from the continuous difference streams.

#### SYMPLE Bandit Curriculum & Anti-Forgetting Replay
- **FR-018**: System MUST implement an EXP3.S non-stationary bandit task scheduler tracking arm selection probabilities $\mathbf{p}_t$ over all $K=524$ sequences using learning progress feedback $r_{i,t} = \hat{p}_i(1 - \hat{p}_i) + |\Delta C_i| + 2.0 \max(0, -\Delta C_i)$ with exploration $\gamma_{\text{exp3}} = 0.15$ and switching rate $\alpha_{\text{exp3}} = 0.05$.
- **FR-019**: System MUST select $B_{\text{active}} = 2$ frontier prompts per training step from the EXP3.S distribution.
- **FR-020**: System MUST allocate dynamic rollout group sizes $G_i \in [8, 16]$ per active prompt via Ada-G: $G_i = \text{clip}(\lceil \frac{\ln(1 - 0.50)}{\ln(1 - \max(\hat{p}_i, 0.02))} \rceil, 8, 16)$ under a fixed active rollout budget ($M_{\text{active}} = 24\text{--}32$).
- **FR-020b**: System MUST implement Virtual Sample Injection: if all $G_i$ on-policy exploratory rollouts fail ($k_i = 0$) for an active sequence that has an existing verified program stored in its EDB bucket $\mathcal{B}_i$, the system MUST inject a synthetic positive return ($r = 1.0$) into the group statistics to generate non-zero contrastive advantages ($\hat{A}^- = -1/\sqrt{G}$) on the failed rollouts, preventing zero-gradient advantage collapse.
- **FR-021**: System MUST maintain an Elite Demonstration Buffer (EDB) storing up to the top-4 shortest canonical AST programs per sequence, deduplicated by Minimum Description Length.
- **FR-022**: System MUST sample $B_{\text{replay}} = 2$ dormant sequences per step from the EDB based on elapsed dormancy $\Delta t_{\text{dormant}} = t_{\text{current}} - t_{\text{last\_visit}}$.
- **FR-023**: System MUST optimize the joint training loss:
  $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta; \mathcal{D}_{\text{active}}) + 0.50 \mathcal{L}_{\text{SFT}}(\theta; \mathcal{D}_{\text{replay}}) + 0.10 \mathcal{L}_{\text{aux}}(\mathbf{z}) - 0.02 \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$
  where $\mathcal{L}_{\text{aux}}(\mathbf{z})$ evaluates Mean Squared Error (MSE) loss for summary token regression heads predicting sequence slope $\frac{\text{Cov}(n, Y)}{\text{Var}(n)}$ and geometric ratio $\text{median}(y_{i+1}/y_i)$.

#### Latent Manifold Normalization & PSLQ Discovery
- **FR-024**: System MUST perform $L_2$ normalization on all sequence representation vectors: $\hat{z}_i = \frac{z_i}{\|z_i\|_2 + 10^{-8}}$ prior to nearest-neighbor indexing and relation search.
- **FR-025**: System MUST execute `VectorRelationSearcher` over normalized embeddings $\hat{z}$ with distance threshold $\varepsilon_{\text{dist}} = 0.8$.
- **FR-026**: System MUST verify discovered relation candidates via arbitrary-precision ($>500$ digits) PSLQ searches ($<10^{-50}$ confidence drop) and generate machine-verified SymPy symbolic proofs.

#### Hardware & Execution Bounds
- **FR-027**: All neural forward and backward operations MUST execute in strict FP32 precision within $\le 3.5\,\text{GB}$ GPU VRAM on Tier 1 hardware (4 CPU cores / 8 threads, 4GB GPU VRAM).
- **FR-028**: 100% of WebAssembly module compilation, DCE optimization, fuel metering (10,000 instruction budget), and batch execution MUST be offloaded to 8 CPU threads via the native Rust `oeis_wasm_evaluator` extension.

---

### Key Entities

- **ASTSkeleton**: Represents an ungrounded WebAssembly program structure containing one or more placeholder tokens (`i64.const_?`), including placeholder locations, linearity classifications, and parameter dependencies.
- **ConstantSolverResult**: Captures the output of Diophantine or SMT constant solving, containing solver type (`DIOPHANTINE_HNF`, `Z3_SMT`), concrete integer solution vector $\mathbf{C}^*$, solve duration in milliseconds, and verification status.
- **CanonicalProgramArtifact**: Stores the output of the optimizing compiler pass, including raw WAT text, optimized binary $B_{\text{opt}}$, disassembled canonical text $P_{\text{opt}}$, instruction token counts, and Syntactic Waste Ratio $\rho_{\text{waste}}$.
- **SYMPLETaskState**: Tracks the curriculum state for a sequence in the 524-benchmark pool, including trailing pass history $W_i$ (window size 20), estimated competence $\hat{p}_i$, competence slope $\Delta C_i$, EXP3.S bandit weight $w_i$, and last visitation timestamp $t_{\text{last\_visit}}$.
- **EliteDemonstrationEntry**: Encapsulates a verified canonical program in the EDB, including sequence ID, canonical WAT code, token length, execution fuel, AST structural hash, and creation timestamp.
- **NormalizedLatentRecord**: Stores $L_2$-normalized continuous embeddings $\hat{z}_i \in \mathbb{R}^{256}$, nearest-neighbor cosine indices, detected relation triples, and PSLQ verification certificates.

---

## 4. Success Criteria *(mandatory)*

### Measurable Outcomes

| Metric ID | Target Metric | Required Threshold | Verification Method |
| :--- | :--- | :--- | :--- |
| **SC-001** | **Linear Solver Latency & Accuracy** | **$< 2.0\,\text{ms}$** solve time with **$100.0\%$** exact recovery | Unit test suite evaluating linear Diophantine recovery on polynomial and affine test suites. |
| **SC-002** | **SMT Fallback Latency & Accuracy** | **$< 250.0\,\text{ms}$** solve time on non-linear skeletons | Contract test suite evaluating Z3 QF_BV constant resolution on modulo and bitwise AST skeletons. |
| **SC-003** | **Syntactic Waste Ratio** | **$\rho_{\text{waste}} < 0.05$** on graduated programs | Compiler canonicalization analysis on synthesized programs across Curriculum Stage 1. |
| **SC-004** | **Advantage Collapse Rate (ACR)** | **$\text{ACR} \le 0.05$** throughout 60 epochs | Real-time telemetry tracking fraction of zero-variance rollout groups during Run 006. |
| **SC-005** | **Curriculum Stage 1 Graduation** | **$C(S_1) \ge 0.85$** rolling competence | Rolling evaluation score across Stage 1 benchmark sequence cohort. |
| **SC-006** | **Anti-Forgetting Retention** | **$\ge 95.0\%$** retention on solved sequences | Replay evaluation on previously solved sequences after 500 intervening gradient steps. |
| **SC-007** | **Extrapolation & Anti-Memorization** | **$100.0\%$** match on $K=100$ terms with **$M_{\text{MDL}} \le 1.20$** | Extrapolation horizon verifier ($n \in [20, 119]$) on all graduated candidate programs. |
| **SC-008** | **Latent Theorem Discovery Yield** | **$\ge 1$ novel verified theorem** | Normalized manifold search yielding PSLQ drop $<10^{-50}$ and SymPy proof export. |
| **SC-009** | **Progressive Pre-Flight Harness** | **$< 30\text{ seconds}$** runtime across all 4 tiers | Pre-flight validation script execution before authorizing production training run. |
| **SC-010** | **VRAM Ceiling Compliance** | **$\le 3.5\,\text{GB}$** peak VRAM usage | Continuous GPU memory telemetry during production 60-epoch training. |

---

## 5. Architectural & System Boundaries

### In Scope
- Implementing `src/oeis_learn/decoder/constant_solver.py` with HNF Diophantine and Z3 SMT solving routines.
- Updating `src/oeis_learn/decoder/grammar_masker.py` to admit `i64.const_?`.
- Integrating online compiler canonicalization (`wasm-opt -O3 --vacuum --dce --remove-unused-locals`) in `runner.py` and `crates/oeis_wasm_evaluator`.
- Implementing Covariant Parsimony Pressure (CPP) and lexicographical ranking in `src/oeis_learn/rl/reward.py`.
- Implementing Partitioned Semantic Entropy and stack-depth temperature scheduling in `src/oeis_learn/rl/egca_grpo.py`.
- Implementing normalized Newton quotients, Prime Fourier Embeddings (PFE), and summary tokens in `src/oeis_learn/encoder/`.
- Implementing `Exp3SBanditScheduler` and `AdaGGroupAllocator` in `src/oeis_learn/curriculum/symple_bandit.py`.
- Updating `src/oeis_learn/rl/elite_buffer.py` for dormancy-weighted vulnerability replay.
- Updating `src/oeis_learn/rl/trainer.py` to orchestrate the unified SYMPLE training loop.
- Fixing $L_2$ normalization in `scripts/run_long_e2e_benchmark.py` and latent discovery tools.
- Launching and monitoring Run 006 production training.

### Out of Scope / Deferred
- Multi-node distributed GPU cluster scaling (Tier 2 $d=768$) — deferred until local Tier 1 Stage 1 and Stage 2 graduation gates pass.
- Non-WebAssembly target compilation (e.g., C/Rust code synthesis) — WebAssembly stack bytecode remains the sole sandboxed execution target.
- Dynamic heap memory growth (`memory.grow`) for Stages 1–3 — restricted to scalar stack operations and local variable buffers.

---

## 6. Assumptions

- **Hardware Profile (Tier 1 Baseline)**: Workstation with 4 CPU cores / 8 threads (Intel Xeon E3-1505M v5 @ 2.80GHz), 64 GB DDR4 RAM, and an NVIDIA Quadro M2000M (4 GB GDDR5 VRAM).
- **GPU Precision & VRAM Limits**: Strict single precision (`torch.float32`) without Automatic Mixed Precision (AMP). Peak VRAM capped at $< 3.5\,\text{GB}$ via micro-batching ($B_{\text{active}}=2$) and dynamic group sizing ($G_i \in [8, 16]$).
- **External Dependencies**: `z3-solver` Python package available for SMT solving; `wasm-opt` binary or Binaryen C-FFI / native Rust passes available in execution sandbox; `mpmath` and `sympy` available for theorem proving.
- **Deterministic Evaluation**: WebAssembly execution without host-system imports provides complete determinism and isolation for mathematical integer computations.
- **Artifact Management**: All Run 006 artifacts (configs, metadata, checkpoints, logs, telemetry, synthesis results, discovered theorems) are automatically archived under `runs/006_phase4_decoupled_symple/`.
