# Research & Architectural Decisions: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine

**Feature**: [specs/004-decoupled-grounding-and-symple-engine/spec.md](specs/004-decoupled-grounding-and-symple-engine/spec.md)  
**Branch**: `004-decoupled-grounding-and-symple-engine`  
**Date**: 2026-09-02

---

## 1. Decoupled Symbolic-Numeric Grounding Engine

### 1.1 Root-Cause Anatomy: The Voronoi Categorical Partitioning Bottleneck
In autoregressive transformer code decoders, continuous sequence context vectors $C_t \in \mathbb{R}^{256}$ are projected via linear classification heads $W_{\text{vocab}} \in \mathbb{R}^{|\mathcal{V}| \times d}$ into discrete token logits. When the grammar requires an immediate integer constant following `i64.const`, linear dot-product attention cannot compute algebraic invariants (such as sequence derivatives $\frac{\Delta y}{\Delta n}$ or recurrence determinants) directly from convex combinations of encoder states. Instead, the categorical head partitions the continuous hidden space into high-dimensional Voronoi polyhedra.

Under uncertainty and sparse 0/1 verification rewards, standard cross-entropy minimization causes the categorical distribution to collapse to the empirical mode of the training distribution (constants `0` and `1`). Consequently, the policy synthesizes the correct topological AST (e.g. `n * C_1 + C_2`), but defaults multipliers to unity ($C_1 = 1$), causing Advantage Collapse under GRPO ($\text{std}(\{R\}) = 0$).

### 1.2 Two-Stage Solver Architecture & Dispatch Strategy

```
                          +----------------------------------------------+
                          |   WAT AST Skeleton with 'i64.const_?' Tokens |
                          +----------------------------------------------+
                                                 |
                                                 v
                          +----------------------------------------------+
                          |         AST Linearity & Operator Parser      |
                          +----------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
            [Linear/Affine in Trace]                             [Non-Linear / Control Flow]
                       |                                                   |
                       v                                                   v
       +-------------------------------+                   +-------------------------------+
       | Exact Diophantine Solver      |                   | Satisfiability Modulo Theories|
       | (Hermite Normal Form / SVD)   |                   | (Z3 QF_BV / QF_NIA)           |
       | Solve: A * C = Y in <1ms      |                   | 250ms Timeout Sandbox         |
       +-------------------------------+                   +-------------------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               | Grounded Executable WASM Module   |
                               +-----------------------------------+
```

#### Decision 1: Exact Diophantine Linear Solver (Hermite Normal Form / Integer Row Reduction)
- **Mechanism**:
  1. For an AST skeleton $P_{\mathbf{C}}$ with $k$ placeholders ($1 \le k \le 4$), substitute basis indicator vectors for constants and execute the partial program in the sandbox for $n \in \{0, \dots, 19\}$ to construct system matrix $A \in \mathbb{Z}^{20 \times (k+1)}$:
     $$P_{\mathbf{C}}(n) = c_0 + c_1 f_1(n) + \dots + c_k f_k(n) \implies A_{n, j} = f_j(n), \quad A_{n, 0} = 1$$
  2. Compute exact integer row reduction or exact rank determination. If $\text{rank}(A) = k+1$ and $Y$ lies in the column space of $A$ over $\mathbb{Z}$, solve $A \mathbf{C} = Y$.
  3. For underdetermined systems ($k > \text{rank}(A)$), select the minimal $L_1$-norm integer solution $\mathbf{C}^* = \arg\min \|\mathbf{C}\|_1$ to favor parsimonious constants.
- **Latency**: $<1.0\,\text{ms}$ per candidate using optimized NumPy / exact integer matrix routines.
- **Alternatives Considered**: Continuous BFGS / Levenberg-Marquardt with post-hoc rounding. *Rejected because non-linear integer rounding fails on tight modular/recurrence constraints, whereas exact Diophantine reduction is deterministic and globally exact.*

#### Decision 2: Z3 SMT Fallback Engine (`QF_BV`)
- **Mechanism**:
  1. When placeholders appear inside non-linear operations (`i64.rem_u`, `i64.shl`, `i64.shr_u`, `i64.div_u`, or inside `if/else`/`br_if` conditions), lower the AST to an SMT-LIB2 formula over 64-bit BitVectors (`QF_BV`).
  2. Formulate constraints $\bigwedge_{n=0}^{19} (\llbracket P_{\mathbf{C}} \rrbracket(n) = y_n)$ with bounded constant variables $c_1, \dots, c_k \in \text{BitVec}(64)$.
  3. Invoke Z3 solver with a strict $250\,\text{ms}$ thread timeout.
- **Latency**: $5\text{--}80\,\text{ms}$ for $k \le 4$.
- **Alternatives Considered**: CVC5, MiniSat, purely unconstrained random search. *Rejected because `z3-solver` provides Python C-bindings, robust QF_BV bitvector logic, and sub-second solving for small parameter counts.*

#### Decision 3: Policy Gradient Attribution Decoupling
- **Mechanism**:
  - The autoregressive policy $\pi_\theta$ generates the **skeleton containing `i64.const_?` placeholders**.
  - On-policy GRPO surrogate loss $\mathcal{L}_{\text{GRPO}}$ is evaluated and backpropagated through the **placeholder token sequence** using the grounded program's reward ($R_{\text{exec}} = 1.0$).
  - The **grounded program** (with concrete constants spliced in) is ingested into the **Elite Demonstration Buffer ($\mathcal{D}_{\text{elite}}$)** to supervise off-policy SFT replay $\mathcal{L}_{\text{SFT}}$.
- **Rationale**: Ensures structural policy gradients are reinforced without requiring the network to solve Diophantine equations internally, while SFT replay anchors the network on valid concrete syntax.

---

## 2. Compiler-in-the-Loop Canonicalization & Anti-Padding RLVR

### 2.1 Root-Cause Anatomy: Grammar-Induced Dead-Code Attractors
Because the dynamic grammar masker strictly enforces stack balance ($\Sigma_t = 1$) at function termination, an uncertain policy gradient discovers that emitting balanced no-op pairs (e.g. `local.get $n` followed by `drop`, or repeated register overwrites `i64.const 1`, `local.set $a`) preserves stack equilibrium. This allows the model to satisfy closing rules and collect $+0.1$ validity rewards without risking execution traps. Under GRPO, relative advantage normalization rewards the best-padded candidate, collapsing policy entropy over meaningful mathematical operators.

### 2.2 Canonicalization & Parsimony Architecture

```
                                +-----------------------------------+
                                | Raw Generated WAT Program P       |
                                +-----------------------------------+
                                                  |
                                                  v
                                +-----------------------------------+
                                | In-Memory Assembly to Binary B    |
                                +-----------------------------------+
                                                  |
                                                  v
                                +-----------------------------------+
                                | wasm-opt -O3 --vacuum --dce       |
                                |         --remove-unused-locals    |
                                +-----------------------------------+
                                                  |
                                                  v
                                +-----------------------------------+
                                | Canonical Binary B_opt & WAT P_opt|
                                +-----------------------------------+
                                                  |
                                                  v
                                +-----------------------------------+
                                | Syntactic Waste Ratio rho_waste   |
                                | Covariant Parsimony Pressure CPP  |
                                | Lexicographical Group Advantage   |
                                +-----------------------------------+
```

#### Decision 4: Online Binaryen Compiler Pass (`wasm-opt`)
- **Mechanism**:
  - Compile raw candidate $P \to B$.
  - Run optimization passes: `--vacuum` (removes unused expressions and push-drop pairs), `--dce` (prunes dead control flow), and `--remove-unused-locals` (eliminates redundant register writes).
  - Disassemble optimized binary $B_{\text{opt}} \to P_{\text{opt}}$.
  - Compute Syntactic Waste Ratio: $\rho_{\text{waste}}(P) = \frac{|P|_{\text{tokens}} - |P_{\text{opt}}|_{\text{tokens}}}{|P|_{\text{tokens}}}$.
- **Performance**: $<1.5\,\text{ms}$ per module via native Rust PyO3 bindings in `crates/oeis_wasm_evaluator`.

#### Decision 5: Hard Syntactic Waste Cutoff & Covariant Parsimony Pressure
- **Mechanism**:
  1. **Hard Waste Cutoff**:
     $$R_{\text{validity}}(P) = \begin{cases} 0.1 \cdot \exp(-2.0 \cdot \rho_{\text{waste}}(P)) & \text{if } \rho_{\text{waste}}(P) \le 0.30 \\ 0.0 & \text{if } \rho_{\text{waste}}(P) > 0.30 \end{cases}$$
     If $>30\%$ of emitted tokens are dead code, all validity rewards are zeroed out.
  2. **Dense Log-Distance Return**:
     $$R_{\text{dense}}(P, Y) = \frac{1}{20} \sum_{n=0}^{19} \frac{1}{1 + \log_{10}(|P(n) - y_n| + 1)}$$
  3. **Covariant Parsimony Pressure (CPP)**:
     $$c_k = \frac{\operatorname{Cov}_{i \in [1, G]}(\ell(P_i), R_{\text{dense}}(P_i))}{\operatorname{Var}_{i \in [1, G]}(\ell(P_i)) + \epsilon}$$
     $$R_{\text{CPP}}(P_i) = R_{\text{dense}}(P_i) - \max(0, -c_k) \cdot (\ell(P_i) - \ell_{\min}) - 0.20 \cdot \rho_{\text{waste}}(P_i)$$
- **Rationale**: When length correlates negatively with reward (bloat), $-c_k > 0$ applies adaptive length penalties. When length correlates with algorithmic progress (e.g., loops), $-c_k \le 0$ and no penalty is incurred.

#### Decision 6: Lexicographical Group Ranking
- **Mechanism**:
  - Order candidate rollouts by strict dominance: $P_i \succ P_j \iff (R_{\text{exec}}(P_i) > R_{\text{exec}}(P_j)) \lor (R_{\text{exec}}(P_i) = R_{\text{exec}}(P_j) \land |P_{\text{opt}, i}| < |P_{\text{opt}, j}|)$.
  - Assign normalized ordinal advantage:
    $$\hat{A}_i^{\text{lex}} = \frac{2 \cdot (\operatorname{rank}(P_i) - 1)}{G - 1} - 1 \in [-1, 1]$$
- **Rationale**: Completely removes padding incentives among equal-validity candidates while ensuring functionally correct programs strictly dominate compact incorrect ones.

#### Decision 7: Partitioned Semantic Policy Entropy & Stack-Depth Temperature
- **Mechanism**:
  - Partition valid actions $\mathcal{A}(s_t) = \mathcal{A}_{\text{sem}}(s_t) \cup \mathcal{A}_{\text{struct}}(s_t)$.
  - Apply positive entropy bonus to semantic tokens ($\mathcal{A}_{\text{sem}}$: arithmetic, loops, variable references) and penalize structural tokens ($\mathcal{A}_{\text{struct}}$: `drop`, `nop`):
    $$\mathcal{L}_{\text{ent}}(\theta) = 0.02 \cdot \frac{\mathcal{H}_{\text{sem}}(\pi \mid s_t)}{\log |\mathcal{A}_{\text{sem}}(s_t)| + \epsilon} - 0.05 \cdot \max(0, P(\mathcal{A}_{\text{struct}} \mid s_t) - 0.15)$$
  - Scale sampling temperature by stack height:
    $$T(s_t) = T_{\text{base}} \cdot \left(1 - 0.60 \cdot \frac{\Sigma_t}{\Sigma_{\max}}\right)$$
- **Rationale**: Tightens temperature as stack height approaches the return arity ($\kappa=1$), ensuring clean termination without wandering into padding loops.

---

## 3. Tri-Stream Encoder v2 & Linear Invariant Representation

### 3.1 Root-Cause Anatomy: Modulatory Entanglement in Hierarchical FiLM
In Phase 1–3, the two-stage Hierarchical FiLM module used complex Fourier phase vectors ($S_2$) to modulate signed log-magnitude ($S_1$) and differences ($S_3$). While effective for categorical style conditioning, applying multiplicative modulation across continuous arithmetic signals warped the latent surface. Bilinear query-key cross-attention could not compute linear slopes $\frac{\Delta y}{\Delta n}$ without disentangling oscillating phase surfaces.

### 3.2 Encoder v2 Architectural Specifications

```
  +--------------------+   +-----------------------+   +--------------------+   +---------------------+
  | S1: Log-Magnitude  |   | S2: Prime Fourier PFE |   | S3: Newton Quots   |   | S4: Summary Tokens  |
  | sign(y)*log(|y|+1) |   | 16 Orthogonal Primes  |   | D^(k) = Delta^k/k! |   | z_affine, z_geom    |
  +--------------------+   +-----------------------+   +--------------------+   +---------------------+
             \                         |                         /                        /
              +------------------------+------------------------+------------------------+
                                       | (Direct Linear Concatenation)
                                       v
                     [Linear Projection to d_model = 256]
                                       |
                                       v
               [Bidirectional Transformer Self-Attention (FP32)]
                                       |
                                       v
                  Continuous Latent Representation Matrix Z in R^{22 x 256}
```

#### Decision 8: Newton Forward Difference Quotients ($S_3$)
- **Formulation**:
  $$D^{(1)}_i = y_{i+1} - y_i, \quad D^{(2)}_i = \frac{y_{i+2} - 2y_{i+1} + y_i}{2}, \quad D^{(3)}_i = \frac{\Delta^3 y_i}{6}$$
- **Rationale**: Newton's forward difference formula states $P(n) = \sum_{k=0}^d \binom{n}{k} \Delta^k y_0$. Factorial normalization represents polynomial Taylor coefficients directly as static constants across positions, allowing linear cross-attention to isolate polynomial multipliers via single query-key dot products.

#### Decision 9: Orthogonal Prime Fourier Embeddings (PFE, $S_2$)
- **Formulation**:
  $$\text{PFE}(y) = \bigoplus_{p \in \mathcal{P}_{16}} \left[ \cos\left(\frac{2\pi y}{p}\right), \sin\left(\frac{2\pi y}{p}\right) \right]$$
  where $\mathcal{P}_{16} = \{3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59\}$.
- **Rationale**: Prime fields are mathematically orthogonal. Linear projections over PFE decompose into block-diagonal operators, isolating modular residues into independent channels and preventing spectral cross-talk.

#### Decision 10: Global Latent Summary Tokens ($\mathbf{z}_{\text{affine}}, \mathbf{z}_{\text{geom}}$)
- **Formulation**:
  Prepend two learnable summary tokens to the encoder sequence:
  - $\mathbf{z}_{\text{affine}}$: Supervised via auxiliary MSE regression head predicting sequence-wide slope $\frac{\text{Cov}(n, Y)}{\text{Var}(n)}$.
  - $\mathbf{z}_{\text{geom}}$: Supervised via auxiliary MSE regression head predicting geometric ratio $\text{median}(y_{i+1}/y_i)$.
- **Auxiliary Loss**: $\mathcal{L}_{\text{aux}}(\mathbf{z}) = \text{MSE}(\hat{m}, m_{\text{true}}) + \text{MSE}(\hat{r}, r_{\text{true}})$, scaled by $\lambda_{\text{aux}} = 0.10$ in the joint objective.

#### Decision 11: Direct Concatenation & Bidirectional Self-Attention
- **Mechanism**: Retire Hierarchical FiLM in favor of direct vector concatenation $[S_1; S_2; S_3]$ followed by linear projection to $d_{\text{model}} = 256$ and 4-layer bidirectional self-attention in strict FP32 precision.

---

## 4. SYMPLE Multi-Task Engine: Bandit Curriculum & Anti-Forgetting Replay

### 4.1 Root-Cause Anatomy: Task Dilution & Parameter Drift in Run 005
In Run 005, uniform sampling across 524 sequences with batch size $B=8$ resulted in an average visitation interval of 65 gradient steps per sequence. Over 65 intervening steps, parameter updates on unrelated tasks erased newly discovered solutions. Furthermore, shallow rollout groups ($G=4$) on hard sequences ($p=0.01$) yielded a hit probability of only $3.9\%$, causing batch-level advantage collapse ($P(\text{Batch Gradient}=\mathbf{0}) = 72.5\%$).

### 4.2 SYMPLE Architecture & Execution Loop

```
  +-------------------------------------------------------------------------------------------------+
  |                                    SYMPLE EXECUTION LOOP                                        |
  +-------------------------------------------------------------------------------------------------+
  |                                                                                                 |
  |  1. Task Selection: EXP3.S Bandit selects B_active = 2 frontier sequences based on ZPD.         |
  |                                                                                                 |
  |  2. Dynamic Group Sizing: Ada-G allocates G_i in [8, 16] (Total M_active = 24..32 rollouts).    |
  |                                                                                                 |
  |  3. Rollout Sampling: Autoregressive decoder emits AST skeletons with 'i64.const_?'             |
  |                                                                                                 |
  |  4. Grounding: Diophantine / SMT Solvers compute concrete constants C*.                         |
  |                                                                                                 |
  |  5. Verification: wasm-opt canonicalization, waste ratio rho_waste, execution reward R_exec.    |
  |                                                                                                 |
  |  6. Virtual Sample Injection: Injects r=1.0 if all k_i=0 but sequence exists in EDB.           |
  |                                                                                                 |
  |  7. Buffer Ingestion: Verified grounded programs stored in Elite Demonstration Buffer (EDB).    |
  |                                                                                                 |
  |  8. Dormancy Replay: Samples B_replay = 2 dormant sequences from EDB for SFT consistency loss.  |
  |                                                                                                 |
  |  9. Joint Optimization: Backpropagates L_GRPO + 0.50 L_SFT + 0.10 L_aux - 0.02 D_KL.            |
  |                                                                                                 |
  |  10. Bandit Update: Updates EXP3.S weights w_i via binomial dispersion r_{i,t}.                 |
  +-------------------------------------------------------------------------------------------------+
```

#### Decision 12: EXP3.S Non-Stationary Bandit Task Scheduler
- **Mechanism**:
  - Maintains weights $w_i$ over all $K=524$ sequences.
  - Arm selection probability:
    $$p_{i, t+1} = (1 - \gamma_{\text{exp3}}) \frac{w_{i, t+1}}{\sum_{j=1}^K w_{j, t+1}} + \frac{\gamma_{\text{exp3}}}{K}$$
  - Weight update with switching factor $\alpha_{\text{exp3}}$:
    $$w_{i, t+1} = w_{i, t} \exp\left(\frac{\gamma_{\text{exp3}} \hat{r}_{i, t}}{K}\right) + \frac{e \cdot \alpha_{\text{exp3}}}{K} \sum_{j=1}^K w_{j, t}$$
  - Learning Progress Feedback in Zone of Proximal Development:
    $$r_{i, t} = \hat{p}_i (1 - \hat{p}_i) + |\Delta C_i| + 2.0 \cdot \max(0, -\Delta C_i)$$
    where $\hat{p}_i(1 - \hat{p}_i)$ is binomial dispersion (peaks at $p=0.5$) and $\Delta C_i$ is competence slope over trailing 20 attempts.
- **Parameters**: $\gamma_{\text{exp3}} = 0.15$, $\alpha_{\text{exp3}} = 0.05$.

#### Decision 13: Dynamic Group Sizing (Ada-G)
- **Mechanism**:
  - For active batch size $B_{\text{active}} = 2$, allocate variable group size per prompt:
    $$G_i = \text{clip}\left( \left\lceil \frac{\ln(1 - 0.50)}{\ln(1 - \max(\hat{p}_i, 0.02))} \right\rceil, 8, 16 \right)$$
- **Guarantee**: Ensures $P(\text{Hit} \ge 1) \ge 0.50$ for frontier sequences ($p \approx 0.05$), scaling positive advantage $\hat{A}^+ \approx \sqrt{15} = 3.87$ to firmly anchor discoveries.

#### Decision 14: Virtual Sample Injection
- **Mechanism**: If all $G_i$ rollouts for an active task fail ($k_i = 0$), but the sequence has a verified program stored in its EDB bucket $\mathcal{B}_i$, inject a synthetic positive return ($r=1.0$) into group normalization.
- **Advantage Effect**: Generates non-zero negative advantages $\hat{A}^- = -1/\sqrt{G}$ across all failed rollouts, penalizing the failed paths without evaluating to zero gradient.

#### Decision 15: Elite Demonstration Buffer (EDB) & Vulnerability Replay
- **Mechanism**:
  - Store top-4 shortest canonical AST programs per sequence in associative archive $\mathcal{D}_{\text{elite}}$, deduplicated by structural AST hash and MDL.
  - Sample $B_{\text{replay}} = 2$ dormant sequences based on elapsed dormancy $\Delta t_{\text{dormant}} = t_{\text{current}} - t_{\text{last\_visit}}$.
  - Joint multi-objective loss:
    $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta; \mathcal{D}_{\text{active}}) + 0.50 \cdot \mathcal{L}_{\text{SFT}}(\theta; \mathcal{D}_{\text{replay}}) + 0.10 \cdot \mathcal{L}_{\text{aux}}(\mathbf{z}) - 0.02 \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

#### Decision 16: Procedural Synthetic Generator with Randomized Affine Sweeps
- **Mechanism**: Apply randomized affine transformations $\tilde{Y} = \alpha Y + \beta$ ($\alpha \sim \pm 10^{\mathcal{U}(0, 5)}, \beta \sim \mathcal{U}(-10^5, 10^5)$) across all forward-generated AST templates during synthetic SFT warmup dataset creation.
- **Rationale**: Breaks correlation between structural idioms and literal constants, forcing cross-attention to rely on continuous difference streams.

---

## 5. Latent Manifold Normalization & PSLQ Discovery Pipeline

### 5.1 Root-Cause Anatomy: Euclidean Scale Distortion in Vector Search
In Run 005, sequence embeddings $z_i \in \mathbb{R}^{256}$ had raw norms $\|z_i\|_2 \approx 10.0\text{--}15.0$. Querying vector triples $(z_A + z_B \approx z_C)$ with a fixed Euclidean distance radius $\varepsilon = 0.8$ resulted in zero candidate pairs, because even minor angular misalignments resulted in Euclidean distances $>3.0$.

### 5.2 Manifold Normalization & Verification Pipeline

#### Decision 17: $L_2$-Normalized Embedding Manifold
- **Mechanism**:
  Normalize all sequence representations prior to indexing:
  $$\hat{z}_i = \frac{z_i}{\|z_i\|_2 + 10^{-8}}$$
- **Effect**: Transforms the latent space into a unit hypersphere where Euclidean distance directly corresponds to cosine distance ($d^2(\hat{u}, \hat{v}) = 2(1 - \cos(\hat{u}, \hat{v}))$), making $\varepsilon_{\text{dist}} = 0.8$ correspond to an angular cone of $\approx 47^\circ$.

#### Decision 18: High-Precision PSLQ & SymPy Automated Theorem Proving
- **Mechanism**:
  1. For each candidate triple $(\hat{z}_A, \hat{z}_B, \hat{z}_C)$ where $\|\hat{z}_A + \hat{z}_B - \hat{z}_C\|_2 \le 0.8$, query high-precision sequence generators for $N=100$ terms at 500-digit precision via `mpmath`.
  2. Run PSLQ integer relation detection over vectors $[a(n), b(n), c(n), 1]$.
  3. If PSLQ confidence ratio drops $<10^{-50}$, pass the integer linear relation to SymPy to construct formal symbolic proofs and export Markdown theorem records.

---

## 6. Summary of Architectural Decisions & Alternatives

| Subsystem | Selected Architecture | Key Hyperparameters / Constraints | Rejected Alternative | Why Rejected |
| :--- | :--- | :--- | :--- | :--- |
| **Numeric Solver** | Dual HNF Diophantine + Z3 QF_BV Fallback | Linear: $<1\,\text{ms}$; SMT: $250\,\text{ms}$ timeout; $k \le 4$ | Pure continuous BFGS regression | Non-convex local minima and integer rounding failures on modular logic |
| **Compiler RLVR** | Online `wasm-opt` DCE + CPP + Lexicographic Ranking | $\tau_{\text{thresh}} = 0.30, \lambda_{\text{waste}} = 0.20$, $R_{\text{exec}} \succ -|P_{\text{opt}}|$ | Static length penalty $-\beta |P|$ | Induces semantic cliff; collapses programs to constant returns |
| **Encoder** | Newton Differences + Prime Fourier + Summary Tokens | $D^{(k)} = \Delta^k y / k!$, 16 primes, $\mathbf{z}_{\text{affine}}, \mathbf{z}_{\text{geom}}$ | Hierarchical Two-Stage FiLM Fusion | Non-linear phase modulation warps latent surface, blocking linear slope extraction |
| **Curriculum** | EXP3.S Bandit + Ada-G + EDB Dormancy Replay | $\gamma=0.15, \alpha=0.05, B_{\text{active}}=2, G \in [8, 16], B_{\text{replay}}=2$ | Staged gating ($C(S_k) \ge 0.85$) with uniform sampling | 65-step prompt dilution and catastrophic parameter drift |
| **Discovery** | $L_2$-Normalized Unit Hypersphere + PSLQ | $\hat{z} = z / \|z\|_2$, $\varepsilon_{\text{dist}} = 0.8$, 500 digits, drop $<10^{-50}$ | Unnormalized Euclidean distance search | Vector norm mismatch ($\|z\| \approx 10$) yielded 0 candidate relations |
