# Research & Technical Decisions: OEIS Learn Neuro-Symbolic Synthesis

**Feature**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md)  
**Branch**: `001-oeis-neurosymbolic-synthesis`  
**Date**: 2026-08-30

---

## 1. Tri-Stream Continuous Integer Encoder & Numerical Precision

### Decision
Implement a 3-axis continuous numerical encoder combining:
1. **Magnitude Stream ($S_1$):** Signed continuous log-magnitude scalar $v_i = \text{sign}(x_i) \cdot (1 + \log_{10}(|x_i| + 1))$ projected via a 2-layer MLP with GELU activations.
2. **Modulo-Spectrum Stream ($S_2$):** Continuous sine/cosine Fourier phase embeddings across 100 base moduli ($m \in \{2, \dots, 101\}$) yielding a 200-dimensional trigonometric residue vector $\mathbf{\Phi}_i = \bigoplus_{m=2}^{101} [\sin(2\pi(x_i \bmod m)/m), \cos(2\pi(x_i \bmod m)/m)]$ projected linearly into $\mathbb{R}^d$.
3. **Local Difference & $p$-Adic Stream ($S_3$):** Logarithmic first difference $d_i^{(1)} = \text{sign}(\Delta x_i) \cdot (1 + \log_{10}(|\Delta x_i| + 1))$, second difference $d_i^{(2)} = \text{sign}(\Delta^2 x_i) \cdot (1 + \log_{10}(|\Delta^2 x_i| + 1))$, and ordinal embeddings $\mathbf{E}_{\text{padic}}(x_i) \in \mathbb{R}^{d_p}$ for $p$-adic valuations $v_p(x_i) = \max\{k : p^k \mid x_i\}$ for $p \in \{2, 3, 5, 7, 11, 13\}$ capped at $k_{\max}=16$.
4. **Hierarchical Two-Stage FiLM Fusion:**
   - Stage 1: $\boldsymbol{\gamma}_i^{(1)}, \boldsymbol{\beta}_i^{(1)} = \text{Split}(\mathbf{W}_{\text{FiLM1}} \mathbf{S}_{2,i} + \mathbf{b}_{\text{FiLM1}})$; $\mathbf{H}_{12,i} = \boldsymbol{\gamma}_i^{(1)} \odot \mathbf{S}_{1,i} + \boldsymbol{\beta}_i^{(1)}$.
   - Stage 2: $\boldsymbol{\gamma}_i^{(2)}, \boldsymbol{\beta}_i^{(2)} = \text{Split}(\mathbf{W}_{\text{FiLM2}} \mathbf{S}_{3,i} + \mathbf{b}_{\text{FiLM2}})$; $\mathbf{Z}_i = \boldsymbol{\gamma}_i^{(2)} \odot \mathbf{H}_{12,i} + \boldsymbol{\beta}_i^{(2)}$.
5. **Strict FP32 Precision:** All encoder modules, forward/backward passes, and optimizer states MUST operate in 32-bit single precision (`torch.float32`). Automatic Mixed Precision (AMP / FP16 / BF16) is strictly disabled.

### Rationale
- **OOV Elimination & Dynamic Range:** Sequences in OEIS span from single digits to $>10^{30}$. Standard tokenization (BPE/WordPiece) breaks down with OOV tokens or length explosions. Signed log-scaling handles astronomical magnitude dynamics.
- **Arithmetic Homomorphism:** Modulo spectrum directly encodes cyclic group properties ($\mathbb{Z}/m\mathbb{Z}$) and Chinese Remainder Theorem (CRT) congruences. Research shows Euler's totient ratio $\varphi(m)/m$ strongly correlates ($r = -0.851$) with information gain in sequence prediction.
- **Local Recurrences:** $S_3$ finite differences and $p$-adic factorizations expose step dynamics and divisibility properties essential for synthesizing linear recurrences, holonomic sequences, and number-theoretic algorithms.
- **Precision Stability:** Phase functions $\sin(2\pi r/m), \cos(2\pi r/m)$ produce subtle gradient updates that underflow or trigger catastrophic cancellation in FP16/BF16.

### Alternatives Considered
- **Discrete Subword / Digit Tokenization:** Rejected due to OOV errors, quadratic sequence length expansion $\mathcal{O}(L^2)$, and loss of numerical proximity.
- **xVal Continuous Scalar Embedding:** Rejected because LayerNorm/RMSNorm squashes single-scalar continuous activations in deep Transformer layers, and xVal lacks modular/algebraic representations.
- **Single-Step Concatenation Fusion:** Rejected because simple linear addition/concatenation does not allow modular congruences and local step dynamics to conditionally modulate growth magnitude.

---

## 2. Grammar-Guided WAT Decoding & Lexical Scope Tracking

### Decision
1. **Target Representation:** WebAssembly Text (WAT) format S-expressions defining typed `(func $generate_term (param $n i32) (result i64) ...)` routines.
2. **Grammar Engine:** Integrate `llguidance` (with `XGrammar-2` compatibility layer) for dynamic Earley-based parsing over byte-level tries, guaranteeing sub-$100\,\mu\text{s}$ per-token masking.
3. **Environment-Indexed Grammar ($\mathcal{G}_{\Gamma_t}$):** Maintain dynamic environment state $\Gamma_t = (\text{Vars}_t, \text{Types}_t, \text{Depth}_t)$ during decoding:
   - $\text{Vars}_t$: Tracks declared function parameters and local variables (`(param $n i32)`, `(local $temp i64)`).
   - $\text{Types}_t$: Tracks operand stack types to enforce valid binary arithmetic operands.
   - $\text{Depth}_t$: Tracks block/loop control nesting to ensure valid branch targets (`br`, `br_if`).
   - Logit masks restrict index tokens in `local.get`, `local.set`, and `local.tee` exclusively to $\text{Vars}_t$, enforcing **No-Ghost Soundness** and 100% syntactically valid WASM compilation.

### Rationale
- **Zero-Defect Code Generation:** Unconstrained autoregressive sampling suffers high syntax and uninitialized variable failure rates. Dynamic grammar masking eliminates malformed programs before execution.
- **Low Latency & Flat Tail:** `llguidance` evaluates Earley states and regex derivatives dynamically on tries without upfront state-table compilation delays ($0.05\text{--}2\,\text{ms}$ cold startup, $40\text{--}60\,\mu\text{s}$ per-token ITL).

### Alternatives Considered
- **Outlines (DFA Matrix Lookup):** Rejected due to massive state explosion and compilation bottlenecks ($>1\,\text{minute}$) when handling recursive, nested S-expressions.
- **Unconstrained Python/C++ Code Synthesis:** Rejected due to non-deterministic execution times, lack of instruction-level fuel metering, and security vulnerabilities during evaluation.

---

## 3. High-Throughput Sandboxed WASM Execution & GIL-Free Parallelism

### Decision
1. **Core Runtime:** Standardize on `wasmtime` (via Wasmtime Cranelift JIT engine).
2. **In-Memory Compilation:** Compile WAT text strings directly to binary WASM bytecode in-memory via `wat::parse_str` (in Rust) or `wasmtime.wat2wasm` (in Python) without disk I/O.
3. **Resource Bounding & Traps:**
   - Fuel limit: Exactly 10,000 instruction units injected via `store.set_fuel(10_000)`.
   - Linear memory limit: Capped at 16 MiB (256 pages $\times$ 64 KiB) via `store.set_limits`.
   - Traps for `OUT_OF_FUEL`, integer division by zero, stack overflow, or out-of-bounds access are caught gracefully, returning structured status codes.
4. **Native PyO3 + Rayon Bridge (`oeis_wasm_evaluator`):**
   - Rust extension releasing Python GIL via `py.allow_threads`.
   - Batch evaluation of 1,000+ WAT programs across 8 CPU threads concurrently using `rayon::par_iter()`.

### Rationale
- **Deterministic Halting:** Cranelift basic-block fuel metering provides instruction-exact budget enforcement with minimal 5–15% CPU overhead. Infinite loops are halted in $<1\,\text{ms}$.
- **GIL Elimination & Throughput:** Python's Global Interpreter Lock bottlenecks multi-threaded evaluation. PyO3 + Rayon achieves $>500$ module evaluations per second on a 4-core / 8-thread workstation.

### Alternatives Considered
- **Python `ProcessPoolExecutor`:** Rejected due to heavy IPC serialization overhead and memory consumption of multiple Python processes.
- **Python `ThreadPoolExecutor`:** Rejected due to severe GIL lock contention during `wasmtime.Store` object allocation and call invocations.
- **Wasmer:** Rejected due to AST-level fuel metering bytecode bloat and maintenance lag in Python bindings.

---

## 4. Reinforcement Learning: EGCA-GRPO with Asymmetric Prompt Weighting

### Decision
1. **Algorithm:** Execution-Guided Credit Assignment GRPO (EGCA-GRPO) with Asymmetric Prompt Weighting.
2. **Reward Function:** Strict binary outcome reward:
   $$R(x, y) = \begin{cases} +1 & \text{if } \text{WASM}_y(n) == A(n), \; \forall n \in \{0, \dots, N-1\} \\ -1 & \text{otherwise} \end{cases}$$
3. **Execution-Guided Credit Assignment (EGCA):** Traces candidate execution state against ground-truth sequence generation. Identifies the exact instruction token index where output deviates from $A(n)$, concentrating gradient updates on the localized error window.
4. **Asymmetric Prompt Weighting:** For rollout groups where all $G$ completions fail ($R_i = -1, \forall i$), non-zero negative gradients are applied directly to the execution divergence span, preventing zero-advantage collapse ($A_i = 0$) on hard prompts.

### Rationale
- **Critic-Free Memory Efficiency:** GRPO eliminates the need for an auxiliary critic model, reducing memory consumption by $\sim 50\%$ and enabling larger rollout group sizes ($G \ge 16$) on workstation hardware.
- **Solving Binary Reward Sparsity:** Standard GRPO produces zero gradient updates when all samples fail. Asymmetric prompt weighting and trace-grounded token attribution maintain informative learning signals throughout curriculum progression.

### Alternatives Considered
- **PPO:** Rejected due to doubling memory footprint with a value critic network and critic lag on non-differentiable execution rewards.
- **Continuous Pass-Rate Rewards ($r = \text{passed}/N$):** Rejected because partial sequence matches heavily reward incorrect polynomial approximations that coincide on initial terms by accident.
- **Vanilla REINFORCE:** Rejected due to catastrophic variance under sparse binary rewards.

---

## 5. Curriculum Learning Engine & Anti-Memorization Verification

### Decision
1. **5-Stage Taxonomic Curriculum:**
   - Stage 1: Primitives & Polynomials (`easy`, `core`, `nonn`) $\rightarrow$ direct scalar loops.
   - Stage 2: Linear Recurrences & Rational GFs (`core`, `frac`, `cons`, `mult`) $\rightarrow$ sliding-window state buffers.
   - Stage 3: Holonomic & D-Finite Sequences (`nice`, `cofr`, `tabl`, `tabf`) $\rightarrow$ dynamic polynomial coefficients.
   - Stage 4: Combinatorics & Number Theory (`hard`, `base`, `eigen`) $\rightarrow$ prime sieves, dynamic programming buffers.
   - Stage 5: Exhaustive Search & Graph Invariants (`hard`, `bref`, `more`) $\rightarrow$ heap memory, backtracking search.
2. **Graduation Gates:** Stage graduation requires:
   - Rolling Task Competence: $C(S_k) = \frac{1}{|S_k|} \sum_{x \in S_k} w_x \hat{\rho}_x \ge 0.85$.
   - Coverage Equilibrium: $\min_{x \in S_k} (\hat{\rho}_x) \ge 0.50$.
   - Policy Stability: Consecutive epoch variance $\mathbb{Var}[C_e(S_k)] \le \varepsilon_{\text{var}}$.
   - Mixture transition sampling: $70\%$ new stage, $20\%$ previous stage, $10\%$ earlier stages.
3. **Anti-Memorization Verification:**
   - **Extrapolation Horizon ($N+K$):** Evaluates candidate WASM binaries on $K=100$ unseen future terms ($n \in [20, 119]$). Requires 100% exact match ($G_{\text{ext}} = 1$).
   - **Minimum Description Length ($M_{\text{MDL}}$):** Compares WASM byte size $|y|_{\text{bytes}}$ against sequence Lempel-Ziv complexity $C(A_N)$. Requires $M_{\text{MDL}} = \frac{|y|_{\text{bytes}}}{C(A_N)} \le 1.2$ to reject bloated lookup tables and Lagrange polynomials.

### Rationale
- Prevents early exploration collapse on complex sequences while eliminating false positives caused by memorization.

---

## 6. Self-Supervised Latent Discovery Pipeline & Theorem Verification

### Decision
1. **Non-Contrastive Representation Learning:** Train sequence embeddings using **VICReg** (Variance-Invariance-Covariance Regularization) over positive algebraic operator pairs:
   - Partial Sums ($\mathcal{S}$), First Differences ($\Delta$), Binomial Transforms ($\mathcal{B}$), Euler Transforms ($\mathcal{E}$), Shifts ($\mathcal{T}_k$).
   - Loss: $\mathcal{L}_{\text{VICReg}} = \lambda s(Z_a, Z_b) + \mu [v(Z_a) + v(Z_b)] + \nu [c(Z_a) + c(Z_b)]$.
2. **Manifold Clustering:** GPU-accelerated cuML UMAP (768D/384D to 2D) and HDBSCAN density clustering to discover unannotated sequence families and flag anomalies (noise label $-1$).
3. **4-Stage Mathematical Discovery Verification Engine:**
   - Stage 1 (Geometric Retrieval): HNSW index finds candidate vector triples $(\vec{v}_A, \vec{v}_B, \vec{v}_C)$ where $\|\vec{v}_A + \vec{v}_B - \vec{v}_C\|_2 < \epsilon$.
   - Stage 2 (Arbitrary-Precision Sampling): Sample sequence terms up to 1,000 terms at $>500$ decimal digits using `mpmath`.
   - Stage 3 (Integer Relation Search): Run the **PSLQ algorithm**; require confidence ratio drop $<10^{-50}$ and norm bound $M$ validation.
   - Stage 4 (Symbolic Theorem Proving): Pass detected relations to SymPy/SageMath (`sympy.rsolve`, generating function identities, Wronskian telescoping) to generate machine-verified algebraic proofs.

### Rationale
- VICReg avoids the class-collision flaw of contrastive InfoNCE (where two sequences assumed negative may actually share an unproven identity) while preventing dimension collapse.
- PSLQ coupled with SymPy turns geometric vector observations into formal, mathematically verified theorems.

---

## 7. Workstation Scaling Strategy (Tier 1 Baseline)

### Decision
Enforce strict resource budgets for local Tier 1 development:
- **Embedding Dimension ($d$):** $d = 256$ (or $d = 384$) for local workstation baseline (Xeon E3-1505M v5, 4 cores / 8 threads, 64 GB RAM, Quadro M2000M 4GB VRAM).
- **Batch Size & Gradient Accumulation:** Micro-batch size of 4–8 sequences on GPU with 4–8 accumulation steps (effective batch size 32).
- **Division of Labor:** 100% of GPU resources allocated to FP32 neural tensor operations; 100% of WASM executions offloaded to 8 CPU threads via PyO3 + Rayon.
- **Dataset Subsetting:** Prototype on filtered Stage 1 & 2 subsets (10,000 to 25,000 sequences) before cluster scale-up.

---

## 8. Technology Stack & Component Mapping

| Subsystem | Primary Library / Technology | Fallback / Alternative | Purpose |
| :--- | :--- | :--- | :--- |
| **Language & Core Framework** | Python 3.11+ / PyTorch 2.3+ | Native C++ / libtorch | Neural modeling, training loop, discovery pipeline |
| **Native Execution Extension** | Rust (2021 edition), PyO3 0.20+, Rayon 1.8+ | C++ ctypes / cython | GIL-free multi-threaded batch WASM evaluation |
| **WASM Runtime Engine** | Wasmtime 20.0+ (`wasmtime` Rust crate / `wasmtime-py`) | Wasmer | Sandboxed fuel-metered execution and trap interception |
| **In-Memory WAT Parser** | `wat` 1.0+ Rust crate / `wasmtime.wat2wasm` | WABT C++ | In-memory text S-expression to WASM bytecode compilation |
| **Grammar-Guided Decoding** | `llguidance` | `XGrammar-2` | Sub-$100\,\mu\text{s}$ dynamic Earley trie logit masking |
| **Database & Ingestion** | DuckDB / SQLite | PostgreSQL | Ingestion and querying of OEIS sequence data and metadata |
| **Manifold & Clustering** | RAPIDS cuML (UMAP, HDBSCAN) / scikit-learn | FastTree / FAISS | High-dimensional geometric representation discovery |
| **Arbitrary-Precision Math** | `mpmath` | GNU MPFR / gmpy2 | $>500$-digit evaluation for integer relation discovery |
| **Integer Relation Search** | PSLQ Algorithm (native mpmath / C) | LLL Lattice Reduction | Identification of integer relation vectors $a_i \in \mathbb{Z}$ |
| **Symbolic Verification** | SymPy 1.12+ / SageMath | Maxima / Giac | Closed-form recurrence solving and algebraic proofs |
| **Test Framework** | `pytest`, `pytest-benchmark`, `cargo test` | `unittest` | Unit, integration, numerical stability, and contract tests |
