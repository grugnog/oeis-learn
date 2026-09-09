# Research & Architectural Decisions: 4 × i64 Multi-Limb Migration & Curriculum Scaling

**Feature Branch**: `006-multilimb-curriculum-scaling`  
**Date**: 2026-09-06  
**Status**: Completed Phase 0 Analysis

---

## 1. Multi-Limb Arithmetic Representation

### Decision
Adopt a fixed $4 \times i64$ (255-bit signed integer, `i256x4_v1`) representation across token generation, sandboxed execution, and verification, returning 4 64-bit integer limbs on the WebAssembly value stack in little-endian order ($[l_0, l_1, l_2, l_3]$).

### Rationale
- **Empirical Corpus Census ($N = 399,005$ OEIS sequences)**:
  - 64-bit signed scalars ($1 \times i64$) fail on **28.6%** of sequences and almost all classic algebraic recurrences at 100 terms (e.g. Fibonacci $F_{93} \approx 1.22 \times 10^{19} > 2^{63}-1$).
  - 255-bit signed quad-limb ($4 \times i64$) covers **99.51%** of all 399,005 OEIS sequences and **99.56%** of all 74,986 computable `jOEIS` classes.
  - The $>255$-bit tail ($0.48\%$) contains sequences with an average of only 9.2 published terms (isolated multi-hundred-digit prime searches, hyper-exponential graph bounds), with only 1 sequence having $\ge 50$ terms.
- **WASM Value Stack Compatibility**: WebAssembly natively supports multi-value returns (`(result i64 i64 i64 i64)`). Four 64-bit registers map cleanly onto CPU registers and SIMD lanes without memory allocations.
- **Hardware Bound**: Preserves the project's Tier 1 workstation profile ($<16\text{ MiB}$ memory, $10{,}000$ instruction fuel cap, sub-microsecond evaluation per term).

### Alternatives Considered
1. **Dynamic Arbitrary-Precision Heap Allocation (BigInt)**:
   - *Rejected*: Requires linear memory allocation, pointers, memory allocators, and garbage collection in WASM. Introduces memory leak risks, out-of-memory traps, breaks SMT constant solving, and increases execution overhead by $>50\times$.
2. **Inline Primitive Multi-Limb Emission**:
   - *Rejected*: Emitting inline 64-bit carry chains for 256-bit operations directly in generated code balloons AST lengths to 250–500 tokens, degrading reinforcement learning credit assignment ($\gamma^{25} \gg \gamma^{350}$) and leading to advantage collapse.
3. **Double-Limb 128-bit ($2 \times i64$)**:
   - *Rejected*: Fails at $n=186$ for Fibonacci and leaves $>12\%$ of OEIS sequences uncomputable at 100 terms.

---

## 2. WebAssembly Lowering & Macro-Synthesized Preamble

### Decision
Adopt the **Hybrid Macro-Synthesized Architecture**:
- The autoregressive policy emits high-level macro tokens: `i256.add`, `i256.sub`, `i256.mul_scalar`, `i256.const`, alongside structural loop operations (`loop`, `br_if`, `local.get`, `local.set`).
- The execution engine links candidate programs against a static, pre-verified WebAssembly preamble containing pure value-stack implementations of:
  - `$i256_add`: 52 instructions, 0 heap memory, 55 fuel units
  - `$i256_sub`: 53 instructions, 0 heap memory, 56 fuel units
  - `$mul64_wide`: 60 instructions ($64 \times 64 \to 128$-bit widening multiplication via 32-bit splitting)
  - `$i256_mul_scalar`: 268 instructions, signed two's-complement multi-limb multiplication with sign-mask correction, 271 fuel units

### Rationale
- Keeps synthesized candidate programs extremely compact (**25–40 tokens** for order-2 linear recurrences).
- Guarantees zero linear memory usage (`max_memories(0)`).
- Ensures sub-microsecond execution per term while maintaining strict deterministic fuel accounting.

### Alternatives Considered
- **Linking External Host Imports**:
  - *Rejected*: Crossing the host-guest WASM boundary per arithmetic instruction introduces unacceptable context-switching latency (~200 ns per call vs ~5 ns internal).
- **Macro Expansion Pre-Compilation in Python**:
  - *Rejected*: Expanding macros to raw WASM bytecodes before compilation inflates AST size and obfuscates structural provenance.

---

## 3. Decoupled Three-Tier Symbolic Grounding Pipeline

### Decision
Replace SMT bit-blasting (`QF_BV`) with a three-tier symbolic grounding pipeline:
1. **Tier 1: Mersenne-61 Fast Modular Filter (<0.5 ms)**:
   - Project 256-bit terms into finite field $\mathbb{F}_{2^{61}-1}$ via fast bitwise limb folding: $\text{fold}_{61}(u) = (u \ \& \ (2^{61}-1)) + (u \gg 61)$.
   - Perform Gaussian elimination over $\mathbb{F}_{2^{61}-1}$ to evaluate rank and consistency via Rouché-Capelli theorem.
   - Reject inconsistent or underdetermined skeletons immediately with a $-0.5$ grounding penalty.
2. **Tier 2: Dixon 1-Step Bounded Diophantine Solver (<1.0 ms)**:
   - For linear recurrence skeletons with coefficients $c_i \in [-1000, 1000]$, invert maximal square minor in $\mathbb{F}_{2^{61}-1}$.
   - Reconstruct candidate coefficients $\mathbf{c} \in \mathbb{Z}^k$ and certify exactness via $A \mathbf{c} = \mathbf{b}$ over $\mathbb{Z}$.
3. **Tier 3: Tactical Non-Linear Constraint Solver (Z3 QF_NIA, <240 ms)**:
   - Model multi-precision constants as integers in $\mathbb{Z}$ and constrain unknowns in $[-1000, 1000]$.
   - Execute tactical pipeline: `(then simplify solve-eqs purify-arith (try-for qfnia 240))`.

### Rationale
- **Failure of SMT Bit-Blasting (`QF_BV`)**: A 256-bit multiplier produces 32,896 partial products, generating $\approx 1.96 \times 10^6$ variables and $\approx 1.11 \times 10^7$ CNF clauses for a 20-term recurrence system. CDCL resolution thrashes across carry-save addition trees, causing an **88.6% timeout rate** (>250 ms).
- **Sub-Millisecond Solving**: The Mersenne-61 filter evaluates in $<5\text{ ns}$ per coefficient; Dixon inversion solves linear systems in $<0.5\text{ ms}$; QF_NIA CAD decomposition operates over a bounded $[-1000, 1000]^k$ box without bit-blasting in $15\text{--}45\text{ ms}$.

### Alternatives Considered
- **Fraction-Free Bareiss Elimination (FFLU)**:
  - *Rejected*: Pivot bit-lengths grow linearly with elimination depth, scaling to $>1000$ bits for $k=4$, requiring expensive multi-precision integer divisions.
- **Hermite Normal Form (HNF) over $\mathbb{Z}$**:
  - *Rejected*: Intermediate coefficient explosion causes exponential bit growth, yielding $15\text{--}120\text{ ms}$ latency even for small orders.

---

## 4. Continual Learning & Architectural Migration Strategy

### Decision
Adopt **Strategy C: Progressive Supervised Fine-Tuning (SFT) Bridge Transfer**:
1. **Phase 1: Vocabulary Surgery & Convex Hull Projection**: Expand token embeddings $E_{\text{in}}$ and linear head $W_{\text{out}}$ with new 4-limb tokens, initializing from semantic ancestors.
2. **Phase 2: Embedding Warmup (1,500 Steps)**: Train only newly allocated embedding and head rows on synthetic multi-limb data with encoder and decoder backbones frozen.
3. **Phase 3: Joint SFT Bridge (5,000 Steps)**: Unfreeze decoder backbone and top 2 encoder layers, training with discriminative learning rates ($\text{enc}=10^{-5}, \text{dec}=5\times 10^{-5}, \text{heads}=10^{-4}$).
4. **Transition Gate Evaluation**: Enforce Advantage Collapse Rate (ACR) $\le 15\%$, Grammar Validity $\ge 98.5\%$, and Candidate Pass Rate $\ge 75\%$ before RL authorization.
5. **Phase 4: Reference Policy Re-Anchoring**: Re-anchor $\pi_{\text{ref}} \leftarrow \pi_{\text{SFT\_bridge}}$, initialize $\beta_{\text{KL}} = 0.00$ (annealing to $0.04$), and recalibrate EXP3.S task bandit ($\gamma_{\text{floor}} = 0.25$).
6. **Phase 5: Production GRPO Reinforcement Learning (Run 011)**.

### Rationale
- **Strategy A (Cold Start)**: Discards 10+ hours of mathematical induction learning in the Tri-Stream Encoder, incurring a $3.5\times\text{--}4.5\times$ compute cost penalty to rediscover mathematical invariants.
- **Strategy B (Direct RL Warm-Start)**: Guaranteed **100% advantage collapse**. Scalar policy cannot emit 4-limb syntax, resulting in uniform 0 rewards, zero advantage variance ($\sigma_{\mathcal{R}} = 0$), vanishing gradients, and policy collapse.
- **Strategy C (Progressive SFT Bridge)**: Protects learned encoder representations while smoothly adapting the decoder syntax manifold, ensuring $\sigma_{\mathcal{R}} > 0$ when RL commences.

### Alternatives Considered
- **Direct SFT on Encoder + Decoder without Warmup**:
  - *Rejected*: Unaligned new token embeddings generate massive attention logit spikes that immediately corrupt pre-trained attention heads in early steps.

---

## 5. Vocabulary Surgery & Attention Stabilization

### Decision
1. **Convex Hull Semantic Projection for Input Embeddings ($E_{\text{in}}$)**:
   $$\mathbf{e}_u = \sum_{j \in \mathcal{S}_u} \alpha_j \mathbf{e}_j, \quad \sum \alpha_j = 1, \quad \alpha_j \ge 0$$
   - `result_i64_x4`: $0.50 \cdot \mathbf{e}_{\text{result}} + 0.50 \cdot \mathbf{e}_{\text{i64}}$
   - `$a0`..`$d0`: $0.85 \cdot \mathbf{e}_{\text{base}} + 0.15 \cdot \mathbf{e}_{\text{i64}}$
   - `$a1`..`$d3`: $0.70 \cdot \mathbf{e}_{\text{base}} + 0.15 \cdot \mathbf{e}_{\text{offset}} + 0.15 \cdot \mathbf{e}_{\text{idx}}$
   - `i256.add` / `i256.sub`: $0.80 \cdot \mathbf{e}_{\text{i64.add/sub}} + 0.20 \cdot \mathbf{e}_{\text{call}}$
   - `i256.mul_scalar`: $0.80 \cdot \mathbf{e}_{\text{i64.mul}} + 0.20 \cdot \mathbf{e}_{\text{call}}$
2. **Unembedding Head Norm Calibration ($W_{\text{out}}$)**:
   $$\mathbf{w}_u = \tilde{\mathbf{w}}_u \cdot \frac{\bar{R}}{\|\tilde{\mathbf{w}}_u\|_2}, \quad \bar{R} = \frac{1}{|V_{\text{old}}|} \sum_{k \in V_{\text{old}}} \|\mathbf{w}_k\|_2$$
3. **Logit Soft-Capping in Decoder**:
   $$\text{logits}_{\text{capped}} = C_{\text{cap}} \cdot \tanh\left(\frac{\text{logits}}{C_{\text{cap}}}\right), \quad C_{\text{cap}} = 30.0$$

### Rationale
- Eliminates logit anomalies, gradient divergence, and token suppression/hijacking during vocabulary expansion.

### Alternatives Considered
- **Random Gaussian Initialization**:
  - *Rejected*: Places new tokens outside the pre-trained manifold, causing severe attention spikes.
- **Zero Initialization**:
  - *Rejected*: Produces zero-magnitude key/value projections, destabilizing layer normalization in deep decoders.

---

## 6. Corpus-Informed Curriculum Scaling & Scaffolding Macro-Templates

### Decision
Define four standardized macro-scaffolding templates mirroring dominant `jOEIS` class structures:
1. **Stage 1 (Closed-Form Polynomials, `Sequence0`/`Sequence1`)**: Single-variable evaluation on `local.get $n$, zero loops.
2. **Stage 2 (Bounded Linear Recurrences, `LinearRecurrence`)**: Orders $k=1..4$, sliding window registers with constant coefficient linear combinations.
3. **Stage 3 (Holonomic & D-Finite Sequences, `HolonomicSequence`)**: Polynomial-coefficient recurrences; loop index $i$ cast to 64-bit (`local.get $i i64.extend_i32_u`) and multiplied into accumulator.
4. **Stage 4 (Multiplicative / Divisor Loops, `Jaguar.factor` / `MemoryFunction`)**: Nested control loops with divisor testing (`local.get $n local.get $d i32.rem_u i32.eqz`).

### Rationale
Directly addresses the structural distribution of OEIS sequences uncovered in our census, providing inductive scaffolding that matches mathematical taxonomy.

---

## 7. Handling Integer Signs & Negative Numbers

### Decision
Fully support signed two's-complement multi-limb arithmetic:
- Expand demonstration generation to include negative initial conditions ($c_i \in [-3, 3]$) and alternating sign patterns ($(-1)^n$).
- Ensure the grammar masker permits signed operations (`i256.sub`, `i64.div_s`, `i64.rem_s`, bitwise negation).
- Two's-complement reconstruction in host Python:
  $$U = \sum_{j=0}^3 (\ell_j \bmod 2^{64}) 2^{64j}, \quad Y = \begin{cases} U - 2^{256} & \text{if } U \ge 2^{255} \\ U & \text{if } U < 2^{255} \end{cases}$$

### Rationale
The empirical census showed that **10.2%** of the OEIS corpus contains negative integers. Prior scalar systems frequently failed on alternating recurrences.

---

## 8. Flexible Extrapolation Protocol

### Decision
Adopt a dual-horizon extrapolation protocol:
1. **Benchmark Canaries (`trustworthy_synthesis_v1.json`)**: Exact 120 terms (20 observed + 100 unseen) verified against extended b-files.
2. **Broad OEIS Corpus Evaluation**: Evaluate over $\min(100, N_{\text{avail}} - N_{\text{obs}})$ terms, enforcing an unseen margin of at least $\max(15, 0.4 \times N_{\text{avail}})$.

### Rationale
Prevents false failure signals on legitimate OEIS sequences that have fewer than 120 published terms in standard `stripped.gz` distribution.
