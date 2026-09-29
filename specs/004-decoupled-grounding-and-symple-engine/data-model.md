# Data Model: Decoupled Symbolic-Numeric Grounding, Parsimony-Regularized RLVR & SYMPLE Multi-Task Engine

**Feature**: [specs/004-decoupled-grounding-and-symple-engine/spec.md](specs/004-decoupled-grounding-and-symple-engine/spec.md)  
**Branch**: `004-decoupled-grounding-and-symple-engine`  
**Date**: 2026-09-02

---

## 1. Domain Entities & Schemas

### Entity: `ASTSkeleton`
Represents an ungrounded WebAssembly program structure containing one or more placeholder tokens (`i64.const_?`), including placeholder locations, linearity classifications, and parameter dependencies.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `raw_wat` | `String` | Raw WAT string containing `i64.const_?` tokens | Valid S-expression syntax |
| `placeholder_count` | `int` | Number of placeholders $k$ in the skeleton | $1 \le k \le 4$ |
| `is_linear` | `bool` | Whether all placeholders appear linearly in the trace | Boolean |
| `placeholder_indices` | `List[int]` | Token offset indices of placeholders within `raw_wat` | Non-empty list of unique integers |
| `basis_signatures` | `List[String]` | Extracted basis functions $f_j(n)$ for linear systems | List of valid partial WAT sub-expressions |

---

### Entity: `ConstantSolverResult`
Captures the output of Diophantine or SMT constant solving, containing solver type, concrete integer solution vector $\mathbf{C}^*$, solve duration, and verification status.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `solver_type` | `String` | Solver engine used | One of: `DIOPHANTINE_HNF`, `Z3_SMT`, `FAILED` |
| `constants` | `Optional[List[int]]` | Solved concrete 64-bit integer coefficients $\mathbf{C}^*$ | List of integers in $[-2^{63}, 2^{63}-1]$ or null |
| `solve_duration_ms` | `float` | Elapsed time spent in solver | Non-negative float ($<2.0\,\text{ms}$ Diophantine, $<250.0\,\text{ms}$ SMT) |
| `is_sat` | `bool` | Whether an exact integer solution was found | Boolean |
| `grounded_wat` | `Optional[String]` | Concrete WAT with constants spliced into placeholders | Valid WAT syntax or null |

---

### Entity: `CanonicalProgramArtifact`
Stores the output of the optimizing compiler pass (`wasm-opt`), including raw WAT text, optimized binary $B_{\text{opt}}$, disassembled canonical text $P_{\text{opt}}$, instruction token counts, and Syntactic Waste Ratio $\rho_{\text{waste}}$.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `raw_wat` | `String` | Raw synthesized WebAssembly Text | Non-empty string |
| `opt_wat` | `String` | Disassembled canonical WebAssembly Text post-DCE | Valid WAT syntax |
| `raw_token_count` | `int` | Token count of raw program $|P|_{\text{tokens}}$ | Positive integer |
| `opt_token_count` | `int` | Token count of optimized program $|P_{\text{opt}}|_{\text{tokens}}$ | Positive integer ($\le \text{raw\_token\_count}$) |
| `waste_ratio` | `float` | Syntactic Waste Ratio $\rho_{\text{waste}} = \frac{|P| - |P_{\text{opt}}|}{|P|}$ | Float $\in [0.0, 1.0]$ |
| `passes_applied` | `List[String]` | Compiler passes executed | Subset of `["--vacuum", "--dce", "--remove-unused-locals"]` |
| `is_waste_exceeded` | `bool` | Flag whether $\rho_{\text{waste}} > 0.30$ | Boolean |

---

### Entity: `ParsimonyRewardRecord`
Encapsulates parsimony-adjusted rewards, continuous log-distance return, and lexicographical group ranking.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `dense_return` | `float` | Continuous log-distance return $R_{\text{dense}}(P, Y)$ | Float $\in [0.0, 1.0]$ |
| `covariance_coef` | `float` | Group covariance coefficient $c_k = \frac{\operatorname{Cov}(\ell, R)}{\operatorname{Var}(\ell) + \varepsilon}$ | Finite FP32 float |
| `parsimony_penalty` | `float` | Dynamic length penalty $\max(0, -c_k)(\ell - \ell_{\min})$ | Non-negative float |
| `waste_penalty` | `float` | Penalty for syntactic dead code $\lambda_{\text{waste}} \rho_{\text{waste}}$ | Non-negative float |
| `cpp_reward` | `float` | Net parsimony return $R_{\text{CPP}}$ | Finite FP32 float |
| `lexicographic_rank`| `int` | Ordinal rank in rollout group ($R_{\text{exec}} \succ -|P_{\text{opt}}|$) | Integer $\in [1, G]$ |
| `ordinal_advantage` | `float` | Normalized ordinal advantage $\hat{A}^{\text{lex}} \in [-1, 1]$ | Float $\in [-1.0, 1.0]$ |

---

### Entity: `SYMPLETaskState`
Tracks the curriculum state for a sequence in the 524-benchmark pool, including trailing pass history $W_i$, estimated competence $\hat{p}_i$, competence slope $\Delta C_i$, EXP3.S bandit weight $w_i$, and last visitation timestamp $t_{\text{last\_visit}}$.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `oeis_id` | `String` | Unique sequence identifier (e.g. `A000217`) | Valid OEIS ID pattern `A\d{6}` |
| `pass_history` | `List[int]` | Rolling binary verification results $W_i$ (window=20) | List of 0s and 1s, length $\le 20$ |
| `competence` | `float` | Estimated pass probability $\hat{p}_i = \text{mean}(W_i)$ | Float $\in [0.0, 1.0]$ |
| `competence_slope` | `float` | Score velocity $\Delta C_i = \text{mean}(W_i^{\text{late}}) - \text{mean}(W_i^{\text{early}})$ | Float $\in [-1.0, 1.0]$ |
| `bandit_weight` | `float` | EXP3.S unnormalized weight $w_i$ | Positive float $> 0$ |
| `selection_prob` | `float` | Active sampling probability $p_i \in \Delta^K$ | Float $\in (0.0, 1.0)$, sums to 1 across pool |
| `last_visited_step` | `int` | Optimization step index when prompt was last sampled | Non-negative integer |
| `dormancy` | `int` | Elapsed steps since last visitation $\Delta t_{\text{dormant}}$ | Non-negative integer |
| `has_elite_solution`| `bool` | Whether sequence has $\ge 1$ verified solution in EDB | Boolean |

---

### Entity: `EliteDemonstrationEntry`
Encapsulates a verified canonical program in the EDB, including sequence ID, canonical WAT code, token length, execution fuel, AST structural hash, and creation timestamp.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `oeis_id` | `String` | Target OEIS sequence identifier | Valid OEIS ID pattern `A\d{6}` |
| `canonical_wat` | `String` | Grounded canonical WebAssembly Text program | Valid, compilable WAT |
| `token_length` | `int` | Token count of the canonical program | Positive integer |
| `fuel_consumed` | `int` | Number of instruction fuel units consumed | Integer $\in [1, 10000]$ |
| `ast_hash` | `String` | SHA-256 hash of canonicalized AST structure | 64-character hex string |
| `discovery_step` | `int` | Training step at which solution was discovered | Non-negative integer |
| `mdl_score` | `float` | Description length score $-\alpha_{\text{len}} |P| - \alpha_{\text{time}} \text{Fuel}$ | Finite float |

---

### Entity: `NormalizedLatentRecord`
Stores $L_2$-normalized continuous embeddings $\hat{z}_i \in \mathbb{R}^{256}$, nearest-neighbor cosine indices, detected relation triples, and PSLQ verification certificates.

| Field | Type | Description | Validation Rules |
| :--- | :--- | :--- | :--- |
| `oeis_id` | `String` | Reference OEIS identifier | Valid OEIS ID pattern `A\d{6}` |
| `raw_embedding` | `List[float]` | Raw latent representation $z_i \in \mathbb{R}^{256}$ | List of 256 FP32 floats |
| `normalized_embedding`| `List[float]` | $L_2$-normalized vector $\hat{z}_i = z_i / (\|z_i\|_2 + 10^{-8})$ | List of 256 FP32 floats, $L_2$ norm $\approx 1.0$ |
| `cluster_id` | `int` | Density cluster index assigned by HDBSCAN | Integer ($\ge -1$) |
| `affine_slope_pred`| `float` | Summary token $\mathbf{z}_{\text{affine}}$ slope prediction $\hat{m}$ | Finite FP32 float |
| `geom_ratio_pred` | `float` | Summary token $\mathbf{z}_{\text{geom}}$ ratio prediction $\hat{r}$ | Finite FP32 float |
| `discovered_relations`| `List[String]` | Formally verified relation equations | Non-empty strings or empty list |

---

## 2. State Lifecycle & Transitions

```
+---------------------------------------------------------------------------------------------------+
|                                 SYMPLE ENTITY LIFECYCLE                                           |
+---------------------------------------------------------------------------------------------------+

           [524 OEIS Sequences in Pool]
                       |
                       v
           +-----------------------+
           |   SYMPLETaskState     |  <--- Updates pass_history, bandit_weight, dormancy
           +-----------------------+
                       |
                       | (EXP3.S Bandit selects B_active = 2 prompts)
                       v
           +-----------------------+
           |     ASTSkeleton       |  <--- Autoregressive decoder emits with 'i64.const_?'
           +-----------------------+
                       |
                       | (Diophantine HNF / Z3 SMT Solver)
                       v
           +-----------------------+
           | ConstantSolverResult  |  <--- Concrete integer constants C*
           +-----------------------+
                       |
                       | (wasm-opt --vacuum --dce --remove-unused-locals)
                       v
           +-------------------------------+
           |   CanonicalProgramArtifact    |  <--- opt_wat, waste_ratio
           +-------------------------------+
                       |
                       | (Covariant Parsimony Pressure & Lexicographic Ranking)
                       v
           +-------------------------------+
           |    ParsimonyRewardRecord      |  <--- R_CPP, A_lex, Virtual Sample Injection
           +-------------------------------+
                       |
                       | (If R_exec == 1.0, ingested into EDB)
                       v
           +-------------------------------+
           |   EliteDemonstrationEntry     |  <--- Top-4 shortest canonical ASTs per sequence
           +-------------------------------+
                       |
                       | (Sampled via dormancy priority for SFT replay)
                       v
           +-------------------------------+
           |     Joint Policy Update       |  <--- L_GRPO + 0.50 L_SFT + 0.10 L_aux - 0.02 D_KL
           +-------------------------------+
```
