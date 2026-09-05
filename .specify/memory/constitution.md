<!--
Sync Impact Report:
- Version change: Uninitialized → 1.0.0
- List of modified principles:
  - I. Exact Multi-Axis Number Representation & Strict FP32 Precision (Added)
  - II. Provably Sound Grammar-Guided WAT Synthesis (Added)
  - III. Sandboxed Deterministic Execution & Strict Resource Bounding (Added)
  - IV. Workstation-First Feasibility & Tiered Architectural Scaling (Added)
  - V. Rigorous Curriculum Progression & Anti-Memorization Verification (Added)
  - VI. Localized Execution-Guided Credit Assignment & Non-Contrastive Discovery (Added)
- Added sections:
  - Hardware Constraints & Operational Division of Labor
  - Development Workflow, MVP Acceptance Gates & Quality Standards
- Removed sections: None
- Follow-up TODOs: None
-->

# OEIS Learn Constitution

## Core Principles

### I. Exact Multi-Axis Number Representation & Strict FP32 Precision
The neural encoder MUST process integer sequence terms without unconstrained tokenization, continuous float approximations, or out-of-vocabulary truncation. Integer representations MUST use a Tri-Stream Continuous Neural Architecture combining:
1. **Magnitude Stream ($S_1$):** Signed logarithmic transformation $v_i = \text{sign}(x_i) \cdot (1 + \log_{10}(|x_i| + 1))$ projected via MLP.
2. **Modulo-Spectrum Stream ($S_2$):** Continuous sine/cosine Fourier phase embeddings across 100 moduli ($m \in \{2, \dots, 101\}$) to capture periodicity and modular congruences.
3. **Local Difference & $p$-Adic Stream ($S_3$):** Logarithmic first difference ($\Delta x_i$), second difference ($\Delta^2 x_i$), and ordinal embeddings for $p$-adic valuations ($v_p(x_i)$ for $p \le 13$).

Streams MUST be unified using Hierarchical Two-Stage FiLM Fusion ($S_2$ modulates $S_1$ to form $H_{12}$; $S_3$ modulates $H_{12}$ to yield final unified embedding $Z_i$).
All encoder forward passes, backward passes, and intermediate state computations MUST run in strict FP32 precision (no FP16/BF16 mixed precision) to prevent phase function gradient underflow and catastrophic cancellation.

### II. Provably Sound Grammar-Guided WAT Synthesis
Program generation MUST directly target WebAssembly Text (WAT) S-expressions rather than unconstrained source code or arbitrary token sequences. Program generation MUST enforce the following constraints:
- The autoregressive Transformer Decoder MUST condition on latent sequence embeddings $Z$ and be strictly constrained by dynamic Earley-based grammar masking engines (`llguidance` or `XGrammar-2`) operating over byte-level tries.
- Per-token grammar evaluation latency MUST NOT exceed $100\,\mu\text{s}$.
- The grammar MUST be Environment-Indexed to maintain lexical scope tracking of declared local variables and evaluation stack depth, guaranteeing 100% syntactically valid WASM compilation and strict No-Ghost Soundness (zero invalid variable references or uninitialized stack operations).

### III. Sandboxed Deterministic Execution & Strict Resource Bounding
All generated algorithms MUST be compiled and executed within a strictly isolated, deterministic WASM sandbox:
- In-memory translation from WAT to WASM MUST use `wasmtime` or native Rust `wat::parse_str`.
- Every module execution MUST be injected with a non-negotiable fuel budget capped at 10,000 instructions and a strict memory ceiling of 16 MiB linear memory to guarantee prompt termination of infinite loops or unbounded memory allocations.
- WASM evaluations MUST be decoupled from Python's Global Interpreter Lock (GIL) and executed via a native Rust PyO3 extension utilizing Rayon worker pools across CPU cores. Host Python environments MUST never block on or crash from untrusted user/model-generated WAT execution.

### IV. Workstation-First Feasibility & Tiered Architectural Scaling
System components MUST follow a strict two-tier execution model to ensure full local prototyping, testability, and algorithmic validation before scaling to high-compute clusters:
- **Tier 1 (Local Workstation Baseline):** Target hardware is bounded to 4 CPU cores / 8 threads (e.g., Intel Xeon E3-1505M v5), 64 GB DDR4 RAM, and 4 GB VRAM (e.g., NVIDIA Quadro M2000M Maxwell). Model backbones MUST use scaled embedding dimensions ($d = 256$ or $d = 384$), context horizons capped at $N = 10 \dots 20$ terms, GPU micro-batches of 4–8, and 100% of WASM executions offloaded to 8 CPU threads.
- **Tier 2 (High-Performance Scale-Up):** Full dataset expansion (390,000+ OEIS sequences), full hidden dimension ($d = 768$), and multi-GPU cluster training (A100/H100) MUST only be initiated after Tier 1 achieves verified graduation ($>85\%$ pass rate on Curriculum Stage 2).

### V. Rigorous Curriculum Progression & Anti-Memorization Verification
Data ingestion and model training MUST progress through a 5-stage taxonomy-aligned curriculum derived from `jOEIS` and `oeisdata` metadata:
1. *Stage 1 (Primitives & Polynomials):* Closed-form polynomials, linear loops (`easy`, `core`, `nonn`).
2. *Stage 2 (Linear Recurrences & Rational GFs):* Order-$k$ linear recurrences, sliding-window buffers (`core`, `frac`, `cons`, `mult`).
3. *Stage 3 (Holonomic & D-Finite):* P-finite recurrences, lower-triangular sequence arrays (`nice`, `cofr`, `tabl`, `tabf`).
4. *Stage 4 (Combinatorics & Number Theory):* Divisor sums, prime factorizations, digital roots (`hard`, `base`, `eigen`).
5. *Stage 5 (Exhaustive Search & Graph Invariants):* Backtracking searches, dynamic heap allocations, graph algorithms (`hard`, `bref`, `more`).

Stage graduation MUST satisfy: Rolling Task Competence $C(S_k) \ge 0.85$, Coverage Equilibrium $\min(\hat{\rho}_x) \ge 0.50$, and low epoch variance $\mathbb{Var}[C_e(S_k)] \le \varepsilon_{\text{var}}$.
Candidates MUST pass Extrapolation Horizon Testing ($N+K$ terms with $N=20, K=100$) and maintain a Minimum Description Length (MDL) ratio $M_{\text{MDL}} \le 1.2$ relative to sequence Lempel-Ziv complexity to eliminate lookup tables and Lagrange polynomial memorization.

### VI. Localized Execution-Guided Credit Assignment & Non-Contrastive Discovery
Reinforcement learning and mathematical discovery pipelines MUST enforce deterministic attribution and collapse-free representation learning:
- Policy optimization MUST use Execution-Guided Credit Assignment GRPO (EGCA-GRPO) with Asymmetric Prompt Weighting and binary reward ($\pm 1$). The execution trace MUST pinpoint the exact token where sequence output deviates from ground truth $A(n)$, concentrating gradients onto localized error windows to prevent zero-advantage collapse on hard prompts.
- Latent sequence representations MUST be trained using non-contrastive VICReg (Variance-Invariance-Covariance Regularization) over positive sequence transformations (partial sums, first differences, binomial transforms, shift operators).
- Conjectured latent vector relations ($\vec{v}_A + \vec{v}_B \approx \vec{v}_C$) MUST undergo arbitrary-precision validation via `mpmath`, integer relation discovery via the PSLQ algorithm, and symbolic theorem proving via SymPy or SageMath before acceptance.

## Hardware Constraints & Operational Division of Labor

The system enforces strict segregation of computational responsibilities between host CPU and GPU devices:

- **GPU Subsystem Scope:** Dedicated exclusively to neural forward and backward passes (Tri-Stream Encoder, Transformer Decoder, and VICReg projection heads). Tensor allocations MUST fit within 4 GB VRAM in Tier 1 via strict micro-batching (4–8) and gradient accumulation.
- **CPU Subsystem Scope:** Dedicated to data ingestion, feature generation, SQLite/DuckDB index queries, and multithreaded WASM sandbox execution via native Rust Rayon worker pools (8 concurrent worker threads).
- **Zero-Crash Resilience:** Sandboxed WAT execution MUST handle infinite recursion, runtime traps (division by zero, out-of-bounds memory), and fuel exhaustion gracefully without raising unhandled panics or segfaults in the host Python process.

## Development Workflow, MVP Acceptance Gates & Quality Standards

All development, testing, and contribution activities MUST strictly satisfy the following quality gates prior to integration or promotion:

1. **Test-Driven Foundation (TDD):** Every subsystem (Tri-Stream Encoder, WASM parser/runtime, Rayon worker bridge, and grammar maskers) MUST have comprehensive unit tests validating edge cases and numerical bounds before implementation merges.
2. **Data Ingestion Gate:** Successful ingestion and indexing of `joeis` and `oeisdata` into local SQLite/DuckDB databases with validated Stage 1 and Stage 2 subsets.
3. **Encoder Numerical Stability Gate:** The Tri-Stream Encoder MUST process 1,000+ benchmark OEIS sequences spanning values from $-10^6$ to $10^{30}$ in FP32 with 0 NaN, Inf, or gradient underflow/overflow anomalies.
4. **Grammar Masking Soundness Gate:** 100% of WAT code synthesized under `llguidance` / `XGrammar-2` MUST assemble into valid WASM binaries without syntax or environment errors.
5. **Execution Sandboxing & Fuel Trap Gate:** Intentional infinite loops and memory hogs generated in WAT MUST terminate within 10,000 fuel units in $<1\,\text{ms}$ without resource leaks or host instability.
6. **Parallel Execution Throughput Gate:** The native PyO3/Rayon execution engine MUST demonstrate sustained throughput exceeding 500 WASM module evaluations per second across 8 CPU threads on Tier 1 hardware.
7. **Synthesis Benchmark Gate:** The system MUST achieve $\ge 80\%$ pass rate on Curriculum Stage 1 (polynomials) program synthesis within Tier 1 resource limits.

## Governance

This Constitution represents the supreme architectural and technical governance document for the `oeis-learn` project. It supersedes all informal architectural proposals, conflicting code conventions, and ad-hoc practices.

- **Supremacy & Compliance:** All pull requests, subsystem implementations, and architectural specifications MUST be validated against the principles, constraints, and quality gates defined in this document. Any implementation introducing floating-point shortcuts in the encoder, bypassing grammar masking, removing sandbox fuel limits, or violating Tier 1 hardware bounds MUST be rejected.
- **Amendment Procedure:** Amendments to this Constitution require:
  1. A formal written RFC detailing the proposed change and explicit architectural rationale.
  2. Proof of feasibility or benchmark results on the target hardware tiers.
  3. Explicit approval and consensus from project maintainers.
  4. An accompanying migration and backward compatibility plan.
- **Versioning Policy:** This Constitution adheres to Semantic Versioning (`MAJOR.MINOR.PATCH`):
  - `MAJOR`: Fundamental redefinition, breaking changes, or removal of core principles/governance rules.
  - `MINOR`: Addition of new principles, stages, hardware tiers, or significant expansion of architectural guidance.
  - `PATCH`: Non-semantic clarifications, typographical corrections, or wording refinements.
- **Runtime Guidance:** Developers and AI agents MUST consult this Constitution and `.specify/templates/` during every phase of specification, design, task planning, and implementation.

**Version**: 1.0.0 | **Ratified**: 2026-08-30 | **Last Amended**: 2026-08-30
