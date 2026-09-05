# OEIS Learn: Neuro-Symbolic Synthesis & Mathematical Discovery

`oeis-learn` is a Neuro-Symbolic AI framework designed to learn continuous mathematical representation spaces, perform automated theorem discovery, and synthesize exact generating algorithms in WebAssembly Text (WAT) format for integer sequences from the Online Encyclopedia of Integer Sequences (OEIS).

---

## 🏛️ System Architecture

```mermaid
graph TD
    Seq[OEIS Integer Sequence] --> S1[S1: Signed Log-Magnitude]
    Seq --> S2[S2: 100-Moduli Fourier Phase]
    Seq --> S3[S3: Differences + p-Adic Valuations]
    S1 --> FiLM1[Stage 1 FiLM: S2 modulates S1]
    S2 --> FiLM1
    FiLM1 --> FiLM2[Stage 2 FiLM: S3 modulates H12]
    S3 --> FiLM2
    FiLM2 --> TransEnc[Bidirectional Transformer Encoder (Strict FP32)]
    TransEnc --> LatentZ[Continuous Latent Representation Z]
    
    LatentZ --> WatDec[Transformer Decoder with llguidance Grammar Masker]
    WatDec --> WATCode[Synthesized WebAssembly Text]
    
    WATCode --> SandBox[Native Rust PyO3 Rayon WASM Sandbox (10,000 Fuel Limit)]
    SandBox --> ExecRes[Execution Output & Divergence Trace]
    
    ExecRes --> EGCA[EGCA-GRPO RL Optimizer with Asymmetric Weighting]
    EGCA --> WatDec

    LatentZ --> VICReg[VICReg Self-Supervised Manifold]
    VICReg --> UMAP[UMAP / HDBSCAN Clustering]
    UMAP --> PSLQ[mpmath PSLQ Arbitrary Precision Search (>500 digits)]
    PSLQ --> SymPy[SymPy Machine-Verified Theorems & Proofs]
```

---

## 🚀 Key Features

1. **Tri-Stream Continuous Neural Encoder (Strict FP32)**:
   - $S_1$ Continuous signed logarithmic scalar projection ($v_i = \text{sign}(x_i) \cdot (1 + \log_{10}(|x_i| + 1))$)
   - $S_2$ 200-dimensional trigonometric Fourier phase embeddings across 100 base moduli ($m \in [2, 101]$)
   - $S_3$ Finite differences ($\Delta x_i, \Delta^2 x_i$) and 6-prime $p$-adic valuation embeddings ($p \in \{2, 3, 5, 7, 11, 13\}$, $k_{\max}=16$)
   - Hierarchical Two-Stage FiLM Fusion in strict single-precision float32.

2. **Provably Sound WAT Synthesis & Environment-Indexed Grammars**:
   - Dynamic Earley trie logit masking with sub-$20\,\mu\text{s}$ per-token latency.
   - Lexical variable scope ($\text{Vars}_t$), stack depth, and parameter `$n` binding tracking enforcing **No-Ghost Soundness** (100% syntactically and semantically valid WASM bytecode).

3. **High-Throughput Sandboxed Execution (`oeis_wasm_evaluator`)**:
   - Native Rust PyO3 extension compiling WAT to WASM in-memory with Cranelift JIT.
   - Exact 10,000 instruction fuel limit trapping infinite loops in $<0.1\,\text{ms}$.
   - 16 MiB linear memory ceiling.
   - Rayon multi-threaded worker pool executing $>5,500$ module evaluations per second releasing the Python GIL.

4. **Anti-Shortcut Regularization & Demonstration Co-Training**:
   - Non-triviality reward gating: empirical output variance $\mathbb{Var}_n[P(n)]$ and input sensitivity $\mathcal{S}_{\text{input}}(P) = \sum |P(n+1) - P(n)|$ zeroing surrogate rewards for constant shortcuts.
   - Batch-level cross-input mutual information proxy ($R_{\text{MI}}$) penalizing identical outputs across distinct prompts.
   - Hybrid co-training blending online policy gradients with teacher-forced SFT demonstration loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \beta_{\text{SFT}}\mathcal{L}_{\text{SFT}}$) and unbiased Schulman per-token KL divergence ($\beta_{\text{KL}}$).

5. **Fine-Grained Execution-Grounded Credit Assignment (EGCA) & PBRS**:
   - Sequence divergence to token span mapping ($k^* \to T_{k^*}$) with downstream token zero-masking ($t > \max T_{k^*}$) and total advantage mass conservation ($\sum a_{i,t} = A_i$).
   - Potential-Based Reward Shaping (PBRS) over AST structural phases and variable bindings ($\phi_{\text{bind}}$) guaranteeing policy invariance.
   - Down-sampled lexicase rollout selection over randomized test cases.

6. **5-Stage Taxonomy Curriculum & Generalization Extrapolation**:
   - Stage 1 (Polynomials) $\rightarrow$ Stage 2 (Linear Recurrences) $\rightarrow$ Stage 3 (Holonomic) $\rightarrow$ Stage 4 (Combinatorics) $\rightarrow$ Stage 5 (Search & Graph Invariants).
   - Extrapolation Horizon ($N+K$, $K=100$) and Minimum Description Length ($M_{\text{MDL}} \le 1.20$) anti-memorization verification.

7. **Self-Supervised Latent Discovery & Theorem Prover**:
   - Non-contrastive Kernel VICReg learning with additive homomorphism loss ($\mathcal{L}_{\text{add}}$).
   - Arbitrary-precision ($>500$ digits) `mpmath` sampling with PSLQ integer relation solver ($<10^{-50}$ confidence drop) and SymPy automated symbolic proofs.

---

## 💻 CLI Usage

```bash
# 1. Ingest OEIS records into local DuckDB storage
python -m oeis_learn.cli.main ingest --subset-stage 1 --db-path data/oeis_learn.duckdb

# 2. Convert legacy checkpoint to strict Checkpoint v2 format
python -m oeis_learn.cli.main convert-checkpoint --input-checkpoint checkpoints/stage1.pt --config configs/train_tier1.yaml --output-checkpoint checkpoints/stage1.v2.pt

# 3. Run Progressive Readiness validation with versioned policy gating
python -m oeis_learn.cli.main test-progressive --policy configs/readiness_tier1_v1.json --output-report reports/readiness_report.json

# 4. Synthesize WebAssembly algorithm with 20-observed and 100-unseen extrapolation verification
python -m oeis_learn.cli.main synthesize --oeis-id A000290 --checkpoint checkpoints/stage1.v2.pt --benchmark-manifest data/benchmarks/trustworthy_synthesis_v1.json --candidate-budget 8 --seed 42 --output-json reports/synthesis_A000290.json

# 5. Run paired experiment ablations (inference or curriculum training)
python -m oeis_learn.cli.main run-ablations --manifest configs/experiments/trustworthy_inference_v1.json --output-directory reports/experiments

# 6. Discover mathematical theorems with exact numerical validation and SymPy reduction
python -m oeis_learn.cli.main discover --checkpoint checkpoints/stage1.v2.pt --benchmark-manifest data/benchmarks/trustworthy_synthesis_v1.json --protocol configs/discovery_protocol_v1.json --definitions data/benchmarks/symbolic_definitions_v1.json --output-json reports/discovery.json
```

---

## 🧪 Running Tests

```bash
# Run all Python tests
pytest -v

# Run Rust WASM evaluator unit tests
cd crates/oeis_wasm_evaluator && cargo test
```
