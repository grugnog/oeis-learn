# Quickstart & Validation Guide: OEIS Learn

**Feature**: [specs/001-oeis-neurosymbolic-synthesis/spec.md](specs/001-oeis-neurosymbolic-synthesis/spec.md)  
**Branch**: `001-oeis-neurosymbolic-synthesis`  
**Date**: 2026-08-30

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **OS:** Linux (Ubuntu 22.04 LTS or newer recommended)
- **CPU:** 4+ Cores / 8+ Threads (e.g., Intel Xeon E3-1505M v5)
- **RAM:** 64 GB DDR4
- **GPU:** NVIDIA Quadro M2000M (4 GB VRAM) with CUDA 12.0+ support
- **Toolchains:** Python 3.11+, Rust 1.75+ (Cargo), DuckDB CLI

### Build & Installation Commands

```bash
# 1. Clone repository and create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python build dependencies and PyTorch (strict FP32)
pip install --upgrade pip setuptools maturin wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install duckdb mpmath sympy llguidance pytest pytest-benchmark pyyaml

# 3. Compile the native PyO3 Rust WASM evaluation extension
cd crates/oeis_wasm_evaluator
maturin develop --release
cd ../..

# 4. Verify native extension is accessible in Python
python3 -c "import oeis_wasm_evaluator; print('WASM Evaluator loaded successfully!')"
```

---

## 2. End-to-End Validation Scenarios

### Validation Scenario 1: Data Ingestion & Indexing
Ingest core Stage 1 & Stage 2 sequences into local DuckDB storage.

```bash
# Ingest OEIS records and classify curriculum stages
python -m oeis_learn.cli.main ingest --subset-stage 1 --db-path data/oeis_learn.duckdb
```

**Expected Outcome:**
- Database table `sequences` populated with $\ge 10,000$ Stage 1 sequence entries.
- Initial terms, tags, and Lempel-Ziv complexity computed without errors.

---

### Validation Scenario 2: Tri-Stream Encoder FP32 Stability Gate
Verify numerical stability of the 3-axis continuous encoder across extreme integer dynamic ranges ($-10^6$ to $10^{30}$).

```bash
pytest tests/unit/test_tri_stream_encoder.py -v
```

**Expected Outcome:**
- 1,000 benchmark sequences pass through $S_1$ (magnitude), $S_2$ (100 moduli), and $S_3$ (differences + $p$-adics) with Hierarchical FiLM fusion.
- Zero `NaN`, `Inf`, or gradient underflow anomalies detected under strict FP32 precision.

---

### Validation Scenario 3: Sandboxed Execution & 10,000 Fuel Traps
Verify in-memory compilation, linear memory bounding, and deterministic fuel trap handling for infinite loops.

```bash
pytest tests/unit/test_wasm_sandbox.py -v
```

**Expected Outcome:**
- Non-terminating loop programs terminate within 10,000 fuel units in $<1\,\text{ms}$ with status `OUT_OF_FUEL`.
- Valid programs compute correct sequence terms with exact fuel accounting.

---

### Validation Scenario 4: Multi-Threaded Batch Throughput Benchmark
Verify GIL-free parallel batch execution across 8 CPU threads.

```bash
pytest tests/integration/test_batch_throughput.py -v -s
```

**Expected Outcome:**
- Evaluates a batch of 1,000 WAT programs in parallel across 8 CPU threads.
- Measured throughput exceeds **500 WASM evaluations per second** on Tier 1 hardware.

---

### Validation Scenario 5: Grammar-Constrained WAT Synthesis & Extrapolation
Synthesize a candidate algorithm for a polynomial sequence (Stage 1), verifying grammar soundness and anti-memorization extrapolation.

```bash
# Synthesize program for Triangular Numbers (A000217: 0, 1, 3, 6, 10, 15, 21...)
python -m oeis_learn.cli.main synthesize \
    --oeis-id A000217 \
    --checkpoint checkpoints/stage1_best.pt \
    --extrapolate 100
```

**Expected Outcome:**
- Generated WAT code compiles in-memory with 0 syntax or scope errors.
- Extrapolation horizon test passes 100/100 unseen future terms ($K=100$).
- Minimum Description Length ratio satisfies $M_{\text{MDL}} \le 1.2$.

---

### Validation Scenario 6: Latent Manifold Clustering & PSLQ Theorem Discovery
Extract latent sequence embeddings, perform cuML UMAP/HDBSCAN clustering, and verify candidate relations via PSLQ and SymPy.

```bash
python -m oeis_learn.cli.main discover \
    --checkpoint checkpoints/vicreg_latent.pt \
    --max-candidates 10 \
    --precision-digits 500
```

**Expected Outcome:**
- Identifies geometric relation candidates satisfying $\|\vec{v}_A + \vec{v}_B - \vec{v}_C\|_2 < \epsilon$.
- PSLQ confirms integer relation vector with confidence ratio drop $<10^{-50}$.
- SymPy generates a symbolic recurrence proof saved to `reports/discovered_proofs.md`.
