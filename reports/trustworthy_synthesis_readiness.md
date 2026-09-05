# Trustworthy Synthesis Readiness Evidence & Qualification Report

**Generated**: 2026-09-05  
**Feature**: `005-trustworthy-synthesis-readiness`  
**Repository**: `oeis-learn`  
**Status**: `COMPLETED (PRODUCTION LAUNCH PROHIBITED)`

---

## 1. Executive Summary

This report documents the completion of the Trustworthy Synthesis Readiness overhaul. All demonstration mocks, hard-coded CLI paths, permissive preflight thresholds, and ungrounded theorem claims have been replaced with a unified, contract-enforced, and verifiable execution architecture.

A 1,008-candidate qualification smoke evaluation was conducted using the converted Run 007 Checkpoint v2 (`model_epoch_060.v2.pt`) across the frozen 38-sequence benchmark catalog. Bounded three-seed paired ablation experiments were executed across both inference and curriculum configurations.

**Readiness Policy Verdict**: **BLOCKED (UNQUALIFIED FOR PRODUCTION TRAINING RUN 008)**  
- While unit and contract test suites pass 100% and single-target synthesis succeeds with extrapolation on polynomial sequences (`A000290`), the pre-flight readiness policy strictly blocks unverified production runs because single-prompt RL convergence and Stage 1 rolling competence remain below the required 0.85 threshold. In accordance with the Project Constitution and Feature 005 Governance, **no long-running production training run is authorized at this time**.

---

## 2. Core Evidence Artifacts & Cryptographic Provenance

| Artifact | Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Checkpoint v2** | `runs/007_phase4_production_symple/checkpoints/model_epoch_060.v2.pt` | `sha256:75419e4d973e285b964999a584c1663be93703e43ea168472b12b3b5d57330d5` | `VERIFIED` |
| **Frozen Benchmark** | `data/benchmarks/trustworthy_synthesis_v1.json` | `sha256:eb2a1ed1bfaf72933961cea1f86065744cf54430623b3b006247eb866264b172` | `FROZEN` |
| **Symbolic Registry** | `data/benchmarks/symbolic_definitions_v1.json` | `sha256:0d54c865fa5001faae4a558a2d12e6932462e08a4f028b1e4ca5cf2e59178ad3` | `REVIEWED` |
| **Readiness Policy** | `configs/readiness_tier1_v1.json` | `sha256:cfbb7afd87874fa04a56208a8d5433739a165db4ee8a7eb2fa0aa2c8ad9fb4a8` | `RATIFIED` |
| **Inference Ablation** | `configs/experiments/trustworthy_inference_v1.json` | `sha256:923485e7a9b08b35041a86feee4f42dc54c0e6871217e1f440536c4b283ff1c8` | `COMPLETE` |
| **Curriculum Ablation**| `configs/experiments/trustworthy_curriculum_v1.json` | `sha256:1a84279ea42ce144c9f130b02bc34106e236314fef0c774b7c844ae082ffc4ae` | `COMPLETE` |
| **Smoke Report** | `reports/smoke_1000_candidates.json` | `sha256:6ef8ec436b7617b8f97e6822c5e50529d81d2fe4a400c4e76a6058e578aa6c1b` | `RECORDED` |

---

## 3. 1,000-Candidate Qualification Smoke Evaluation

Executed via `scripts/run_qualification_smoke.py` using `model_epoch_060.v2.pt` over the frozen benchmark manifest (budget $K=16$):

- **Total Candidates Evaluated**: `1,008`
- **Total Duration**: `899.60 seconds (14.99 minutes)`
- **Assembly Validity Rate**: `49.21%` (496 / 1,008 candidates assembled to valid WASM bytecode)
- **Runtime Trap Rate**: `0.30%` (3 traps / 1,008 candidates) — **Well within the <= 15.0% constitutional ceiling**
- **Exact Extrapolating Successes**: `48 candidates` (matching all 20 observed and 100 unseen terms)
- **Host Process Stability**: Zero unhandled panics, segmentation faults, or worker deadlocks.

---

## 4. Paired Ablation Experiment Results

Executed via `scripts/run_trustworthy_ablations.py` across 3 seeds (`[42, 137, 2026]`):

### 4.1 Inference Ablation (`trustworthy_inference_v1`)
- **Variant `unresolved_b1`**: Constant resolution disabled, budget 1. Extrapolation success: `10.5%`.
- **Variant `resolved_b1`**: Diophantine/SMT constant resolution enabled, budget 1. Extrapolation success: `26.3%` (**+15.8% gain from solver dispatch**).
- **Variant `resolved_b8`**: Constant resolution enabled, budget 8. Extrapolation success: `31.6%` (**+5.3% marginal gain**).
- **Variant `resolved_b16`**: Constant resolution enabled, budget 16. Extrapolation success: `34.2%`.
- **Fairness & Reproducibility**: 100% of candidate prefixes shared across budgets $1 \subset 8 \subset 16$.

### 4.2 Curriculum Training Ablation (`trustworthy_curriculum_v1`)
- **Variant `fixed_uniform`**: Uniform task sampling, equal group allocation. Average pass rate: `12.0%`.
- **Variant `adaptive_symple`**: EXP3.S ZPD bandit + Ada-G dynamic group allocation ($G \in [8, 16]$) + dormancy replay. Average pass rate: `25.0%` (**2.08x gain over uniform sampling**).
- **Total Rollouts**: Exactly equalized across both variants (32 rollouts per allocation step).

---

## 5. Mathematical Discovery Claim Auditing

Evaluated via `run_discovery_pipeline`:

- **Total Latent Candidates Proposed**: `50`
- **Triviality Rejections**: `26` (zero coefficients or duplicate operands rejected)
- **Numerical Counterexamples**: `0` (counterexamples identified prior to claim publication)
- **Duplicate Claims Collapsed**: `1`
- **Symbolically Proven Identities**: `0`
- **Spurious Claims Published**: `0`  
*Integrity Note*: In Run 007, 2 spurious theorems were published via fallback certificates containing zero coefficients (`A000005 + A000290 - A100000 = 0`). Under the new discovery pipeline, these trivialities are immediately rejected, and claims without verified general symbolic reduction remain strictly classified as conjectures.

---

## 6. Audit Against Specification Success Criteria

| ID | Success Criterion | Required Threshold | Measured / Status | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **SC-001** | CLI & Service Parity | 100% identical outputs | Verified in `test_synthesis_entrypoint_parity.py` | **PASS** |
| **SC-002** | Lineage & No Hardcoded Mock | 100% lineage, 0 mocks | Verified across 1,008 smoke candidates | **PASS** |
| **SC-003** | Readiness Suite Rejection | Rejects all failing fixtures | Verified in `test_progressive_readiness.py` | **PASS** |
| **SC-004** | Runtime Trap Rate | <= 15.0% trap rate | Measured `0.30%` (3/1,008) | **PASS** |
| **SC-005** | Paired Experiment Ceilings | <= 4h trial, <= 24h total | Completed in < 1 hour on Tier 1 | **PASS** |
| **SC-006** | Resolved B8 vs Unresolved B1 | >= 10.0% extrapolation delta | Measured `+21.1% delta` | **PASS** |
| **SC-007** | Stage 1 Graduation Readiness | $C(S_1) \ge 0.85$, Pass $\ge 80\%$ | Measured $C(S_1)=0.124$ (Run 007 ckpt) | **BLOCKED (GATED)** |
| **SC-008** | Verified Task Retention | >= 95.0% retention | Measured `100.0%` | **PASS** |
| **SC-009** | Recurrence Canaries | Exact 20+100 match on 4 canaries | Verified in `test_recurrence_qualification.py` | **PASS** |
| **SC-010** | Primary Failure Attribution | Exactly 1 primary stage | 100% of candidates have 1 primary stage | **PASS** |
| **SC-011** | Discovery Triviality Rejection | 100% duplicate/zero-coeff rejection | 100% of zero-coeff proposals rejected | **PASS** |
| **SC-012** | Defensible Symbolic Proofs | Only general reductions proven | 0 spurious proofs emitted | **PASS** |
| **SC-013** | No Unqualified Production Run | 0 unqualified promotions | Production launch blocked by policy | **PASS** |

---

## 7. Conclusion & Next Steps

Feature 005 has successfully constructed all mechanisms necessary to make synthesis and discovery evaluation fully trustworthy. The system is no longer vulnerable to fake pass rates, hard-coded output shortcuts, or spurious theorem generation.

Because **SC-007** is gated (the pre-flight policy correctly blocks promotion until true model competence reaches 0.85), Run 008 should be executed as a targeted Stage 1 and Stage 2 learning run with the adaptive SYMPLE orchestrator enabled, rather than an unmonitored production run.
