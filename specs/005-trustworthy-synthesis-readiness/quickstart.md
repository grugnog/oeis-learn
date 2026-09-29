# Quickstart: Trustworthy Synthesis Readiness

**Feature**: [spec.md](spec.md)  
**Plan**: [plan.md](plan.md)  
**Branch**: `005-trustworthy-synthesis-readiness`  
**Date**: 2026-09-04

This guide describes post-implementation validation. It intentionally stops before a production-length training run. A run is authorized only when the final readiness report is `AUTHORIZED`.

## 1. Prerequisites

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'

cd crates/oeis_wasm_evaluator
cargo build --release
maturin develop --release
cd ../..
```

The implemented dependency set must include `z3-solver` for nonlinear constant resolution and `jsonschema` for contract validation.

Confirm the native evaluator, strict FP32 runtime, and proof dependencies:

```bash
python - <<'PY'
import mpmath
import sympy
import torch
import z3
import oeis_wasm_evaluator

assert torch.get_default_dtype() == torch.float32
print('Runtime dependencies available')
PY
```

**Expected outcome**: Imports succeed and the native evaluator is available. Qualification must not continue through the fallback evaluator.

## 2. Validate Contracts

```bash
pytest -q \
  tests/contract/test_benchmark_manifest_contract.py \
  tests/contract/test_synthesis_evaluation_contract.py \
  tests/contract/test_readiness_report_contract.py \
  tests/contract/test_experiment_manifest_contract.py \
  tests/contract/test_discovery_report_contract.py
```

**Expected outcome**: Every positive fixture validates, every malformed fixture is rejected, and exact integers round-trip as decimal strings without precision loss.

Contract references:

- [Benchmark manifest](contracts/benchmark-manifest.schema.json)
- [Checkpoint metadata](contracts/checkpoint-metadata.schema.json)
- [Evaluation protocol](contracts/evaluation-protocol.schema.json)
- [Synthesis result](contracts/synthesis-evaluation.schema.json)
- [Readiness report](contracts/readiness-report.schema.json)
- [Experiment manifest](contracts/experiment-manifest.schema.json)
- [Symbolic definitions](contracts/symbolic-definitions.schema.json)
- [Discovery report](contracts/discovery-report.schema.json)
- [CLI behavior](contracts/cli-interface.md)

## 3. Build the Frozen Benchmark

Set `OEIS_SNAPSHOT_DIR` to a versioned local OEIS snapshot containing the stripped catalog and b-files:

```bash
python scripts/build_benchmark_manifest.py \
  --stripped "$OEIS_SNAPSHOT_DIR/stripped" \
  --bfiles "$OEIS_SNAPSHOT_DIR/bfiles" \
  --source-revision "$OEIS_SNAPSHOT_REVISION" \
  --output data/benchmarks/trustworthy_synthesis_v1.json
```

Validate the result:

```bash
pytest -q tests/unit/test_benchmark_manifest.py tests/contract/test_benchmark_manifest_contract.py
```

**Expected outcome**:

- every qualified target contains exactly 20 observed and 100 unseen terms;
- offsets and source digests are present;
- excluded targets have explicit reasons;
- Fibonacci, Lucas, Pell, and powers of two select `i256x4_v1`;
- repeated term or program fingerprints are detected before training.

The current DuckDB files contain at most 50 terms and are not valid substitutes for this step.

## 4. Convert and Validate Run 007

```bash
python -m oeis_learn.cli.main convert-checkpoint \
  --input-checkpoint runs/007_phase4_production_symple/checkpoints/model_epoch_060.pt \
  --config runs/007_phase4_production_symple/config.yaml \
  --output-checkpoint runs/007_phase4_production_symple/checkpoints/model_epoch_060.v2.pt

pytest -q tests/unit/test_checkpoint_loader.py
```

**Expected outcome**: The converted checkpoint records the legacy digest, full model constructors, vocabulary digest, FP32 precision, and strict state-key digest. Missing or altered metadata is rejected before generation.

## 5. Verify Shared Synthesis Behavior

Run a fixed-seed evaluation:

```bash
python -m oeis_learn.cli.main synthesize \
  --checkpoint runs/007_phase4_production_symple/checkpoints/model_epoch_060.v2.pt \
  --benchmark-manifest data/benchmarks/trustworthy_synthesis_v1.json \
  --oeis-id A000290 \
  --candidate-budget 8 \
  --seed 42 \
  --constant-resolution \
  --fuel-per-invocation 10000 \
  --memory-limit-mib 16 \
  --mdl-max 1.20 \
  --output-json reports/readiness/a000290-seed42.json
```

Run the focused parity tests:

```bash
pytest -q \
  tests/unit/test_evaluation_protocol.py \
  tests/unit/test_sampler_determinism.py \
  tests/unit/test_synthesis_pipeline.py \
  tests/integration/test_synthesis_entrypoint_parity.py
```

**Expected outcome**:

- the CLI and benchmark adapter return the same semantic result for the same protocol;
- budgets 1 and 8 are prefixes of budget 16 candidate generation;
- placeholders are either resolved with retained evidence or fail at `CONSTANT_RESOLUTION`;
- every failure has one primary stage;
- exact execution checks 120 outputs and never truncates the unseen horizon;
- assembly success alone is never counted as sequence success.

## 6. Verify Native Sandbox and Result Profiles

```bash
cargo test --manifest-path crates/oeis_wasm_evaluator/Cargo.toml
pytest -q \
  tests/unit/test_wasm_sandbox.py \
  tests/unit/test_wide_result_profile.py \
  tests/integration/test_native_resource_limits.py
```

**Expected outcome**:

- scalar and four-limb signed values reconstruct exactly;
- Fibonacci, Lucas, Pell, and $2^n$ values through index 119 do not overflow;
- fuel resets for each `compute(n)` call and never exceeds 10,000 per invocation;
- the 16 MiB limit is observable in result evidence;
- fuel exhaustion and execution traps are distinct classifications;
- no candidate can crash or destabilize the host.

## 7. Prove Readiness Gates Reject Bad Evidence

```bash
pytest -q \
  tests/unit/test_readiness_policy.py \
  tests/integration/test_readiness_failure_injection.py \
  tests/integration/test_readiness_override.py
```

Run preflight without an override:

```bash
python -m oeis_learn.cli.main test-progressive \
  --max-tier 3 \
  --policy configs/readiness_tier1_v1.json \
  --output-report reports/readiness/preflight.json
```

**Expected outcome**:

- zero exact successes fail the single-prompt tier;
- syntax, runtime-trap, competence, coverage, variance, pass-rate, and retention failures remain distinct;
- failed mandatory gates return a nonzero exit and state `BLOCKED`;
- a diagnostic override preserves every failed gate, produces `OVERRIDDEN_UNQUALIFIED`, and cannot update graduation or best-run state.

## 8. Run Paired Inference Ablations

```bash
python -m oeis_learn.cli.main run-ablations \
  --manifest configs/experiments/trustworthy_inference_v1.json \
  --output-directory runs/008_readiness_ablations/reports/experiments
```

**Expected outcome**:

- seeds are exactly `17`, `42`, and `101`;
- constant-resolution on/off variants consume the same cached raw candidates;
- budgets 1, 8, and 16 use ordered prefixes;
- every trial reports exact observed pass rate, 100-term extrapolation pass rate, unique candidates, failure funnel, MDL distribution, solver time, candidate evaluations, and wall time;
- no trial exceeds 4 hours and the experiment is not `COMPLETE` until every pair finishes.

The promotion target is at least a 10 percentage-point exact extrapolation gain for resolved best-of-8 over unresolved single-candidate inference without median MDL exceeding 1.20.

## 9. Run Fixed-versus-Adaptive Training Ablations

First validate orchestration in isolation:

```bash
pytest -q \
  tests/unit/test_symple_curriculum.py \
  tests/unit/test_adag_allocator.py \
  tests/unit/test_elite_buffer_replay.py \
  tests/unit/test_curriculum_orchestrator.py \
  tests/integration/test_ablation_fairness.py
```

Then run the bounded comparison:

```bash
python -m oeis_learn.cli.main run-ablations \
  --manifest configs/experiments/trustworthy_curriculum_v1.json \
  --output-directory runs/008_readiness_ablations/reports/experiments
```

**Expected outcome**:

- every variant receives exactly 32 active rollouts and two replay examples per decision;
- adaptive runs emit task probabilities, group allocations, feedback components, dormancy selections, and retention events;
- evaluations occur at decisions 0, 100, 200, 300, 400, and 500;
- verified-task retention is at least 95%;
- incomplete seed pairs remain `PARTIAL` and cannot support promotion.

## 10. Qualify Recurrence State Transitions

```bash
pytest -q \
  tests/unit/test_recurrence_tracker.py \
  tests/unit/test_synthetic_recurrence_generator.py \
  tests/integration/test_recurrence_qualification.py
```

Evaluate the fixed canaries through the shared pipeline:

```bash
for oeis_id in A000045 A000032 A000129 A000079; do
  python -m oeis_learn.cli.main synthesize \
    --checkpoint checkpoints/recurrence_candidate.v2.pt \
    --benchmark-manifest data/benchmarks/trustworthy_synthesis_v1.json \
    --oeis-id "$oeis_id" \
    --candidate-budget 16 \
    --seed 42 \
    --constant-resolution \
    --max-tokens 192 \
    --output-json "reports/readiness/${oeis_id}.json"
done
```

**Expected outcome**: Each canary and one held-out recurrence has at least one compact, bounded program matching all 20 observed and 100 unseen exact integers. Training/evaluation leakage checks must pass first.

## 11. Validate Discovery Claims

```bash
pytest -q \
  tests/unit/test_relation_identity.py \
  tests/unit/test_numerical_validator.py \
  tests/unit/test_symbolic_prover.py \
  tests/contract/test_discovery_report_contract.py \
  tests/integration/test_discovery_pipeline.py
```

Run checkpoint-backed discovery:

```bash
python -m oeis_learn.cli.main discover \
  --checkpoint runs/007_phase4_production_symple/checkpoints/model_epoch_060.v2.pt \
  --benchmark-manifest data/benchmarks/trustworthy_synthesis_v1.json \
  --protocol configs/discovery_protocol_v1.json \
  --definitions data/benchmarks/symbolic_definitions_v1.json \
  --seed 42 \
  --output-json reports/discovery/discovery-v1.json \
  --output-markdown reports/discovery/discovery-v1.md
```

**Expected outcome**:

- permutations, sign/scalar equivalents, zero coefficients, repeated operands, aliases, and reducible supports do not create claims;
- coefficients freeze before unseen terms are opened;
- a numerical relation without complete definitions remains `NUMERICALLY_VERIFIED_CONJECTURE`;
- only an exact general symbolic reduction reaches `SYMBOLICALLY_PROVEN_IDENTITY`;
- Markdown status counts exactly match authoritative JSON.

## 12. Final Qualification

Run the complete suites and qualification smoke test:

```bash
pytest -q
cargo test --manifest-path crates/oeis_wasm_evaluator/Cargo.toml
python scripts/run_progressive_validation.py \
  --max-tier 3 \
  --policy configs/readiness_tier1_v1.json \
  --qualification-candidates 1000 \
  --output-report reports/readiness/final.json
```

Inspect the machine-readable readiness result:

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('reports/readiness/final.json').read_text())
assert report['qualification_state'] == 'AUTHORIZED'
assert report['overall_passed'] is True
assert all(gate['passed'] for gate in report['gate_results'])
print('Production run authorized by', report['report_id'])
PY
```

**Expected outcome**: All tests pass and every mandatory readiness gate passes without an override. If the assertion fails, retain the evidence and do not launch another production-length run.
