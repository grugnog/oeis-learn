# Quickstart: Validation and Bounded Research

Phase 5 code is available in [PR #59](https://github.com/grugnog/oeis-learn/pull/59).
T092 retains its reviewed historical HIP PASS. The new scope-specific preflight
and one diverse32 learning outcome need current-source HIP execution; full
Phase 5 qualification and Phase 6 promotion remain pending. See
[the gate table](validation/phase-5.md) and [budget registration](validation/screen-preflight.md).

## Phase 5 qualification runner

For the targeted cost follow-up, reuse the already checked inputs. This runs
actual HIP regression/parity checks and fixed-state measurements, writes
independent diverse32 and transfer-screen receipts, and launches no learning:

```bash
phase5_preflight="tmp-validation/008-phase5-scoped-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --performance-only \
  --cost-scope diverse32 --config configs/foundation/bigint_smoke_phase5.yaml \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "$phase5_preflight"
```

Share the resulting ZIP, including a BLOCKED outcome. Review the diverse32
receipt and preregistered 14,400-second budget before running the next command.
A passing diverse32 receipt cannot admit the transfer screen.

```bash
python3 scripts/run_bigint_diverse32.py --engine podman \
  --reviewed-preflight "$phase5_preflight/performance/benchmark.json" \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "tmp-validation/008-phase5-diverse32-$(date -u +%Y%m%dT%H%M%SZ)"
```

The second command reuses the reviewed image/costs, checks inputs, claims exactly
one seed20261016 attempt and archives learning/pack results even on failure.
Keep the reviewed directory in `phase5_preflight` if starting another shell.
A negative outcome preserves all thresholds and blocks promotion; an interrupted
attempt must resume its original run/ledger. See [learnability.md](validation/learnability.md).

For full CPU/native/retirement/containment qualification rather than the targeted
follow-up, omit `--performance-only`; use `--prepare` instead of `--inputs PATH`
only when new preparation is necessary. Use a fresh output for measurement
retries. Both Docker and Podman evidence is required for T091. The former
integrated `--learnability` flag now directs users to the separate reviewed step.
Commands and archive details are in [phase-5.md](validation/phase-5.md).

## Phase 2 qualification available now

Run from the new PR checkout on the HIP/ROCm machine. The standard-library host
helper uses the existing Docker/Podman image, explicit GPU-device selection,
resource limits and workspace GPU lock. The image builds through the root
Dockerfile with the existing cached dependency layers and committed locks.
It launches no training or final-cohort evaluation.

```bash
# Build/reuse the current checkout image and run CPU/native algorithm fixtures.
python3 scripts/qualify_bigint_phase2.py --engine podman \
  --output tmp-validation/008-phase2-cpu-001

# Run actual HIP forward/backward, parameter update, device/cache and generation parity.
python3 scripts/qualify_bigint_phase2.py --engine podman --gpu \
  --output tmp-validation/008-phase2-hip-001
```

Use `--engine docker` if that is your configured engine. With multiple GPUs, add
`--render-node /dev/dri/renderD128` using the intended device's render node. An
already-built immutable image can be selected with `--image IMAGE`; use one
containing this PR's source, not the previously merged image. Output directories
must be new; choose a new suffix for a retry.

Each directory contains `host.json` (image/runtime identity), `summary.json`,
`tests.xml`, `stdout.log` and `stderr.log`. A successful HIP run also writes
`gpu/encoder-update.json` with the actual device, HIP version and changed weight
hashes. The helper requires zero failed or skipped tests; CPU fallback is a
failure for `--gpu`. Share those reports/logs after the commands finish. Native
Rust unit tests also run in the PR's separate native CI job; this helper is not
a replacement for that gate. The user has run this helper successfully on HIP; see [the recorded result](validation/gpu.md).

## Prerequisites

Use a clean checkout after implementation Phases 1–4, a working Docker or Podman connection and workspace disk/memory sufficient for the declared recipes. GPU suites require a supported ROCm GPU, correct device permissions and a qualified host driver. No sudo, special project account, broker or SSH is part of these commands. Substitute `docker` for `podman` consistently if preferred.

Run from the repository root. Persistent paths below are workspace-relative. The image pins native/WASM dependencies and exposes readable artifacts to arbitrary non-root users. Do not reuse an interrupted old-contract run name for new-profile training.

## 1. Build and CPU/retirement qualification

```bash
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-cpu --output tmp-validation/bigint-cpu-001
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-retirement --output tmp-validation/bigint-retirement-001
```

Expected: a result summary and per-stage logs; exact factorial0..99, signed boundary/cancellation arithmetic, independent reference agreement, affine/SMT scope outcomes, invalid imports/handles, resource exhaustion/recovery and legacy-profile controls pass. Retirement proves supported calls route to one native fitter and that installed Python fitting implementations/dependencies are absent. Record actual runtime/native solver/toolchain/lock hashes. A failed build is not permission to fall back to Python.

## 2. Actual GPU input/generation checks

```bash
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-gpu --output tmp-validation/bigint-gpu-001
```

Expected: actual ROCm device name, concrete parameter device, changed parameters after backward/optimizer steps, finite large-input features at256/512/1024/4096bits and 4097-bit differences, correct padding and cached/uncached generation parity. CPU fallback is a failure. No claim about sustained GPU utilization follows merely from this check.

If local GPU execution is unavailable to the implementation agent, these are the commands to ask the user to run; report CPU/native evidence separately and leave the HIP gate pending.

## 3. Prepare one frozen generic dataset

```bash
python3 scripts/foundation_run.py --engine podman data fetch \
  --config configs/foundation/bigint_literal_screen.yaml
python3 scripts/foundation_run.py --engine podman prepare \
  --config configs/foundation/bigint_literal_screen.yaml \
  --output tmp-bigint-inputs-001
python3 scripts/foundation_run.py --engine podman check \
  --inputs tmp-bigint-inputs-001
```

Expected: pinned bulk snapshot cache, paired concrete/parameterized generic records, 512train/128parameter-dev/128structural-dev manifest, explicit overlap/resource exclusions and private OEIS scoring payload. Both target views share one admitted behavior dataset. Preparation fails clearly if any requested split cannot be filled; do not reuse the old accept-shortfall shortcut for a qualified comparison.

## 4. Diverse tiny synthesis gate

Use the two-step Phase 5 commands above with the reviewed measured recipe.
The original `bigint_smoke.yaml` remains a historical 7,200-second registration;
`bigint_smoke_phase5.yaml` is the explicit 14,400-second resource revision.
Do not bypass receipt binding through CLI budget overrides. The unchanged gate
requires at least 31/32 independently correct greedy-plus-fitting continuations,
32/32 complete coverage and 32/32 greedy termination, with explicit raw,
shuffled, absent and template results. Negative results retain useful bundles
and block promotion. Passing does not establish held-out structural transfer.

## 5. Three matched literal/parameterized screens

Run only after the preceding gates pass, the resolved experiment contract is reviewed and complete-screen cost preflight fits the frozen budget. Periodic synthesis uses a declared stratified panel; end-of-screen reports cover the complete fixed development sets. Recipes set `evaluation.primary_transfer_metric: continuation_accuracy_100` and `evaluation.final_access: sealed`. Ordinary training finalization does not evaluate or expose the final OEIS cohort. Each run is capped at7,200seconds; six sequential runs can consume up to12hours **combined**, including diagnostics. This is a matched screen, not one automatic full-scale training run.

```bash
set -euo pipefail
for seed in 20261016 20261017 20261018; do
  for arm in literal parameter; do
    python3 scripts/foundation_run.py --engine podman train \
      --profile bigint-bootstrap \
      --config "configs/foundation/bigint_${arm}_screen.yaml" \
      --inputs tmp-bigint-inputs-001 --seed "$seed" \
      --run "runs/bigint-${arm}-${seed}-001"
    python3 scripts/foundation_run.py --engine podman pack \
      --run "runs/bigint-${arm}-${seed}-001" \
      --output "tmp-analysis/bigint-${arm}-${seed}-001.zip"
  done
done
python3 scripts/foundation_run.py --engine podman compare \
  --runs runs/bigint-literal-20261016-001 runs/bigint-parameter-20261016-001 \
         runs/bigint-literal-20261017-001 runs/bigint-parameter-20261017-001 \
         runs/bigint-literal-20261018-001 runs/bigint-parameter-20261018-001 \
  --output tmp-analysis/bigint-representation-001
```

Expected: immutable contracts, all paired counts/coverage and phase costs, selected checkpoint identities, recorded seed/family variation and explicit promote/stop recommendation using [contracts/experiments.md](contracts/experiments.md). Transfer success requires both production and independent agreement at all100 indices for the candidate sealed using visible terms, under the exact-fitting equal-budget portfolio. Visible20-term fit and raw/rescue results are separate metrics. A non-improving screen is a completed experiment, not a reason to discard the seed or retune its scoring denominator.

## 6. Diagnose, resume and share

```bash
python3 scripts/foundation_run.py --engine podman status \
  --run runs/bigint-parameter-20261016-001
python3 scripts/foundation_run.py --engine podman diagnose \
  --run runs/bigint-parameter-20261016-001 \
  --ablate-inputs all --output tmp-analysis/bigint-input-ablation-001
```

For an interrupted eligible run only, use `resume --run runs/bigint-parameter-20261016-001` with the same effective image/config contract. For a changed algorithm/profile/image, create a new explicitly linked run; never edit a budget ledger or delete state to reclaim time.

Share the six bounded run ZIPs plus comparison/ablation summaries. Core artifacts must include run/status/selected checkpoint identity, split/provenance hashes, raw/fitted/conditioning/template outcomes, size/complexity tables, phase timings and omission manifest. Include selected exact/incomplete examples and fitting/reference evidence. Weights are optional separate exports; private final terms are not required for ordinary report analysis.

Assisted and input-retraining studies follow only after selecting a recipe and freezing reviewed demonstration/input-study manifests. `bigint_assisted_screen.yaml` intentionally fails until its foundation checkpoint and reviewed assisted manifest are supplied. Do not invent a demonstration set, enable replay or start a12+hour training arm merely to make the quickstart complete. Longer-run configuration is a subsequent explicit decision based on transfer, coverage and measured hardware throughput.

## 7. Explicit final evaluation after all study decisions

After the intended representation/curriculum/replay/assistance/input studies, freeze their decision manifest and create `tmp-analysis/final-evaluation-plan.json` conforming to `final-evaluation/v1`. It must identify the still-sealed cohort hash, every registered arm/seed and already-selected checkpoint hash, effective recipes/portfolios, fixed denominators/controls and explicit aggregate/per-arm budgets. Missing decisions/budgets or a changed checkpoint make validation fail. This is the Phase 6 final-study task, not part of the six training commands above.

For each arm registered in that immutable plan, use its matching run/config/output. For example, if the following parameter arm is registered:

```bash
python3 scripts/foundation_run.py --engine podman evaluate \
  --run runs/bigint-parameter-20261016-001 --split final --fitting paired \
  --config configs/foundation/bigint_parameter_screen.yaml \
  --final-plan tmp-analysis/final-evaluation-plan.json \
  --output tmp-analysis/bigint-final-parameter-20261016-001
```

Every arm consumes the same final-study ledger subject to its registered sublimit. Exposure is durably recorded before the first final-prompt evaluation; interruption cannot restore freshness or reissue budget. Only the identical plan may finish its remaining arms, and unstarted/failed arms stay in the fixed report. Any later tuning uses a new cohort for a fresh final claim. Share the final plan/decision hashes and result bundles along with the earlier development reports. These remain post-implementation commands.
