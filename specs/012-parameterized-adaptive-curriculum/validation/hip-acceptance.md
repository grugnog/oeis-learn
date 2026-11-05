# Phase E — Actual workstation acceptance

## T068: Original session and independent verification

The operator supplied `report(10).zip` after the instructed acceptance-only command,
using the original prepared/pilot roots and `pac-012-acceptance-004` output. The
[recorded scope](operator-handoff.md) identifies the exact plan and public archive.
This closeout consumes that successful session; no new GPU work was launched.
The invocation below is the command supplied in the conversation, not a claim
that private raw argv was exported inside the ZIP:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --seed 20261016 \
  --output run-state/generalization/pac-012-acceptance-004
```

| Stage | Application | Process | Elapsed seconds | Cleanup |
| --- | --- | --- | ---: | --- |
| Admission | PASS | COMPLETE, exit 0 | 1126.679780 | Acknowledged, within bound |
| Qualification | PASS | COMPLETE, exit 0 | 768.669247 | Acknowledged, within bound |
| Calibration | PASS | COMPLETE, exit 0 | 354.490339 | Acknowledged, within bound |
| Acceptance | PASS | COMPLETE, exit 0 | 351.252565 | Acknowledged, within bound |

All four receipts are fresh (`reused=false`); summed stage elapsed time is
2601.091931 seconds (43.3515 minutes), excluding unreported bootstrap/export time.
The evidence error list is empty. Original terminal status COMPLETE is bound to
its evidence and verification bytes; independent `verify_archive` reproduces
integrity PASS, software PASS, feature qualified and scientific UNMEASURED.
This establishes software acceptance, not a completed scientific comparison.

Qualification contains **1,499 CPU tests and 27 HIP tests**, with zero failures,
errors or skips. It records Podman 5.8.6/crun, rootless, network off, read-only root,
no new privileges/capabilities/runtime socket; FP32; AMD Radeon Graphics;
120,259,084,288 device-visible bytes; Python 3.12.3;
`torch 2.9.1+rocm7.2.1.gitff65f5bc`; HIP `7.2.53211-e1a6bc5663`; driver `6.18.50`.
These are observed labels, not an inferred exact GPU product name.

| Binding | Original value |
| --- | --- |
| Source | `sha256:a5140e5d2be137411d6ae82bf6b6e28ec37ad27a8d71d3a283a4ce1ed814ac5c` |
| Qualification image | `sha256:8c5814b6da8ff9edb4b9f8b01d653b937f92941e1051cbdacab713be20a84c4e` |
| HIP image | `sha256:abfd06f7cafe415cc26bba28ac06f903d7e1a77013d868523fcb6a1c6520cdd4` |
| Execution policy | `sha256:7e8b6150696841d883936a6599c05d80b7055cb6d8c518846c29bc0897f5d71d` |
| Requested and observed helper | `sha256:430aa7c09551926fbcacab0fe1bc0c6b0e3ee4ffa4a113fc9bc547ff081418bd` |
| Qualification | `sha256:164f508c704d028d1581b0e3bfd24b15ddf99ba81d45c30bfd4056c11087d7a4` |
| PAC acceptance | `sha256:cea6bcf72fb2cd968063ba047558e5f76b44fd7051776dea1abe4e4a0c7acab0` |

The source inventory matches the checkout at the start of this closeout. Production
source, tests, locks, contracts and orchestration helper are unchanged by the
Phase E documentation; qualified runtime bytes need not be rebuilt for prose.

## Failure history and smallest retry boundary

The earlier original scope failed completed post-exclusion quotas; its authorized
v2 amendment preserves draws/exclusions but requires new admission. The `-002`
run passed admission, then failed qualification because the protected structural
artifact was omitted from the custodian projection. The `-003` run passed admission
and qualification, then calibration indexed a nonexistent learner-row source field.
Those production fixes required new source-bound plans/images; their earlier
successful stages could not be rebound silently. `-004` is the first supplied
complete acceptance PASS. Earlier failed directories and historical 011 tasks
remain unchanged. Their detail is recorded in the implementation ledger; this
closeout does not pretend those archives were newly re-uploaded or rerun.

The current PASS requires no repair/retry. Compatible unchanged interruptions may
use the existing `resume`; report-only recovery uses `recover` and never launches
training. Missing proof is a gap, not a reason to repeat every successful stage.

## T070: Finite HIP updates, paired targets and loaded checkpoint

The two original training proofs each contain uninterrupted and resumed runs with
16 actual updates, 752 charged target tokens, changed finite weights, and identical
initial-weight identity:
`sha256:94c08692e84ea85dacf272a1ee09a52675bc5e7faf862e05320c84257b0fcee8`.
All final weights are
`sha256:923796ae278ef95fe4842c9925671872836e3ee2fd7c3258f1f2d63d71f83520`.
The uninterrupted final checkpoint is
`sha256:2c2638b18991f2c1d0f99fe73bf104ccba5cfd17d045238cdc6c56cbe9112f8c`.
Proof identity/byte hashes bind the original reports, rather than regenerated
CPU measurements. The handler requires actual device RNG and checks finite changed
parameters through the installed checkpoint reader; qualification additionally
records finite forward/backward and changed parameters on HIP.

A fresh load of that exact final checkpoint passes HIP forward evaluation on two
training probes: 47 tokens, finite loss sum 81.84005737304688, target recall
`[false,false]`. This confirms real checkpoint evaluation, not model accuracy,
semantic success or structural generalization. Failure to recall these probes is
not a software acceptance failure after this tiny diagnostic run.

Production admission retains all 32,768 concrete rows, 32 families and 1,024 rows
per family. Every family reports 1,024 supported paired targets and zero whole-row
fallbacks: 32,768 supported / 0 fallback in the original production pool. This does
not prove every supported row contains a free parameter; parameter-count detail
is not exported. Native/independent grounding occurs before the admission receipt.
The reduced mechanical acceptance deliberately includes both supported and fallback
strata, and its source-bound PASS checks both received committed exposure. Thus the
fallback path was exercised even though production's reported fallback count is zero.

| Complete corpus encoding | Target tokens |
| --- | ---: |
| Literal | 2,543,653 |
| Parameterized | 2,064,384 |
| Reduction | 479,269 (18.84%) |

These are corpus token totals, not learner exposure or speedup. The report's
production exposure fields remain null because no 25M-token B/C arm ran. Calibration
is separately labeled: 64 actual production-shape updates, 122,464 charged tokens,
six valid rewards; its weights cannot initialize research. The assembly uses one
fresh random initializer/recipe/order/pool identity for the declared B/C profiles,
and the production calibration checks that frozen initializer. Actual completed
B-versus-C learner equality is not asserted from an acceptance-only report.

The unchanged inference contract gives compared arms identical eight attempts,
60-second portfolio, twenty visible/100 scoring terms, at most eight fitted
parameters, and final-source validation. Paired-target/solver/role parity is covered
by the source-bound qualification tests. No full B/C structural inference occurred;
empirical raw/fitted success, continuation and wall-time comparisons remain absent,
not fabricated as parity results. Tiny acceptance tensors remain diagnostic only.

## T071: Actual reward/review path and measured cost

The original acceptance receipt reports **six valid rewards** and **two graded
anchors**. The source-bound producer loads actual final controller/policy state,
requires its last observation to be neither warmup, degenerate nor censored, and
requires acquired/exposed anchors from both target strata. The installed controller
records the same pre/post panel around one contiguous family block, completes the
kernel/scaler transaction before its next decision, and commits anchor grades only
after successful optimizer work. These are checked producer attestations, not
fabricated observation rows or a claim that private state was re-read locally.

The full-state proof additionally compares the entire PAC policy and observer,
including block/selection probability, kernel, scaler reservoir/RNG, row/probe
streams and anchor states, excluding elapsed observer seconds. Numeric raw gains,
six scaled rewards and individual anchor grades are not separately exported;
no positive learning gain, semantic mastery or family retention is inferred.

The acceptance fixture has two families, initially uniform. The first five valid
scaler observations are warmup zeros. Graves sharing with observed zero preserves
uniform weights, so **each of its first six block selections has probability 1/2**,
including the first non-warmup opportunity. This probability is independently
*derived* from the source-bound fixture/kernel and recorded valid clock, not an
exported raw selection trace. The six valid updates advance the scaler seen clock
to six, and the sixth kernel update uses alpha=1/7. Source checks reject duplicate
feedback and retain the probability recorded at selection. Production has 32
families and its distinct exploration bounds 1/128 and 97/128; acceptance does not
empirically establish production family exposure or 128-anchor service coverage.

| Clock | Reduced HIP acceptance | Production policy |
| --- | ---: | ---: |
| Family block updates | 2 | 8 |
| Bandit/review cycle updates | 8 | 32 |
| Review interval base updates | 1 | 512 |
| Common-panel cadence updates | 8 | 256 |

Actual HIP acceptance reaches 16 committed updates and both pause boundaries, with
an encoded-token minimum of 1,024; actual consumed tokens are 752. The minimum is
a conservative cap allowing all boundaries, not a claim that all of it was spent.
Production 8/32 clocks, missing-feedback pauses and all review-cycle offsets have
independent installed CPU/qualified regression coverage. They are distinct from
the reduced device proof.

Production-shape calibration separately performs 64 updates, six valid rewards
and a real complete-state save/load. All fifteen required sample categories are
present; one or two samples per category are a limited calibration, not a robust
worst-case performance survey.

| Sample | Count | Largest observed seconds |
| --- | ---: | ---: |
| Short update | 2 | 0.616624449 |
| Long update | 2 | 0.897658407 |
| Cold native | 2 | 0.188891724 |
| Warm native | 2 | 0.006493950 |
| Reward probe | 2 | 0.029068204 |
| Anchor grade | 2 | 0.015180335 |
| Common panel | 1 | 1.938198486 |
| Semantic attempt | 2 | 0.368159817 |
| Complete checkpoint save | 1 | 2.834473869 |
| Complete checkpoint load | 1 | 2.041430625 |

The remaining early/later shard, RNG restore, publication and cleanup samples are
present in the original calibration record. These timings are measurements of
those individual callbacks, not all-inclusive startup or cold-shard throughput;
the already-loaded one-shard row lookup is warm. Do not interpret its microseconds
as initial corpus loading speed.

Both B/C forecasts use a conservative 26,940-update bound: update work
24,182.917491 seconds plus diagnostics 17,724.305655 seconds, or **11.640895 hours
per arm**. The diagnostic projection charges the full five-second semantic budget
even though the two sampled attempts finished sooner. It reserves 53,176.793361
diagnostic seconds, with a 239,284.008460-second (**66.467780-hour**) learner ceiling.
The projection is a planning bound from short samples and scientific budgets;
the ceiling is not an ETA. Preparation/generation forecasts remain explicit gaps.
No wall limit or scientific budget was silently enlarged during research; no
research arm ran. The observed production calibration and reduced correctness
acceptance serve separate purposes.

## T072: Verified clean resume/export; actual HIP crash debit BLOCKED

Both original pause proofs report `resume_equal=true`, `pac_equal=true`, matching
initial/final weight identities and sixteen committed updates/752 tokens. The
installed proof compares model tensors, optimizer tensors, host/device RNG, data,
replay and token ledger exactly, plus all PAC kernel/scaler/review/observer state.
Only elapsed observation seconds are excluded from deterministic policy comparison;
wall timings are not required to be identical. Device RNG must be present in the
actual HIP acceptance producer. The two pause points are:

| Boundary | Original pending state | Actual comparison |
| --- | --- | --- |
| After update 1 | Inside the two-update family block | Clean resumed state equals uninterrupted state |
| After update 2 | Gradients complete, post-feedback still pending | Clean resumed state equals uninterrupted state |

The recorded no-Lt execution-policy identity matches qualification, calibration
and acceptance source/image/environment bindings. Its source-owned process-start
requirements are `DISABLE_ADDMM_CUDA_LT=1`, `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`ROCBLAS_USE_HIPBLASLT=0`, `ROCBLAS_DEFAULT_ATOMICS_MODE=0`, with accepted cublas
preference. The producer validates them before device work; raw command/environment
logs remain outside this bounded closeout. A changed backend cannot inherit PASS.

The success ZIP independently verifies original terminal/evidence/verifier/member
hashes. Its original 869-byte `acceptance/failure.zip` also independently verifies:
summary FAIL, exactly the mandatory `summary.json` and `failure.json`, and no private
members. It was made after actual updates with primary
`injected_after_actual_updates`. It demonstrates negative-scope packaging and
private-payload exclusion, **not an actual crashed HIP learner**.

The qualified CPU XML includes passing
`test_pac_real_crash_discards_unsaved_work_without_policy_replay`. That test executes
real CPU updates, injects a crash after the third optimizer call, restores the
initial complete generation, verifies positive discarded tokens and bounded
charged exposure, and keeps lost newer policy from being replayed. Its config is
explicitly `device='cpu'`. Neither the 27-test GPU qualification selection nor the
PAC acceptance handler executes the corresponding unsaved-segment HIP crash.
There is no actual HIP discarded-crash receipt in the supplied ZIP.

**T072 therefore remains unchecked**, despite verified clean device resume and
verified success/negative exports. Do not infer HIP crash-debit proof from the old
011 run, the CPU counterexample, the synthetic negative ZIP, or the aggregate
`feature_qualified` flag. The current public verifier's software acceptance scope
is narrower than this task's complete device/failure evidence requirement.

The missing evidence requires an actual HIP crash/recovery exercise with original
compatible input/checkpoint access: fail after uncheckpointed optimizer work,
restore a complete generation, charge positive lost tokens/time, preserve the old
model-bound policy rather than newer events, and export a source/image/environment-
bound receipt. No engine or HIP workstation is available in the implementation
session, and private tensors are intentionally absent from the upload. A re-upload
of the same successful ZIP cannot establish an exercise that never occurred.
A future proof must use canonical installed learner/persistence logic; any needed
production/test-source change requires affected qualification and its own frozen
scope. Preserve the existing PASS for its original closure; do not relabel it FAIL
or automatically rerun all already-verified work.
