# Validation and first-pilot guide

## Current next step: read-only recovery evidence

The prospective run `observable-eval-20261104T193226Z` completed current-source
qualification, then failed fresh cohort quotas after original-corpus exclusions.
Neither retained evaluation started. Do not repeat the full run yet: use the
[read-only diagnostic command](validation/prospective-evaluation-recovery-20261104.md)
to recover the missing family shortfalls from that directory. This needs no image
build, containers or HIP. Preserve that run, its qualification and the original
pilot/preparation inputs. T175 fixes reporting; T176 awaits the original ledger;
T174 remains incomplete. No assessment scoring is authorized.

## Historical completed full pilot with hourly snapshots (2026-11-04)

The reviewed `training-hourly-probe-20261104T083935Z` passed current-source
qualification, native preparation, both actual HIP learners and finite changed-weight
proofs. Its conservative forecast is 18,619.51 seconds (5h10m) per 25M-token arm;
the frozen learner cap remains 21,600 seconds (6h). This is a short-run forecast,
not a generalization result or a guarantee for long-run/full-shard throughput.

Use this command on the existing branch. No image rebuild, new qualification,
full corpus regeneration or additional operator probe is requested. Keep the
complete original preparation, hourly probe (including its private canonical
receipts/checkpoints and qualification), imported recipe inputs and qualified images.
At least 224 GiB must be free. Do not modify these directories while running.

```bash
(
  set -e
  cd /home/owen/workspace/oies-learn
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 -S specs/011-controlled-generalization/validation/fresh_pilot.py run \
    --engine podman \
    --rebind-prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --training-probe run-state/generalization/training-hourly-probe-20261104T083935Z \
    --output "run-state/generalization/fresh-p1-hourly-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

The temporary handoff verifies original bytes, qualification source/images/resources
and original probe measurements, then creates a distinct study with an explicit
hourly persistence amendment. Data, model, batch16, seed20261016, 25M tokens/arm,
six learner hours/arm and 4/8/16/25M milestones are unchanged. Static repeats the
first 32,768-row shard; refreshed traverses the 12 prepared shards (393,216 rows).
Both start from the same fresh random initialization; the previous failed learner
is preserved and supplies no weights or refunded exposure to this new study.

One command runs current full-data custodian preflight, the small paired
learning/generation/scoring smoke, fresh serial full learners, selection evaluation
and canonical report verification. Cached native admission remains attributed to
the original preparation; current preflight validates its resolved execution
closure. Canonical consumers independently recheck hardware qualification. A failed
gate blocks later learners and produces diagnostic evidence. Assessment stays sealed.

The short probe suggests roughly 2.3 hours of steady learning per arm; budgeting
5h10m per arm includes a 2× margin, measured snapshot overhead and another 30 minutes.
Full-data validation previously took about 15 minutes per invocation; hashing/copying
and the small smoke add overhead. Full selection generation/scoring throughput is
still unmeasured: each has a six-hour inner ceiling per arm (up to 24 hours combined).
The controller retains its generous 96-hour outer safety ceiling and reporting
reserve. Allow an overnight/multi-day window; those ceilings are not duration forecasts.
Initial/hourly/milestone/final snapshots retain resumability; small progress journals
and bounded loss/throughput trends continue between snapshots. The final JSON names
the `report.zip` or `recovery.zip` to share.

Everything below describes earlier qualification/continuation procedures. The old
`--resume-prepared` command is **not** the next command after this source transition.

All nine generalization commands are registered. Acceptance-016 passed the
pinned CPU/native and actual HIP qualification, all five feature cases and canonical
archive verification. See [the evidence review](validation/acceptance-016-review.md).
The scientific status remains UNMEASURED.

## Historical handoff: continue the prepared 20261103T035534Z run

The CPU/native probe passed. The subsequent campaign passed acceptance, full
source feasibility and all 16 native preparation cells (6675 seconds), then stopped
before smoke assembly because only about 61.2 GiB remained free. No full learner
started. Preserve the complete original run directory, throughput directory,
imported artifacts and both exact qualified images.

Free at least **224 GiB on the workspace filesystem** before continuing: 64 GiB
for remaining research, 64 GiB for the separate smoke, another 64 GiB kept free for canonical
checks, and 32 GiB of headroom.
`df -h .` checks filesystem availability; `podman system df -v` helps identify
container storage usage. Do not blindly prune all unused images: the exact image
pair recorded in the original `acceptance/report/qualified/qualification.json`
is required. No image rebuild is needed for this temporary-helper update.

```bash
(
  set -euo pipefail
  git switch 011-controlled-generalization
  git pull --ff-only origin 011-controlled-generalization
  python3 -u specs/011-controlled-generalization/validation/fresh_pilot.py run \
    --engine podman \
    --resume-prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --output "run-state/generalization/fresh-p1-continued-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

The wrapper verifies the original inventories, qualification/feature evidence,
source and image identities, prepared selections and matching throughput receipt
(still subject to its 72-hour freshness limit). Only the known predecessor host
controller may migrate; measured preparation helpers and installed source must
match exactly. The original failed run remains unchanged. New outputs and a
public continuation record go into the new directory; large input data stays in
its original location. Do not move or alter the old run during continuation.

The corrected small smoke keeps its descriptor/cursor refresh authority with two
2048-draw prefixes and one selected row per family. It runs before the full paired
learners, evaluation and canonical packaging. Failed smoke still blocks learners.
Each full arm retains its original six-hour/25M-token cap; the new invocation has
its own bounded 96-hour campaign envelope because no full learner exposure has
been consumed. A run that already started a full learner is rejected by this
specific continuation; its checkpoint state requires a separate explicit resume.

Read-only `--dry-run` prints the plan but does not verify originals or run hardware.
The final JSON prints `report.zip` or a diagnostic `recovery.zip` to share.

The first continuation (`fresh-p1-continued-20261103T142058Z`) reached small
native preparation and a successful two-update static HIP learner, then correctly
blocked full training at the three-update smoke gate. Its old 3723-token cap came
from a different subset of the parent shard. T151–T152 now source-select the exact
small prefixes before freezing the cap and compare the native-prepared token arrays
before any learner starts. Retry using the command above, still naming the original
`fresh-p1-v2-20261103T035534Z` parent. A new output timestamp preserves this failed
smoke. Full preparation and acceptance are reused again; images need no rebuild.

## CPU/native throughput readiness for a new campaign

Run this CPU-only diagnostic before another full campaign. It needs no imported
weights, GPU image or HIP device. It prepares four full protected cohorts, the full
first training shard, and middle/end samples across all32 families, then checks two
fresh native workers. The container invocation is bounded to 90 minutes; image build
and small host setup/export costs are separate. Failure/timeout also produces a
verified aggregate report ZIP. No training or full campaign starts automatically.

```bash
(
  set -e
  git switch 011-controlled-generalization
  git pull --ff-only origin 011-controlled-generalization
  PILOT_ENGINE="${PILOT_ENGINE:-podman}"
  PROBE_RUN="run-state/generalization/native-throughput-$(date -u +%Y%m%dT%H%M%SZ)"
  "$PILOT_ENGINE" build --target qualification -t oeis-learn:generalization-cpu .
  python3 -u specs/011-controlled-generalization/validation/fresh_pilot.py throughput \
    --engine "$PILOT_ENGINE" \
    --cpu-image oeis-learn:generalization-cpu \
    --output "$PROBE_RUN"
)
```

Share the printed `<PROBE_RUN>/report.zip`. PASS requires complete exact native
agreement and family quotas, restart consistency, acknowledged cleanup, and an
estimate inside the preparation work budget: three times the slowest measured training
cost per draw extrapolated to 12 full shards, three times protected preparation cost,
and two additional hours for unmeasured full-study publication/validation. This is
an engineering margin, not a statistical confidence interval or guarantee for
learning/evaluation. BLOCKED means insufficient timing headroom; FAIL includes
incomplete/timeout/containment/identity evidence. Neither permits a full run.

## Subsequent step: fresh paired P1 pilot

The operator approved a fresh start on 2026-11-01 Pacific. Defer E0 and P0; preserve
the historical artifacts without using them as fresh validation or paired seeds.
P1 uses static versus refreshed mechanical training from identical random weights,
seed20261016, literal targets, fixed model/support/mixture and no replay.

T139–T142 revise the failed v1 design. The temporary runner first checks all four
protected cohorts and all12 training shards on CPU, then performs current-source
acceptance, canonical native preparation, a small same-model paired smoke, both
full learners, matched selection evaluation, reporting and archive verification. Keep sealed
assessment out of adaptive decisions. Acceptance-016 remains original evidence for
ef886a7; the production changes below require fresh acceptance. No fresh HIP or
scientific result is claimed until this command runs on the operator's machine.

After reviewing a PASS throughput report, set `PILOT_THROUGHPUT` to that exact
output directory (the directory, not its ZIP). The original receipt is valid for
72 hours only on matching source, helpers, immutable CPU image, engine/host facts
and resources. It does not replace qualification. Then run the following.
It assumes the already imported files remain at
`run-state/generalization/uniform-import`. Set `PILOT_ENGINE=docker` before the block
if using Docker; otherwise it uses Podman. Both images must be rebuilt from this
checkout, with the existing pinned versions and numerical execution policy.

```bash
(
  set -e
  git switch 011-controlled-generalization
  git pull --ff-only origin 011-controlled-generalization

  PILOT_ENGINE="${PILOT_ENGINE:-podman}"
  : "${PILOT_THROUGHPUT:?Set this to the PASS native-throughput output directory}"
  PILOT_RUN="run-state/generalization/fresh-p1-v2-$(date -u +%Y%m%dT%H%M%SZ)"
  "$PILOT_ENGINE" build --target qualification -t oeis-learn:generalization-cpu .
  "$PILOT_ENGINE" build --target gpu -t oeis-learn:generalization-gpu .
  python3 -u specs/011-controlled-generalization/validation/fresh_pilot.py run \
    --throughput "$PILOT_THROUGHPUT" \
    --engine "$PILOT_ENGINE" \
    --cpu-image oeis-learn:generalization-cpu \
    --gpu-image oeis-learn:generalization-gpu \
    --imported run-state/generalization/uniform-import \
    --output "$PILOT_RUN"
)
```

The final JSON prints the archive to upload: normally `<PILOT_RUN>/report.zip`, or
`<PILOT_RUN>/recovery.zip` if the research report cannot be verified. A nonzero exit
still requires reading that JSON and sharing the archive. Preserve the full private
run and imported files locally; the shareable archive deliberately excludes tensors,
optimizer/RNG state and protected target payloads. Existing outputs are never
overwritten. Failed CPU feasibility starts no hardware. Failed acceptance/native preparation
starts no smoke learner; failed smoke starts no full learner.

This freezes a **96-hour total ceiling after image builds**, not an estimated runtime.
Each learner gets at most6 hours/25M target tokens; each arm's generation and scoring
gets up to6 hours each. Setup, qualification, validation, checkpoint persistence,
cleanup and final reporting have separate reservations. CPU feasibility and the
paired smoke each have a6-hour outer ceiling; native preparation has a 24-hour operational allowance
(with two minutes reserved for cleanup/publication) and a 28-hour enclosing command
for validation/setup headroom. These limits supersede the earlier six-hour
preparation limit; the learner budgets remain unchanged. CPU reference generation uses four ordered
workers; canonical preparation uses up to three plus one native client, with
existing native execute requests batched eight at a time. GPU work stays serial. Qualification and research
use64GiB/four CPUs, with64GiB reserved per study root. The smoke has its own
bounded root and retains diagnostic checkpoints; allow additional disk headroom. Checkpoint/input projections
and image storage need free disk space as well; measured preflight remains mandatory.
Partial learners are evaluated for diagnosis, but unequal/partial exposure cannot
support a matched25M comparison. No assessment scoring or automatic next experiment
is launched. See [the frozen handoff and local evidence](validation/fresh-pilot-readiness.md).

Use `--dry-run` on the final Python command to print reservations without containers
or HIP. Advanced reuse through `--qualified PATH` requires an original current-source
qualified root at the exact64GiB/four-CPU resources; acceptance-016 does not qualify
this changed source. The default above handles that requirement automatically.

For a preparation failure, export the preserved run's original control records before
starting any retry. This command performs no engine or HIP work, leaves the original
run untouched and includes a public projection of the failed cell, quota deficits and
rejection counts. Existing target/source/term payloads remain private:

```bash
python3 -S specs/011-controlled-generalization/validation/fresh_pilot.py diagnose \
  --run run-state/generalization/fresh-p1-20261101T235037Z \
  --output "run-state/generalization/fresh-p1-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)"
```

Share the archive printed by this command. The original v1 failure is now explained:
20,634/32,768 training rows admitted,11 empty families and43,817 cross-role exclusions.
Its three-parameter filter allowed fixed arithmetic segments that collided with
protected20-term windows. Do not rerun that frozen design unchanged.

V2 keeps the exact isolation rules and A4/B6/C2/D20 mixture, screens for numeric
variation, and explicitly widens coefficients from[-16,16] to[-1024,1024] for both
arms. The local full reference check selected393,216 unique-source training rows
and30,525,661 target tokens in about34 minutes. This is CPU/source evidence only;
it does not predict native preparation or HIP throughput. The smoke uses the full
model, two32-row training shards, at least three updates/arm and two selection
prompts. It verifies finite changed weights and fresh-row consumption. Recorded
negative/censored synthesis is allowed; missing results or failed accounting block
full training. Smoke weights and scores never select or initialize research.

### Historical diagnostic: untraced fresh validation with explicit addmm bypass

Processes-001 reproduced native Linear variation across fresh GPU processes with
identical operands. Original and earlier rocBLAS settings still logged hipBLASLt
calls and varying solution indices. All three explicit no-Lt processes agreed,
including every formulation. Test that specific bypass in the full learner path:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/probe_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --start fresh \
    --policy no-lt \
    --output run-state/generalization/resume-no-lt-fresh-001
)
```

Upload `run-state/generalization/resume-no-lt-fresh-001.zip`, pass or fail. Keep
`diagnostic-001` intact. Nine updates across two independent fresh uninterrupted
runs, a fresh update1 pause, and its continuation. Every learner call requests
`DISABLE_ADDMM_CUDA_LT=1`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, the previous rocBLAS
handle defaults and accepted hipBLAS preference before BLAS initialization. This
is the explicit bypass tested in processes-001, not merely a library preference.
Bounded hipBLASLt logging is enabled for diagnostic provenance.

No tensor observers or readbacks are installed during training; CPU comparison
runs after learning. Canonical compatibility still gates the original image/data,
not qualification of the experimental numerical configuration. `repeat_equal` and
`split_equal` must both be true in exact comparison before considering production
promotion. Historical endpoint hashes may legitimately differ under another
numerical path; they are context, not a replacement criterion. One clean operand
experiment does not establish full training reproducibility. No rebuild, broad test
suite, search or evaluation is needed. Stage ceilings remain67 minutes, measured
fresh-matrix times so far roughly3–5 minutes. T064–T067 remain open.

### Completed diagnostic: saved operands across fresh processes

Linear-001's isolated loops were stable: native/flattened Linear match the captured
output, while separate-bias formulations round differently but are stable. The full
controls still differ, and capture changes a traced endpoint. The next test uses
the already saved operands, with zero model loading or training updates:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/replay_linear.py \
    --engine podman \
    --run run-state/generalization/resume-linear-001 \
    --output run-state/generalization/linear-processes-001
)
```

Upload `run-state/generalization/linear-processes-001.zip`, pass or fail. Keep
`resume-linear-001` intact: it supplies the trusted private operand capture. The
runner verifies the original/copy hash before loading, inspects the immutable
learner image and mounts only helper code, a copied operand directory and private
output. It launches nine GPU processes serially, interleaving three repeats each
of original settings, the earlier rocBLAS defaults, and an explicit addmm-Lt bypass
request (`DISABLE_ADDMM_CUDA_LT=1`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, plus the
rocBLAS defaults). These are diagnostic configurations, not qualifying receipts.

Each process runs the same64 operand-level evaluations. Public results compare
variant output hashes across processes and report within-process/layout stability,
actual versions/GPU architecture, requested flags and BLAS preference. Bounded
hipBLASLt logging is enabled; absence of log lines alone is not proof of routing.
The bypass arm tests an upstream hypothesis; that upstream report ultimately had
a different embedding-backward cause. No library cause is assumed here.

All original tensor contents remain private and are excluded from the archive.
Partial results and bounded process/cleanup logs survive ordinary failures; missing
cleanup acknowledgment or interruption stops new GPU work. PASS means measurement
completion, not reproducibility. The nine process ceilings sum to27 minutes,
plus bounded image/device setup and packaging; actual duration is unmeasured.
No model run, rebuild, broad qualification, search or evaluation is performed.
T064–T067 remain open.

### Completed diagnostic: isolated Linear operand capture

Deferred-001 found the first differing module: `encoder.s1_magnitude.mlp.3`
(`Linear`), with matching input hashes and differing output hashes. The preceding
Linear, GELU and Dropout match. Restored parameter hashes match, but original tensor
layout/alignment was not recorded. Observer effects still occur; this is a concrete
measured boundary, not a proven library defect.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/trace_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --linear-probe \
    --output run-state/generalization/resume-linear-001
)
```

Upload `run-state/generalization/resume-linear-001.zip`, pass or fail. Keep the
new output directory intact as well as both inputs. This retains the same controls
and eight training updates. In the first traced forward only, it captures the target
layer's input, weight, bias and actual output on-device, then probes those fixed
operands after whole-forward completion. Broad module tracing and BLAS overrides
are absent. No rebuild, broad qualification, search or evaluation is required.

Each captured operation gets64 forward evaluations: native Linear, flattened Linear,
explicit matrix multiplication plus separate bias, and Linear without bias plus
separate bias, each with four repeats at four allocation offsets. Reports distinguish
within-offset stability from equality across offsets, match counts against the
actual captured output, and maximum differences versus a CPU FP64 computation.
Different formulations/layouts can legitimately round differently; neither FP64
agreement nor a changed hash supplies a new acceptance tolerance. Original operand
shapes/strides/storage offsets and alignment modulo256, plus logical-value hashes,
are reported. No memory addresses or tensor values are exported.

The captured tensors (at most64 MiB) remain in private run-owned
`linear-operands.pt` files under each traced role output; their hashes accompany
the report. Retaining them allows follow-up isolated operation tests without another
model run. Do not upload those private files as part of the diagnostic archive.
The probe still perturbs timing/allocation, so interpret its endpoint versus its
control, and do not declare a fix from one clean repetition. Stage ceilings remain
67 minutes; operand probe overhead on HIP is unmeasured. T064–T067 remain open.

### Completed diagnostic: deferred module capture

Layers-001 verified, but all202 module events and traced endpoints agree while
controls differ. One traced endpoint also differs from its control. Per-module
CPU readbacks are therefore an intervention in the observed behavior; this run
did not locate the original divergence. The existing helper now supports deferred
module capture, keeping the same controls and schedule:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/trace_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --layers \
    --defer-layers \
    --output run-state/generalization/resume-deferred-001
)
```

Upload `run-state/generalization/resume-deferred-001.zip`, pass or fail. Preserve
both input directories. Eight updates, original numerical policy, no rebuild,
broad qualification suite, search or evaluation. Compatibility is rechecked first.

During the first traced forward, hooks clone observed tensor inputs/outputs on
their existing device. Fingerprints are computed only after the entire original
forward returns. The module hooks perform no CPU readbacks or explicit device
synchronization; existing before/after update fingerprints still synchronize.
Clones preserve values against later mutation. Capture is bounded to1 GiB of
tensor data and100000 nodes per forward,2048 module events, and2 MiB per exported
trace. All capture remains private until converted to hashes. The allocation and
queued-copy work can still affect behavior, so compare controls and traced
endpoints before attributing a first differing module. Missing reproduction is
not a fix, and this probe cannot identify a backend kernel by itself. The original
67-minute sum of stage ceilings remains; deferred capture's HIP overhead is
unmeasured. Feature qualification remains false and T064–T067 remain open.

### Completed diagnostic: synchronized module tracing

Blas-fresh-001 failed exact full/resume equality without tensor observers: fresh A
and B agree, while the resumed endpoint differs in weights/optimizer and matches
the earlier divergent endpoint. All initial comparisons and old/new pause values
match. The combined BLAS configuration is insufficient; do not promote it into
production. Continue with the original numerical policy and more specific forward
observations in the same existing helper:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/trace_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --layers \
    --output run-state/generalization/resume-layers-001
)
```

Upload `run-state/generalization/resume-layers-001.zip`, pass or fail. Keep both
input directories intact. Eight updates across the same four serial continuations:
old/new controls and old/new traced. No candidate BLAS override, rebuild, broad test
suite, search or evaluation. Qualification compatibility is rechecked first.

The first traced forward records up to2048 ordered module events with input/output
hashes and fixed model-module names/types. All input values stay private. Hooks
include kwargs and are removed after forward success or failure. A comparison
reports the first differing module, whether its inputs agree and whether outputs
agree. Different inputs localize an upstream boundary; equal inputs with different
outputs narrow the observed operation. Functional operations between modules and
backend kernels remain unobserved, so do not infer a specific kernel cause.
Traces are capped at2 MiB each and remain mandatory archive members on failure.
They synchronize execution and may alter timing; compare traced endpoints against
the uninstrumented controls. If the mismatch disappears under observation, report
non-reproduction rather than a fix. Stage ceilings remain67 minutes; finer tracing
has unmeasured actual HIP overhead. T064–T067 remain open.

### Rejected candidate: fresh full/resume validation without tracing

Blas-001 produced identical modified old/new continuation traces and final states,
while the original-policy controls still differed. The modified endpoints reproduce
the original uninterrupted endpoint. The next single command tests the candidate
configuration on fresh full and paused/resumed learners without tensor observers:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/probe_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --start fresh \
    --policy rocblas \
    --output run-state/generalization/resume-blas-fresh-001
)
```

Upload `run-state/generalization/resume-blas-fresh-001.zip`, pass or fail. Keep
`diagnostic-001` intact. This is nine updates: two independent complete fresh runs,
a fresh update1 pause, and continuation to completion. All four calls apply the
same candidate BLAS configuration. The bootstrap configures the library and emits
a bounded accepted-preference record; it installs no tensor hashing, gradient hooks
or trace readbacks. CPU comparison runs after training. No rebuild, broad test suite,
search or evaluation is performed. Original compatibility is rechecked first.

`requested_policy=rocblas` and `tensor_tracing=false` accompany metadata/branch
results. Comparison remains exact in weights, optimizer, RNG, data, replay and
ledger, plus the initial/pause comparisons described below. Interpret fresh A/B
and A/split first; comparisons with historical original-policy states remain context,
not acceptance of a changed runtime. A passing matrix supports a production policy
repair, which still requires explicit source/qualification binding and fresh final
acceptance. This diagnostic configuration itself grants no feature qualification.

Stage ceilings remain67 minutes. Earlier matrices took roughly3–5 minutes, but
this exact fresh configuration's duration is unmeasured. `--dry-run` shows the plan
without engine work.

### Completed diagnostic: explicit BLAS diagnostic configuration

Trace-001 reproduced the numerical mismatch. The first observed difference is
forward logits in the first continuation update (overall update2), despite equal
restored input/weights/optimizer/RNG hashes. Both traced endpoints exactly match
their respective controls. The association reverses relative to the earlier matrix:
old now equals the original uninterrupted endpoint and new equals old-resumed.
This is not a deterministic distinction between the old/new checkpoint contents.

The next invocation tests a concrete runtime intervention in the existing helper:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/trace_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --policy rocblas \
    --output run-state/generalization/resume-blas-001
)
```

Upload `run-state/generalization/resume-blas-001.zip`, pass or fail. Four serial
continuations perform eight updates total; retain both input directories. The
controls use the original policy. Only the two traced learner processes request
PyTorch's `cublas` (hipBLAS on ROCm) preference, `ROCBLAS_USE_HIPBLASLT=0` and
`ROCBLAS_DEFAULT_ATOMICS_MODE=0`, before BLAS handles are created. No rebuild,
broad test suite, search or evaluation is needed. The installed code is unchanged.
Canonical compatibility still gates the original inputs/images; it does not
qualify this experimental numerical configuration.

The header records the accepted preference and requested environment. These are
not measurements of every selected kernel or the effective handle atomics mode;
the library API may override the environment. The diagnostic combines backend
selection and handle defaults, so success would identify a candidate mitigation,
not isolate one cause. `intervention_effect` compares controls to modified runs;
`observer_effect` is null because policy and tracing both change. Compare paired
modified endpoints and initial hashes before interpreting any improvement. New
hashes of all named buffers (including non-persistent caches) and module training
modes supplement the earlier pre-update state observations. Exact equality remains
required and feature qualification remains false. A production change requires
fresh qualification and uninterrupted/resumed validation under the chosen policy.

The four-branch stage ceiling remains67 minutes; actual HIP overhead is unmeasured.
Add `--dry-run` to inspect the plan without engine work.

### Completed diagnostic: paired controls and update tracing

Layout-001 confirms equal tensor representations and identical serialized numerical
state. No learner or host production fix occurred between the differing and passing
runs; all intervening changes were diagnostic-only and both use the same immutable
learner image. Stop repeating checkpoint-only inspections. The next single command
runs old/new paused-state continuations without instrumentation, then repeats both
with bounded phase fingerprints. Each continuation performs two updates (eight
updates total), reusing the frozen data/pool/configuration and original qualification.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/trace_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --output run-state/generalization/resume-trace-001
)
```

Upload `run-state/generalization/resume-trace-001.zip`, pass or fail. Keep both
input directories intact. No rebuild, broad qualification suite, new input assembly,
search or evaluation is performed. Canonical compatibility is rechecked first;
all four learner branches then run independently and serially. Ordinary branch
failures do not prevent the others; failed cleanup or interruption stops new work.

The trace records hashes of the input batch, restored weights/optimizer/RNG,
forward logits, raw gradients, clipped gradients/norm and post-update state. It
also records bounded execution flags, versions and selected numerical-runtime
environment settings. No tensor values, training programs or memory addresses are
exported. Readbacks synchronize the device and can change timing: the independent
uninstrumented controls are required to detect that effect. No backend or numerical
policy is changed. Hashing is a diagnostic observation, not qualification.

`comparison.json` contains all available endpoint pairs (including original saved
endpoints), both traces, the first differing observed phase and control-versus-traced
equality for each starting state. Standalone partial traces are mandatory archive
members even if final comparison fails. Missing endpoints are explicit. If controls
or traces no longer reproduce the mismatch, that is non-reproduction, not a repair.
If a phase differs, it narrows the execution boundary; it does not by itself identify
a defective kernel. T064–T067 remain open.

Stage ceilings total67 minutes, including compatibility and CPU comparison. Prior
untraced matrix runs took roughly3–4 minutes; this observer's actual HIP overhead
is unmeasured, so do not assume that duration. `--dry-run` shows the plan without
engine work. The operator HIP invocation is the remaining block.

### Completed diagnostic: CPU inspection of equal-valued pause checkpoints

Fresh-001 finished in176.265 seconds. Both fresh full runs and fresh pause/resume
match the original uninterrupted endpoint exactly. All initial comparisons and the
new/old pause value comparison also match. The old resumed endpoint still differs.
Do not treat this as a diagnosed repair or weaken exact equality. Inspect the saved
pause representations before spending more HIP time:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/inspect_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --matrix run-state/generalization/resume-fresh-001 \
    --output run-state/generalization/resume-layout-001
)
```

Upload `run-state/generalization/resume-layout-001.zip`, pass or fail. Keep both
input directories intact. This loads existing private checkpoints in the original
immutable image with GPU access disabled, read-only inputs and no network. It does
not train, rebuild, search, evaluate or rerun qualification. The external process
allowance is270 seconds plus30 seconds cleanup; that is a ceiling, not an estimate.

`inspection.json.pause_pair` contains the exact six-component value comparison,
tensor layout/storage-alias and dictionary-order comparison, bounded old/new layout
metadata, serialized component hashes, and differences in scheduler/history/pins/
selection. Paths/keys in layout metadata are hashed; tensor contents and addresses
are never exported. Some bookkeeping differences are expected and are not causes
by themselves. Inspection reflects CPU-loaded representations, not historical GPU
allocations or backend state. Input matrix source/images/result/checkpoint bindings
are checked before inspection. Successful inspection remains diagnostic-only.

### Completed diagnostic: fresh-start and first-pause matrix

Matrix-001 completed in219.628 seconds. Both continuations from the saved paused
checkpoint, the extra-restart path and the earlier resumed endpoint match exactly.
Only the original uninterrupted endpoint differs. This does not identify a backend
or restore defect. The next temporary matrix repeats the original fresh-start path:

- A and B independently train from the frozen initializer to completion.
- A third fresh run pauses after update1 and resumes to completion.
- CPU comparisons cover all final endpoints, the new versus old update1 pause,
  and complete initial numerical/RNG/data/replay/ledger states of new and old runs.

Nine updates total; no broad suites, rebuild, search, evaluation or new inputs.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/probe_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --start fresh \
    --output run-state/generalization/resume-fresh-001
)
```

Upload `run-state/generalization/resume-fresh-001.zip`, pass or fail. Keep
`diagnostic-001` intact. The prior matrix output is useful evidence but is not an
input to this command. Canonical qualification compatibility is rechecked. Helpers
and tests are separately hashed; production source and image authority are unchanged.
The frozen model, seed,512-token cap, pool, prepared data and initializer all remain
unchanged. The intentional pause is accepted only at exactly one completed update.

`comparison.json` distinguishes completed measurement from exact equality. Final
pairs remain in `comparisons`; `boundaries` contains full initial and pause state
comparisons with checkpoint references. `branches.json` preserves starting/final
checkpoint IDs, initializer hashes and update counts without exporting private
payloads. All comparisons are exact; scheduler elapsed time and provenance/history
are excluded as in the original comparison. Missing branches are explicit, and
independent branches still run after an ordinary failure. No outcome confers feature
qualification. Equal fresh repeats plus a different split path do not alone locate
the first divergent update; interpret the initial and pause comparisons as well.

Allow roughly5–10 minutes based on the earlier retained-model measurements; this
is an estimate, not a guarantee. The sum of stage ceilings remains67 minutes,
including compatibility/comparison and cleanup reserves. Add `--dry-run` to inspect
the plan without engine work. Full acceptance remains pending after a causal repair.

### Completed diagnostic: same-checkpoint continuation matrix

Inspection-002 found no retained uninterrupted snapshot at the paused update
count. Further CPU-only inspection cannot recover that missing boundary. The
temporary matrix below reuses the saved one-update paused checkpoint in three
paths: A continues to completion; B independently repeats A; the third path
continues to update2, saves, then resumes to completion in another process.
This is six new updates across four learner invocations. It performs no replay
search, model evaluation, new input assembly or broad qualification suite.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/probe_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --output run-state/generalization/resume-matrix-001
)
```

Upload `run-state/generalization/resume-matrix-001.zip`, pass or fail. Keep
diagnostic-001's private checkpoints, resolved study and replay pool intact. No
image rebuild is required: helper changes remain outside the existing source
inventory, while canonical source/image/device compatibility is rechecked first.
All branches use the original learner configuration, seed, batch order, pool,
model and512-token cap. Frozen full-state compatibility is still enforced. The
split's intentional PARTIAL result counts as a completed diagnostic stage only
when the exact update2 boundary is reached.

One final CPU container compares all available endpoint pairs, including both
original endpoints. It reports exact weights/optimizer/RNG/data/replay/ledger
equality and bounded numerical statistics. `repeat_equal=false` demonstrates
divergence between repeated continuations from the same saved state. If repeats
match but `split_equal=false`, the extra restart distinguishes the paths. If both
match, compare the original endpoint pairs before interpreting the earlier failure.
None of these observations alone identifies a backend cause or licenses a tolerance.
PASS means the selected diagnostic executions/comparison completed; feature
qualification remains false, and equality is reported separately.

Independent branches continue after ordinary failures; interruption or missing
cleanup acknowledgment stops new work. The split-rest stage requires split-first.
The comparison preserves available pairs even if a new branch failed. GPU work
remains serial; training uses existing isolated production roles. The outer stage
ceilings total67 minutes, including the compatibility check and final comparison;
this is not a measured duration. Add `--dry-run` for the plan without engine work.

### Prior saved-state inspection on CPU

Inspection-001 established exact initial weights/RNG/data/replay/token-accounting
agreement but different final weights and optimizer moments. The updated inspector
also compares every retained uninterrupted checkpoint at the paused update count,
to distinguish divergence already present before resumption from later divergence.
It reports global tensor statistics even when individual detail rows are omitted.
Do not repeat training to obtain this missing comparison detail. Keep
the original diagnostic-001 directory and its private checkpoints, then run:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 specs/011-controlled-generalization/validation/inspect_resume.py \
    --engine podman \
    --run run-state/generalization/diagnostic-001 \
    --output run-state/generalization/resume-inspection-002
)
```

Upload `run-state/generalization/resume-inspection-002.zip`. This uses the existing
immutable learner image with no GPU devices, no network and read-only mounts of
the three saved learner directories. It loads trusted local full-state artifacts
after hash validation; do not point it at untrusted checkpoints. No build,
qualification suite, model update or evaluation is run. The container has a
five-minute total work/cleanup envelope; this is a ceiling, not a measured duration.

The report separates weights, optimizer, RNG, data cursor/pending batch, replay
and token-ledger equality. Tensor differences report counts and maximum finite
absolute difference, never tensor values. Initial weight comparisons distinguish
initializer differences from later divergence. Wall-time accounting is recorded
but excluded from the equality verdict, matching the canonical proof's scope.
No tolerance is introduced. Report status PASS means inspection completed;
`comparison_equal` is the equality result. Feature qualification remains false.
The pause-boundary subreport compares weights, optimizer and RNG only: a retained
pre-update snapshot may already have prefetched/debited the next batch, so its data,
replay cursor and ledger are not treated as the same stopping point. Missing
matching retained snapshots are explicitly unavailable. A matching boundary does
not prove a particular backend or resume defect; it localizes the divergence.

### General selective diagnostic command

Use the temporary helper below before another complete acceptance session. It
consumes the existing acceptance-015 qualification and prepared inputs, checks
their compatibility, and does not rebuild images or rerun the broad suites.
The helper/tests live under `validation/` and are not in the existing production
source inventory; this is verified against acceptance-015's uploaded source
closure. Their own hashes are recorded in the diagnostic ledger. Production
source, build, image or hardware incompatibility still blocks reuse.

From the existing checkout on `011-controlled-generalization`:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization

  python3 specs/011-controlled-generalization/validation/diagnose_hip.py \
    --engine podman \
    --manifest run-state/generalization/acceptance-inputs-015/generalization-acceptance.json \
    --previous run-state/generalization/acceptance-015 \
    --output run-state/generalization/diagnostic-001 \
    --dry-run

  python3 specs/011-controlled-generalization/validation/diagnose_hip.py \
    --engine podman \
    --manifest run-state/generalization/acceptance-inputs-015/generalization-acceptance.json \
    --previous run-state/generalization/acceptance-015 \
    --output run-state/generalization/diagnostic-001
)
```

Upload `run-state/generalization/diagnostic-001.zip`, even on failure. The parent
directory retains private checkpoints locally; only bounded control evidence and
logs enter the shareable archive. It is explicitly diagnostic-only and cannot
satisfy the canonical feature verifier. Exit0 means all selected diagnostics
passed, not feature qualification. Exit1 includes failed/blocked cases;130 means
interrupted. The dry-run performs host manifest validation, no engine work.

Default order: original qualification compatibility; one real native 100-term
admission/restore probe; actual HIP two-update uninterrupted/paused/resumed learner
comparison; traced reproduction using both original retained checkpoints and
their original prompts/budgets; then canonical refresh, parameter-targets,
failure-packaging, retained-eval and replay-resume cases. Tracing records bounded
queue/start/end/cancel events with hashed job IDs and remaining time. It does not
walk thread stacks, alter deadlines or suppress worker failures. The traced probe
does not publish an initializer, so uninstrumented canonical retained-eval still
runs to provide replay's required dependency. No reduced-model result substitutes
for retained-weight load/persistence or complete replay acceptance.

Ordinary probe/case failures do not stop independent cases. Incompatible original
qualification, interruption or unacknowledged cleanup stops further work. Replay
is BLOCKED if its canonical retained-eval prerequisite fails; selecting replay
automatically includes that prerequisite. A new output root is required each time.

For a faster targeted follow-up, append `--probes retained --cases` to run only the
traced reproduction, or `--probes native resume --cases` for the tiny probes.
`--probes --cases parameter-targets` runs just that canonical case. Previously
passing cases are omitted explicitly; their status is never copied or relabeled.
The previous root remains the original session, not a diagnostic archive.

The full diagnostic selection has230 minutes of stage ceilings, plus bounded
packaging; this is not a runtime prediction. The tiny selection has25 minutes of
stage ceilings (including compatibility), and traced-only25 minutes. Actual
durations will be recorded. Acceptance-015 spent24 minutes on preflight and broad
qualification; those suites are absent from this loop. After the individual pieces
pass, use the complete acceptance command below for final qualification.

## Assemble real inputs first

Reconstruction starts from verified Git commit `6e63421`; the earlier conversational
`dae1469` reference was unsupported and must not be used. The commands below are now
implemented. They perform no HIP work. Place the two supplied ZIPs in the checkout
root, retain the original private weight files, and import the verified uniform arm:

```bash
python3 scripts/generalization_inputs.py import-uniform \
  --bundle study-20261016.zip --metadata uniform-import-metadata.zip \
  --weights tmp-validation/010-followup-004/study-20261016/stages/followup.uniform/train \
  --output run-state/generalization/uniform-import
```

This checks the actual files before importing. It produces `best-checkpoint.json`
and `latest-checkpoint.json` with original provenance. Best is at 17,885,681 tokens /
18,433 updates; latest is at 24,999,970 tokens / 25,764 updates for the uploaded run.
Neither is a full-state resume or a P4 selection-designated initializer.

The imported weights remain nonresumable. The acceptance workflow now evaluates
both on newly frozen selection supervision and publishes a new initializer from
that measured decision. P4 resolves that explicit dependency later in the same
session; an inconclusive selection stops it without substituting weights.

After updating the existing PR branch, build both pinned targets from that exact
checkout, then assemble the reduced inputs automatically:

```bash
docker build --target qualification -t oeis-learn:generalization-cpu .
docker build --target gpu -t oeis-learn:generalization-gpu .
python3 scripts/generalization_inputs.py acceptance \
  --engine docker \
  --imported run-state/generalization/uniform-import \
  --cpu-image oeis-learn:generalization-cpu \
  --gpu-image oeis-learn:generalization-gpu \
  --output run-state/generalization/acceptance-inputs
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/generalization/acceptance-inputs/generalization-acceptance.json \
  --output run-state/generalization/acceptance-001 --dry-run
```

For Podman, substitute `podman` for `docker` in both builds and `--engine` values.
Setup inspects immutable image IDs, verifies the imported checkpoint bytes, records
the source closure and freezes all four studies. It generates bounded generic
mechanical fixtures with finite isolation and a declared acceptance-only replay
diagnostic. It does not run a model or issue native/HIP qualification. Canonical
preflight/prepare still enforce measured storage, native admission and all quotas.
Fresh refresh/target-format cases use the existing smaller 256-wide model; retained
evaluation and P4 use the exact original model. P4's diagnostic proposal is never
reported as a learned discovery. No manual recipe, initializer receipt or session
JSON is required for this reduced acceptance path.

Setup and run destinations must be new. Preserve a failed setup/run directory and
choose a fresh suffix for a justified retry. Do not rebuild images, edit source or
replace frozen inputs between setup and execution. After the dry run succeeds,
start the one normal HIP invocation:

```bash
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/generalization/acceptance-inputs/generalization-acceptance.json \
  --output run-state/generalization/acceptance-001
```

The shareable result is `run-state/generalization/acceptance-001.zip`, including
partial/failure evidence when a stage fails. Both normal and recovery archives
include bounded control diagnostics and `diagnostics/index.json`, so preflight
timeout records and supervision logs need no separate manual export. The index
identifies missing, omitted and truncated files; private inputs/state are excluded.
Runtime validation, learner/generator roles and image inspection automatically
request Podman startup debug logs and emit timestamped Python progress on stderr.
Roles start timestamp-only30-second heartbeats before imports; inspection uses
the same heartbeat. Automatic cross-thread traceback dumps are disabled after
acceptance-009 crashed during a dump. Heartbeats indicate Python liveness only;
a blocked GIL may suppress them, while the external supervisor still enforces
the deadline. Learner phase markers identify input/model/initializer loading,
update start/completion, checkpoint saving, resume/replay restoration and roster
publication. At most128 phase markers plus a terminal marker are emitted per
learner. These diagnostics remain under existing log caps and external deadlines.
An absent Python-start marker alone does not prove where startup stalled; inspect
the accompanying engine log. No separate diagnostic command is required.
To supplement an older archive using only existing run files, without launching
containers or modifying that run, export to a fresh sibling directory:

```bash
PYTHONPATH=src python3 -S -m oeis_learn.pipeline_validation.generalization_diagnostics \
  --run run-state/generalization/acceptance-010 \
  --output run-state/generalization/acceptance-010-diagnostics
```

The resulting sibling ZIP is explicitly diagnostic-only. It includes bounded
terminal case results, public case proofs and report/verifier role failures in
addition to startup/qualification logs. Bounded metadata discovery also captures
nested replay audit, training, evaluation and proof role supervision records,
plus bounded status/reason/error projections from their result files. Other result
fields and private input/state payloads are excluded. Negative role outcomes also
appear on stderr in future runs, even if the process exits0. The index maps original
paths to exported copies and identifies projected, missing, truncated or omitted
records, preserving the original hash for projections. It neither reruns
work nor upgrades the original acceptance status.

After the acceptance-014 small-model persistence repair (T113), use fresh `-015` paths.
This single shell invocation rebuilds both images, freezes inputs, checks the
serial plan, executes qualification and all cases, and packages/verifies results.
Run from the project checkout after pulling `011-controlled-generalization`:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  test ! -e run-state/generalization/acceptance-inputs-015
  test ! -e run-state/generalization/acceptance-015
  test ! -e run-state/generalization/acceptance-015.zip
  podman build --target qualification -t oeis-learn:generalization-cpu .
  podman build --target gpu -t oeis-learn:generalization-gpu .
  python3 scripts/generalization_inputs.py acceptance --engine podman \
    --imported run-state/generalization/uniform-import \
    --cpu-image oeis-learn:generalization-cpu \
    --gpu-image oeis-learn:generalization-gpu \
    --output run-state/generalization/acceptance-inputs-015
  python3 scripts/generalization_hip_serial.py --engine podman \
    --manifest run-state/generalization/acceptance-inputs-015/generalization-acceptance.json \
    --output run-state/generalization/acceptance-015 --dry-run
  python3 scripts/generalization_hip_serial.py --engine podman \
    --manifest run-state/generalization/acceptance-inputs-015/generalization-acceptance.json \
    --output run-state/generalization/acceptance-015
)
```

Share `run-state/generalization/acceptance-015.zip`, including failures. Existing
runs are preserved. T113 also reserves small-model persistence and proof time; the
395-minute (6h35) serial ceiling applies, plus image-build time. No research is started.

A successful command must still have
original real-device evidence; setup, CPU tests and a dry run cannot establish it.
The existing `study` and `session` commands remain available for independently frozen
custom/research inputs using the explicit blueprints.

## Freeze inputs before execution

Use Python's standard library on the host and the pinned Dockerfile/locks for
installed roles. Prebuild `qualification` and `gpu` targets on the operator's engine,
inspect their immutable `sha256:...` IDs, then freeze code and contracts. Do not
change host drivers. `validate --suite bigint-cpu` is CPU/native only;
`pipeline all-ready` may select HIP and is not a CPU-only precheck.

Each executable study has a sealed v3 `study.json` and v2 `runtime.json` beside it. The runtime binds
the resolved recipe, CPU/GPU image pair, role views, storage reservations and original
feature acceptance. References are relative to the study directory; use a dedicated
study input/output tree. Explicit `--runtime FILE` selects a sealed sidecar without
overwriting an earlier one. See [artifact formats](contracts/artifacts.md).
The templates in `configs/foundation/pipeline_studies/` are unresolved recipe
commitments, not ready-to-run study manifests.

Preparation views name frozen cells. `prepare` publishes a new study/runtime with
exact role files; its `result.json` reports `result.manifest` and `result.runtime`.
Use those emitted workspace-relative paths for subsequent commands. Never assume
the resolved study is directly under the command output root.

E0 evaluation must retain actual private weights and exact descriptor identities.
P1 freezes paired static/refresh views with the same family support and fresh variants.
P3 requires independently checked target-pair grounding artifacts. P4 requires a
version 3 study with a verified common retained literal initializer, original
selection/producer/ancestry proofs, paired continued-SFT/replay arms and selection
evaluation. Weights-only artifacts cannot serve as full-state resume checkpoints.

## Four frozen inputs, five acceptance cases

A sealed `generalization-acceptance-session/v1` or `/v2` manifest binds four case entries:
`retained-eval`, `refresh`, `parameter-targets` and `replay-resume`. Each entry has
exact `manifest` and `runtime` ArtifactRefs relative to the session document.
All children use acceptance purpose, the same source closure and exact image pair.
Version 2 permits the explicit retained-evaluation-to-P4 selection dependency. Its
resolution verifies original case, selection and initializer producers; arbitrary
post-preparation model or budget changes remain forbidden.
The failure-packaging case uses the refresh input. The session cannot override
child sidecars with a runtime CLI flag.

Use tiny actual model/data fixtures. Retained evaluation loads an identified
checkpoint. Refresh must consume variants outside the static subset. Parameter
targets must exercise both forms with verified grounding. Replay/resume must
produce at least one independently admitted training alternative and compare
complete resumed state with uninterrupted state. Empty replay yield is valid in
research but cannot establish acceptance coverage of an admitted replay update.

Acceptance inputs have at most one seed and 4,096 target tokens. The loader also
bounds each case's work to fit inspection, validation and work inside its reserved
6000-second replay/resume,1500-second retained-eval/refresh/target-format or900-second failure-packaging
reservation. Preflight validates all four inputs and storage before expensive work.

## One consolidated HIP session

First inspect the plan without launching an engine:

```bash
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/generalization/acceptance-inputs/generalization-acceptance.json \
  --output run-state/evidence/generalization-acceptance-001 --dry-run
```

After the operator completes real image/input/preflight gates, the single normal
invocation is the same command without `--dry-run`. It runs preflight, one original
qualification, preparation, five serial cases, report and independent verify-bundle.
The helper forwards the frozen CPU image. Reservations total 23,100 seconds plus
600 seconds of recovery margin within 395 minutes; they do not authorize research.

The helper preserves original receipts, bounded logs and failure status. A verified
complete or partial report is copied to the session sibling ZIP. Report/verifier
failure produces a minimal recovery archive that does not claim qualification.
A successful complete session publishes `report/qualified/`, containing the original
qualification and feature proof closure for subsequent research consumption.
A partial session cannot qualify research. Generic individual-command ZIPs carry
receipts only; they are not feature qualification archives.

Restart into a fresh output root with `--reuse-run OLD_SESSION_ROOT`. Original
case producers and exact hashes must still verify against unchanged inputs.
Preflight and preparation are revalidated with an explicit repeat reason.
Changed source/images/authority invalidate affected evidence. Do not use unchecked
skip/start-step flags. Learner `train --resume FULL_CHECKPOINT` is a separate route:
it verifies the full state, optimizer/RNG/data/replay compatibility and retained pins.
A weights-only checkpoint is rejected. Selection persistence emits a fresh private
store whose verified full-state pointer can be used for later resume.

## E0 → P0 → P1 operator sequence

Research is separately authorized after acceptance. Freeze research-purpose runtime
inputs with the accepted feature receipt and current source/image identities.
The following paths and descriptor IDs are operator-provided inputs, not artifacts
created by implementation. Set `QUALIFICATION_IMAGE` to the inspected immutable
CPU image and `QUALIFIED_ROOT` to the complete session's `report/qualified/`.

E0 extends evaluation without retraining. First assemble a research-purpose
blueprint with `generalization_inputs.py study`; it must reference the original
completed `feature.json`, qualified source/images and actual research inputs.
Preflight does not prepare data. Run `prepare`, then read its returned manifest
and runtime from `result.json` rather than evaluating the unresolved input study.
Select available descriptor identities from the frozen roster; display names and
unavailable IDs are not runnable checkpoint selections. Freeze unavailable entries
in the accompanying measurement plan so they are not silently dropped from claims.

The following is a **template**, not the ready-to-run E0 invocation. `CHECKPOINT_IDS`
is a shell array of the declared available descriptors. `BANK_ID` is the canonical
digest of the eight frozen unique integer seeds in `runtime.evaluation.bank_seeds`.
The explicit `--proposal-bank` option activates the shared absent-input control;
setting seeds in the sidecar alone does not activate it.

```bash
python3 scripts/foundation_run.py --engine podman --image "$QUALIFICATION_IMAGE" generalization preflight \
  --manifest run-state/inputs/E0/study.json --output run-state/evidence/E0-preflight
python3 scripts/foundation_run.py --engine podman --image "$QUALIFICATION_IMAGE" generalization prepare \
  --manifest run-state/inputs/E0/study.json --qualification "$QUALIFIED_ROOT" \
  --output run-state/evidence/E0-prepared
# Read PREPARED_STUDY/PREPARED_RUNTIME from E0-prepared/result.json's result object.
python3 scripts/foundation_run.py --engine podman --image "$QUALIFICATION_IMAGE" generalization evaluate \
  --manifest "$PREPARED_STUDY" --runtime "$PREPARED_RUNTIME" \
  --checkpoints "${CHECKPOINT_IDS[@]}" --proposal-bank "$BANK_ID" \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/E0-evaluate
```

Report/verify-bundle follow only after assembling the explicit public evidence
root consumed by `generalization report`. An evaluate command's sibling ZIP is a
command receipt, not the complete scientific evidence export. See the readiness
review for the required orchestration and bounded failure packaging.

Retain unavailable checkpoint status rather than substituting weights. Compare
fixed denominators, raw/fitted budgets, shared-bank costs and completion. Selection
uses the declared candidate set. Assessment consumes a frozen authorization and
durable exposure debit; it does not rank new checkpoints.

P0 adds only uniform seeds through the opt-in historical route, preserving its
recipe and selection semantics:

```bash
python3 scripts/foundation_run.py --engine docker pilot-followup \
  --run run-state/runs/historical --admission run-state/inputs/admission.json \
  --inputs run-state/inputs/historical --arms uniform --tokens 25000000 --seed 20261017 \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/P0-seed17
```

Repeat serially for seed20261018 in a fresh root. Include seed16 only with compatible
identified evidence; missing evidence remains missing or requires separately
registered regeneration. Do not silently add historical balancing/decay arms.

P1 is the first paired learning intervention:

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization prepare \
  --manifest run-state/inputs/P1/study.json --qualification "$QUALIFIED_ROOT" \
  --output run-state/evidence/P1-prepared
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization train \
  --manifest "$PREPARED_STUDY" --runtime "$PREPARED_RUNTIME" --arms static refreshed --seeds 20261016 \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/P1-train
```

Set `PREPARED_STUDY` and `PREPARED_RUNTIME` from the prepare result. Compare fixed
exposures and selection-supervision results; freeze checkpoint/portfolio choices
before isolated assessment. Advance only under the registered completion rule.
If screen assessment influenced advancement or any recipe/incumbent choice,
confirmation uses fresh assessment. P2/P3/P4 have separately frozen studies;
they are not automatic follow-on jobs or acceptance-helper actions.

## Report and independently verify

Research `report` consumes **one explicit aggregate public evidence directory**
containing `evidence.json` and its exact inventory. Assemble observations and missing
statuses according to the evidence contract. Do not pass multiple raw stage roots
or infer scientific measurements from private checkpoint files. Acceptance-session
reporting assembles its original case proof closures automatically.

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization report \
  --manifest "$PREPARED_STUDY" --inputs run-state/evidence/P1-public \
  --output run-state/evidence/P1-report
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization verify-bundle \
  --archive run-state/evidence/P1-report.zip --output run-state/evidence/P1-verified
```

Every output and archive path must be new. Preserve actual costs, every registered
seed/cell or explicit missing status, and independently reconcilable bounds.
A successful software receipt does not establish improvement. Private learner
states, sealed truth and checkpoint bytes remain outside sharing archives.

See [local evidence](validation/local-checks.md), [algorithm review](validation/algorithm-review.md),
[hardware status](validation/hip-acceptance.md), [coverage audit](checklists/implementation.md)
and [CLI contract](contracts/cli.md) for implementation and validation boundaries.

Qualification reserves 120 minutes, including 35 minutes for the CPU/native suite.
Preserve earlier frozen inputs and run directories; rebuild images and assemble
fresh numbered inputs/session after source changes. Current timing is below.


### Current acceptance timing (T113)

Fresh acceptance inputs freeze a395-minute (6h35) outer ceiling, including a100-minute replay/resume case and25-minute retained-eval, refresh and target-format cases. Replay learner envelopes remain600 seconds; small-model learners and all full-state proofs get180 seconds. Retained evaluation gets720 seconds total; each post-training policy evaluation gets1200 seconds. This supersedes older timing instructions; it is not an expected runtime. Reassemble inputs after rebuilding the changed source closure. No additional manual diagnostic invocation is required.


### Retry after the verified paired smoke (T153–T154)

The144804Z report proves paired smoke success, followed by a pre-learner shard
storage breach. The wrapper now creates an exact, bounded execution view while
preserving the larger preparation audit tree. Full outputs are under `execution/`;
`execution-view.json` records original/projected manifest hashes and peak storage.
No image rebuild, data regeneration or repeat of original acceptance is required.
The smoke repeats (previously about8 minutes), then the two unchanged learners
have up to6 hours each, followed by frozen generation/scoring. The96-hour campaign
ceiling is a safety envelope, not an ETA; full training throughput is unmeasured.
The original throughput proof must remain under72 hours old and compatible.

From the repository root:

```bash
(
  set -euo pipefail
  git switch 011-controlled-generalization
  git pull --ff-only origin 011-controlled-generalization

  python3 -u specs/011-controlled-generalization/validation/fresh_pilot.py run \
    --engine podman \
    --resume-prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --output "run-state/generalization/fresh-p1-continued-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

Keep the original prepared parent unchanged. Share the final `report.zip` and
`verified.zip`, or `recovery.zip` if canonical reporting cannot complete. The
readiness note records the exact uploaded report identity and observed failure.


### Inspect saved work after the six-hour learner timeout (T155–T156)

The151303Z continuation launched the static learner and saved checkpoints before
its external deadline. Do not repeat the previous pre-learning command: preserve
this run and obtain its saved exposure/time metadata first. This command uses only
host Python, hashes the latest/pinned checkpoint files, and emits a small ZIP. It
does not use HIP, load tensor/optimizer state, change originals or authorize resume.
Expect disk-reading time for several GiB, rather than another training run.

```bash
(
  set -euo pipefail
  git switch 011-controlled-generalization
  git pull --ff-only origin 011-controlled-generalization

  python3 specs/011-controlled-generalization/validation/fresh_pilot_checkpoint.py \
    --run run-state/generalization/fresh-p1-continued-20261103T151303Z \
    --output "run-state/generalization/checkpoint-audit-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

Share the printed archive path. Diagnostic PASS means the saved files/accounting
verified, not that the failed research completed. Exact committed versus pending
exposure and elapsed learner time determine the safe continuation; finalization
reserves and checkpoint serialization overhead remain follow-up work.

## Reliable checkpoint implementation and training probe (T157–T160, 2026-11-04 UTC)

Supersedes the follow-up-work status above. Checkpoint saves remain durable before
and after **every** update. A debit-only save reuses the immediately preceding
weights/optimizer bytes; RNG, cursor, pending batch, history, pins and persisted
clock are still freshly saved. Every generation remains independently restorable,
including on filesystems without hard links. Writer-local verification reuse is
bounded and invalidated by inode/size/mtime/ctime changes; restore, descriptor export
and cold readers still verify full payload hashes. Publication reserves up to
300 seconds (one fifth of short allowances); process exit has a separate reserve
inside the existing cleanup envelope. These reserves do not add learner budget.
An operation can still overrun a boundary; the external supervisor and durable
pending debit remain the final containment/recovery mechanism.

The old failed research state remains evidence: 2,000,078 committed tokens,
2,001,307 charged tokens, 1,229 pending tokens, 1,612 updates and
21,593.43016040977 persisted learner seconds. Do not reset that clock, promote the
incomplete generation, or rerun the old pre-learning continuation command. New
production/test bytes require new images and qualification; the probe starts a
fresh diagnostic study and never migrates those old weights.

From `/home/owen/workspace/oies-learn`, this **one shell invocation** rebuilds the
images, performs current-source acceptance, revalidates a small paired data sample,
runs both training arms serially, checks finite changed weights, and packages the
metadata. It requires the preserved original preparation and imported recipe, plus
the existing conservative 224 GiB free-space gate. It never starts full research.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  podman build --target qualification -t oeis-learn:generalization-cpu .
  podman build --target gpu -t oeis-learn:generalization-gpu .
  python3 -S specs/011-controlled-generalization/validation/fresh_pilot_training_probe.py \
    --engine podman \
    --prepared-run run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --imported run-state/generalization/uniform-import \
    --cpu-image oeis-learn:generalization-cpu \
    --gpu-image oeis-learn:generalization-gpu \
    --output "run-state/generalization/training-probe-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

To reuse a **matching current-source** qualification on a subsequent probe, supply
`--qualified <probe-root>/acceptance/report/qualified` instead of the image/import
options, always with a fresh output root. Original qualification is verified;
stale source/image/resource authority is rejected, not upgraded or bypassed.

The probe uses the original full model, optimizer, numerical execution policy,
seed 20261016 and batch size 16. It freezes two 256-row shards (eight examples from
each of the same 32 families, original descriptor law and refresh cursors), with
original protected cohorts retained only for canonical exclusion/preparation.
Native preparation must match the frozen source order, encodings and batch token
charges. Static repeats shard one; refreshed traverses both. There is no scientific
evaluation, assessment exposure, checkpoint selection or accuracy threshold.
Each arm must complete at least 24 updates and two eight-update timing windows
after four warm-up updates, with consumption beyond shard one and matching initial
weights. The token cap is the lesser of two static passes and both refreshed shards;
this is roughly 32 updates per arm, not a fixed token count.

The estimate uses the slowest complete window across both arms, including durable
pre/post-update saves, multiplies its seconds/token by 25 million and **two**, then
adds **1,800 seconds** for setup/finalization and unmeasured variability. `PASS`
means that conservative short-probe estimate fits the proposed six-hour arm budget;
`BLOCKED` means it does not. Both are software/performance diagnostics, never a
scientific result or guarantee. A small shard cannot fully measure large-shard I/O
or long-run variance. No outcome automatically extends a budget or launches a full
learner; review the measurements before choosing new full-run limits/design.

Timing ceilings are deliberately generous: current acceptance retains its
395-minute ceiling (plus image build time); source selection has up to 15 minutes
per training shard; native preparation gets 30 minutes inside a one-hour outer
bound; each learner gets 30 minutes inside a one-hour command bound; preflight gets
20 minutes and each finite-state proof uses the canonical three-minute bound.
These are failure ceilings, not runtime predictions. Training speed is unmeasured
until this HIP probe; the diagnostic runs tens of updates instead of millions of
tokens. Reusing valid current qualification avoids repeating acceptance.

Share `<probe-root>/report.zip` after any terminal outcome. The archive contains
only the sealed result and its verified inventory: protocol/helper hashes, original
receipt references, per-arm committed/charged counters, checkpoint I/O and timing
windows, proof status and forecast. Failure reports recover bounded timing and
preparation progress where available. Original tensor/data/log trees stay local.
If acceptance itself fails, also share `<probe-root>/acceptance.zip` when emitted.
`--help` and orchestration require only host stdlib; model/native work stays in the
installed canonical container roles. No HIP validation was performed when this
implementation was committed.

## Hourly recovery and independent training progress (T161–T163)

The operator approved up to approximately **60 minutes of lost model progress**
on an interruption. Newly frozen research configurations now use `hourly/v1`:
full model/optimizer/RNG/data/replay snapshots are written at initialization,
required milestone boundaries, approximately hourly, and orderly termination.
A boundary is checked after a completed update; a slow operation can cross it.
The existing publication/cleanup reserves and external deadlines still apply.
Dense `per-update/v1` acceptance remains available, while the installed HIP resume
probe exercises hourly persistence at forced complete boundaries. This supersedes
the every-update full-snapshot description in T157–T160 above.

Small atomic accounting records are written before and after updates, and progress
is refreshed after each completed update. The learner's `timing.json` contains:

- Live committed tokens/updates, total charged tokens, discarded tokens, and the
  last fully recoverable tokens/updates/checkpoint reference.
- Last observed training loss and gradient norm when supplied by the real learner.
- The most recent 64 eight-update throughput/loss windows after four warm-up
  updates; a coalesced whole-invocation trend retains at most 128 bins, preserving
  total measured tokens, time and weighted loss observations.
- Checkpoint count, total and maximum snapshot duration, serialized/reused/verified
  bytes, journal write count, publication time and total invocation time.

No unbounded per-update tensor history or log is needed. Full-run diagnostic
bundles and the small paired probe include aggregate timing/progress/trend data;
source programs, examples, tensors and private batch contents remain local.
A checkpoint audit describes **recoverable** state; live progress may be newer.
Keep each original run's `timing.json`, accounting journal and supervisor receipts
with its checkpoint chain when cleaning up, including failed/partial runs.

On canonical resume, the latest complete snapshot restores exact model, optimizer,
RNG, cursor and replay state. Any later charged exposure in the journal remains
charged as discarded work. Already consumed time is carried forward using the
original supervisor's elapsed time only when process/container cleanup was
acknowledged. Missing crash-time evidence conservatively exhausts the remaining
frozen wall allowance; it does not fabricate a refundable interval. Missing or
malformed journals and older-than-latest periodic resume points are rejected.
Clean checkpoint-boundary stops still resume exactly. Selection metadata copies
preserve closed recovery accounting and reject active journals.

If discarded work prevents reaching the original model-token target before the
charged budget is exhausted, the result is explicitly
`PARTIAL / RECOVERED_WITH_DISCARDED_EXPOSURE`. It cannot silently qualify as the
original matched-exposure comparison. An explicit subsequent decision is needed
for a revised budget/study; this implementation never extends one automatically.

The next hardware check is still the same small paired probe, with current images
and qualification. Its forecast now includes the measured maximum full-snapshot
cost for hourly saves, original research milestones and initial/final saves, then
applies the existing twofold margin and 30-minute allowance. The short probe does
not need to run for an hour to test cadence; local controlled-clock tests exercise
the hourly boundary, and hardware acceptance tests complete checkpoint resume.
Actual speed and runtime remain unmeasured until the new probe completes.

One operator invocation from `/home/owen/workspace/oies-learn`:

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  podman build --target qualification -t oeis-learn:generalization-cpu .
  podman build --target gpu -t oeis-learn:generalization-gpu .
  python3 -S specs/011-controlled-generalization/validation/fresh_pilot_training_probe.py \
    --engine podman \
    --prepared-run run-state/generalization/fresh-p1-v2-20261103T035534Z \
    --imported run-state/generalization/uniform-import \
    --cpu-image oeis-learn:generalization-cpu \
    --gpu-image oeis-learn:generalization-gpu \
    --output "run-state/generalization/training-hourly-probe-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

Upload the printed `report.zip` after any outcome. Current-source acceptance is
performed once within that invocation, and its `acceptance.zip` remains separately
available if that stage fails. Source changes invalidate the previous probe's
qualification; do not pass its old `--qualified` path. Builds and generous stage
ceilings remain as described above. No full research learner launches automatically.
