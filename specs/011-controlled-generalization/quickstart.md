# Validation and first-pilot guide

All nine generalization commands are registered. The implementation has local
CPU/reference and fixture integration evidence. Actual pinned-container/native
qualification and HIP acceptance have **not** run in this environment. No research
campaign is started by these instructions or by the acceptance helper.

## Faster diagnostic iteration after acceptance-015

### Current next step: defer module reads until forward completion

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

E0 extends evaluation without retraining. Select exact descriptor identities from
the frozen runtime checkpoint roster; display names are not IDs:

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization preflight \
  --manifest run-state/inputs/E0/study.json --output run-state/evidence/E0-preflight
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization evaluate \
  --manifest run-state/inputs/E0/study.json --checkpoints "$EARLY_ID" "$SELECTED_ID" "$UNIFORM_BEST_ID" "$UNIFORM_LATEST_ID" \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/E0-evaluate
```

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
