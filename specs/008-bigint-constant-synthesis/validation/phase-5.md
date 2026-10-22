# Phase 5 qualification (T089–T098)

The initial qualification tooling merged in PR #58. Scope-specific cost
admission and the reviewed single-attempt runner are implemented in
[PR #59](https://github.com/grugnog/oeis-learn/pull/59). **The Phase 5 execution
and promotion gate remains BLOCKED. T089/T090/T091 are complete for their recorded implementation; remaining
external gates and new-source validation are listed below.** No Phase 6 screen or final-cohort study ran.

The reviewed `008-phase5-scoped-20261021T115212Z` run passes all nine actual HIP
tests and twelve fixed-state generation parity/deadline groups. All24 archived
file hashes and the source/image/shape/recipe/schedule bindings verify. Diverse32
projects7,926.648s within its11,520s admission allowance (80% of the preregistered
14,400s); its10,082s of role ceilings also fit the complete budget. **T095 is
complete and admitted only the one registered T096 attempt.** The independently
calculated transfer screen remains BLOCKED at29,637.220s versus5,760s allowed.
That single learnability run is now reviewed and **T096 is complete with a
negative outcome**:31/32 correct rescue continuations,31/32 complete rescue
evidence and32/32 greedy termination. The required32/32 completeness fails.
See [the scoped review](phase-5-scoped-hip-review.json) and
[the learnability result](phase-5-diverse32-review.json). Promotion stays BLOCKED.

The earlier HIP run closed T092 and retains its original failed larger-screen
estimate at47,749.33s; [that historical receipt](phase-5-batched-hip-review.json)
is preserved. Remaining slowdown and operations evidence is still required
before overall Phase5 promotion. Historical-model task T093 is user-deferred.

Entry is merged Phase 4 PR #57 at
`be6070bead5f30147774b6517c0f50eb1cb3f6cc`. T088 is closed by reviewed required
CI jobs and digest-verified installed/native artifacts at `8556cce`; see
[phase-4-exit.json](phase-4-exit.json). The non-critical Podman rebuild was
cancelled, separately from successful Podman execution. Historical evidence is
not relabelled as execution of this new image.

The requirements checklist passes16/16. All design documents, task list,
contracts, AGENTS and constitution were reviewed. No extension hooks exist.
Native timing observes the same production run method. Generation scheduling
now batches up to eight portfolios; numerical, solver, encoder, selection and
resource policies remain frozen.
Third-party manifests/locks and long Docker dependency layers are unchanged.

| Task | Current execution status / evidence |
| --- | --- |
| T089 | [Arithmetic](arithmetic.md): COMPLETE,184 selected managed-image cases with zero failures/errors/skips; exact source/runtime/compiler/locks verified. |
| T090 | [Fitting](fitting.md): COMPLETE,37 selected cases plus six complete non-scoring fitting classes; deadline correction and successful exact visible replay reviewed separately. |
| T091 | [Containment](containment.md): COMPLETE for the recorded implementation;64 cases per engine, earlier29 host/3 public tests per engine, verified offline/read-only inspect and cleanup0.015571s Docker /0.048828s Podman against5s. |
| T092 | [GPU](gpu.md): PASS,9 actual HIP tests with zero failures/errors/skips; reviewed source/image-bound evidence closes this task. |
| T093 | [Checkpoint rescue](checkpoint-rescue.md): USER-DEFERRED2026-10-22; historical model comparison is outside the current completion gate. No result claimed. |
| T094 | [Performance](performance.md): [current Podman CPU adoption](phase-5-engine-adoption-review.json) passes334 arithmetic/37 fitting/64 containment tests and75 comparison rows. Small batch100 improves5.18–5.45x versus zero, but remains13.29–13.40x retained; timer lifecycle and single-term uncertainty remain open. No new whole-workload HIP/cost receipt. |
| T095 | [Data and cost](screen-preflight.md): COMPLETE. Checked512/128/128 plus tiny32; diverse32 PASS at7,926.648s versus11,520s allowed,10,082s reserved within14,400s. Independently scoped transfer-screen remains BLOCKED at29,637.220s versus5,760s allowed. |
| T096 | [Learnability](learnability.md): COMPLETE, NEGATIVE. One seed20261016 run,1800 updates; rescue31/32 correct and31/32 complete, termination32/32. The full-coverage gate fails; no retry/resume or promotion follows. |
| T097 | [Operations](operations.md): both-engine public validate/rerun and arbitrary-UID installed retirement PASS; complete current-scope command/export coverage remains open; T093-dependent historical diagnosis is deferred. |
| T098 | This report, [local checks](phase-5-scoped-local.json) and [current HIP review](phase-5-scoped-hip-review.json) retain their exact identities; overall qualification remains BLOCKED. |

## Current implementation evidence reconciliation — 2026-10-21

Reviewed eight artifacts from completed workflow39686633400 at head
`9f86c4d10ab96a0e7a95bc44752c54a08afd3f94`. All archive CRCs/SHA256 values,
mandatory JUnit hashes, portable receipt hashes and matching native identities
verify. The tested source snapshot is
`sha256:942afe05c9cbe3ba95f839ed071ca7a44855b1222d2e0b03d82f0c39eeeef562`,
and the shared image is
`sha256:2792f2f01f8f7380cb468ca16c63096fb3c1f1f7442e2863bb3e1cdf9fb9145c`.
The subsequent audit-only run39693058662 retains that implementation snapshot
and supplies both-engine successful inspect/timed cleanup, closing T091; see
[the containment receipt](phase-5-containment-review.json). The new runtime
profiling implementation changes the snapshot and awaits matching native CI. The earlier
stale disposition binding no longer fails either portable CLI job. See the
[bounded execution review](phase-5-execution-review.json).

The diagnostic/reliability follow-up is complete: [the reviewed user replay](phase-2-fit-deadlines-replay-review.json)
returns five independently verified fits for the original candidate/seed within
its unchanged worker allowance. This does not retroactively change T096 or
satisfy32/32 qualification. No same-case replay or further training is needed
to assess that defect.

Remaining work, in dependency order (updated2026-10-22):

1. Current [Podman CPU adoption evidence](phase-5-engine-adoption-review.json)
   passes435 tests without skips, verifies450 detailed/ordinary pairs and225
   cross-setting groups, and confirms the64KiB policy optimization. That exact
   source/image now has CPU regression/comparison evidence. The earlier failed
   fixture-identity run remains FAIL in its separate review.
2. T094 stays open: small batch100 execution remains13.29–13.40x retained;
   three batch1 rows regress and only two warm samples exist. Timer start/stop
   is roughly half of profiled small-case execution. The user now requests the
   [whole-pipeline HIP plan](phase-5-pipeline-profile-plan.md) before choosing
   further timer/WASM optimization. Any change must preserve deadline,
   fuel, store isolation and cleanup semantics. A new design has no measured
   speedup yet. Eventual-image HIP and whole-workload cost evidence are separate.
3. T093 historical-model comparison is user-deferred and missing historical
   weights no longer block current Phase5. Complete the supported current-scope
   command/export checklist for T097. Current Podman adoption adds operation evidence;
   it does not constitute new Docker/HIP execution or close the full checklist.
4. Reconcile T098's final pass/fail/pending report. Negative T096, missing
   prerequisites and the unadmitted transfer screen continue to block promotion.
   No new qualification learning attempt is launched.

T092/T095 remain completed historical tasks for their recorded source/image,
not permission to reuse old HIP/cost receipts for new training. No numerical,
solver, encoder, selection, data, budget or threshold policy changed in this
reconciliation.

The CI partitions reuse the existing shared qualification image and uploaded
image provenance. They require nonempty zero-failure/error/skip JUnit, clean
arbitrary-UID packaging and unchanged retained numerical controls. GPU execution
is separate. New CI is left running after pushes as requested.

## Initial PR #58 local checks and limits

The affected CPU/policy/codec/reference and tooling set passes **135 tests with
one explicit packaged-native measurement skip**, zero failures/errors. Ruff
F/E9, Python compilation, workflow YAML/shell syntax, Spec Kit prerequisites,
artifact structure and source retirement pass. The source/JUnit hashes and
per-task statuses are in [phase-5-local.json](phase-5-local.json). Small mocked
measurement tensors test checkpoint and fixture plumbing only; they are not
reported as hardware performance, actual full-model training or learned accuracy.
This agent environment has no Docker/Podman/Rust/HIP toolchain. Native compilation,
new-image execution/benchmarks, real data preparation and GPU learning are not
claimed locally. Required CI/image runs fail rather than skip missing artifacts.

## Earlier scoped-cost implementation validation

The PR #59 affected local suite passes **235 tests**, zero failures/errors,
with six explicit packaged-native skips and one GPU deselection. The broad
suite covers bigint policy, codec, reference, CPU encoder/training/checkpoint,
workflow contracts, generation and retirement callers; a separate registration
test verifies that only the smoke budget changes. Ruff F/E9 and formatting,
standard-library-only host help, documented Bash syntax, Spec Kit prerequisites
and whitespace checks pass. See [the source-bound local receipt](phase-5-scoped-local.json).
No container build, native timing or new HIP/learning execution is claimed here.

PR #59 separates admission authority for diverse32 and transfer-screen. Its
forecast and coordinator use identical functions for exact panels, attempt
rounds, raw/fitted/hidden checks, full teacher forcing, checkpoints and cleanup
reservations. A full-budget ceiling is checked in addition to the 80% p95 limit.
The diverse32 worker uses exactly its 32 training prompts for periodic synthesis;
transfer-screen retains its three 32-row panels and complete 128+128 end screen.

Receipts bind source, image, numerical/model/search profile, prepared shape,
scope and registered schedule. Legacy receipts remain readable but cannot
authorize new scoped training. Both receipts are written from the same measured
p95 inputs; one passing scope gives no authority to the other. The reviewed
runner reuses those measurements and the image, enforces seed20261016, validates
sealed report/recipe identities and durably prevents a second fresh attempt.
It archives public check/learnability/pack results, including negative and
interrupted outcomes. Resume continues the original ledger.

The [budget preregistration](phase-5-scoped-preregistration.json) is explicitly
an inference from historical p95s, not new hardware evidence. It calculates
7,999.085s for diverse32 and 10,183s of role reservations against a 14,400s cap.
The independently corrected transfer estimate is 29,723.688s and remains over
its original budget. Neither this calculation nor mocked orchestration tests
establish learned accuracy. T095 is now closed by the reviewed fresh HIP
receipts; T096 is closed by the reviewed negative learning outcome. The one
attempt is consumed; its finalized ledger cannot be resumed for more training.

## Bounded CPU runtime profiling

The opt-in `bigint-runtime-profile/v1` implementation measures ten execution
phases on the production run path, including deadline-thread and store cleanup.
Each detailed call is paired with an ordinary aggregate call, with alternating
order; full values/failure/fuel/coverage/resource identity must match. The report
keeps per-case overhead, phase call counts, residual time and a descriptive
phase ranking. Successful results retain the independent reference/production
checks; expected exclusions are not throughput successes. Normal candidate and
worker protocols do not expose profiling.

Earlier profiler checks passed55 tests with two native-only skips; the source-
bound [receipt](phase-5-runtime-profile-local.json) is retained. Latest nested-input
checks pass103 tests, with five native-only contract skips; see
[the new local receipt](phase-5-input-load-local.json). The uploaded cache
comparison is now reviewed; new observation compilation,
regressions and actual nested timings remain pending; no suitable
engine/toolchain exists here. The CI adds a dedicated CPU partition using the
shared image and audited socket-free launcher. No dependency layer/lock changes
or measured speedup is claimed. T094 remains open until new cache safety and
ordinary comparison results are reviewed. No further HIP is needed for this
step, and no training or T096 replay is launched.

```bash
profile_out="tmp-validation/008-runtime-profile-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman \
  --runtime-profile-only --profile-input-load --output "$profile_out"
```

With `--profile-input-load`, each detailed sample also partitions its input-load
parent into seven nested phases with partial-work counters. Add
`--compare-host-lookups` to repeat the optional third baseline. This builds/reuses the current
checkout image, then runs only the fixed CPU corpus
(25cases, batches1/20/100, three samples per arm). The work role is bounded
to600seconds with separate5s cleanup; image building is outside that work limit.
Share `$profile_out.zip`, including a failed or partial result. Outputs include
source/image/binary/build identities, phase/control measurements, logs, actual
inspection and cleanup, with a bounded archive/omission manifest. Use a fresh
output on retry and `--engine docker` for Docker. An explicit `--image` must
contain the exact current checkout; stale source fails binding. No inputs,
checkpoint, `--gpu`, `--prepare` or additional measurement scope is accepted.

## Copy-pastable hardware commands

The diagnostic/reliability pass requires no further HIP execution. The command
below is retained for a future separately scoped current-source device/cost
receipt; it is not the immediate next step and does not authorize training.
It reuses prepared inputs and measures both scopes without training:

```bash
phase5_preflight="tmp-validation/008-phase5-scoped-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --performance-only \
  --cost-scope diverse32 --config configs/foundation/bigint_smoke_phase5.yaml \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "$phase5_preflight"
```

Review/share the resulting ZIP and both `performance/cost-SCOPE.json` receipts.
The [separate single-attempt command](learnability.md) has now run and is
documented for provenance; do not repeat the consumed attempt. There is no automatic learning
or Phase 6 launch. A new code/profile/data/image identity requires a new receipt.

The preceding CI supplied both containment audit records; new-source CI will
repeat them. No manual full-suite rerun is needed while those jobs execute. The following complete
CPU/native/retirement/operations commands remain available for reproduction:

```bash
python3 scripts/qualify_bigint_phase5.py --engine docker \
  --output "tmp-validation/008-phase5-docker-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman \
  --output "tmp-validation/008-phase5-podman-$(date -u +%Y%m%dT%H%M%SZ)"
```

To combine those partitions with the HIP preflight, omit `--performance-only`
from the first command. For new preparation replace `--inputs PATH` with
`--prepare`; retain the pinned bulk cache. Add `--render-node
/dev/dri/renderD128` when device selection is needed. An explicit `--image IMAGE`
must contain this exact source; stale images fail source binding.

Each invocation creates a bounded ZIP on success/failure and prints its path.
It retains summaries, source/image/artifact hashes, receipts, JUnit, logs and
observed isolation/cleanup, with an omission manifest. Private inputs, weights
and caches are excluded. Use a fresh output directory for measurement retries;
learning uses the original attempt/ledger instead.

T093 historical-model diagnosis is user-deferred; `--checkpoint-run` remains
a supported optional operation rather than a current Phase5 prerequisite.
No historical-model result is claimed. See [the disposition](checkpoint-rescue.md).

## Historical follow-ups

The sections below describe earlier source identities and commands. Use the
current commands above; the former integrated `--learnability` flow is replaced
by a separate reviewed receipt step.

## CI discovery follow-up

At `c2a91dcb44d99278ef17e45e4f397a5a6f7dccca`, all five new bigint
qualification partitions passed, as did native qualification, both preparation
partitions, the image/Podman builds and portable Podman execution. Foundation
review run39364753190 passed. Container run39364753246 failed only portable
Docker's CPU regression: the two-program archive test returned no relation
claims (1089 other tests passed). These job statuses do not replace review of
the qualification artifacts or qualify the follow-up source/image.

The test passes locally. Its exact relation can be silently omitted when the
contained fitter exhausts its default two-second spawn/import/IPC deadline.
A controlled 2.2-second cold SymPy import reproduced UNKNOWN at two seconds
(2.021s total), then FIT with coefficients `[1, -1]` at ten seconds (2.526s).
The original CI log did not retain the fit status, so its precise resource
failure cannot be recovered. The discovery entry point now explicitly uses
the existing ten-second maximum per candidate and records every fitting
disposition, including UNKNOWN, in the sealed report. The killable worker,
1-GiB memory cap, coefficient bounds and independent finite validation remain
in force. No bigint numerical/solver profile or training recipe changed.

Regression coverage checks the successful real archive path, propagation of
the configured deadline, preservation of UNKNOWN without a claim, rejection
of unbounded deadlines, and the fitter's existing immediate-timeout behavior.
The earlier [local report](phase-5-local.json) remains evidence for its original
source, not this follow-up. HIP and matched-checkpoint gates remain pending.

Follow-up local checks: **68 passed, 3 skipped** across complete discovery,
Phase 5 runner/benchmark and symbolic-proof suites. All three skips require the
packaged native certificate solver. Ruff F/E9, formatting, whitespace checks
and standard-library-only runner help pass. A broader CLI-contract attempt had
two environment failures because this sandbox lacks
`/proc/sys/kernel/random/boot_id`; those container-only paths await CI rather
than changing their boot-identity checks. Follow-up CI is left unmonitored.

## Bulk fetch hook compatibility follow-up

The HIP attempt `008-phase5-hip-20261020T185302Z` reached bulk fetch but stopped
at `git lfs install --local`: Git LFS 3.4.1 was present and rejected an existing
pre-push hook. This is initialization failure, not a missing image dependency
or HIP test result. Fetch now uses `git lfs install --local --skip-repo` to
configure repository filters while preserving all hooks, including custom
`core.hooksPath`. The download cache never pushes. Pinned plans, selected paths,
checksums and batched downloads are unchanged.

Actual Git/LFS regression tests reproduced the failure before the fix and now
verify repeated cached fetches, filter setup, untouched older/custom hooks and
unchanged plan/receipt hashes. Bulk and qualification-runner tests pass23/23;
Ruff and whitespace checks pass. No image dependency layer changed. New CI is
left running without monitoring; actual HIP qualification remains pending.

After updating this branch, rerun the timestamped `--gpu --prepare --learnability`
command above. Retain `tmp-oeis-bulk`: its pinned snapshot, plan and verified
downloads are reused. Use a fresh output directory; do not use `--inputs` for
this attempt because fetching failed before preparation produced complete
inputs. The wrapper rebuilds for the new source and prints the new result ZIP.

## Unresolved LFS object recovery

The next HIP attempt `008-phase5-hip-20261020T191140Z` passed retirement, then
bulk fetch reported success for a batch but checkout left
`files/A373/b373464.txt` as a 130-byte pointer. User diagnostics confirm that
its plan, pinned Git blob and working pointer all identify the same
29,047-byte object, SHA-256
`c0896a08bd0248bc1ee96779af5fdf1d21a1d69f2be38fc330d7b1e9849218f8`,
at pinned commit `f0fb257e855fb356cc50a580e41e081a41443bd0`.
A clean Git/LFS status therefore did not establish materialized input data.
The original cause of the batch omission has not been established.

Fetch still batches the frozen selected paths and verifies every size/hash.
For an unresolved pointer or missing file only, it now checks the exact Git
pointer against the existing plan and requests that object through `git lfs
smudge`. It clears smudge-skip/filter exclusions for that command, allows one
120-second attempt under the existing aggregate fetch/container deadline,
monitors output size, and kills/reaps the process group on failure. Output
stays in a temporary workspace file until exact size/SHA-256 verification
passes, then replaces the pointer atomically. A changed working file is never
overwritten. An unavailable object fails with its path/OID and leaves the
pinned plan, pointer and earlier verified downloads intact. No HTTP API,
cohort change, new snapshot, threshold change or dependency layer was added.

Local tests use actual Git LFS 3.4.1 and a local file-backed remote, with the
observed successful-but-incomplete batch injected explicitly. They cover
successful recovery and reuse, missing remote data, changed working files,
stale plans, temporary-file cleanup and untouched unselected pointers. The
bulk, qualification-runner and benchmark tooling suites pass **34/34**, with
zero skips. Ruff and whitespace checks pass. The exact remote omission and
actual HIP execution still require the user's run; CI is left unmonitored.

To test fetch first without repeating the qualification suites:

```bash
git pull --ff-only
python3 scripts/foundation_run.py --engine podman data fetch \
  --config configs/foundation/bigint_smoke.yaml
```

After fetch succeeds, run the timestamped `--gpu --prepare --learnability`
command above. Keep `tmp-oeis-bulk`; verified objects and the frozen plan are
reused. If recovery fails, share its `RECOVER LFS object` line and subsequent
stderr; the object is not silently excluded or counted as complete.

The user's recovery attempt subsequently exited successfully from Git LFS,
but independent output verification rejected its result. The initial error
did not record actual output size/hash. That message now includes actual
bytes, SHA-256 and any parsed pointer, alongside the expected identity.
A real Git LFS regression confirms that a same-size corrupted local cache
object can produce a zero exit while returning wrong bytes; verification
rejects it and retains the original working pointer. This establishes a
possible failure mode, not the cause on the user's machine. Recovery with the
existing pre-push hook also passes. The user's cached object must be inspected
before declaring a remote/client/cache diagnosis or deleting any objects.

### Confirmed upstream pointer size defect

The user's cache inspection resolves the failure: object
`c0896a08bd0248bc1ee96779af5fdf1d21a1d69f2be38fc330d7b1e9849218f8`
is **27,916 bytes**, and its SHA-256 exactly matches the pinned pointer/plan.
The upstream pointer's **29,047-byte** size is overstated by 1,131 bytes.
The payload is correct; changing the source pin or excluding the record would
change the study unnecessarily. Git LFS's size check repeatedly treats this
valid object as missing, and our earlier exact-size check rejected its output.

The corrected importer treats declared bytes as the existing hard allowance
and SHA-256 as the exact content identity: shorter content is accepted only
with the pinned hash; oversized or wrong-hash content is rejected. Fetch now
checks the configured local LFS object store before invoking LFS fetch/checkout,
copies a verified object atomically to replace only its matching pinned pointer,
and reuses the verified working file on subsequent runs. The immutable snapshot
and plan retain their original commit, OID and declared size.

Every such case is printed and recorded in `fetch-result.json` as
`size_discrepancies` with path, SHA-256, declared bytes and actual bytes. The
import report retains that audit as `lfs_size_discrepancies`, so prepared-source
provenance records the upstream defect. Normal receipts retain their existing
shape. No data member is dropped, and download/disk allowances do not increase.

Regression tests use an actual Git commit containing an overstated LFS pointer
and a correct shorter cached object. They verify recovery without any fetch,
unchanged pin/plan hashes, deterministic discrepancy receipts, repeat execution,
import-report propagation, and rejection of oversized or wrong-hash objects.
The relevant bulk, qualification-runner and benchmark-tooling suites pass
**39/39** with zero skips; Ruff, formatting and whitespace checks pass.
Use the fetch-only command above first, retaining `tmp-oeis-bulk`, then rerun
the timestamped HIP command once fetch completes. Actual HIP gates remain
pending; new CI is left running unmonitored.


## Current performance direction — user decision2026-10-22

T093 is deferred, not passed, and removed from current Phase5 prerequisites.
Before further timer/WASM optimization, use the
[whole-pipeline profiling plan](phase-5-pipeline-profile-plan.md) to measure real
HIP generation, scratch update cost, CPU fitting/execution/reference work and
resource/waiting timelines. The paired harness is now implemented as
`scripts/profile_bigint_pipeline.py`; see the plan/quickstart for the bounded
Podman/Docker command. Actual HIP collection/review (W005) is pending. No new
qualification attempt or training experiment is authorized by this diagnostic.
