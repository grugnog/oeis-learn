# 011 reliability review and 012 prevention gates

Reviewed 2026-11-04. These are committed 011 evidence reconciliations and current
code findings, not freshly reproduced hardware results. Preserve distinctions
between established failure, plausible mechanism and missing evidence. Every row
must have an implementation/test owner in tasks before the next HIP handoff.
B/C denote the critical phases in [plan.md](plan.md); D owns combined local gates.

The [task ownership matrix](tasks.md#reliability-ownership-before-hardware) maps
all 14 original classes and four added policy risks to implementation tasks,
counterexamples, evidence and recovery. These are planned owners, not passing
results. T064 must reconcile actual local checks before handoff; actual HIP proof
remains in T068–T072.

| Failure class and source | Prevention / owner | Required regression or injection | Exported evidence and recovery |
| --- | --- | --- | --- |
| Engine identity and qualification/production environment mismatch ([audit](../011-controlled-generalization/validation/hip-failure-audit.md), T107) | C: canonical engine inspection and shared `container_runtime.private_runtime_environment`; compare actual role commands, mounts, threads, scratch | Docker and Podman command construction for every new role; mismatched engine/image/source rejected before work | Requested/observed engine and policy identities; requalify only incompatible closure |
| Fresh-directory ownership and host dependency leakage (audit 002/007) | D: one owner per output; stdlib host assembly; canonical validation inside pinned role | Preexisting/partial output, host `python -S`, fresh-directory validator, escaping path | Exact failing stage and input identity; new sibling output or validated resume, never delete originals |
| Cold-start inspection/overlay delays (audit 003–006; mechanism not established) | C/D: relaxed measured envelopes for import, image inspection, load and exit; no tiny timeout derived from warm timings | Virtual-clock cold start and long exit; stale PASS cannot bypass process status | Startup/work/exit/cleanup separately timed; enlarge frozen operational plan before retry |
| Nested learner/evaluation/persistence timeouts (T096, T106, T111–113) | B: full deadline graph; C: propagate remaining parent budgets and protect publication/cleanup | Delayed load/save/hash/copy/export; nominal child cap exceeding parent; every acceptance branch including resumed proof | Requested/effective deadlines and first exhausted boundary; resume newest complete snapshot |
| Long preparation and repeated compile/hash cost ([readiness](../011-controlled-generalization/validation/fresh-pilot-readiness.md)) | D: reuse existing per-worker compiled/runtime artifacts with fresh isolated execution state; native all-family cold/warm calibration | Deterministic parallel/single worker equivalence; dirty worker reset, cancellation, later-shard costs; no unbound cache | Draw/validation/admission throughput and quota counters; reuse only sealed cells |
| Finite quota shortage mistaken for timeout ([quota audit](../011-controlled-generalization/validation/prospective-quota-audit-20261104.md)) | B: all post-exclusion structural quotas and train-probe quotas before research; preserve original 16 exclusions | Independent low-window-diversity collisions; full-draw shortage, missing family, native disagreement; reject hidden extra seed/draw | Before/after exclusion counts, rule counts where newly measured, deficits; stop scientific branch, no blind retry |
| Smoke failed to reach real boundaries (fresh readiness; T109) | C: tiny installed path with actual encoded-token/update coverage; separate production-size calibration | Exercise feedback, due review, parameter fallback, pending debit, load, resumed update and proof; no mocked learner substitute | Actual update/token/feedback/review counts and checkpoint hashes; diagnostic weights cannot initialize research |
| Disk/tmpfs and peak simultaneous copy growth ([repairs](../011-controlled-generalization/validation/operational-repairs.md), T108/T110) | B/C: peak-live graph including old/new snapshots, proofs, data copies, scratch, recovery and reporting; private disk-backed test/runtime scratch | Exact-fit/over-cap virtual sizes; ENOSPC during publication; nested symlink rejection; fixture cleanup | Free/required space by stage/category; preserve last valid snapshot, never auto-prune original runs |
| Full snapshots dominated updates; progress ended early (fresh readiness, hourly repairs) | C: preserve hourly full snapshots, milestone deduplication, compact whole-run progress; measure synchronization and serialization | Many updates with bounded save count; delayed save/partial atomic publication; trend beyond old 128-window boundary | Useful vs discarded tokens; save count/time, whole-run throughput and forecast; latest verified generation |
| Backend process-to-process numerical divergence ([HIP evidence](../011-controlled-generalization/validation/hip-acceptance.md)) | C: existing `generalization_execution.ENVIRONMENT` no-Lt process-start policy, source/image binding | Uninstrumented installed fresh/pause/resume comparison; changed environment rejected | Exact weights/optimizer/RNG/data/adaptive equality under qualified policy; upgrades need new evidence |
| Diagnostic interference and watchdog handler lifetime (audit 009/F5) | C/D: preserve non-invasive progress and safe watchdog ownership; no cross-thread stack walking | Slow scanner/cancel/nested contexts; no signal after restored handler; cleanup under failure | First error plus signal/cleanup status; bounded external termination and evidence preservation |
| Cross-image restore identity and newer-journal/older-model state (audit 010; persistence code) | B/C: exact producer identity, atomic whole-state snapshot, event IDs and discarded-work accounting | CPU/GPU identity mismatch; changed policy/inventory; crash before/after feedback and review; duplicate event | Original and expected identities, committed/pending/discarded ledger; no scheduler-only restoration |
| Missing nested outcome, COMPLETE projection, quota ledger; secondary error masked primary ([recovery audit](../011-controlled-generalization/validation/prospective-evaluation-recovery-20261104.md)) | C/D: shared bounded collector and finalization contract; distinguish application/process/cleanup | PASS file plus failed process/verifier, missing role result, truncated logs, export failure after original failure | Ordered errors, explicit omissions, terminal completion binding; independently verified minimal or read-only recovery |
| Packer rejected legitimate helper/build paths; stale provenance reuse (audit 007 and prospective recovery) | D: real allowlisted public projection, one canonical payload envelope; bind helper plan separately | Actual success/recovery packer with all new paths, path traversal and private payload rejection; stale receipt mismatch | Archive hash/member manifest and producer closure; retain original failed artifact, rerun smallest incompatible dependency |

## Hardware handoff rule

Every row needs passing CPU/contract counterexamples plus an explicit actual-device
item where relevant. Document a skipped native/HIP check as unavailable, never PASS.
The one consolidated session exercises the new installed learner path and all
terminal packaging paths. Existing compatible evidence can avoid redundant work;
new production source invalidates affected qualification. A helper-only change does
not justify rerunning an otherwise unchanged qualified learner, but its new plan,
paths and packaging must be validated.

## Timeout policy strengthened by operator feedback

The operator explicitly prioritizes generous margins over an arbitrary five/six-hour
ceiling. Apply [execution.md](contracts/execution.md) bottom-up to **every** path,
including tiny acceptance, calibration, resumed proofs and failure export. Check
schema ceilings as well as launcher values; raising only the outer wrapper repeats
the 011 failure. Default limits are ceilings, not promised runtime. Never reduce an
already justified allowance merely because one warm run was faster.

## Additional policy risks from the research revision

| Risk | Owner and prevention | Required evidence |
| --- | --- | --- |
| An EXP3-named heuristic, altered cap/share equation or wrong probability receives a published-algorithm label | B: explicit Graves Eq.1 kernel; independent numeric fixtures; reject legacy state; remove custom priority/cap | Named variant, hyperparameters, selected probability, raw/scaled reward and valid-update index |
| Periodic cross-family loss changes are credited to the last arm, or a pending reward is applied twice | B/C: matched pre/post probe around one contiguous block; atomic scaler/kernel transaction; resume pending observation before another update | Model/update/panel bindings, block cursor, exactly-once reward IDs, censored/discarded events |
| Review uses another example's success, ignores incidental practice, or overpromises service for 128 anchors | B/C: per-anchor pre-update grade, acquired flag and committed exposure; queue sizing at actual 24+8 cadence | Target-recall versus semantic evidence; all 32 offset simulations; grade/update fault injection; terminal pending anchors |
| Reward/grade observation quietly reintroduces short timeouts or dominates training | C/D: representative complete-cycle cost forecast and relaxed enclosing limits; preserve required reward reserve | Actual probe/grade/panel counts and costs; first exhausted boundary; no hidden frozen-policy continuation |

The original 14 failure classes remain in force. The additional rows are design
risks, not claims that these failures already occurred on HIP. Retain generous
12-hour learner/24-hour preparation floors, grow them when forecasts require, and
sum every selected arm plus protected finalization before launch.

## Implementation task ownership (T005)


The full prevention/evidence/recovery definitions remain in [reliability-review.md](reliability-review.md).
Each row below names both implementation and required regression tasks; D's T064
must link actual check results, not merely cite this table.

| Original failure class / additional design risk | Implementation owner | Counterexample / proof | Required evidence and recovery |
| --- | --- | --- | --- |
| Engine/qualification environment mismatch | T029, T040 | T039, T062, T072 | Requested/observed engine/source/image/env; requalify incompatible closure |
| Fresh directory/host dependency | T044, T051, T053 | T043, T050, T053, T062 | Input/output owner; explicit root or safe sibling/resume |
| Cold startup/inspection/overlay delay | T025, T040, T052 | T024, T039, T053 | Startup/work/exit costs; amend frozen operational plan before retry |
| Nested learner/evaluation/persistence timeout | T025, T040 | T024, T039, T052, T062 | All requested/effective limits and exhausted boundary; last complete snapshot |
| Repeated preparation compile/hash cost | T031, T052–T053 | T030, T052–T053 | Cold/warm/later-shard throughput; sealed-cell reuse only |
| Finite quota deficit | T008, T010, T031 | T007, T010, T030, T069 | Original draws/exclusions, deficits; no reseed/blind timeout increase |
| Smoke never reaches actual boundary | T034, T041 | T033, T036–T037, T070–T072 | Actual update/reward/review/load counts and checkpoint hashes |
| Disk/tmpfs/live copy exhaustion | T025, T038, T040 | T024, T037, T053 | Peak required/free categories; preserve complete prior state/no auto-prune |
| Excess snapshots/truncated progress | T038, T049 | T037, T048, T057 | Save count/time and whole-run totals; hourly/latest verified generation |
| Process-to-process backend divergence | T038, T040 | T037, T039, T072 | Actual full-state equality and no-Lt closure; changed backend requalifies |
| Diagnostic interference/watchdog lifetime | T035, T053 | T033, T053, T057 | Mode/RNG/handler/cleanup status and ordered errors |
| Cross-image/newer-journal restore | T021–T022, T038 | T020, T022–T023, T037, T072 | Identity/pending/discarded segments; no policy-only replay |
| Missing nested outcome/COMPLETE/primary cause | T055–T056 | T039, T054, T057, T072 | App/process/cleanup/verifier, quota, omissions and ordered errors |
| Packer helper paths/stale producer | T055–T056 | T054, T056–T057, T072 | Verified archive and original producer/helper closure; minimal rerun |
| Wrong EXP3 variant/probability | T014–T015 | T013, T015, T036, T071 | Named kernel, selected p, raw/scaled reward, valid clock |
| Cross-family/double reward credit | T021, T034, T038 | T020, T023, T033, T037, T071–T072 | Panel/model/block IDs, exactly-once feedback, censored/discarded events |
| Wrong anchor grade/service bound | T017, T021, T034 | T016, T020, T033, T036, T071 | Anchor grade/exposure/queue, cycle offsets, pending terminal items |
| Observation cost/short timeout recurrence | T019, T025, T035, T052 | T018–T019, T024, T039, T048, T052, T071 | Complete probe/grade/panel counts/costs; required-feedback pause |


These owners were checked against the current handlers during phase A. A task ID
is an assignment, not proof that a failure is prevented. The first 14 rows retain
their original evidence qualifications; the final four remain prospective design
risks. Local proof is recorded incrementally in validation/implementation-ledger.md;
installed CPU evidence is now recorded for C. Delivery, actual engine/native/HIP
validation and end-to-end evidence reconciliation remain D/E.


## Phase C local evidence boundary

Installed CPU checks now bind contiguous feedback, target recall and committed
review to actual SFT tensors, model-bound atomic snapshots, clean resume and
separately charged crash recovery. Canonical PAC role/CLI tests cover immutable
source/image/backend binding, explicit engine selection, protected work/publication
and cleanup reservations, disk-backed scratch, peak-live storage arithmetic and
failed-process rejection despite application PASS. The reduced acceptance handler
loads initial/final tensors, compares both pending-block resume boundaries and
verifies an actual negative-scope recovery archive. See the T042 command/results
in the implementation ledger.

These are local CPU/command-construction checks. Original workstation corpora,
actual native extensions, Docker/Podman execution, representative calibration and
HIP equality remain unverified here. The new HIP test is implemented for E, not
executed. D's full reporting/wrapper/export matrix and T064 reconciliation remain
unchecked; no historical 011 failure or missing hardware task is changed.
