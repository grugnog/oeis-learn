Latest collection: the [third operator review](operator-20261024T035922Z-review.md) records passing CPU suites and a filesystem free-space stop during GPU inspection. Resolve workstation storage before rerunning; precise disk reporting is implemented and passes 89 host tests.

Latest operator collection: see the [second archive review and rerun](operator-20261024-review.md). Its 15 HIP tests pass; CPU/workflow failures are the Docker fixture issue already repaired by `5f0c568`. Combined current-source validation remains pending.

# Implementation handoff: Spec 009

Latest follow-up: the [operator archive review and repair handoff](operator-20261023-review.md) supersedes the original evidence status and next-command guidance below. It records 593 CPU/14 actual-HIP passing tests on the producer, three repaired failure paths, 85 current host tests and the remaining current-source gates.

Original implementation handoff status: code and available host verification complete; target runtime/engine qualification and separately admitted research remain pending. PR [#60](https://github.com/grugnog/oeis-learn/pull/60) uses the existing `009-pipeline-throughput` branch and remains stacked on #59. The optimized options remain experimental; serial rollback/defaults are retained. This is not whole-feature completion.

## Implementation and verification

| Area | Implemented behavior | Evidence / remaining gate |
| --- | --- | --- |
| Shared / phase 1 | Stdlib contracts, memory resolution before Host, static DAG, immutable receipts, resource/failure stop, current image inspection, bound CPU reuse, bounded supervision and one verified archive | Actual public CLI and 80 passing stdlib tests; real-engine/permissions/packaging pending |
| Phase 1 measurement / phase 2 | Frozen finite workloads/calibration, full versus reduced denominators, explicit decoder bookkeeping candidate and rollback, long cached/full CPU/HIP parity fixtures | Host comparison fixtures pass; Torch/cache/numerical/HIP execution pending |
| Phase 3 | Timing-free v2 selection/retention, stable jobs, exactly-once debits, bounded 1/2/4 slots/caches, unique generations, affinity, shared cancellation/reaping deadline and native thread limits | Real fake-worker subprocess/cache/policy fixtures pass; Rust/native exactness, RSS/fault and scaling runs pending |
| Phase 4 | Single GPU owner, bounded overlap/lookahead, cumulative checkpoint barriers, finite batch/cache/capacity branches, fresh five-pair cold/warm whole-pipeline confirmations and full 32 fixed-budget assessment | Host scheduling/roster/decision fixtures pass; full Torch checkpoint roundtrip and actual HIP stress/performance pending |
| Phase 5 | Existing supported-command/retirement/operations adapters, current-source qualification report, material-stratum and exact-scope full-control cost forecasts | Host gate/cost regressions pass; installed-image, both-engine, HIP and complete measured costs pending |
| Phase 6 | Explicit per-kind reviewed admission, source/image/resource/engine consistency, whole-arm finite supervision, durable reservations, independent multi-study batch with one archive, full-denominator reporting and final exposure/remaining-budget checks | Stdlib authority/budget/final-state fixtures pass; pinned-image finalization fixtures pending; no actual study admitted or run |

The consolidated entry point reuses existing profile preparation, `CheckpointStore`, training/evaluation, independent admission/scoring and matched-study reports. Some planned target paths are represented by focused modules instead: `benchmarks.py`, `neural.py`, `role.py`, `costs.py`, `policies.py`, `study_batch.py` and host `test_pipeline_*` suites. Engineering workloads/candidate rosters are static/versioned in those adapters; unsupported custom manifest overrides fail closed. Existing trace diagnostics remain separately labelled optional diagnostics; ordinary confirmation is primary and missing observations are not invented. No new trainer, fitter, host dependency or service was introduced.

The approved 111-task plan was restored from its original commit after an earlier tool-output truncation removed tasks048–066. The [task evidence index](task-evidence.json) now distinguishes completed implementation/host tasks, authored fixtures awaiting their runtime, target collection/review, and unadmitted research. Unchecked tasks are not quietly closed by code existence. No historical008 checkbox/result was rewritten.

## Actual local evidence

The final public command ran with `python3 -S`, selected all-ready and both engines, and exited3. `host.contracts` passed **80 tests**. Missing Docker/Podman remained explicit capability blockers; selected scope was PARTIAL, engineering BLOCKED, portability PENDING and feature_complete=false. The resulting archive had 54 manifest-listed members and passed `pipeline_validation.bundle.verify`. Hashes and terminal stage states are in [local-evidence.json](local-evidence.json); the original host receipt and test output are retained alongside it. The raw temporary ZIP is reproducible evidence, not a durable qualification artifact.

`python3 -m compileall -q src scripts tests/integration` and `git diff --check` passed. These do not substitute for the unrun Torch/Rust/native/container suites. There is no Torch, Cargo, Docker or Podman in this workspace; no host packages were installed to circumvent the pinned Dockerfile. Actual HIP was not available. No performance gain, new learning result, matched research execution, assistance comparison, final exposure or longer-run authorization is claimed. PR CI was neither queried nor awaited.

## Next target-machine command

After updating the existing branch, from the repository root on the HIP/ROCm workstation:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both \
  --manifest configs/foundation/pipeline_validation.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-$(date -u +%Y%m%dT%H%M%SZ)"
```

The two input paths are the last recorded compatible008 handoff paths, not guesses about another machine's current filesystem. Substitute their current workspace-relative paths if moved. Use `--engine docker` for a Docker primary. No separate build/test/profile/archive sequence is needed. Return the single sibling output ZIP, including a non-PASS ZIP if any stage fails or is blocked.

Expected duration is **unknown until current-source target calibration**; no measured range can be honestly supplied yet. The 24-hour ceiling includes at most two 14,400-second engine build allowances, phase caps 3,600/10,800/7,200/18,000/14,400 seconds, report/archive 900 and invocation cleanup 300 seconds. Reservations cannot borrow another phase's time. Role cleanup has one shared five-second deadline. Aggregate memory is at most 90% measured physical RAM, further restricted by engine/ancestor limits, with peak-plus-headroom admission and at most 16 CPU equivalents. Missing representative material strata or costs leave adoption/forecast gates blocked.

A rerun uses the same command with a fresh output. Add `--reuse tmp-validation/PRIOR_009_OUTPUT` to import only complete compatible host/pure CPU evidence; current-image checks and timing blocks rerun. A targeted native-only rerun is documented in the [quickstart](../quickstart.md). Do not delete durable `run-state/pipeline-studies` reservations to obtain new research budgets.

## Remaining decisions

Target evidence owns tasks025/026/034/040/041/059/060/061/062/073–076/084–086 and runtime portions of the fixture/operations/packaging tasks in the index. It must be reviewed before performance adoption or engineering qualification. The historical 13.29–13.40x small-value execution gap and >5x review trigger remain visible; no new arithmetic optimization or gap closure is claimed.

Research tasks093–107 require prospective decisions and qualified exact-scope cost. No reviewed manifests or assistance data were fabricated. Representation needs six paired-seed arms and full 128+128 controls; curriculum/replay, assistance, encoder sensitivity/retraining and final plans have distinct decisions. Original008:T093 remains deferred; original008:T096 stays negative at 7,730/14,400 seconds. No automatic retry/resume or 12-hour-plus training is authorized.

No extension hooks are registered (`.specify/extensions.yml` is absent). The implementation handoff stops here without waiting for PR CI or launching research.
