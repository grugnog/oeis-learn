**Current remaining-work plan:** [Full-scale bounded soak and remaining Spec009 tasks](../plan.md). This is the single current sequence: complete all unblocked integration/local checks, then target one combined HIP command/result archive covering qualification, finite batch selection, persistent104.5M work, controlled restart and reports. Two3600s workload windows share a600s checkpoint/restart reserve; old global/phase/resource/artifact caps remain. Any follow-up HIP loop needs a concrete blocker and reuses only proven-compatible evidence. The new scope is PLANNED, with no new executable handoff yet; original counts78/18/15 and accepted source735ede… remain unchanged. Earlier handoffs and implementation snapshots below are historical at their stated producers.

**Latest accepted sizing result**: [model-sizing-operator-20261025T180134Z.md](model-sizing-operator-20261025T180134Z.md). Corrected sizing is accepted at56f6156/source735ede…:170 host,692 CPU tests plus31 subtests,20 actual-HIP tests and all15 trials pass. Actual8.12M/31.90M/104.50M profiles achieve median training5,934/4,219/2,092 tokens/s with2.45/3.36/5.89GiB observed role upper memory. All84 parity records have zero mismatches and640 strict native jobs complete; zero verified generated programs remains explicit.009:M004/T110 close at that source; original counts78 complete/18 runtime pending/15 research held.104.5M is the proposed next engineering candidate, not a production/learning choice. Persistent state/checkpoint/resume, longer-sequence and training-batch efficiency work remains before a longer run. No repeat of this size ladder or multi-day training is requested. Earlier handoffs below are historical.

Current corrected handoff: [sizing execution-policy repair](model-sizing-execution-repair.md). First target passes687 CPU+30 subtests/all CPU counts but fails m8 HIP encoder parity before all15 timing roles. The explicit FP32/math policy and diagnostics are repaired with170 host tests and verified export; corrected CPU/HIP validation remains pending. Counts77/19/15 and learning hold remain. Earlier handoffs below are historical.

Current handoff: [bounded model-size hardware screen](model-sizing-implementation.md).165 host tests and verified local export pass; three explicit fresh profiles8.12M/31.90M/104.50M share one fixed15-role command/archive. Current CPU/HIP sizing collection remains pending; accepted co-scheduling remains complete at5014881. Original counts77 complete/19 runtime pending/15 research held reflect affected-sourceT110 reopening. No learning/default or multi-day run. Earlier handoffs below are historical.

Latest accepted result: [complete GPU/native co-scheduling PASS](coschedule-operator-20261025T151306Z.md).674 CPU tests plus25 subtests,19 actual-HIP tests and all20 component trials pass at5014881. Overlap cuts elapsed21.72/24.99%; ten observed repeats remain39.019–39.504s with4.42GiB sampled role upper peak. Original counts78 complete/18 runtime pending/15 research held. [Hardware convergence](hardware-convergence.md) ends isolated sweeps and identifies the real-generation/persistent-state bridge before a longer training decision. No repeat command is requested and no learning run is admitted. Earlier handoffs below are historical.

Latest accepted scope: [native CPU capacity PASS](native-capacity-operator-20261025T040709Z.md).655 CPU tests and all18 native slots pass at exact producer176ecc9. Four workers are selected experimentally:68.63–70.71% paired interval reductions; all306 wave clocks/debits verify. Counts78 complete/18 runtime pending/15 research held. No duplicate native/GPU screen or learning run requested. Earlier handoffs below are historical.

Latest native collection: [qualification failure and repair](native-capacity-operator-20261025T032351Z.md).652 CPU tests pass; one serial-worker metrics assertion blocks every timing slot. That assertion and the independent CI foundation binding mismatch are repaired.136 host tests/export pass; corrected CPU target validation remains pending. Runtime algorithms, learning hold, accepted GPU evidence and counts77 complete/19 runtime pending/15 research held are unchanged. Use the single corrected native-capacity handoff; no HIP repeat. Earlier handoffs below remain historical.

Current handoff: [combined native CPU capacity screen](native-capacity-implementation.md).134 host tests and verified local export pass; current CPU/native1/2/4 target collection remains pending. Accepted GPU ladder remains complete at adda1b6; do not repeat it. This next command needs no HIP or parent/input run. Counts77 complete/19 runtime pending/15 research held. Earlier handoffs below are historical.

Latest accepted scope: [combined hardware ladder PASS](hardware-ladder-operator-20261024T220811Z.md).639 CPU/18 actual-HIP tests and all22 GPU roles pass at exact producer adda1b6. Batch128 achieves61.13% median replay interval reduction versus16 and about90.5% sampled GPU busy with2.84GiB role memory. Counts78 complete/18 runtime pending/15 research held. No duplicate run is requested; no production/learning admission. Earlier handoffs below are historical at their stated sources.

Current increment: [combined hardware experiments](hardware-ladder-implementation.md) share preparation/reference/archive and serialize timing.123 host tests pass; source-bound CPU/HIP128-row/mode/capacity collection remains pending. Accepted009:B004 remains complete at d74d30e. Counts77 complete/19 runtime pending/15 research held after reopening affected-sourceT110. No learning run admitted.

Latest accepted scope: [batch8/16 hardware replay PASS](replay-batching-operator-20261024T203537Z.md).629 CPU/17 HIP tests and all roles pass;48.98% median interval reduction. Counts78 complete/18 runtime pending/15 research held. Hardware-first, combined-experiment preference remains; no learning run admitted.

Current hardware increment: [equal-work batch8/16 replay](replay-batching-implementation.md).112 host tests and verified local export pass; new affected-source CPU/HIP collection is pending. Original counts77 complete/19 runtime pending/15 research held or blocked. Accepted bookkeeping evidence remains complete at its source. No learning or multi-day run is admitted.

Latest accepted result: [paired replay comparison PASS](replay-comparison-operator-20261024T193940Z.md).622 CPU/16 HIP tests and all roles pass at exact producer; component reduction10.54–11.28%. Counts78 complete/18 runtime pending/15 research held or blocked. User priority is hardware throughput/capacity; no new learning or multi-day run is admitted.

Latest follow-up: [comparison operator failure and fixture repair](replay-comparison-operator-20261024T190425Z.md). The archive verifies and matches source;621 CPU/15 HIP tests pass, one HIP fixture fails before all measured arms. The fixture-thread repair is committed, production guards are unchanged and106 host tests pass. New-source CPU/HIP collection remains pending; original counts stay77 complete/19 runtime pending/15 research blocked. Use the existing combined comparison handoff after updating, with a fresh output.

Current increment: [separately registered cache bookkeeping replay comparison](replay-comparison-implementation.md). Code and105 host tests pass; actual CPU/HIP comparison validation remains pending. The public stdlib invocation exports a verified53-listed-member non-admitting archive when engines are absent. Original-task counts are77 complete/19 runtime pending/15 research blocked, with T110 reopened for new affected-source validation. Three new comparison implementation tasks are complete; target collection009:C004 is pending. Serial defaults and all research gates remain. The following records are historical closeouts at their stated sources.

Latest closeout: [baseline replay scope accepted](replay-operator-20261024T172754Z.md). The archive and full source match; 613 CPU/16 HIP tests and all five replay trials pass, with a 60-second component reservation. T110 and R005 close; original task counts are 78 complete / 18 runtime-evidence pending / 15 research blocked. No duplicate run or new code change is requested; complete pipeline/cost/research remain pending.

Current handoff: [baseline replay viability implementation](replay-implementation.md). This is the separately versioned narrowed scope requested after the fourth archive, with 99 passing host tests and new pinned CPU/HIP execution pending. The original full-pipeline no-adoption decision and missing costs remain unchanged.

Latest collection: the [fourth operator review](operator-20261024T141450Z-review.md) supersedes the earlier storage/rerun handoff. The source matches exactly, selected Podman prerequisites pass, and the finite measurement scope ends with no adoption. Seven fixture tasks and three decision-report tasks now close; 19 runtime/evidence tasks and 15 research tasks remain. No unchanged-scope rerun is requested.

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
