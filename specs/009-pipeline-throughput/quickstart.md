**Latest accepted sizing result**: [model-sizing-operator-20261025T180134Z.md](validation/model-sizing-operator-20261025T180134Z.md). Corrected sizing is accepted at56f6156/source735ede…:170 host,692 CPU tests plus31 subtests,20 actual-HIP tests and all15 trials pass. Actual8.12M/31.90M/104.50M profiles achieve median training5,934/4,219/2,092 tokens/s with2.45/3.36/5.89GiB observed role upper memory. All84 parity records have zero mismatches and640 strict native jobs complete; zero verified generated programs remains explicit.009:M004/T110 close at that source; original counts78 complete/18 runtime pending/15 research held.104.5M is the proposed next engineering candidate, not a production/learning choice. Persistent state/checkpoint/resume, longer-sequence and training-batch efficiency work remains before a longer run. No repeat of this size ladder or multi-day training is requested. Earlier handoffs below are historical.

Current corrected handoff: [sizing execution-policy repair](validation/model-sizing-execution-repair.md). First target passes687 CPU+30 subtests/all CPU counts but fails m8 HIP encoder parity before all15 timing roles. The explicit FP32/math policy and diagnostics are repaired with170 host tests and verified export; corrected CPU/HIP validation remains pending. Counts77/19/15 and learning hold remain. Earlier handoffs below are historical.

Current handoff: [bounded model-size hardware screen](validation/model-sizing-implementation.md).165 host tests and verified local export pass; three explicit fresh profiles8.12M/31.90M/104.50M share one fixed15-role command/archive. Current CPU/HIP sizing collection remains pending; accepted co-scheduling remains complete at5014881. Original counts77 complete/19 runtime pending/15 research held reflect affected-sourceT110 reopening. No learning/default or multi-day run. Earlier handoffs below are historical.

Latest accepted result: [complete GPU/native co-scheduling PASS](validation/coschedule-operator-20261025T151306Z.md).674 CPU tests plus25 subtests,19 actual-HIP tests and all20 component trials pass at5014881. Overlap cuts elapsed21.72/24.99%; ten observed repeats remain39.019–39.504s with4.42GiB sampled role upper peak. Original counts78 complete/18 runtime pending/15 research held. [Hardware convergence](validation/hardware-convergence.md) ends isolated sweeps and identifies the real-generation/persistent-state bridge before a longer training decision. No repeat command is requested and no learning run is admitted. Earlier handoffs below are historical.

Latest accepted scope: [native CPU capacity PASS](validation/native-capacity-operator-20261025T040709Z.md).655 CPU tests and all18 native slots pass at exact producer176ecc9. Four workers are selected experimentally:68.63–70.71% paired interval reductions; all306 wave clocks/debits verify. Counts78 complete/18 runtime pending/15 research held. No duplicate native/GPU screen or learning run requested. Earlier handoffs below are historical.

Latest native collection: [qualification failure and repair](validation/native-capacity-operator-20261025T032351Z.md).652 CPU tests pass; one serial-worker metrics assertion blocks every timing slot. That assertion and the independent CI foundation binding mismatch are repaired.136 host tests/export pass; corrected CPU target validation remains pending. Runtime algorithms, learning hold, accepted GPU evidence and counts77 complete/19 runtime pending/15 research held are unchanged. Use the single corrected native-capacity handoff; no HIP repeat. Earlier handoffs below remain historical.

Current handoff: [combined native CPU capacity screen](validation/native-capacity-implementation.md).134 host tests and verified local export pass; current CPU/native1/2/4 target collection remains pending. Accepted GPU ladder remains complete at adda1b6; do not repeat it. This next command needs no HIP or parent/input run. Counts77 complete/19 runtime pending/15 research held. Earlier handoffs below are historical.

Latest accepted scope: [combined hardware ladder PASS](validation/hardware-ladder-operator-20261024T220811Z.md).639 CPU/18 actual-HIP tests and all22 GPU roles pass at exact producer adda1b6. Batch128 achieves61.13% median replay interval reduction versus16 and about90.5% sampled GPU busy with2.84GiB role memory. Counts78 complete/18 runtime pending/15 research held. No duplicate run is requested; no production/learning admission. Earlier handoffs below are historical at their stated sources.

Current hardware handoff: [one combined overhead/capacity run](validation/hardware-ladder-implementation.md), covering ordinary-versus-telemetry and conditional batch32/64/128 comparisons in one archive.123 host tests pass; new CPU/HIP collection is pending. Accepted batch8/16 evidence stays complete; no learning-quality work is launched.

Latest batching result: [registered target PASS](validation/replay-batching-operator-20261024T203537Z.md),629 CPU/17 HIP tests and all roles complete. Batch16 cuts equal-work interval48.98% median, with GPU busy about44% and1.71GiB sampled role memory. No duplicate run requested.

Current hardware handoff: [equal-work batch8 versus batch16](validation/replay-batching-implementation.md). The accepted bookkeeping comparison showed10.54–11.28% component improvement. New batching code and112 host tests pass; target CPU/HIP batching validation is pending. Learning-quality/curriculum work remains held.

Latest accepted result: [comparison PASS](validation/replay-comparison-operator-20261024T193940Z.md):622 CPU/16 HIP tests and all registered roles pass; exploratory component reduction10.54–11.28%. No duplicate run is requested. Current priority is CPU/GPU/memory performance; learning-performance/curriculum work is held for a later decision.

Latest target review: [comparison preflight failure and fixture repair](validation/replay-comparison-operator-20261024T190425Z.md).621 CPU tests passed; HIP qualification had15 passes and one fixture thread-setting failure. No measured baseline/candidate arms ran. The fixture now configures actual threads in a fresh bounded process;106 host tests pass and corrected CPU/HIP validation is pending. Use the existing combined comparison command with a fresh output after updating the branch.

Current handoff: [bounded baseline-versus-hoisted replay comparison](validation/replay-comparison-implementation.md). Its separate manifest now registers one excluded candidate warmup and two fixed AB/BA exploratory pairs after current-source baseline calibration. Code and105 host tests pass; target CPU/HIP validation is pending. Use that one combined command for this increment. Original whole-pipeline/cost/research gates remain open.

Latest status: [baseline replay accepted](validation/replay-operator-20261024T172754Z.md) at its exact producer source. That baseline remains accepted. The new separately registered comparison above collects new-source baselines for its own prospective contract; full-pipeline and research gates remain separate.

Latest narrowed follow-up: [replay viability command and limits](validation/replay-implementation.md). The user requested continuing this separately versioned baseline-only scope; its registered target collection is now PASS as reviewed above. The original full-pipeline comparison retains its no-adoption result.

Current measurement decision: see the [fourth operator review](validation/operator-20261024T141450Z-review.md). The unchanged finite scope has no-adoption/inconclusive comparisons and incomplete costs; do not repeat it merely to seek a favorable completion. Any future measurement revision requires a new prospective identity under FR-011. Commands below remain the supported interface, not a request to retry the same scope.

# Operator guide: Spec 009

The pipeline CLI is implemented on `009-pipeline-throughput` in PR #60, stacked on PR #59. The current local evidence is **80 passing stdlib tests**, an actual stdlib CLI invocation, and a verified failure/capability archive. Pinned-image CPU/Torch/native, actual HIP and both-engine evidence are pending. No throughput gain, optimized default, study admission or scientific result is claimed. See [implementation closeout](validation/implementation-closeout.md) and the [task evidence index](validation/task-evidence.json).

## Default handoff: one combined command

Run from the repository root on the HIP/ROCm workstation:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both \
  --manifest configs/foundation/pipeline_validation.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-$(date -u +%Y%m%dT%H%M%SZ)"
```

These are the last recorded compatible parent/input paths from the [008 profile handoff](../008-bigint-constant-synthesis/validation/phase-5-pipeline-profile-plan.md#implemented-command-and-interpretation). If moved, replace only those two paths with their current workspace-relative locations. Use `--engine docker` to select Docker as primary. `both` explicitly records a missing second engine as a portability blocker; `available` checks the primary and other discoverable engine, while `selected` checks only the primary. One engine cannot qualify both.

The parent must be a completed, sealed `bigint-run/v1` with selected checkpoint, seed 20261016 and hybrid-chunks/v1 inputs. Prepared data identity, selected weights, native-admission status, visible examples and synthetic diagnostic truth hashes are checked. The parent is read-only. GPU roles receive only frozen diagnostic weights and visible synthetic data; the independent CPU scorer runs after GPU cleanup. Final/OEIS hidden inputs are not part of engineering validation. Missing neural inputs block their dependent stages while safe CPU stages remain eligible.

No separate build, native test, HIP test, profiler or packaging command is required. All engineering phases 1–5 are registered. Actual qualification depends on their results; `all-ready` never includes phase 6 research. Measurements and builds run serially, with start/end messages and 30-second heartbeats. Each engine shares one 14,400-second build/inspection allowance across its targets.

The hard global ceiling is 86,400 seconds, including build/report/cleanup, with at most 84,000 seconds declared reservations. Phase caps remain 3,600/10,800/7,200/18,000/14,400 seconds; export 900 and invocation cleanup 300 seconds are reserved. These are ceilings, not expected duration. **No current-source target calibration exists yet, so a defensible expected range is unavailable.** Calibration either freezes a bounded complete subset and reservations before candidates, or records an inconclusive/blocked scope. It never trims failed attempts to create a speedup or extends another phase's cap.

The workstation policy resolves before Host's older 16 GiB default. CPU/GPU/native work shares `min(floor(0.90 * physical RAM), effective engine/ancestor limit)`, rounded down to MiB, with at most 16 CPU equivalents. Expected active peak plus `max(1 GiB,10% peak)` headroom must fit known current capacity. Historical 16/capacity64 are named comparisons. Missing device/shared-memory accounting prevents resource qualification. Native configurable numerical threads are one; Torch uses two intraop/one interop threads.

## Results and reruns

Return the single sibling `OUTPUT.zip` printed by the command. It includes the resolved public contract, immutable stage receipts, compact attempt/job evidence, outcomes, source/policy hashes, bounded logs, omissions and a size/hash-verified manifest. Weights, private truth, full inputs and caches are excluded. Limits are 8 MiB per stream, 8 GiB transient, 128 MiB optional traces and 512 MiB ZIP. A physical reporting reserve protects minimal failure export; if the full export fails, a verified compact **FAIL** archive records omitted details and surviving paths. A recovery archive cannot qualify omitted stages.

The summary separates selected scope, engineering, learning, research, portability and longer-run admission. Exit codes: 0 selected-scope PASS; 1 FAIL; 2 invalid invocation; 3 BLOCKED/PARTIAL/INCONCLUSIVE; 130 interruption. `feature_complete` remains false while feature gates are pending. A valid no-adoption report is distinct from performance adoption or learning success. Baseline serial/v1 defaults remain supported; hoisted metadata, v2, native slots, overlap and batch/cache candidates remain experimental opt-ins until current evidence is reviewed.

Use the same command with a fresh `--output` for a rerun. Optionally add `--reuse tmp-validation/PRIOR_009_OUTPUT`. Reuse requires the complete prior directory or a manifest-verified extracted bundle, including all bound public members. Current installed-image inspection always reruns. Only compatible host/pure CPU PASS evidence is imported, with the original producer, costs and artifacts retained. Timing/calibration/confirmation always rerun; an interrupted five-pair block restarts in full. Research budgets are never refunded by a new output directory.

For a CPU/native-only targeted rerun, without a neural parent:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase 3 --engines selected \
  --output "tmp-validation/009-native-$(date -u +%Y%m%dT%H%M%SZ)"
```

For scope inspection only:

```bash
python3 -S scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both --scope-report
```

`--max-seconds` and resource/artifact caps may only lower the committed finite bounds. The committed engineering manifest uses a static, versioned workload/comparison roster. Nonempty custom `workloads` or `comparisons` overrides are rejected; a new roster requires a reviewed implementation/manifest version. Optional `source`, input/weight content references and up to 16 reuse references are checked before engine execution. Paths alone are not identities.

## Research manifests and one batch command

There is no admitted research manifest in this branch. Tasks 093–107 remain blocked on actual qualification, exact-scope cost and prospective scientific decisions. Original008:T096 stays negative at 7,730/14,400 seconds and008:T093 stays deferred. No synthetic authority or assistance data has been created to clear these gates.

After an independently reviewed study or batch exists, the same entry point runs it:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase 6 \
  --study-manifest configs/foundation/pipeline_studies/reviewed-batch.json \
  --output "tmp-validation/009-study-$(date -u +%Y%m%dT%H%M%SZ)"
```

The example path denotes a future reviewed file, not an available admission. Add `--scope-report` to inspect admission without executing. Engineering `--manifest` cannot be mixed with phase 6. An explicitly admitted final study additionally requires its identical `--final-plan`; exposure is recorded before the first prompt and survives interruption.

A single `pipeline-study/v1` record freezes kind, decision ID, hypothesis, seeds, lineage, arms, visible 20/horizon 100 denominators, stop/promotion rules, typed gates, cost, budgets and authority. Arms bind inputs/config/run/checkpoint by `{path,sha256}` and specify operation, seed and finite wall/work/disk amounts. Representation requires six train arms with `view=literal|parameterized` paired at 20261016/17/18. Curriculum and replay are separate decisions. Encoder sensitivity uses diagnose with four component masks plus baseline; selected retraining uses separately reviewed train arms. Assistance requires reviewed assisted lineage.

A combined manifest has this shape (illustrative references, **not executable authority**):

```json
{
  "version": "pipeline-study-batch/v1",
  "state": "FROZEN",
  "studies": [{"path": "configs/foundation/pipeline_studies/study-a.json", "sha256": "sha256:<content hash>"}],
  "authority": null,
  "budgets": {"wall_seconds": 7200, "work": 1000000, "disk_bytes": 1073741824, "cleanup_seconds": 30, "archive_seconds": 60},
  "identity": "sha256:<canonical payload hash>"
}
```

One to 16 independent study references are allowed. Their full reservations plus batch overhead must fit the batch budget, and each retains its own admission. A batch cannot approve missing child authority, nest batches, substitute seeds, invent inputs or expose final prompts. Blocked children retain their unstarted rosters in one combined archive.

Use `pipeline_validation.identity.seal` for canonical records. The reviewed authority's `decision_payload` comes from `studies.decision_payload`; authority/gate/cost/state references are excluded from that prospective payload to avoid a hash cycle. Study JSON under `configs/foundation/pipeline_studies` is independently content-addressed authority input, not compiled source. Runtime code/schema/defaults remain source-bound.

Typed `pipeline-study-gate/v1` records contain scope,state,source,image,resources,producer,engine and identity. They are reviewed attestations to retained producer evidence, not self-issued PASS values. All current gate envelopes must agree. Resource identity uses `resources.execution_identity` of the measured resolved policy; current availability is rechecked but does not rename the envelope. A smaller CPU/memory request needs corresponding reviewed qualification. `pipeline-study-cost/v1` binds scope,source,decision payload,projected seconds,budget seconds and within-budget decision. Projection must fit 80% of the exact reviewed allowance. The reduced 576-attempt diagnostic cannot admit full 1,024-attempt diverse32 controls or 8,192-attempt transfer controls; missing material production strata keep forecasts blocked.

Durable reservations live under `run-state/pipeline-studies`, keyed by decision ID, and are written before dispatch. Unknown work stays charged. Work units count updates, teacher rows, candidate services and scoring events according to `studies.validate_recipe`; this is a mixed-work ceiling, not CPU seconds. Nonfinal interrupted arms do not restart under a fresh output. Only an identical final plan with predeclared smaller attempt reservations may finish within its remaining cumulative allowance. No manifest authorizes automatic twelve-hour-plus training.

See the [operator](contracts/operator.md), [scheduling](contracts/scheduling.md), [comparisons](contracts/comparisons.md), [research](contracts/research-gates.md) contracts and [carryover register](carryover.md). Target evidence must be reviewed before checking the corresponding runtime or research tasks.
