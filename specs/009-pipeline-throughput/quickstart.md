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

The workstation policy resolves before Host's older16 GiB default. CPU/GPU/native work shares `min(floor(0.90 * physical RAM), effective engine/ancestor limit)`, rounded down to MiB, with at most 16 CPU equivalents. Expected active peak plus `max(1 GiB,10% peak)` headroom must fit known current capacity. Historical 16/capacity64 are named comparisons. Missing device/shared-memory accounting prevents resource qualification. Native configurable numerical threads are one; Torch uses two intraop/one interop threads.

## Results and reruns

Return the single sibling `OUTPUT.zip` printed by the command. It includes the resolved public contract, immutable stage receipts, compact attempt/job evidence, outcomes, source/policy hashes, bounded logs, omissions and a size/hash-verified manifest. Weights, private truth, full inputs and caches are excluded. Limits are 8 MiB per stream,8 GiB transient,128 MiB optional traces and 512 MiB ZIP. A physical reporting reserve protects minimal failure export; if the full export fails, a verified compact **FAIL** archive records omitted details and surviving paths. A recovery archive cannot qualify omitted stages.

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

There is no admitted research manifest in this branch. Tasks093–107 remain blocked on actual qualification, exact-scope cost and prospective scientific decisions. Original008:T096 stays negative at 7,730/14,400 seconds and008:T093 stays deferred. No synthetic authority or assistance data has been created to clear these gates.

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
