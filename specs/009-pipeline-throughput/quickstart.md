# Planned operator guide: Spec 009

**Status**: These pipeline options are the implementation target, not available commands in this planning PR. No validation or training run is requested now. [tasks.md](tasks.md) defines the implementation sequence; phase 1 is next.

## Default handoff: one combined command

After implementation, from the repository root, with the completed bigint run and its compatible prepared inputs available:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready \
  --manifest configs/foundation/pipeline_validation.json \
  --run runs/EXISTING_COMPLETED_RUN \
  --inputs tmp-fresh-inputs-host \
  --output "tmp-validation/009-$(date -u +%Y%m%dT%H%M%SZ)"
```

The implementation handoff must replace `runs/EXISTING_COMPLETED_RUN` with the actual compatible local path and confirm the input path. No separate build, native test, HIP test, profiler or packaging command should be necessary. Use `--engine docker` for Docker. The command defaults to checking other usable supported engines for portability while running GPU measurements only on the selected primary engine. A missing second engine can remain an explicit portability gap or be satisfied by a compatible imported receipt; it does not turn into a false PASS.

The new manifest defaults to sixteen CPU equivalents and up to90% of measured physical RAM, as authorized by the user on the128 GB shared-memory workstation. CPU/GPU/native work shares that aggregate allowance. The old16 GiB limit remains a named historical comparison profile, not the default ceiling. Current host/engine/ancestor limits and availability are checked and reported before work. Available memory is not a requirement to fill it; the bounded comparison determines whether larger batches/caches help.

All ready phases run without pausing at phase boundaries. Builds and compatible prerequisite checks are shared, measurements competing for CPU/GPU are serialized, progress appears at least every30 seconds and one archive is produced. The24-hour combined external ceiling includes builds; actual expected duration is reported from the resolved scope and calibration. This is an operational maximum, not a prediction or permission to train for24 hours. Selected stages have smaller finite caps and the command stops admitting work when cleanup/archive reserves would be threatened.

## Results to return

The command prints the fresh output directory, `summary.json` and sibling ZIP path. Return that single ZIP. It contains stage receipts, compact outcomes for every prompt/arm, relevant bounded logs, resource summaries, exclusions and a verifiable file manifest. Weights and private/final truth are excluded.

The summary separately reports:

- the phases/stages selected and their PASS/FAIL/BLOCKED/PARTIAL/INCONCLUSIVE outcomes;
- current engineering/device/portability qualification;
- measured adoption or no-adoption, including complete-work and fixed-budget results;
- remaining learning/research gates and historical T093/T096 dispositions.

Missing device capability, incomplete comparisons and failed checks remain visible. A combined run can collect useful independent evidence after one check fails, but it cannot report overall selected-scope PASS when a mandatory selected check failed. Phase6 research is not part of `all-ready`.

## Targeted phase or rerun

The same entry point supports one phase plus prerequisites. For example, after the concurrency implementation is ready:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase 3 \
  --manifest configs/foundation/pipeline_validation.json \
  --reuse tmp-validation/EXISTING_VALIDATION \
  --output "tmp-validation/009-native-$(date -u +%Y%m%dT%H%M%SZ)"
```

Phase3 native fixed-candidate comparisons do not require a model or GPU; the manifest supplies the qualified candidate fixture corpus. If a selected prerequisite does require checkpoint/device evidence, it is imported only if compatible or reported blocked. `--reuse` is optional. Always create a fresh output root. Failed bounded stages rerun from the beginning; the command preserves old failures, costs and hashes rather than overwriting evidence or resuming training.

Choose `--phase 1`, `2`, `3`, `4` or `5` as needed; neural phases require the declared `--run`/`--inputs`. Prefer `all-ready` when multiple checks can be collected together, even when it takes longer. `--scope-report` prints the resolved work and ceilings without starting it.

## What each handoff resolves

| Phase | Question answered | Hardware / authority |
| --- | --- | --- |
| 1 | Can one command validate, measure and export a trustworthy baseline, including failures? | CPU host/fixture work plus target HIP baseline where applicable |
| 2 | Does reduced decoder bookkeeping preserve candidates and improve generation? | Actual HIP and frozen diagnostic weights; no reusable training |
| 3 | Does deterministic selection plus bounded native concurrency preserve exact admission and scale? | CPU/native; explicit serial policy-only comparison before concurrency |
| 4 | Does GPU/CPU overlap and selected batch/cache/resource policy improve the complete pipeline? | Actual HIP, shared-memory capacity, source-bound paired comparisons |
| 5 | Are current-source operations and runtime/device gates complete, and which learning scopes fit a measured budget? | Relevant CPU/HIP and both-engine coverage; engineering authority only |
| 6 | What do already admitted learning/research studies show? | Separate prospective study decisions/cost admission; final exposure is explicit |

## Research handoff

Only after the relevant gates pass and a study has an explicit admitted manifest, the implementation supplies a command of this shape:

```bash
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase 6 \
  --study-manifest PATH_TO_ADMITTED_STUDY \
  --output "tmp-validation/009-study-$(date -u +%Y%m%dT%H%M%SZ)"
```

That one invocation runs the authorized batch and its subordinate checks/controls/reporting. It cannot invent future recipe choices or final-cohort authority. Final evaluation additionally requires an explicit immutable `--final-plan`; it will receive its own concrete handoff after decisions are frozen. T093 remains deferred and original T096 remains negative. No new learning or final evaluation is admitted by this guide.

See [operator contract](contracts/operator.md), [comparison contract](contracts/comparisons.md), [research gates](contracts/research-gates.md) and [carryover register](carryover.md) for exact bounds and interpretations. There is no need to repeat every earlier Spec 008 diagnostic command simply to start Spec 009.
