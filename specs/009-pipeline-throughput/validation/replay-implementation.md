Subsequent increment: [cache bookkeeping replay comparison](replay-comparison-implementation.md) now has a separate registered manifest and target command. This baseline-only scope remains accepted at its recorded producer; its historical command below is not the current comparison handoff.

Latest target status: **registered replay scope PASS** at exact source `69807d38` / producer `c9be16a`. CPU613/HIP16 tests and the reference plus all five baseline trials pass. See the [accepted operator review](replay-operator-20261024T172754Z.md). R005 and affected-source T110 close; no duplicate target run is requested. The command below is retained for reproducibility, not a request to repeat it.

# Baseline replay viability: implementation handoff

The user requested continuing the narrowed follow-up after the source-matching fourth archive. Code now registers that path under its own manifest/version/registry. The [exact replay contract](../contracts/replay.md) extends the plan without changing the original validator, numerical rules, production/native deadlines, phase/global ceilings or original no-adoption result. No candidate, research study or final access is enabled.

## Implemented behavior

- Existing current-image, CPU and actual-HIP prerequisites run through the same public entry point and static DAG. Parent preparation reuses completed/sealed parent, selected weight/checkpoint and visible-data hashes, returning before hidden truth is read. The first eight fixed visible training rows supply a decoder-memory fixture; this does not reduce the original generation-success denominator.
- A separate CPU-only role computes frozen encoder memories and independent full-prefix expected logits at batch 1/8, offsets 1/32/128/513/1024. Baseline mixed/finished/reordered/inactive cache behavior is included. Tensor shapes/dtypes/content hashes and a serialized reference hash are sealed before any GPU timing. Raw tensors stay private, and no parent weights or optimizer are changed.
- Exactly five fresh HIP roles run cold, excluded warmup and three baseline repetitions. Each uses eager FP32 baseline bookkeeping, forced BOS input streams and the same reference/weight bytes. CPU/HIP logit tolerances and exact top-one outputs are enforced. No generated program, hidden scoring, solver or update work is measured. Fresh roles do not imply persistent process-cache warming.
- Each role has a 600-second external bound and an internal 570-second deadline, reserving startup/cleanup within the external ceiling. Five GPU trial ceilings total 3000 seconds; CPU reference has its own 600 seconds within phase 1. The existing 3600/10800 phase ceilings and 86400 global ceiling remain. Aggregate memory admission, device isolation and five-second shared cleanup remain mandatory.
- A failure blocks later trial execution. Independent reporting retains every missing/unstarted slot. Each role durably plans all 12 boundaries, distinguishes queued/unverified work from verified completion, and retains partial outcomes if interrupted; a missing terminal is never inferred complete. Only complete reference-matching, resource-qualified, device-consistent results can produce the three-observation duration bound. Every replay report denies optimization adoption and research-cost admission. The top-level summary explicitly labels `engineering-replay`.
- Existing bounded 1 Hz CPU/GPU observations retain missing-sensor reasons and actual thread settings. No per-token synchronization or trace is added. Observation overhead is uncalibrated and no production performance claim is made.

## Available validation and pending gates

The new host tests cover closed manifest authority/overrides, scope selection under `python3 -S`, five bounded trials, frozen streams/mapping, missing/mismatched/resource-inconsistent results, immutable reference dispatch, separate CPU/GPU roles, hidden-truth-free preparation, twenty-term visibility limits and rejection as complete research-cost evidence. CPU and actual-HIP fixtures exercise real serialized reference tensors and role functions with tiny frozen test weights, plus numerical boundaries, changed references, nonfinite/missing outputs and expiry. These runtime fixtures are registered in the existing image suites; they are not substitutes for the parent-weight target measurement.

Final source/archive identities and the actual public CLI result are recorded in [replay-local-evidence.json](replay-local-evidence.json). The stdlib suite passes **99 tests**. Missing local Docker/Podman yields exit 3 and a non-admitting partial archive; no local Torch/native/HIP result is claimed. Compilation and whitespace checks pass. The 007 implementation binding is refreshed while preserving all 68 dispositions and incomplete acceptance. The original 111-task ledger now records 78 complete / 18 runtime-evidence pending / 15 research blocked; all five supplemental replay items are complete for their stated scope.

Older source-matching CPU/HIP passes remain historical after these code changes. New affected-source validation is necessary. The fourth archive's empty full-work baseline, incomplete native calibration, missing full-control cost and no-adoption decision are preserved. Replay success cannot close those collection/adoption gates, Docker portability, learnability, studies, final exposure or longer-run authority. PR CI is not awaited.

An observed earlier CI run at `4aef0f6` exposed the same new replay fixture failure in native qualification (`120929423760`) and arithmetic qualification (`120929423901`): the CPU prompt was created under `inference_mode`, while the decoder cache correctly requires mutation-version counters. The replay/reference paths now use `no_grad`, preserving the existing cache contract rather than bypassing its guard. The registered CPU fixture exercises that exact path; this correction is now validated by the source-matching target CPU/HIP run and both successful producer CI workflows recorded in the accepted operator review. Earlier Foundation CPU and portable-engine successes are not represented as successful replay validation.

## One target command

After updating the existing branch, from the repository root on the HIP/ROCm workstation:

```bash
git pull --ff-only origin 009-pipeline-throughput
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_replay_validation.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-replay-$(date -u +%Y%m%dT%H%M%SZ)"
```

Substitute input locations only if moved. No separate build/test/reference/profile/export commands are needed. Return the single sibling ZIP, including a non-PASS result. No automatic retry follows a partial trial. Podman-only selection leaves Docker portability pending. Current images/source are inspected, and changed dependency closures may invalidate old CPU reuse. Builds use existing dependency layers where compatible; no skip or complete runtime estimate is promised. There are six replay/reference role ceilings of 600 seconds plus prerequisites/builds/reporting within the unchanged 24-hour maximum. These ceilings are not expected durations.

The decision this run can resolve is whether the registered baseline-only replay work completes reproducibly enough to establish its own finite observational bound. It cannot select a hoisted/native/overlap winner or authorize a new research run. Those remain later prospective decisions with the original full-pipeline and complete-cost requirements.
