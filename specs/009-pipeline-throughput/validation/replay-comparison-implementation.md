# Cache bookkeeping replay comparison: implementation handoff

The next phase-2 investigation is implemented under the separate
[prospective contract](../contracts/replay-comparison.md) and
`configs/foundation/pipeline_replay_comparison.json`. Existing default and
baseline-only manifests retain their identities. The accepted baseline replay
archive stays historical PASS; this new comparison has not run on HIP yet.

## Implemented scope

The one-command graph shares current images, CPU qualification, actual-HIP tests
and a visible-only CPU reference. It collects the five registered baseline roles
at the new source, then seals a child before any candidate dispatch. The child
binds source/validation/weights/reference/workload, all baseline receipts,
calibration, device/memory/threads, ordered slots and limits. Host file hashes and
isolated expected child/slot identities reject changed authority. Exactly one
excluded `hoisted/v1` warmup and two exploratory pairs (`AB`, `BA`) follow.

Both arms use frozen BOS streams, eager FP32, batch1/8 and all12 dense/mixed cache
boundaries. Each checks finite logits, existing HIP tolerances and exact top-one
tokens against the same private CPU reference. Fresh roles/weights/KV caches,
bounded aggregate1Hz observations, device/resource audits, boundary progress and
external cleanup are shared. No per-token synchronization, decoder/model change,
native fitting, hidden truth, learning or final access is introduced.

The comparison reports complete replay intervals through verification/export,
both pair durations and descriptive gain median/range only if all five scheduled
slots complete consistently. Missing, changed or partial arms remain visible and
produce INCONCLUSIVE without ratios from a favorable subset. Both admission
flags are always false. Aggregate telemetry overhead is still uncalibrated;
these are exploratory component observations, not production confirmation.

## Completed available checks

The full stdlib host registry passes **105 tests**. New regressions cover the
closed manifest/graph, prospective freezing, complete baselines, immutable child
and slot custody, deadline allowances, resource/thread mismatches, partial slot
reporting and denied adoption even for favorable synthetic timings. The existing
host orchestration fixture now drives reference, all five baselines, freeze and
all five comparison roles with their actual host adapters and checks the final
report. No fixture timing is hardware evidence.

Pinned CPU and actual-HIP fixtures now also exercise hoisted replay against the
serialized CPU oracle; the HIP fixture invokes both real comparison arms and
rejects changed slots. They are authored/registered, **not executed locally**.
No Torch, pytest, Docker or Podman is available here, and no host dependencies
were installed. Compilation and whitespace checks pass.

The actual public stdlib invocation exits3 with missing-engine PARTIAL and
INCONCLUSIVE comparison, all five candidate/comparison slots unstarted, and a
verified53-listed-member archive. Source/manifest/archive hashes and stage states
are in [replay-comparison-local-evidence.json](replay-comparison-local-evidence.json).
The refreshed007 package binding preserves all68 dispositions and incomplete
foundation acceptance. All original111 tasks remain accounted for: **77 complete,
19 pending runtime evidence,15 blocked research**. T110 is reopened only for the
new affected source; its earlier exact-source closure remains historical.

## One target command

After the code work above, run from the repository root on the HIP/ROCm machine:

```bash
git pull --ff-only origin 009-pipeline-throughput
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_replay_comparison.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-replay-comparison-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use the same completed parent/input directories; replace their locations only if
moved. Return the **single sibling ZIP**, including any non-PASS outcome. No
separate build/test/reference/profile/packaging command and no automatic retry.
Podman-only selection leaves Docker portability pending.

The earlier baseline invocation took15m21s including prerequisites/images; its
three replay intervals were29.60–29.78s. Candidate performance at this new source
is unknown. Five additional baseline-like roles would add roughly150s of replay
work if they behave similarly; this is planning context, not a candidate forecast
or full invocation estimate. Build/qualification cost can dominate.

Each baseline/reference role retains its600s external/570s internal ceiling.
The child derives a component reservation from exactly three new complete baseline
intervals, then adds30s for internal setup and30s for outer startup/cleanup.
With the prior30s observations this would mean60s component,90s internal and120s
external, **not** a60s whole-container deadline. If the derived external bound
exceeds600s, candidates are blocked rather than capped or enlarged. Static maxima
are ten GPU roles×600s, plus HIP preflight/reporting within phase2's10800s;
CPU reference is phase1. Phase/global/build/disk/memory limits are unchanged.

## Remaining original gates

This can measure the single bookkeeping intervention on complete forced replay.
It cannot close free-generation equivalence, full fixed-budget coverage,
whole-pipeline five-pair confirmation, complete diverse32/transfer costs,
both-engine operations or research admission. Original no-adoption/inconclusive
archives and32/576/full-control denominators stay unchanged. Serial defaults
remain.008:T093 stays deferred and008:T096 negative at7730/14400s. No research,
final exposure or longer run is launched. PR CI is not awaited.
