# Consolidated HIP validation and evidence reuse

This is the execution design for [tasks.md](tasks.md), not a claim that device tests have run. It refines FR-011/018/024/038 and the existing execution contract without relaxing source closure or actual-device requirements.

## Intended number of handoffs

Aim for **one planned workstation handoff after the implementation and CPU/native tests are stable**. That handoff has two lifecycle phases, because qualification precedes preparation/freeze and acceptance must validate the frozen campaign:

1. **Q — consolidated qualification**: one serial batch of the selected device capability cases against one immutable source/image pair. Include reusable feature probes in this batch when their exact diagnostic inputs are already fixed.
2. **A — post-freeze acceptance**: reconcile the frozen campaign with the original Q producers. Reuse matching case evidence and run only missing campaign-dependent cases together in one bounded delta batch. If all required device cases already match, acceptance can bind/reconcile those original receipts without another GPU test batch; it still performs its non-device checks.

This is a target of **one consolidated qualification batch plus at most one planned acceptance delta for unchanged scope**, not a promise of one process or permission to omit a test. Separate role processes, isolated outputs, controlled worker restarts and crash recovery remain necessary. Newly discovered defects or new execution scopes can require more validation.

Freeze a separate bounded DIAGNOSTIC_ONLY acceptance campaign from the sealed tiny fixtures when research preparation lacks fresh support or other scientific prerequisites. This lets all-stage software/device acceptance finish without manufacturing a research cohort. The diagnostic campaign never authorizes research; a later real campaign needs its own complete freeze and acceptance scope binding, reusing only genuinely matching original capability producers. Required source/hardware/containment prerequisites still apply to the diagnostic campaign.

Do not run one HIP session per user story, per test-file edit, per training seed, or simply because a new checkpoint was produced. Scientific training/evaluation still executes its actual GPU work, and paired seeds are not collapsed into one run. Qualification reuse reduces repeated validation, not the experiments' scientific replication.

## Prepare everything before Q

- Finish the full implementation, independent CPU arithmetic/metric tests, real CPU learning/update/recovery tests, native acceptance, role isolation, installed-wheel checks, configuration/lock changes and the selected device test inventory.
- Resolve the actual retained model recipe and every candidate shape before qualification. Trial proposals may vary only inside the already declared qualified capabilities. Do not treat hardware-only model-sizing fixtures as research constructors.
- Construct fixed DIAGNOSTIC_ONLY fixtures for structural refresh, archive admission, component use and four-term source input. These may force a verified candidate to demonstrate machinery; they establish no autonomous discoveries or scientific gains.
- Compute a concrete case matrix, exact IDs and finite aggregate deadline. Share native build/image preparation and bounded immutable fixture inputs; allocate separate owned result/checkpoint directories per case. Use one GPU owner, with CPU services under the frozen shared-memory envelope.
- Commit the source/test/policy group before Q. Do not edit code during the session. Record missing hardware as BLOCKED and retain the ready-to-run inventory; a skipped device test is not qualification.

All of Q and A, preparation, packaging and any separately allocated sustained acceptance consume the relevant selected campaign reservations. The full-path 12h qualification/acceptance envelope is not automatically available to smaller subsets. If measured work cannot fit, stop for a prospective redesign rather than silently enlarging it.

## Case matrix and original producers

Each case records source closure, inspected installed package and native binary identities, CPU/GPU image IDs, engine/runtime/device and driver facts, precision, model architecture/shape, effective dropout, input profile/length, optimizer/update path, fixture/policy identities, numerical/language profile, declared limits, actual measurements and its original producer references. Include the supported parameter domain where a producer intentionally qualifies a bounded configuration family; absence of that evidence is not wildcard coverage.

| Case family | Device evidence that must actually exist | Consolidation rule |
| --- | --- | --- |
| Model/update capability | Finite forward/backward, nonzero parameter update, finite optimizer state, actual parameter count and measured memory for each selected shape/dropout path | Deduplicate exact scope keys and run serially in Q. Cover all allowed selected paths in advance; do not run another identical test for each seed/trial |
| Logical accumulation | Correct complete-update behavior, accumulation/non-accumulation path and interrupted-microbatch accounting on HIP | Mathematical weighting equivalence is first proved by independent CPU tests with dropout disabled; HIP adds actual device execution and compatible restore evidence |
| Full recovery | Save/restore model, optimizer/scheduler, device/framework RNG and dynamic stream/allocation state; compare compatible uninterrupted/continued results | Use one shared production-path scenario with separately sealed subcase receipts. Check deterministic behavior under the declared policy; no new tolerance selected after a mismatch |
| Static/refreshed support | Actual device updates consume verified new structural shards while preserving ordered cursors and charges | Fold into the full recovery scenario; use known small diagnostic data, not a live diversity research result |
| OEIS admission/replay | Candidate passes native/reference admission, committed target enters a real update, archive cut restores; prefix-only negative stays excluded | Native truth checks need no GPU; device target consumption can be a phase of the integrated scenario |
| Learned components | A mined training component affects an expanded candidate, is independently accepted and participates in an actual update with library-state binding | Reuse decoder vocabulary and normal model path; do not add a model-size matrix just for transparent component expansion |
| Short source profile | Real four-input encoding/loss/update, full restore, generation and nine-index score path | At least one actual selected shape/dropout must cover this distinct profile. A 20-input receipt does not cover it. If the later selected source model differs, qualify that exact missing scope |
| Decode and native handoff | Actual checkpoint generation, grammar completion under declared limits, fit/final-source acceptance and deadline accounting | Share with the integrated candidate paths and keep drain. Do not substitute canonical programs for checkpoint-generated evaluation evidence |
| Cleanup and restart | Bounded owned-worker interruption, durable state, reclaimed resources and successful restart on the selected engine/device | Deliberate process restarts occur inside the same operator handoff; never count a process launch as a separate research experiment |

The full supported hyperparameter domain must be explicit. CPU validation covers scalar constraints, shape/head divisibility, optimizer equations and schedule semantics. Device receipts must cover the claimed model/dropout/input/update paths and their declared memory/work envelope. Do not infer that one small inference test qualifies every architecture, longest target or training batch. Where broad qualification has not been demonstrated, narrow the selectable configurations or add the missing bounded case to Q/A before that configuration runs.

The test module uses the repository GPU marker and mandatory-device mode. It must verify both HIP build/runtime availability and actual device tensors/updates; a torch `cuda` API name alone does not distinguish ROCm from another backend. Mandatory qualification fails on unavailable HIP, CPU fallback, skipped required cases, nonfinite state, absent parameter change or missing producer evidence.

## What acceptance may reuse

Reuse means validation of **original producer bytes**, not copying a PASS field into a new receipt. The acceptance collector must check:

1. Original producer inventory, hashes, result/test identity, requested versus executed cases and successful mandatory-device outcome.
2. Current source/package/image/native/runtime/device facts and every relevant scope field against the frozen request.
3. Exact fixture/input identity for the behavior being claimed. A tiny fixture can prove mechanics only; it cannot prove the real cohort's availability, checkpoint performance or scientific gain.
4. Correct ancestry and costs. Reused evidence references its original execution once; no double charging and no erasure of that cost. Reuse verification overhead remains measured.
5. Required post-freeze cases whose inputs, selected capabilities or behavior were not covered. Collect those into A; report an empty delta only when genuinely complete.

Cache **qualification evidence** by compatible scope. Do not use it to cache away experiment generation, admission or scoring work whose execution/budget is part of the scientific comparison. Full reports retain which cases ran, which reused evidence, why reuse was allowed and what remains missing.

## Invalidation and scope changes

`pipeline_validation/closures.py` currently uses a conservative whole-source dependency closure. Preserve it. A changed mapped source, test, dependency, installed policy/schema or native binary invalidates affected reuse under that rule, even if the edit looks small. Do not add ad hoc path exclusions, downgrade provenance checks or introduce a new fine-grained cache just to meet a target session count.

| Change | Required action |
| --- | --- |
| Different seed/checkpoint produced by the same qualified compatible learning policy | Verify checkpoint lineage and scope; no blanket requalification solely for the new weights |
| Different already-covered recipe settings | Check exact receipt domain and current facts; reuse only within it |
| Uncovered shape, dropout path, input horizon, precision, language or update semantics | Freeze the concrete child scope, add independent CPU/native tests, then batch its missing HIP cases before production |
| Code/test/policy/lock/image/runtime/device change that breaks compatibility | Invalidate under the existing closure rules; fix and complete CPU preflight before the next consolidated device batch |
| Presentation-only findings update genuinely excluded by the existing closure policy | Preserve valid producers while rechecking current source/package facts; do not assume all spec documentation is presentation-only |
| Hardware unavailable or receipt lost/truncated | Report BLOCKED/INCOMPLETE; CPU evidence and reference hashes alone cannot supply the missing producer |

Use CPU/native tests for Docker/Podman portability where those tests establish the relevant boundary. HIP evidence identifies the actual engine used. Running on one engine does not prove the other's GPU behavior; only repeat hardware testing on another engine if that GPU scope is selected and required.

## Sustained execution without a redundant training run

Short Q/A tests establish mechanics, not hours of stable execution. Before a sustained run, freeze actual target duration, observation cadence, task boundaries, interruption/recovery event, memory/disk/thermal/throughput observations, acceptance rules and total allowance. Missing sensors are explicit nulls with reasons.

Prefer collecting this evidence during the first already-eligible two-task research path after shortened acceptance, using the same installed production code and complete checkpoints. The scientific design must declare any interruption/recovery test, include charged/discarded work and preserve matched comparison budgets. Do not alter a frozen research schedule after outcomes merely to satisfy an engineering checklist.

If the selected research cannot supply the required sustained scenario, use a separately preallocated acceptance path or keep sustained evidence pending. Do not impose an extra 24h soak by default, claim a two-minute clock fixture demonstrates 24h behavior, or run a failed candidate longer just to collect duration. No individual training task exceeds 24 active hours, and actual measured duration is always reported.

## Handoff output

Implement the canonical `qualify → prepare → freeze → acceptance` sequence from quickstart using the existing portable host interface. It may be wrapped in one operator script after those commands exist; do not create a new host service or an arbitrary-step bypass. Stop at the first blocking dependency and preserve the first producer cause. Resume compatible remaining cases without overwriting completed outputs.

Return one inventory of missing inputs upfront, a scope/reuse table, per-case real outcomes, measured resource/timing data, complete failure details and one bounded diagnostic package. Repository publication contains concise findings and original receipt identities; weights and full outputs remain workstation artifacts.


## Packaged lifecycle case inventory

The shared [progressive-acceptance.json](../../src/oeis_learn/generalization/policies/progressive-acceptance.json)
inventory registers the CPU installed-wheel lifecycle test and queues
`progressive-logical-update-restore` and `progressive-two-task-continuation-pause`
for the consolidated T090/T091 actual-HIP batch. Their QUEUED status supplies no
device authority. The shortened lifecycle receipt records actual subprocess
microseconds/hours and the future sustained protocol; its simulated task ceilings
are never reported as elapsed research hours. Use the existing Q/A handoff and
selected campaign reservations described above.
