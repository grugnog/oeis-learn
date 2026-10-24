This scope is now accepted at producer d74d30e; [target review](replay-batching-operator-20261024T203537Z.md). The command below is historical reproduction guidance, not a request to repeat accepted work.

# Equal-work batch8/16 hardware screen: implementation handoff

Current priority is CPU/GPU/memory throughput and capacity, per the user's latest
instruction. Learning-quality investigations, curriculum changes and multi-day
training remain held. The [accepted bookkeeping comparison](replay-comparison-operator-20261024T193940Z.md)
supports carrying hoisted/v1 forward experimentally; its10.54–11.28% component
reduction is preserved, without production adoption or a repeat request.

## Implemented hardware work

The separate `configs/foundation/pipeline_replay_batching.json` manifest registers
the [equal-work batching contract](../contracts/replay-batching.md). It shares
existing image/CPU/actual-HIP prerequisites and visible-only parent preparation.
It freezes32 distinct-ID training rows with20 visible terms and32768 supplied BOS
tokens. A CPU-only role freezes encoder memory and full-prefix outputs at five
positions through1024 for every row:160 row boundaries per measured role.

Both arms use hoisted/v1, eager FP32, the same frozen weights/memories/streams and
all32 rows. Batch8 processes four contiguous groups; batch16 processes two. Only
decoder batching changes. No generated-source, fitting, scheduling, precision,
model-size, training-batch or learning-algorithm intervention is introduced.

One cold batch8 role, one excluded warmup and three baseline calibrations precede
the sealed child. Qualified real resources, actual2/1 threads, reference/workload,
all receipts, slot order, limits and memory projection are bound before candidate
timing. Candidate memory prediction is at least8GiB or twice the largest observed
baseline role upper bound, whichever is larger, plus headroom. It must leave10%
of the enforcement ceiling unreserved. Actual dispatch independently accounts for
weight-derived allocations and currently available capacity; these cannot borrow
that10% reserve. Unknown/insufficient capacity blocks candidates.

Then one excluded batch16 warmup and exactly two fresh pairs AB/BA execute.
All160 row logits must be finite, match CPU/HIP tolerances and have exact top-one
tokens. Durable row plans, queued/unverified events and terminal states account
for all work. A failure blocks later roles; independent reporting keeps every
missing slot. Aggregate1Hz CPU/GPU observations, memory qualification, device
isolation, external supervision and cleanup remain shared. Reports give complete
intervals, pair gains and **supplied** tokens/second, with both adoption/admission
flags false. Telemetry overhead remains uncalibrated.

No batch32 timing or open-ended sweep is registered. A later named branch would
need batch16 evidence and its own finite contract/capacity check. Larger memory
use is useful only if it improves completed throughput within the bounds.

## Available validation

All **112 stdlib host tests pass**. New coverage checks closed authority/graph,
equal denominators, hidden-truth-free32-row preparation, complete calibration,
missing/wrong-size arms, capacity prediction, actual weight headroom, immutable
child/slot/reference custody and the complete host reference/calibration/pair/
report path. The actual public `python3 -S` command exits3 when the engine is
absent and exports a verified53-listed-member non-admitting archive. Exact source,
manifest and archive hashes are in [local evidence](replay-batching-local-evidence.json).

CPU and actual-HIP fixtures now exercise batch8 and16 over real serialized32-row
oracle tensors, both real comparison roles, changed slots/references, nonfinite/
missing outputs and expiry. They use the corrected isolated-process thread
fixture. They are authored/registered, **not executed locally**: Torch/pytest and
container engines are unavailable. Compilation and whitespace checks pass. No
host dependency was installed. The007 implementation binding is refreshed with
all68 dispositions and incomplete foundation acceptance preserved.

## One target command

After completing the code work above, run from the repository root:

```bash
git pull --ff-only origin 009-pipeline-throughput
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_replay_batching.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-replay-batching-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use the existing completed parent/input paths unless moved. Return the single
sibling ZIP, including non-PASS outcomes. No separate build/test/profile/export
commands or automatic retries. This is a new equal-work batching scope, not a
repeat of the accepted bookkeeping comparison. Podman selection leaves Docker
portability pending.

The previous comparison invocation took19m18s, but used a different replay work
shape. The new32768-token/32-row interval has no target calibration yet, so a
reliable total-duration forecast is unavailable. The command establishes its
baseline before any candidate. Each reference/baseline has600s external/570s
internal maximum. Derived candidate bounds are component reservation plus30s
internal setup and another30s outer startup/cleanup; exceeding600s blocks the
candidate. Ten GPU roles×600s plus HIP/reporting fit the unchanged10800s phase2
cap. Builds share14400s per engine;86400s global/report/cleanup caps remain.

## Remaining scope

This screen can establish whether batch16 improves complete forced-replay
throughput and how CPU/GPU/memory observations change. It cannot adopt a full
pipeline configuration, close generated-program equivalence or admit training.
Native worker scaling, bounded overlap, complete pipeline timing and sustained
operations remain subsequent hardware tasks. A separately planned multi-day
training experiment, after hardware qualification, will inform organic/stochastic
versus curriculum-driven learning. It is not launched here. Historical results,
denominators and all numerical/provenance/final-data protections remain.

Original task counts are77 complete/19 runtime-evidence pending/15 research held
or blocked after reopeningT110 for newly affected source. Accepted009:C004 stays
complete at its recorded source. Batching implementation tasks are complete;
target batching collection remains pending. PR CI is not awaited.
