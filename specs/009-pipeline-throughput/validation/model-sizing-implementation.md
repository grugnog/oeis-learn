**Superseded handoff**: [First target review](model-sizing-operator-20261025T170755Z.md) records a HIP encoder parity prerequisite failure before all15 timing roles. Use the [corrected execution-policy handoff](model-sizing-execution-repair.md):170 host tests/export pass; actual corrected CPU/HIP remains pending. The original source-bound details below are historical.

# One-command model-size hardware screen

The user requested increasing model capacity before the long-run choice. This is
a prospective revision to [hardware convergence](hardware-convergence.md), after
the accepted co-scheduling result. The [new contract](../contracts/model-sizing.md)
and `configs/foundation/pipeline_model_sizing.json` implement a separate hardware
scope. No accepted prior size/worker/overlap experiment needs repetition.

| Profile | Width | Encoder/decoder layers | Heads | Parameters | Prospective peak reservation before headroom |
| --- | ---: | --- | ---: | ---: | ---: |
| m8 |256|4 /4|4|8,118,294|11.24GiB|
| m32 |512|4 /4|8|31,903,510|14.95GiB|
| m100 |768|6 /6|12|104,496,662|24.61GiB|

Parameter counts are calculated from the exact constructors and asserted against
actual parameters in the pinned runtime; they have not been executed locally.
Reservations are conservative estimates, not measured peaks. Existing headroom,
known current availability, ten-percent unreserved capacity and aggregate90%-
physical/ancestor/engine enforcement remain mandatory. Native4/Torch2/1 share one
at-most16-CPU quota with one GPU role; CPU references have no accelerator devices.

The production/default model remains frozen256/4+4. An explicit closed hardware
profile generalizes only the encoder projection dimensions and transformer shapes.
All mathematical input groups/features, codec, FP32 eager backend, positions,
dropout, AdamW/LR, training batch8 and learning strategy remain. The new factory
initializes all three models freshly with seed20261016, including the8M reference;
it neither resizes nor inherits the completed parent's trained weights. The parent
supplies checked read-only data/recipe lineage only. The package fixture checks
that the8M factory has byte-identical initial weights/buffers to the existing one.

## Frozen work and interpretation

One checked preparation exports128 actual training rows,32 teacher-forced validation
rows and32 distinct visible generation prompts, without reading hidden truth.
All shapes/data/recipe/source/row identities freeze before timing. Each shape has
one CPU initialization/full-prefix reference (two visible prompts at1/32/128 tokens).
Before updates, the actual HIP cached decoder and encoder must match that CPU
oracle and greedy IDs. This short parity oracle does not claim full1024 numerical
equivalence; the real1024-body grammar workload is separately retained.

Each profile has exactly five fresh GPU roles: cold, excluded warmup, three measured
calibrations. Here cold means a fresh model/optimizer/GPU process/KV state, not
flushed machine/engine/filesystem caches; every later role is also fresh. Each role
does32 actual finite AdamW updates, the same32 validation rows and real grammar
generation for32 prompts at each batch8/32. Batch order8→32 for m8/m100 and32→8
for m32 is frozen. Each prompt keeps its seed,1024-body cap and group's original60s
clock. No extra attempts or replicated prompts fill idle capacity. Native4 then
independently fits/verifies only complete emitted programs, in bounded waves with
immutable4s admission/owner clocks. No fixture/template work substitutes for failures.
GPU/native timing is serialized in this size-attribution screen; accepted overlap
remains the later integration direction.

The64 attempts/trial and all15 trial slots remain visible on failure. Partial
tokens never count as decoded or independently verified programs. Exactly three
complete calibration results plus the cold/warmup audits are required for a profile's
training medians. Zero generated programs stays zero; fresh-model results say
nothing about learned quality. All profiles use the same bounded1Hz observations;
telemetry overhead is uncalibrated. A sample exceeding16KiB fails collection rather
than truncating it. Cgroup CPU/memory.stat, process/device residency and host memory
remain; bulk host-wide CPU/load text is omitted for this scope.

Reports include actual training tokens/s and updates/s, validation throughput,
partial/completed generation tokens, decoded/verified programs per minute, batch
denominators, native clocks/debits and full resource/component costs. Both in-role
and engine-startup-through-cleanup program rates are explicit. A training-only72h
token extrapolation is labelled incomplete and cannot admit research. No automatic
learning-model selection, optimization/default adoption or curriculum change occurs.

Initial weights/oracles are hard-linked into read-only role inputs, preventing
per-trial private weight copies. Public native evidence retains every independent-
admission digest/identity and complete clocks/debits; exact responses stay private
inside the existing8GiB invocation watchdog. Bounded telemetry reserves about261MiB
uncompressed over the maximum15 external-role intervals. No trials/evidence are
trimmed and no archive/disk/stream/cleanup caps increase. The archive is a hardware
report, not a fitting-semantic requalification or a trained checkpoint.

## Available verification

**165 host stdlib tests pass**, including11 new sizing cases: exact closed profiles/
counts, manifest/phase caps, partial-work denominators, all15 unstarted slots,
incomplete calibration/mismatched resource/thread/reference rejection, real scheduler
clock/debit handling with a fake client, safe filenames, a full host-adapter graph
with shared quotas/hard-linked inputs and a verified archive, plus telemetry byte
failure. Simulated execution is not hardware evidence.

The actual public CLI passes host checks and exits3 for unavailable Podman, retaining
all15 blocked trials and a verified63-listed-member partial archive plus manifest.
[Local evidence](model-sizing-local-evidence.json) binds the exact719-entry source,
manifest and archive. Compilation/whitespace checks pass. The007 implementation
binding is refreshed with the actual foundation serializer; all68 dispositions and
full007=false remain. No host Torch, pytest, native toolchain or engine was installed.

Authored/registered pinned fixtures cover actual counts/all three model shapes,
default baseline byte compatibility, CPU/HIP finite updates and full/cached parity,
inactive rows, real grammar batches8/32 and real native generated-program admission/
clocks. Their4-token sampler cap bounds fixture time; the operator workload retains
1024. These fixtures and the complete screen remain target-pending.

## One target command

From the existing checkout after the code work above:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_model_sizing.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-model-sizing-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return the single sibling ZIP, including partial/failure archives. No separate build,
fixture/profile/export commands or manual branches are needed. Isolate this command
from other heavy compute; its model roles are deliberately serialized on the shared
GPU/memory workstation. Previous accepted GPU/native component screens are not rerun.
Only the applicable current-source package/CPU/HIP prerequisites repeat.

Planning estimate: roughly1–3 hours with warm caches, **uncalibrated for the new
sizes**. Every GPU role has1080s internal/1110s external;15 external reservations
total4h37m30s. CPU references have570s internal/600s external each; ordinary host/
CPU/HIP checks and preparation are additional. Phase4 declares16680s including
the30s report, below18000s; phase1 references/preparation/readiness fit3600s and
HIP600s is phase2. Build targets share14400s per engine and the invocation has an
86400s ceiling with unchanged900s archive/300s emergency reserves. Limits, not
planning estimates, control admission; an oversized or incomplete profile stays
explicitly blocked/partial.

009:M001–M003 code/local tasks are complete.009:M004/current-sourceT110 await target
review. Original counts77 complete/19 runtime pending/15 research held or blocked.
Earlier O004/T110 closure remains accepted at5014881. The model choice and later
real persistent-state/checkpoint/operations soak depend on these measurements;
neither is launched here. Original full-pipeline/five-pair/cost and learning gates,
008:T093 deferral and original008:T096 negative remain unchanged. Do not wait for
new PR CI after publication.
