Target status: [accepted corrected collection](native-capacity-operator-20261025T040709Z.md) at176ecc9. All18 slots and655 CPU tests pass;4 workers win experimentally. The command below is retained for reproducibility, not a duplicate-run request. Earlier failure/repair evidence remains historical.

Latest target collection: [qualification failure and repair](native-capacity-operator-20261025T032351Z.md).652 CPU tests pass; one serial metrics fixture assertion blocks all performance slots. The fixture and independent CI disposition-serializer issue are repaired;136 host tests pass. Use the same combined command below after updating, with a fresh output. No GPU repeat.

# Native CPU capacity: combined target handoff

The [accepted GPU ladder](hardware-ladder-operator-20261024T220811Z.md) selects128
experimentally within its finite roster:61.13% median equal-work interval reduction
versus16, about90.5% sampled GPU busy and2.84GiB accounted role memory. No larger
GPU sweep or duplicate ladder is requested. Learning-quality work remains held.

The next [prospective contract](../contracts/native-capacity.md) combines a queued
native diagnostic, full18-row wave1 baseline calibration, independent2/4-worker
AB/BA comparisons, and separate observed1/4-worker trials in one CPU-only invocation.
All timing trials are serial to avoid contention. Deliberate GPU/native overlap,
useful full-pipeline generation and sustained operations remain later hardware work.

## Implemented behavior and available checks

The old queued-eight policy retains its one diagnostic. New slot-sized waves defer
admission of a later wave until its preceding wave drains. Once admitted, the same
four-second request deadline includes dispatch/queue/owner observation and never
renews; the original whole-trial parent clock remains. At most one request per
native slot is admitted in each wave. Production scheduler/defaults are unchanged.

Every ordinary trial creates a fresh client/pool and empty native caches. The full
18-row corpus is frozen before timing, including large integer cancellation,
affine/nonlinear and negative fits. There is no cold-selected subset. Complete
baselines freeze source/manifest/corpus/semantics, five receipt identities,
calibration, memory ceiling, slots and decision rules before any candidate timing.
The two named worker hypotheses remain independent. Partial work, changed worker
or semantic identity, late observations or missing debits cannot produce a ratio.
Unsupported gains retain1 worker; fewer workers win within5% of the fastest.

All18 nested trial slots are planned first. Each wave trial also persists its
18-row job plan, synchronized admission/dispatch/completion/observation events,
terminal or unstarted status, per-job ledger/debits and ordinary/inner intervals.
The primary interval includes client shutdown and trial serialization. A primary
wall overrun invalidates the trial. Resource failure, nonquiescent workers and
interruption retain partial evidence and stop globally. Unknown aggregate accounting
cannot qualify or select a candidate; an8GiB expected peak plus1GiB headroom must
fit known current capacity before launch. Native per-slot1GiB/aggregate4GiB and
shared CPU/quota/cache/artifact bounds remain; external cleanup stays within5s.

**136 host stdlib tests pass**, including12 native-capacity tests and a foundation-serializer regression. Coverage includes the finite graph,
full work, strict clocks/debits, immutable child, independent partial outcomes,
negative gains, primary overruns, fatal stops and a complete simulated graph with
one verified archive. The actual public stdlib CLI exits3 for the unavailable
local Podman, records all18 undispatched slots as UNSTARTED, and exports a verified
23-listed-member partial archive. [Current repair-local evidence](native-capacity-fix-local-evidence.json)
binds the exact709-entry source, manifest and archive. Compilation and whitespace
checks pass. The007 code binding is refreshed with all68 dispositions and incomplete
foundation acceptance preserved. Existing manifests and accepted evidence are unchanged.

Three packaged-native fixtures for1/2/4 workers are registered in the pinned CPU
suite. They use18 distinct simple affine prompts to exercise real slots rather
than cache-only hits, independently assert exact3/7 assignments, and validate actual
admission/owner/debit clocks. These fixtures are authored, **not executed locally**;
Torch/pytest/native binaries/engines were not installed on the host. The operator's
full18-stratum performance corpus remains separate from this simple correctness
fixture. Actual HIP is not required or selected for this CPU-only increment.

## One command and one archive

After the completed code work above, run from the existing branch checkout:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_native_capacity.json \
  --max-seconds 86400 \
  --output "tmp-validation/009-native-capacity-$(date -u +%Y%m%dT%H%M%SZ)"
```

No parent/checkpoint/input directories are needed for this fixed native fixture.
Return the single sibling ZIP even if the investigation stops or is inconclusive.
No separate build/test/profile/export steps, manual worker selection or second
command is needed between these predeclared experiments. Isolate the timing from
other compute-heavy work. No GPU ladder/HIP replay is repeated.

Planning estimate: roughly10–25 minutes with warm dependency caches and complete
native calibration; this is an extrapolation, not a calibrated runtime guarantee.
The external CPU role has4500s (75min), with4470s internal. Each queued/baseline
trial has180s; a baseline-only child reserves `ceil(max(30,2*max calibration))`,
maximum270s, for each of ten candidate and two observed trials. An incomplete or
oversized baseline blocks candidates. No extra sample, scope narrowing or cap
clamping. Phase3 declares4530s below7200s. Existing14400s shared build per engine,
86400s global,900s archive/300s emergency cleanup and disk/memory limits remain.

Affected-sourceT110 reopens only for this new CPU/native/packaging source. Original
counts77 complete/19 runtime pending/15 research held or blocked.009:H004 remains
complete at adda1b6;009:N001–N003 are complete and N004 awaits target review.
No worker/default/pipeline adoption, research cost, learning or multi-day authority
is granted. PR CI is not awaited after publishing.
