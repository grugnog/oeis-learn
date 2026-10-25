# Native capacity collection: qualification failure and repair

The uploaded `009-native-capacity-20261025T032351Z.zip` verifies all49 listed
members plus its manifest and exactly matches producer
`e8b89cf67be372adabee59c729a32671d3ad3bfa`,709 source entries,
`sha256:723ac9742b7ff09c2e43325d59a2508ddf645b3c6e2d59367c6b25b54bc16564`.
The1,312,310-byte archive SHA256 is
`6667de427ccea856abb32ad6b8498cf37efa91ac7a2cf00c2f7a16bb8be5d442`.
[Machine-readable review](native-capacity-operator-20261025T032351Z.json) retains
source/receipt/manifest identities, exact qualification failure and reviewed CI.

## Observed outcome

The134 host tests pass. The pinned CPU suite has652 passes, one failure, one
explicitly deselected case and14 passing subtests. The only failed case is
`test_real_native_slot_waves_preserve_full_corpus_and_clocks[1]`, with
`KeyError: 'workers'` at its metrics assertion. Its earlier full18-row semantics,
assignments and strict clock/debit assertions had completed successfully; the
real2/4-worker parameterizations pass. This does not constitute a passing CPU
qualification suite.

The engine and current qualification image pass. The CPU failure blocks
`native.capacity`; all18 planned performance slots are explicitly UNSTARTED in
the independent disposition report. No queued diagnostic, baseline calibration,
candidate pair or observed performance trial ran. There is no worker-selection,
CPU-scaling or throughput result to accept. The selected scope correctly fails
in382.14s (6m22s), with feature completion and all learning/admission flags false.

Review verifies the bundle inventory/size/hash/summary, exact709-entry source
closure, canonical terminal-stage seals and each referenced public-file hash,
qualification XML/failure log and all18 undispatched slot dispositions. Original
archive bytes remain unchanged.009:N004 and affected-sourceT110 stay pending.

## Two validation repairs

The real WorkerPool factory intentionally retains its serial implementation for
one worker. Its metrics are a flat counter dictionary; explicit2/4-slot pools
report `workers` and `slots`. The fixture incorrectly assumed the pooled shape
for all three cases. Repair commit `95fc98d318b6dff7fd103c4b727d6ac3bbfb4211`
checks the actual serial shape for1 and exact pool/slot counts for2/4, while
requiring positive activity in every slot, aggregate/slot counter agreement and
at least18 real native batches from distinct prompts. Assignment, complete-work,
clock and debit checks remain. Production metric schemas are unchanged.

Completed producer CI also exposed an independent Spec007 disposition-binding
error. The prior refresh used ASCII-escaped pipeline serialization; the foundation
implementation identity uses its original UTF-8 canonical serializer. The actual
package implementation hash is
`sha256:052fa2df91bb615f1e20780995593d035cd53e6cb8e17c687987aac357e0df9e`,
not the recorded `sha256:a6f39313ac555b01ad7522a3e75b4f5368b65d0b1ede5ec1e1621da41b9c544f`.
The repair refreshes only that metadata using the actual foundation functions;
all68 dispositions and incomplete foundation acceptance remain.

Both producer workflows fail:40157945940 and40157945849. All six failed jobs
were inspected. Foundation CPU regressions and both portable-engine jobs fail
on the disposition mismatch; the arbitrary-UID contract job fails on the serial
fixture. Both Docker/Podman containment artifacts12584618755/12585568642 have
67 passes and that same single serial-fixture failure. Their numerical/native
algorithms do not require a change. Newly queued CI is not awaited.

Two host regressions reproduce the problems before repair and pass afterward.
One executes the real WorkerPool factory and fixture assertion against actual
serial/pooled stat shapes with clearly synthetic counters; no host process is
launched. The other executes the actual foundation canonical/implementation
functions in stdlib isolation, including UTF-8 behavior, and checks all68 current
dispositions. The public host suite now passes136 tests. A verified23-listed-member
missing-engine archive retains all18 unstarted slots. Compilation and whitespace
checks pass. [Repair-local evidence](native-capacity-fix-local-evidence.json)
binds corrected709-entry source
`sha256:4534bfdc83f9faf7132830a4d281c55fb8e6c52f129f2f4e3339fcbe7504308c`.

Only tests and validation metadata changed; runtime Python bytes, native service
clock, parent reservations,18-row workload, AB/BA schedule and numerical/selection
thresholds are unchanged. The corrected pinned CPU fixture still needs execution.
The prior GPU ladder remains accepted at its stated source; no HIP repeat or
learning-performance investigation is requested.

## One combined rerun after updating

All unblocked repair work is complete. From the existing checkout:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_native_capacity.json \
  --max-seconds 86400 \
  --output "tmp-validation/009-native-capacity-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return the single ZIP, including failures or inconclusive outcomes. This retries
qualification after a concrete fixture repair; it does not resample failed timing
pairs or narrow the performance corpus. No separate build/test/profile/export,
checkpoint/input paths or HIP command is needed. Isolate timing from other heavy
compute. Planning10–25 minutes with warm caches/complete calibration remains an
extrapolation; the4500s external CPU role,4470s internal parent,≤270s derived
trials and existing phase/global/build/artifact/memory/cleanup ceilings remain.
