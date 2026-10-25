# Accepted combined hardware ladder

The uploaded `009-hardware-ladder-20261024T220811Z.zip` passes its finite engineering
scope at producer `adda1b6599786be03c91fa7f811273280fab3d4f`. The archive is
6,488,795 bytes; all565 listed members plus the manifest verify. Archive SHA256 is
`d86b1a4499159acc72c78e546868bf6b33d4915d927f6484ec6010dc6b9558c5`.
The705-entry source closure matches exactly:
`sha256:e1e827a317bac519c3f904b71755bd7d05610cddc69d5fea28ac17462ca34bd7`.
[Machine-readable review](hardware-ladder-operator-20261024T220811Z.json) retains
receipt identities, the reference, baseline-derived children, every pair and role.

All20 outer stages and15 nested roles pass.123 host tests,639 CPU tests (one
explicitly deselected GPU test) and18 actual-HIP tests pass. The new physical
128-row/all-size CPU/HIP fixtures ran. Both source PR workflows had already
completed successfully:40136880779 and40136880793; no new CI was awaited.
Total invocation elapsed2741.47s (45m41s), inside its existing ceilings.

## Equal-work results

Each arm processes131072 supplied BOS tokens, with32 unique visible rows cyclically
mapped into128 physical encoder-memory/KV-cache rows. All22 GPU roles verify640
row boundaries each (14080 total) against the frozen CPU full-prefix oracle.
The CPU reference itself retains640 boundaries. This is capacity/replay work,
not128 independent prompts or generated-program/learning success.

The ordinary/aggregate1Hz pair is98.4084/99.0225s:0.6241% observed overhead selects
aggregate1Hz for every subsequent arm under the predeclared5% rule. This single
pair calibrates policy; it does not estimate population uncertainty or eliminate
mandatory-watchdog overhead.

| Batch | Scored interval, AB / BA (s) | Reduction versus paired16 | Supplied tokens/s, AB / BA | Sampled GPU busy, AB / BA | Accounted role peak, AB / BA (GiB) |
| --- | --- | --- | --- | --- | --- |
|16|99.6356 /99.3498 (32 branch)|reference|1315.5 /1319.3|47.65% /47.39%|1.731 /1.718|
|32|50.6356 /50.5317|49.18% /49.14%|2588.5 /2593.9|84.45% /84.53%|1.879 /1.888|
|64|42.0363 /42.7440|57.71% /56.93%|3118.1 /3066.4|88.84% /88.40%|2.192 /2.204|
|128|38.5495 /38.6287|61.10% /61.17%|3400.1 /3393.1|90.44% /90.64%|2.836 /2.836|

Batch128 is the experimental selection within the fixed16/32/64/128 roster:
61.13% median reduction versus its paired16 controls, about2.57 times throughput.
Its median38.5891s is8.97% lower than64's42.3901s;64 is16.20% lower than32.
All branches satisfy the frozen progression rule. There is no registered branch
above128, and no automatic size extension is justified by spare memory alone.

Sampled role CPU use is about1.013 equivalents at16,1.038 at32,1.112 at64 and1.161
at128 against a16-equivalent quota. The role memory ceiling is112.45GiB; all
observed reservations/watchdogs qualify.128's predicted peak/headroom reservation
is15.18GiB; measured peaks are distinct from that conservative prediction. All
cleanup completes within5s and device/isolation audits pass. GPU busy is a1Hz
whole-device counter, not kernel occupancy or memory bandwidth; memory-busy/BW
counters remain unavailable. These observations support moving attention from
isolated GPU capacity toward native CPU scheduling and useful pipeline work.

## Review and disposition

Review recomputed the archive inventory/hash/summary, exact source closure,
all parent/nested receipt seals and referenced public-file hashes, mode decision,
all baseline-derived child durations/capacity/identities, pair ratios, progression
and final selection. Each GPU role retains a640-boundary plan and1281 durable
plan/queued/verified events; exported tensor hashes match their verification events,
and shapes/dtypes match the reference. Private raw tensors are intentionally
excluded: the archive cannot independently rerun numerical tolerance comparisons;
those are source-bound actual-role results. Warmups remain excluded.

009:H004 and affected-sourceT110 close at this producer. Original counts become
78 complete/18 runtime pending/15 research held or blocked. Previous accepted
scopes remain source-bound. No duplicate ladder run is requested. No production
batch/default changes, full-pipeline confirmation, native worker selection,
Docker portability, complete research costs or learning/multi-day admission are
inferred. The user's hardware-first and combined-experiment priorities remain.
