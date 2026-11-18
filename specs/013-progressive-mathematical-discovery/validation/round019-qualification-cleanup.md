# Reclaim superseded qualification models before continuing round019

The 2026-11-18 02:07:12 diagnostic bundle and workstation du listing show four
roughly 63 GiB superseded sessions: research rounds 005, 006, 007 and 008. Their
bounded large-file inventory already includes about 146 GiB of qualification
weights/optimizer candidate files. Exact reclaimable allocation requires the live
plan because hardlinks, later deletions and smaller tensors affect the total.

Retire diagnostic tensors first, retaining all run directories and original
metadata. Do not delete data, the D reference lineage, current round019, original
preparation002, calibration009 or corrected-panel016 folders. The prepared graph
and session binding reference those histories. tmp-validation and other old
scientific runs require a separate ancestry/exposure review before whole-run
removal; age and directory size alone do not establish independence.

The reusable stdlib module is
`src/oeis_learn/pipeline_validation/qualification_retirement.py`. Fetch its committed
bytes into `run-state/discovery-tools/qualification-retirement.py`, keeping HEAD
pinned at b616b116dad59a05ad1911687583d8ab1626af13. It uses the existing cleanup,
qualified-case validation, filesystem confinement and metadata primitives from
that checkout. This does not invalidate current source qualification.

Use `plan --session PATH` once per selected session and a separate
`--output run-state/maintenance/qualification-cleanup-001`. Planning verifies
original qualification, original source, completed case producers and acknowledged
process/container termination, checks active dependencies, and records exact
checkpoint manifests and file identity/stat information. It does not delete files.
It also inspects the exact current CPU/GPU image IDs without building or pulling.
The uploaded image inventory predates the operator's pruning, so cannot by itself
prove those images still exist.

`apply --output SAME_PATH` locks the GPU and selected/current sessions, revalidates
the plan and metadata, hashes all eligible tensors against their original state
manifests before any deletion, records durable retirement intent, removes only
those weights.pt/optimizer.pt files, and replays the original qualification proof.
It preserves state manifests, RNG state, logs, measurements, reports, raw inputs,
trajectory/calibration/scientific checkpoints and all current dependencies. The
retired diagnostic models cannot resume, which is explicit in the receipt. A
repeated apply can finish an interrupted retirement, without resetting evidence.

The maintenance directory includes cleanup-report.zip, plan.json and
retirement-intent.json/retirement.json. The exact planned files and the conservative
reclaimable-byte count can be reviewed there. No image pruning is performed.
After cleanup, use the existing storage-continuation preflight before freeze;
continue only if its capacity check and required image availability both pass.
If 368G was reported by df -h, that is about 395 decimal GB: close to the campaign
reservation, with little spare space. Reclaiming old test tensors provides useful
headroom rather than weakening the storage reservation.

The updated storage-continuation helper additionally checks the exact qualified
image IDs before any expensive freeze/acceptance/run continuation, so pruning
cannot silently defer a missing-image failure until after another long freeze.
Fetch that helper alongside the cleanup module; keep the original T230 helper
unchanged because it remains bound historical evidence.
