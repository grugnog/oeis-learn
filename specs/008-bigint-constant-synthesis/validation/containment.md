# Phase 5 containment qualification — T091

Status: **COMPLETE for the recorded implementation**. Completed workflow
39693058662 at head46f33bd supplies the successful raw inspection and separately
timed cleanup records missing from the earlier run. Docker and Podman each pass
64 mandatory cases with zero failures/errors/skips. Cleanup takes0.015571s and
0.048828s respectively, within the existing5s allowance. Archive CRC/SHA256,
JUnit, source/native identities and inspect/cleanup file hashes verify in
[the containment review](phase-5-containment-review.json).

Both inspect records show no network, read-only root, no privileged mode and
only the explicit writable `/work/output` bind. The earlier completed workflow
39686633400 supplies29 host/actual-role and three public CLI tests per engine,
covering private compiler storage, public CPU validation and unchanged
receipt-based reruns; see [the execution review](phase-5-execution-review.json).
Together these close T091 for their recorded source/image. The new observation
code changes that source identity: matching new CI remains necessary before
promotion, and historical passes are not relabelled as its execution.

The audited launcher retains `container-inspect.json` and `cleanup.json` on
success with a five-second cleanup bound and no runtime socket in the role.
It now also launches the bounded CPU profiler; its actual new-image execution
is pending. Disposable `.runtime` files are excluded from uploads. The separate
arbitrary-UID packaging suite is unchanged.

The suite runs framed transport, malformed/stale responses, restart/reaping,
RSS/deadline failures, count-first/byte-first pressure, cancellation and partial
input writes, oversized cache bypass, disk failure/symlink rejection, conservative
ledger/checkpoint recovery, isolated visible/scoring views and bounded exports.
It requires zero failed/error/skipped selected cases. The optional nested-engine
test is deselected inside the socket-free role; actual engine launch/inspection/
reclamation is performed by the host helper and the existing portable-engine CI.

The helper uses the existing Engine policy, a read-only root, no network or
runtime socket and a single explicit output mount. Each completed role has a
`container-inspect.json` and timed `cleanup.json`; unexpected bind/network/root
capabilities or cleanup beyond5s fail qualification. CPU evidence cannot replace
GPU qualification. An unavailable second engine remains pending, never passed.
