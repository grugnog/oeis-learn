# Phase 5 containment qualification — T091

Status: pending the new `bigint-qualification-containment-docker` and
`bigint-qualification-containment-podman` artifacts plus host inspection records.

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
