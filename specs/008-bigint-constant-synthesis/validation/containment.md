# Phase 5 containment qualification — T091

Status: **PARTIAL**. Completed workflow39686633400 passes64 mandatory cases
on each of Docker and Podman, with zero failures/errors/skips. Each engine also
passes29 host/actual-role tests and three public CLI tests. The latter verify
private compiler storage, public CPU validation and unchanged receipt-based
reruns. CRC/SHA256, mandatory JUnit and shared source/native identities verify;
see [the execution review](phase-5-execution-review.json).

Actual host-role cases assert no network, read-only root and input mount, exact
input/output binds, no CPU GPU/socket/private-truth access, successful repeated
service use and removal/reaping. Those passing assertions are evidence of the
checks performed. However, the old CI paths only retained raw inspection on
failure, and containment partitions used automatic removal. They do not provide
the successful raw role inspection and separately timed cleanup records needed
to close this task. No elapsed cleanup value is invented from a whole-test time.

The containment CI partitions now call the existing audited `container_stage`
launcher, retaining `container-inspect.json` and `cleanup.json` on success, with
a five-second cleanup bound and no runtime socket inside the role. Host resource
selection uses the same portable policy as workstation qualification; the
arbitrary-UID packaging suite remains independent. Disposable `.runtime` files
are excluded from uploads. Embedded host code passes standard-library-only
Docker/Podman command/inspection/failure-cleanup smoke tests with simulated
engines;47 owning Python regressions pass locally and one actual-engine case is
explicitly deselected. **Actual audited CI artifacts remain pending**, and T091
stays unchecked until they are reviewed. No running build was awaited.

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
