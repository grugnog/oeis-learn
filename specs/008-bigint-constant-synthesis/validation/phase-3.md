# Phase 3 integration checkpoint (T059–T080)

Status: in progress; integration exit T080 is not yet qualified.

Entry is merged main `6d284b6597582e8900260c1397b796d274aad22b` (PR #55).
Phase 2 records 48 completed tasks, passing native/CPU contracts and the user's
actual HIP results. Its algorithms and profiles are the integration boundary.
The requirements checklist is 16/16 checked. No extension hooks are registered.

This PR will add bounded native transport/supervision, strict resolved recipes,
role-isolated preparation/training/scoring and complete durable state, policy
routing, provenance, prioritized exports and public validation suites. Phase 4
retirement and research execution remain outside this request. New CI is left
running without monitoring; unavailable native/HIP execution is recorded with
reproduction commands rather than asserted from source inspection.

## Transport/configuration milestone (T059–T063)

Tests were added before implementation and initially failed collection on the
missing modules. Framed native dispatch, one shared FIFO supervisor, bounded
partial reads/writes, generation/contract checks, external timeout/RSS/stderr and
kill/reap, result/disk reservation policies and closed recipe resolution are now
implemented. Native build packaging includes the worker binary. The unchanged
Phase 2 driver calls the same dispatch as the worker; neither grants admission.

Local targeted result: 10 passed, one explicit packaged-native skip. The
qualification image must execute that native case. Native compilation is pending;
this workspace has no Docker/Podman or Rust toolchain. The configuration record
is resolved inside the image; the host remains standard-library-only. Later
milestones complete orchestration/training/export wiring before T080 qualifies.
