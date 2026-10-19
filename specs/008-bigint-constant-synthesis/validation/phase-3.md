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
