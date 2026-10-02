# Constitution RFC 007 — Adoption Record

**Feature**: 007-experiment-foundation (Trustworthy Experiment Foundation)
**RFC**: [constitution-rfc.md](../constitution-rfc.md) — Version 2.0.0 (proposed)
**Original Ratified Constitution**: 2026-08-30 (v1.0.0 at `aa2c8c4579dd74fdd8114ec9456960a98fcf500e`)
**Amendment Proposed**: 2026-09-16

## Feasibility Evidence (T001)

The standalone Docker GPU feasibility probe has been implemented and validated:

| Artifact | Path | Status |
| --- | --- | --- |
| Dockerfile | `docker/foundation/Dockerfile` | Built — pins resolved base image digest |
| Base image digest | `sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191` | Resolved 2026-09-17 via `docker buildx imagetools inspect` |
| Dependency lock | `docker/foundation/requirements.lock` | 21 packages, all hash-pinned, torch excluded |
| Preflight module | `src/oeis_learn/cli/foundation_preflight.py` | Runnable entry point, independent of later CLI/codec/pool |
| Config | `docker/foundation/preflight.yaml` | `schema_version: preflight/v1`; rejects unknown fields |
| Build script | `scripts/foundation/build_image.sh` | Resolves digest, builds image, writes `runtime.lock.json` |
| Run script | `scripts/foundation/run_container.sh` | Exposes KFD/render, existing numeric GIDs, no privileged mode |
| Evidence (local VM) | `docker/foundation/evidence/preflight-report.json` | `hardware_not_ready`: no /dev/kfd, no render nodes, no torch/HIP |

**Local VM evidence** (exit code 3, expected — this machine has no AMD GPU):

```json
{
  "status": "hardware_not_ready",
  "diagnostics": [
    "/dev/kfd present=False",
    "render nodes present=['none']",
    "torch/hip: ModuleNotFoundError: No module named 'torch'",
    "hip_available=False",
    "silent CPU fallback is prohibited; no driver install; no privileged container"
  ],
  "no_cpu_fallback": true,
  "no_driver_install": true,
  "no_privileged_container": true
}
```

On an AMD NixOS workstation with `/dev/kfd` and a render node, the probe:
1. Detects the AMD device via KFD + render node (or torch/HIP)
2. Runs three finite FP32 forward/backward/AdamW parameter updates at 20 terms / 1,024 tokens
3. Persists actual device name, loss, gradient, parameter-change, timing and memory evidence

## Adoption Decision

**Status: ADOPTED — 2026-09-16**

This document records the results of the feasibility probe and the **explicit maintainer adoption of Constitution 2.0.0** for the 007 feature branch. The user accepted the proposed governance change; adoption is recorded here, not fabricated.

### Adoption Record

- [x] Maintainer: oeis-learn maintainer (user instruction on the 007 branch)
- [x] Adoption date: 2026-09-16
- [x] Rationale (brief): The user explicitly accepted the proposed Constitution 2.0.0 (RFC 007). This supersedes the proposed status and authorizes remaining implementation tasks (T003–T105) to proceed under the new governance, subject to the recorded feasibility evidence and acceptance gates.
- [x] Deviations: none — adopted as proposed. The original 2026-08-30 v1.0.0 ratification at `aa2c8c4579dd74fdd8114ec9456960a98fcf500e` and its historical compatibility map are preserved (see RFC migration table).

### Effect of Adoption

The proposed revision is ratified on this feature branch. `.specify/memory/constitution.md` is updated to record the adopted status/ratification date while preserving the original 2026-08-30 ratification date and the historical compatibility map. Remaining implementation tasks (T003–T105) may now proceed under Constitution 2.0.0, gated by the acceptance gates in [plan.md](../plan.md).

### If Adoption Had Been Declined

Remaining tasks beyond T001 would have been blocked. The standalone preflight remains available as a reusable GPU feasibility probe. The proposal would have required revision or withdrawal; no 007 completion would be claimed.

---

**Updated**: 2026-09-16T00:00:00Z
