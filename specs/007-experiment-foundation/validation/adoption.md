# Constitution RFC 007 — Adoption Record

**Feature**: 007-experiment-foundation (Trustworthy Experiment Foundation)
**RFC**: [constitution-rfc.md](constitution-rfc.md) — Version 2.0.0 (proposed)
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

<!-- Status: PENDING maintainer approval -->

This document records the results of the feasibility probe. **Explicit maintainer adoption of Constitution 2.0.0 is required before remaining implementation tasks (T003–T105) may proceed.**

### If Adoption Is Granted

Maintainer notes go here (date, rationale, any deviations from the proposed RFC):

- [ ] Maintainer name/signature
- [ ] Adoption date
- [ ] Rationale (brief):
- [ ] Deviations (if any, or "none"):

Upon adoption: update `.specify/memory/constitution.md` to record the new status/ratification date while preserving the original 2026-08-30 ratification date and historical compatibility map.

### If Adoption Is Declined

Remaining tasks beyond T001 are blocked. The standalone preflight remains available as a reusable GPU feasibility probe. The proposal must be revised or withdrawn; no 007 completion is claimed.

---

**Updated**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
