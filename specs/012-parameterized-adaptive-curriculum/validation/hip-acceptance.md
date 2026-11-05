# Phase E — Actual workstation acceptance

## T068: Original session and independent verification

The operator supplied `report(10).zip` after the instructed acceptance-only command,
using the original prepared/pilot roots and `pac-012-acceptance-004` output. The
[recorded scope](operator-handoff.md) identifies the exact plan and public archive.
This closeout consumes that successful session; no new GPU work was launched.
The invocation below is the command supplied in the conversation, not a claim
that private raw argv was exported inside the ZIP:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --seed 20261016 \
  --output run-state/generalization/pac-012-acceptance-004
```

| Stage | Application | Process | Elapsed seconds | Cleanup |
| --- | --- | --- | ---: | --- |
| Admission | PASS | COMPLETE, exit 0 | 1126.679780 | Acknowledged, within bound |
| Qualification | PASS | COMPLETE, exit 0 | 768.669247 | Acknowledged, within bound |
| Calibration | PASS | COMPLETE, exit 0 | 354.490339 | Acknowledged, within bound |
| Acceptance | PASS | COMPLETE, exit 0 | 351.252565 | Acknowledged, within bound |

All four receipts are fresh (`reused=false`); summed stage elapsed time is
2601.091931 seconds (43.3515 minutes), excluding unreported bootstrap/export time.
The evidence error list is empty. Original terminal status COMPLETE is bound to
its evidence and verification bytes; independent `verify_archive` reproduces
integrity PASS, software PASS, feature qualified and scientific UNMEASURED.
This establishes software acceptance, not a completed scientific comparison.

Qualification contains **1,499 CPU tests and 27 HIP tests**, with zero failures,
errors or skips. It records Podman 5.8.6/crun, rootless, network off, read-only root,
no new privileges/capabilities/runtime socket; FP32; AMD Radeon Graphics;
120,259,084,288 device-visible bytes; Python 3.12.3;
`torch 2.9.1+rocm7.2.1.gitff65f5bc`; HIP `7.2.53211-e1a6bc5663`; driver `6.18.50`.
These are observed labels, not an inferred exact GPU product name.

| Binding | Original value |
| --- | --- |
| Source | `sha256:a5140e5d2be137411d6ae82bf6b6e28ec37ad27a8d71d3a283a4ce1ed814ac5c` |
| Qualification image | `sha256:8c5814b6da8ff9edb4b9f8b01d653b937f92941e1051cbdacab713be20a84c4e` |
| HIP image | `sha256:abfd06f7cafe415cc26bba28ac06f903d7e1a77013d868523fcb6a1c6520cdd4` |
| Execution policy | `sha256:7e8b6150696841d883936a6599c05d80b7055cb6d8c518846c29bc0897f5d71d` |
| Requested and observed helper | `sha256:430aa7c09551926fbcacab0fe1bc0c6b0e3ee4ffa4a113fc9bc547ff081418bd` |
| Qualification | `sha256:164f508c704d028d1581b0e3bfd24b15ddf99ba81d45c30bfd4056c11087d7a4` |
| PAC acceptance | `sha256:cea6bcf72fb2cd968063ba047558e5f76b44fd7051776dea1abe4e4a0c7acab0` |

The source inventory matches the checkout at the start of this closeout. Production
source, tests, locks, contracts and orchestration helper are unchanged by the
Phase E documentation; qualified runtime bytes need not be rebuilt for prose.

## Failure history and smallest retry boundary

The earlier original scope failed completed post-exclusion quotas; its authorized
v2 amendment preserves draws/exclusions but requires new admission. The `-002`
run passed admission, then failed qualification because the protected structural
artifact was omitted from the custodian projection. The `-003` run passed admission
and qualification, then calibration indexed a nonexistent learner-row source field.
Those production fixes required new source-bound plans/images; their earlier
successful stages could not be rebound silently. `-004` is the first supplied
complete acceptance PASS. Earlier failed directories and historical 011 tasks
remain unchanged. Their detail is recorded in the implementation ledger; this
closeout does not pretend those archives were newly re-uploaded or rerun.

The current PASS requires no repair/retry. Compatible unchanged interruptions may
use the existing `resume`; report-only recovery uses `recover` and never launches
training. Missing proof is a gap, not a reason to repeat every successful stage.
