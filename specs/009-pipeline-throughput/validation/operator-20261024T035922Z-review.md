# Third operator archive: storage stop

`009-20261024T035922Z.zip` verifies with the actual bundle verifier: 106 manifest-listed members plus the manifest, canonical identity, sizes/hashes, confined paths and summary consistency. Its 2,831,626 bytes are unchanged. The 689-file closure hashes to the recorded producer source `sha256:a2e19add9656ff3417b710f93773216e8d65e6bf3c794c048fd671449f0b1ad0` at `322f853`. The run took 843.37 seconds (about 14 minutes), ending selected scope FAIL, engineering FAIL, portability PENDING and feature incomplete. [Exact archive/stage/test observations](operator-20261024T035922Z-review.json) are retained.

| Target suite | Tests | Failures/errors | Skips |
| --- | ---: | ---: | ---: |
| Podman CPU semantics | 600 | 0 | 0 |
| Retirement regressions | 123 | 0 | 0 |
| Supported workflow CPU suite | 1115 | 0 | 9 |

The previous Docker command-fixture regression is resolved on the target. The 87-test producer host suite, Podman engine and qualification image also passed. The GPU image build completed successfully (image `sha256:bee78b344d76a503c3940b641ce0c85f9307f70c500245481dd9676d2cc997b3`). The following package inspection was stopped by the invocation disk watchdog after about 121 seconds; container reclamation was acknowledged within its allowance. No package mismatch was reported before termination; inspection success cannot be inferred.

The recorded artifact peak is **162,878,178 bytes**, below the **8,589,934,592-byte** cap. Trace peak is zero, below 134,217,728 bytes. In this producer's three-way check, the remaining condition is filesystem available space below the **16,777,216-byte reserve**. This identifies the free-space branch, not the actual byte value or storage consumer: those were absent from the historical report. Engine image/cache storage, other activity and quotas cannot be distinguished from this ZIP. The protected filesystem is the invocation output filesystem; engine storage may be on the same or another filesystem. Do not treat this as an artifact-cap overflow or increase caps to bypass it.

Global resource-stop behavior correctly blocked HIP preflight, policy/native/calibration/inference/overlap, qualification reporting and both costs. No HIP tests were executed in this collection. Earlier 15-test HIP evidence remains historical; no new performance, cost, learning, research or longer-run authority is claimed.

## Reporting repair and verification

Disk observations now distinguish artifact, trace and free-space failures and retain the first breach sample, most recent free bytes, minimum observed free bytes and protected filesystem path. A cleanup-time recovery cannot erase the triggering low-space measurement. Existing limits and global-stop behavior are unchanged; no engine-owned cache is deleted. New deterministic regressions cover low-space/recovery persistence and separate artifact/trace causes.

The actual public `python3 -S` all-ready validator passes **89 stdlib tests**. Its 43 manifest-listed archive members verify; unavailable local engines leave exit 3, selected scope PARTIAL, engineering BLOCKED and portability PENDING. [Local evidence](disk-reporting-local.json) records source/archive identities. Compilation and whitespace checks pass. The implementation review binding was refreshed without changing its 68 dispositions or pending acceptance. No Torch/native/container/HIP execution is claimed locally, and new CI is not awaited. Task counts stay 67 implementation/local tasks complete, 29 runtime/evidence tasks pending and 15 research tasks blocked.

## Storage action before another target run

From the repository root, these commands only inspect storage:

```bash
df -h .
podman info --format '{{.Store.GraphRoot}}'
podman system df
```

Inspect free space on the printed engine storage path too if it is on another filesystem. Move/remove unneeded files or unused images/caches you choose, accounting for workspace quotas and other workloads. Keep completed run/inputs and evidence. The validator does not automatically delete engine caches. The 16 MiB reserve protects reporting; it is not an estimate of the space needed to create or materialize a large GPU image. A recovered free-space reading after container cleanup does not invalidate the recorded stop.

After resolving storage, update the existing branch and run the combined scope:

```bash
git pull --ff-only origin 009-pipeline-throughput
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_validation.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-$(date -u +%Y%m%dT%H%M%SZ)"
```

Substitute input locations only if moved. Return the single sibling ZIP, including a non-PASS result. Changed source closures invalidate affected reuse; no skips are promised. Current image checks and timing rerun. Complete duration is unknown and the hard cap remains 24 hours. The prior empty cold-calibration subset can still yield an honest INCONCLUSIVE after storage is resolved; no workload or budget was revised to force a favorable outcome. Podman-only selection leaves independent Docker portability PENDING; the engineering command cannot launch studies or expose finals.
