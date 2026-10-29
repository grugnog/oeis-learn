# Second operator archive review

`009-20261024T013625Z.zip` passes the actual bundle verifier: all 124 manifest-listed members plus the manifest, canonical identity, member sizes/hashes, confined paths and summary consistency. The original 4,820,654-byte ZIP is unchanged. It ran for 1,078.997 seconds (about 18 minutes), ending with selected scope FAIL, engineering BLOCKED and portability PENDING. No learning, research or longer-run admission occurred. [Exact archive, stage and JUnit identities](operator-20261024-review.json) are retained.

The producer source is `sha256:45f20504444c34a5ecef6869f964135bbb998c5bcfdb4840f61e7572810f8ad9`, the recorded snapshot at `170cb10`. Its 689-entry closure hashes to that identity. The current source at `eb3bc93` has the same inventory, with only `container_runtime.py` and `test_pipeline_resources.py` differing: the Docker user-mapping CI repair and its regressions. The operator collection began before that fix was pushed.

| Target suite | Tests | Failures | Skips | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Podman bigint CPU semantics | 598 | 1 | 0 | Docker replay command fixture hit the missing-MemTotal exception |
| Podman supported workflow CPU suite | 1115 | 8 | 9 | All eight failures hit the same exception in Docker command fixtures |
| Retirement regressions | 123 | 0 | 0 | Producer PASS |
| Actual HIP preflight | 15 | 0 | 0 | Producer PASS, including `test_actual_hip_diagnostic_checkpoint_roundtrip` |

There are zero JUnit errors. The 85-test producer host suite, both Podman image stages, engine capability and retirement stage also passed. The nine CPU/workflow failures are the same resource-probe-in-user-mapping regression fixed in `5f0c568`; no additional defect was found. The earlier assessment that the Podman run need not restart overlooked that these suites include Docker command fixtures. This collection cannot advance past the failed CPU prerequisite. Calibration, policy/native/inference/overlap and both costs are BLOCKED, not newly measured. The checkpoint device repair now has actual HIP fixture evidence, but its full diagnostic path still awaits an unblocked run.

On the latest reviewed branch `eb3bc93`, Foundation CI run `40052881112` passed. In container/native run `40052881105`, all previously failing jobs passed: portable Docker and Podman, contract preparation, arithmetic and both containment jobs. Native qualification/preparation, fitting, performance and runtime-profile also passed. The separate Podman build was cancelled, making the aggregate workflow cancelled rather than green. This is an observed result, not an inferred build success; no CI completion was awaited or rerun requested.

No new code change is needed for this archive. Current implementation evidence remains 87 passing host tests with strict resource qualification retained. Task counts stay 67 implementation/local tasks complete, 29 runtime/evidence tasks pending and 15 research tasks blocked. Historical passes do not close changed-source engineering qualification, portability, performance adoption or scientific gates. The earlier empty cold-calibration subset may still legitimately produce INCONCLUSIVE after the CPU prerequisite is repaired; no workloads or budgets have been changed to force completion.

## Combined rerun on the updated branch

From the repository root on the HIP workstation, first update the existing branch, then run:

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

Keep the current input locations or substitute their workspace-relative paths if moved. Return the single sibling ZIP, including a non-PASS result. Changed source closures affect reuse; no previous skips are promised. Image checks and timing rerun. The hard cap is 24 hours and complete duration remains unknown; these failed 18 minutes are not a complete-run estimate. Podman-only selection leaves independent Docker portability PENDING. This engineering command does not launch studies or expose finals.
