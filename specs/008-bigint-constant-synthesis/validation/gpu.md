# Actual HIP qualification — 2026-10-19

**PASS: 2 tests, zero failures, errors or skips.** The user supplied the helper's
summary, host identity, encoder update, stdout and JUnit report. Their hashes,
normalized results, native/lock identities and full reported source manifest are
preserved in [gpu-evidence.json](gpu-evidence.json). Original submitted bytes are
retained in workspace `tmp-validation/phase2-hip-20261019/`; attachments remain
the original evidence. This review does not rewrite those inputs.

| Item | Recorded result |
| --- | --- |
| Device | AMD Radeon Graphics, `/dev/dri/renderD128` |
| Runtime | Podman 5.8.6; 4 CPUs; 16 GiB memory limit |
| PyTorch | `2.9.1+rocm7.2.1.gitff65f5bc` |
| HIP | `7.2.53211-e1a6bc5663` |
| Local image | `sha256:dd47ebca75fd794056d7723e8a5fd274f1a2640f68116c1aca724cf6d0e2e1ff` |
| Host return code | 0 |
| JUnit | 2 passed, zero failures/errors/skips |
| Source comparison | All 283 reported source hashes match PR checkout `01b09ed9de2bb53a377c8fc4b5cdd43bbc105127` |
| Encoder update | Finite forward/backward; tracked projection weight hash changed |

`source_revision` is null in the helper report. The source match above is based
on the complete reported file manifest, rather than an inferred Git revision.
That manifest covers source/crates/scripts but excludes tests. The JUnit names
and stdout identify the expected two tests; the reviewed checkout's test-source
hash is recorded separately and is not mislabelled as an uploaded hash.

## Covered checks

`test_actual_hip_large_input_update` checks model/output placement on the HIP
device, signed inputs at 256/512/1024/4096 bits and 4097-bit differences, finite
outputs/gradients and an actual AdamW update. The tracked
`chunk_encoder.projection.weight` changes from
`e53288ee5856d7e8165eec0585e5c1638de4abee977047c93df8355f6305f2a7`
to `37d3377904e1a67d28ba0f25283709535cdd6d6d1e6c19d221db73f3b8fd44b2`.

`test_actual_hip_parameter_decoder_cache_and_position_growth` checks d256/four
heads/four decoder layers with parameter tokens, checkpoint loading after
position-buffer growth, CPU/HIP and cached/full logit parity at the declared
`atol=2e-5, rtol=2e-4`, and identical cached/full generated tokens and completion
status. Correct, shuffled and absent conditioning all execute these comparisons.

The three PyTorch warnings concern nested-tensor optimization and experimental
AMD attention kernels. They did not cause failure or fallback. These finite
checks establish actual-device correctness for this platform; they do not
measure learning improvement, sustained utilization or long-run throughput.

## Phase boundary

This evidence completes the actual-HIP portion of the Phase 2 algorithm review.
Together with the completed CPU/native CI run, it closes T025/T036/T058. All
48 Phase 2 tasks are checked. Public production routing remains disabled.

T092 stays unchecked because it belongs to Phase 5's integrated
`validate --suite bigint-gpu` path, which depends on Phase 3/4 integration and
retirement. The hardware checks already have this early evidence; the future
CLI and effective integrated image must still be qualified at that task.
