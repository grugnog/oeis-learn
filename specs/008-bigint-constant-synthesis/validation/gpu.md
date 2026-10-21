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

## Phase 3 integrated rerun — 2026-10-19

The user subsequently supplied a passing production CLI
`validate --suite bigint-gpu` run: **3 passed, zero failures/errors/skips**,
including the full checkpoint/update/resume fixture on HIP. See the
[Phase 3 review](phase-3.md#integrated-hip-rerun--2026-10-19) and
[preserved artifacts and hash verification](phase-3-hip/review.json).
The run reports image
`sha256:41b763236ca5f5131ef8b8f07118f79fc4a22d6efbf5f55ec058fdcdfbcbd570`,
AMD Radeon Graphics (`gfx1151`), and the same PyTorch/HIP versions above.
This submission has no source manifest or Git revision; it is not given the
independent source-match claim established for the earlier Phase 2 evidence.

At this earlier Phase 3 boundary, T092 remained unchecked pending Phase 4
retirement and eventual source/image review. This historical evidence retains
its own identity; the current Phase 5 result is recorded below.

## Phase 5 current-source review — T092 complete

The supplied `008-phase5-batched-20261021T001809Z.zip` passes **9 actual HIP
tests, zero failures/errors/skips**. All21 archive file sizes/hashes verify.
The suite and cost benchmark report the expected source snapshot
`sha256:1e8d469e97912d3af3e40ad0b255a2df18ca09d13d9cb1209f67580cb9386a52`
and immutable image
`sha256:3932b17d9612f8eb6e5b8128329fa7a6403839e66a49db54668a4becde89896b`.
See [the preserved review receipt](phase-5-batched-hip-review.json).

The device is AMD Radeon Graphics (`gfx1151`) with HIP
`7.2.53211-e1a6bc5663` and PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`.
Finite actual-device encoder forward/backward/update covers256/512/1024/4096-bit
inputs and4097-bit differences. The tracked weight changes from
`f2737f4711ab0e954abb5ba41a1761e9c4953eb5e00b93207b7a5afea62c444c`
to `bd6d772ddc593f4a55ab13f99d75216645a6fe5f9fb61ed2980cd46756889dd7`.
The original cached/full decoder and checkpoint/update/resume cases pass.
All six new mixed-prompt singleton/batched/reordered parity cases pass across
correct/shuffled/absent conditioning and greedy/stochastic sampling.

This closes T092 after the completed Phase 4 prerequisite. It does not qualify
Phase 5 overall: complete-screen cost admission fails, learnability has not run,
and the remaining execution/review gates stay open. The separate benchmark
also passes all12 frozen-state generation parity/deadline groups; see
[the cost review](phase-5-generation.md#reviewed-current-source-hip-run--2026-10-21).

## Scoped preflight HIP review — 2026-10-21

The subsequent `008-phase5-scoped-20261021T115212Z` run passes all nine HIP
tests with zero failures, errors or skips, including a finite changed-weight
update, checkpoint workflow and mixed-prompt decoding cases. All twelve
fixed-state calibration/control/temperature groups pass parity, frozen-weight
and deadline checks. The source/image, JUnit and all24 archive file hashes
verify; see [the scoped review receipt](phase-5-scoped-hip-review.json).

The actual device remains AMD Radeon Graphics (`gfx1151`), HIP
`7.2.53211-e1a6bc5663`, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, under Podman5.8.6.
The reviewed diverse32 cost receipt admits the single T096 attempt; no
learnability outcome or transfer-screen admission follows from these HIP tests.
