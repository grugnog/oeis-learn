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

T092 stays unchecked because Phase 5 depends on Phase 4 retirement. Its
integrated hardware fixtures now have passing early evidence; qualification
must review that evidence against the eventual source/image and completed
prerequisites. The wrapper's `qualified: false` does not negate the PASS result.

## Phase 5 eventual-image gate — T092

The earlier Phase 2/3 device results above retain their submitted source/image
identities. They do not close T092 for the Phase 5 image. Run from PR #58's
checkout:

```bash
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu \
  --output tmp-validation/008-phase5-hip-001
```

This includes the supported `validate --suite bigint-gpu` call and preserves
its receipts, JUnit and `gpu/encoder-update.json` inside the result archive.
The suite must execute every selected case on actual HIP with zero skips;
CPU fallback, stale-image source hashes or missing changed-weight evidence fail.
Add `--prepare --learnability` to request the gated complete data/cost/tiny work.
No Phase 5 device execution has yet been supplied; T092 remains unchecked.
