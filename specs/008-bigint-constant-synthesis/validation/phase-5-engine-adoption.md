# Explicit64KiB production engine adoption — 2026-10-22

T094 follows the reviewed [zero/64KiB comparison](phase-5-growth-reservation-review.json):
450 paired observations and225 cross-setting groups agree, with5.23–5.27x small
batch100 execution improvement and no corpus-level peak-RSS increase. Three
batch1 regressions and the remaining13–14x retained gap remain visible.

Freeze this focused implementation scope:

- `BigintRuntime::new` selects64KiB spare growth capacity under explicit
  `bigint-engine-policy/v2`. Dynamic memory, zero guards, pinned Wasmtime/guest,
  fresh stores, candidate ABI, visible memory/arena caps, fuel, deadlines, failure
  mapping and cleanup remain unchanged. Reservation does not grant visible memory.
- One shared immutable JSON policy supplies Rust settings and Python metadata.
  Bind its canonical hash into bigint runtime identities, fitting evidence,
  caches/prepared contracts and packaged build artifacts. Retained signed256
  keeps its existing identity shape. Old checkpoints/contracts do not silently
  resume under the changed artifacts or claim the old negative study became PASS.
- The only alternate constructor fixes zero for qualification. No worker/request
  setting or mutable reservation-size override is added. Engine configuration and
  comparison v2 distinguish production64KiB from zero qualification control.
  New comparison rows report production at top level and nest the complete zero
  report. Keep v1 archival readers and immutable historical records unchanged.
- Tests cover default/zero manifests and identities, changed-policy separation,
  exact results/fuel/failures, first-byte growth, visible memory before/after
  accepted growth and denial, and packaged artifact identity. Shared CPU/native
  and Docker/Podman paths exercise the same production constructor/worker.
- Publish available local checks and commands for native/container qualification.
  Do not wait for PR CI. Actual new-image qualification remains pending where
  unavailable locally. Neural/HIP throughput claims require separate evidence;
  no new training, hidden scoring or consumed T096 retry is authorized.

- [X] A001 Add failing shared-policy/identity/default/control regression tests.
- [X] A002 Adopt64KiB and bind explicit policy in Rust/Python/build identities.
- [X] A003 Preserve zero comparison, archive readers and packaged regression routing.
- [X] A004 Verify available checks, push incrementally and provide CPU commands.

T094 remains open for qualification and remaining slowdown/measurement work.
Original negative T096, immutable earlier receipts and promotion block persist.


A001–A004 implementation/available checks are complete. Actual new-image native,
container and eventual-device qualification remains pending. The shared manifest
is copied only into native source-build stages and is included in native source
provenance and Python package data. Dependency/lock inputs stay unchanged.
The native extension exposes its compiled manifest for build-time comparison.

`--engine-adoption-only` runs the existing arithmetic/fitting/containment groups
(max3000s each) followed by the600s v2 comparison, without data/weights/training.
It rejects unrelated scopes/GPU/preparation before engine access, requires the
expected policy in each group's artifacts, binds source/image, and archives
failures under the existing role/audit/cleanup and byte limits. Production is
now64KiB; zero is qualification-only. Archived v1 summaries remain reproducible.
See [phase-5-engine-adoption-local.json](phase-5-engine-adoption-local.json) and
[quickstart](../quickstart.md) for evidence limits and CPU/optional HIP commands.

The first uploaded adoption run is preserved as FAIL in
[the failure review](phase-5-engine-adoption-failure-review.json). All12 manifest
entries verify; arithmetic passes333 cases. Fitting passes34 and fails three
independent-admission cases because the qualification fixture still calculates
the old bigint runtime identity without the engine-policy hash. The native
diagnostic and production client already include it. The fixture now includes
the policy only for bigint; retained signed256 and strict admission are preserved.
A regression reproduces the old mismatch and checks both profiles against the
client and independently assembled manifests. Local regressions pass226 cases;
15 native/artifact/benchmark cases skip explicitly. Containment and comparison
were unstarted in the upload. Rerun the CPU adoption command on the updated
branch; no HIP, data preparation or training is needed for this correction.
Actual corrected-image qualification remains pending and T094 stays open.
