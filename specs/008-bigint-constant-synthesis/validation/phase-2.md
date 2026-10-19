# Phase 2 algorithm exit review (T011–T058)

Status: **PASS — all 48 Phase 2 tasks complete, including T025/T036/T058**.
The qualified code is PR #55 checkout
`01b09ed9de2bb53a377c8fc4b5cdd43bbc105127`; this follow-up changes evidence only.
The completed [container/native run39239102389](https://github.com/grugnog/oeis-learn/actions/runs/39239102389)
passed all seven jobs, including 154 CPU/native contract tests with zero
failures/errors/skips and locked core/retained Rust tests. Foundation review
run39239102311 passed. The user-supplied HIP run passed both actual-device tests;
all 283 reported source hashes match that checkout.

The immutable CPU/native image is
`ghcr.io/grugnog/oeis-learn@sha256:2e9ba5df002e62f966a647f474803aa20bec4fafd19dac8918518c2e0cb9fcf4`.
[HIP evidence](gpu.md) records the separate GPU image/device and full source
comparison. [gpu-evidence.json](gpu-evidence.json) binds uploaded input hashes,
results, source/lock identities, CI jobs and artifact IDs. Earlier
[local-only evidence](algorithm-learning-local.json) remains historical.

Production remains disabled. Phase 3 may route and persist these qualified APIs
when requested; it must return any new algorithmic decision or semantic defect
to Phase 2. Training, final-cohort access, migration and activation are later work.
Only completed runs were reviewed; new evidence-commit CI is left unmonitored.

| Mandatory algorithm-phase.md exit row | Implemented APIs and counterexamples | Acceptance state |
| --- | --- | --- |
| Exact integers | `value`, typed `language`, guest `oeis_bigint_runtime`, `lowering` and `runtime`; canonical zero/sign, adjacent 256–4096-bit values, factorial 0…99, full multiplication/cancellation and actual signed offsets | PASS: current packaged-driver differential cases and locked guest/core tests |
| Containment | Private two-instance guest/store, restricted imports, sticky handles, source/AST/loop/arena/memory/fuel/aggregate/deadline limits; failure then healthy request | PASS: current native limits/containment and bounded-driver checks |
| Service capacity | `service_policy` FIFO one-active/eight-pending/32 MiB accounting, original deadlines, cancellation; Rust/Python bounded deterministic cache reservations, oversize bypass and failed-write release | PASS: Python/Rust policy fixtures; production storage/supervision remains Phase 3 |
| Independent reference | Separately parsed bigint and explicit retained checked-256 sources; exact arithmetic, eager intermediate limits, all actual indices; injected wrong 16-for-4 and reference exhaustion rejected | PASS: current production/reference differential and disagreement fixtures |
| Affine/SMT | `dependence`, `parameters`, `symbolic`, `affine`, `smt`, `fit`, `substitute`; shared/distinct literals, semantic control taint, exact fraction-free elimination, rank-deficient 2p+3q=1, nonlinear square=4, guarded cancellation, worker-local Z3, bounded enumeration and four uniqueness states | PASS: locked native compilation/tests and packaged fitting counterexamples |
| Admission/evidence | `admission`, `evidence`, request/schema identity checks; only final independent all-visible agreement admits; scoped UNSAT rederives source/prompt/profile constraints, unknown/timeouts never permanent negatives | PASS: independent final admission, native assignments and scoped replay |
| Codec/decoder | `bigint-codec/v1` and typed grammar; EOS/literal/parameter round trips; correct/shuffled/absent teacher forcing and actual generated-token cache/full parity; old-vocab mismatch rejected | PASS: CPU and actual HIP cache/full and conditioning parity |
| Encoder | `hybrid-chunks/v1`, finite large-number features, exact 4097-bit differences, shared chunk representation, padding/width invariance; four explicit fusion masks, matching retraining freezes and checkpoint identity | PASS: CPU ablations/features and actual HIP placement, parity and finite update |
| Data/policies | Connected counter-based A–D sampling, canonical typed families, paired 1024-token admission, observed 20-window/100-horizon grouping; diverse32 and 512/128/128 selection with explicit shortfall; visible-only selection, resumable curriculum/stopping/replay and monotone assistance | CPU fixtures pass; positive split fixture isolates selection from separately tested codec; no learned success or universal equivalence claim |
| Comparisons | Fixed denominators, sealed source before hidden 100-index scoring, prefix-only false-success=0, eight-of128 and each-control mean gates, three paired seeds, 2000 seeded family resamples, deterministic capped structural/Pareto alternatives with omissions | CPU fixtures pass; no empirical transfer gain claimed |
| Final access | `state` frozen complete final plan/arm/checkpoint roster, sealed→authorized→exposed, debit-before-work, shared/per-arm budgets, retained partial batches, exact-contract resume, no restored freshness; assistance/exposure ancestry checks | Pure state fixtures pass; command/mount enforcement is Phase 3 and remains disabled |

## Frozen boundaries and limitations

Profiles are `bigint-body/v1`, `bigint-codec/v1`, `bigint-resources/v1`,
`bigint-wasm/v1`, `native-fitting/v1`, and `hybrid-chunks/v1`; exact source/policy
hashes and unchanged counterexample fixture hashes are in the local evidence.
The retained profile is explicitly `wat_i256_checked_v1`: only checked
const/zero/add/sub stack bodies translate for fitting, with every signed-256
intermediate guarded. Scalar, wrapping, bitwise, control and streaming forms
return unsupported. No implicit conversion to bigint semantics is allowed.

The qualification bridge accepts the Phase 1 request field shape inside a
bounded harness envelope containing resolved limits, reference identity and
optional scoped evidence. Its isolated validation additionally permits a
100-index execute horizon and the explicit retained profile; public Phase 1
stubs keep their original restrictions and cannot fit/admit. Execution values
in that envelope are inert placeholders; expected continuations remain in the
test/reference process. Fit/replay still accept at most20 visible pairs.

No algorithmic policy is intentionally deferred to later integration. The current
CPU/native and actual-HIP results satisfy the required Phase 2 exit rows above.
Native artifact uploads now succeed after the cross-UID scratch fix, preserving
reports for review. Source/profile/fixture identities remain frozen in the
referenced records. Any later semantic failure reopens the affected task.

The HIP report has no Git revision field, but all 283 reported source digests
match the tested checkout. Its manifest excludes tests; reported test names and
the reviewed test source establish the scope described in [gpu.md](gpu.md).
No wider-device, sustained-throughput or learned-transfer claim follows.

Use the current [Phase 2 helper](../quickstart.md#phase-2-qualification-available-now)
for repetitions. The integrated Phase 3 CLI and Phase 5 task T092 are still future
work and have not been marked complete by this algorithm qualification.
