# Phase 2 algorithm exit review (T011–T058)

Status: **all unblocked implementation is present; CPU/native exit gate remains closed**.
PR #55 continues merged PR #54. 45 of 48 Phase 2 tasks are checked. The remaining
T025/T036 are execution gates for the common core diagnostic driver and native
fitter; T058 requires their passing results. Production remains disabled and no
Phase 3 work, training, final-cohort evaluation or activation is authorized here.

The user requested completing unblocked algorithm code and leaving new PR CI
running without monitoring. That permits implementation of the downstream pure
policies; it does not waive native acceptance or claim CPU tests as HIP evidence.
Read [the merged evidence audit](phase-2-completion.md),
[fitting evidence](algorithm-fitting.md), [learning evidence](algorithm-learning.md)
and [machine-readable local evidence](algorithm-learning-local.json).

| Mandatory algorithm-phase.md exit row | Implemented APIs and counterexamples | Acceptance state |
| --- | --- | --- |
| Exact integers | `value`, typed `language`, guest `oeis_bigint_runtime`, `lowering` and `runtime`; canonical zero/sign, adjacent 256–4096-bit values, factorial 0…99, full multiplication/cancellation and actual signed offsets | Merged PR #54 guest/core passed; rerun all differential cases through the new common driver at T025 |
| Containment | Private two-instance guest/store, restricted imports, sticky handles, source/AST/loop/arena/memory/fuel/aggregate/deadline limits; failure then healthy request | Merged guest/core checks passed; new driver enforces external RSS/deadline/output/kill/reap; current native execution pending |
| Service capacity | `service_policy` FIFO one-active/eight-pending/32 MiB accounting, original deadlines, cancellation; Rust/Python bounded deterministic cache reservations, oversize bypass and failed-write release | Python fixtures pass; Rust cache fixtures await native job; production storage/supervision is Phase 3 |
| Independent reference | Separately parsed bigint and explicit retained checked-256 sources; exact arithmetic, eager intermediate limits, all actual indices; injected wrong 16-for-4 and reference exhaustion rejected | CPU fixtures pass; production/reference differential runs pending at T025/T036 |
| Affine/SMT | `dependence`, `parameters`, `symbolic`, `affine`, `smt`, `fit`, `substitute`; shared/distinct literals, semantic control taint, exact fraction-free elimination, rank-deficient 2p+3q=1, nonlinear square=4, guarded cancellation, worker-local Z3, bounded enumeration and four uniqueness states | Implemented; locked native compile and counterexamples required at T036 |
| Admission/evidence | `admission`, `evidence`, request/schema identity checks; only final independent all-visible agreement admits; scoped UNSAT rederives source/prompt/profile constraints, unknown/timeouts never permanent negatives | CPU admission/cache fixtures pass; native assignment/replay integration pending |
| Codec/decoder | `bigint-codec/v1` and typed grammar; EOS/literal/parameter round trips; correct/shuffled/absent teacher forcing and actual generated-token cache/full parity; old-vocab mismatch rejected | CPU passes; actual HIP parity is explicitly pending |
| Encoder | `hybrid-chunks/v1`, finite large-number features, exact 4097-bit differences, shared chunk representation, padding/width invariance; four explicit fusion masks, matching retraining freezes and checkpoint identity | CPU passes; actual HIP update/device/CPU parity utility supplied |
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

No algorithmic policy is intentionally deferred to later integration. However,
new native code has not compiled or executed in this workspace, which has no
Rust toolchain or container engine. A native failure reopens its implementation
task. Review the new CI contracts/native artifacts before checking T025/T036/T058;
merged image evidence cannot substitute for these results. Actual HIP evidence
may remain pending after the CPU/native exit, as allowed by the spec, but any
HIP semantic defect reopens the corresponding algorithm.

Use the current [Phase 2 utility commands](../quickstart.md#phase-2-qualification-available-now).
Do not run the future Phase 3 CLI suites or training recipes until implemented
and independently authorized.
