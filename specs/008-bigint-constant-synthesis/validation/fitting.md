# Phase 5 fitting qualification — T090

Status: **COMPLETE** for reviewed implementation source
`sha256:942afe05c9cbe3ba95f839ed071ca7a44855b1222d2e0b03d82f0c39eeeef562`.
The `bigint-qualification-fitting-docker` artifact from completed
workflow39686633400 passes37 cases with zero failures/errors/skips. ZIP CRC,
archive/JUnit hashes, source and shared native artifact identities verify; see
[the execution review](phase-5-execution-review.json). All18 measurements across
six non-scoring fitting classes are complete and honestly classify fit, scoped
UNSAT or unsupported outcomes. Native p95 is0.367194s and independent reference
p95 is0.008184s in this CPU CI measurement; these are not a new HIP cost receipt.

The [deadline correction](phase-2-fit-deadlines.md) additionally has20 passing
Rust core tests and actual Docker/Podman fitting execution. The [exact visible
replay](phase-2-fit-deadlines-replay-review.json) returns five independently
verified assignments before the original worker deadline. The historical
negative T096 result and promotion block remain unchanged.

The mandatory suite exercises native affine and bounded SMT fitting, large/shared
coefficients, integer rank deficiency, singular modular pivots, inconsistency,
nonintegrality, square16-versus4, independent admission, scoped replay/tampering,
and unsupported old forms. Eight parameters/assignments, 4096-bit coefficients,
16/64/256/1024/4096-bit scopes and the2s native plus2s independent allowances
remain unchanged. UNKNOWN, partial coverage and exhausted scopes cannot become
UNSAT or verified solutions. Finite matches and scoped replay are not proofs of
a sequence or novelty. `benchmark.json` separately measures native fit and
independent admission on six non-scoring fitting classes, retaining incomplete
outcomes; incomplete measurements block cost admission.

No Python solver or trial-assignment callback is introduced. See
[phase-5.md](phase-5.md) for reproduction and evidence status.
