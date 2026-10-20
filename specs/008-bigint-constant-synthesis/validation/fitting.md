# Phase 5 fitting qualification — T090

Status: pending `bigint-qualification-fitting-docker` for the new image.

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
