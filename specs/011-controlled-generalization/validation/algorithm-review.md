# Phase B algorithm review

Implemented T009–T044 as one algorithm phase, with a separate commit after every
implementation subheader group. Phase C integration and Phase D qualification
remain pending. No research campaign, learner training, container build or HIP run
was performed.

## Local validation

Executed with Python 3.12 and the isolated Phase-A test environment, using the
repository-locked pytest/jsonschema versions:

```bash
PYTHONPATH=src /workspace/scratch/5b234f116159/phase-a-venv/bin/python -m pytest -q \
  tests/unit/test_generalization_*.py tests/contract/test_generalization_*.py \
  tests/unit/test_bigint_reference.py tests/contract/test_bigint_contracts.py \
  tests/unit/test_pilot_followup.py tests/unit/test_pipeline_reuse.py \
  tests/unit/test_pipeline_bundle.py
```

Result: **113 passed, 4 subtests passed**. `git diff --check` passed.
Each of the six task groups first introduced tests that failed collection because
its policy module did not yet exist, then implemented the policy and reran tests.
The exit review expanded counterexamples and reran the combined regression set.

## Invariants reviewed

| Group | Independent evidence |
| --- | --- |
| Shared | Mutated/fabricated identities, strict fields, file bytes/hashes, traversal/symlinks, ancestry, shortfall refusal, actual target-token count, pending/debited batch restoration and storage failure. |
| US1 | Hand-computed unequal-family fractions, missing-family bounds, paired differences, scorer-only union, paired-seed intervals, charged invalid/duplicate draws, own-visible-prefix bank fitting, unavailable weights and incompatible evidence. |
| US2 | Executed/zero-iteration loops, both/one branch path, >64-bit values, length bins, insufficient active families, deterministic quota shortfalls, cross-role source/group/variant/family/window exclusions and visible-only projection. |
| US3 | Fixed support and numeric law, exhaustion, exact grounding, explicit sharing, loop-carried control taint, nonlinear repeated coefficients, >8 parameter rejection, 15 independently calculated 100-term affine traces, eligibility/ties, actual-exposure pins, atomic-commit refusal, complete-state checks, resumed RNG/next batch/debits, selection-only progression. |
| US5 | Every qualification equality key changed independently, missing capabilities, REUSED rejection, missing feature acceptance, corrupted payloads, immutable terminal receipts, fresh-root restart, reservation/storage/log bounds, private mandatory payload refusal, offline bundle round trip, stale status and tampered summary. |
| US4 | Full 100-term oracle/reference/production agreement, prefix-only failure, forbidden roles/ancestry, codec/length mismatch, task-weight invariance, deterministic per-task/global eviction, finite-behavior duplicate scope, empty pool, snapshot cursor, bounded attempts and one-round budgets. |

## Review corrections

Aligned acceptance case names with the canonical CLI contract. Added explicit
actual-exposure milestone mapping for batches that straddle a threshold. Strengthened
resolved-study validation to check referenced manifests, quota commitments, coverage,
exclusion identity and generator-visible payload schemas. Rejected invalid restored
update-boundary labels. All corrections remain within Phase B.

## Evidence limits and integration obligations

Docker, Podman and Cargo executables are unavailable here. Native differential
execution and actual GPU state/update equivalence have **not** been demonstrated;
they remain required by the registered C/D integration/qualification gates. Existing
Rust parameter protection and affine/SMT fitting are reused, not replaced. Local
Python reference checks cannot qualify the native implementation or hardware.

Synthetic qualification and replay callbacks in tests test policy decisions only.
Production adapters must provide actual inspected identities, bounded native
execution, trusted mechanical lineage, complete serialized state, immutable mounted
roots and independently verified original producer payloads. Replay restore's
verification callback must rerun admission; the simple identity callback used by the
sampler-only test is not a production verifier. Qualification receipts establish
integrity relative to trusted producer/current inspection, not authenticity against
a malicious producer that fabricates every artifact.

Public command stubs remain explicitly blocked. The algorithm interfaces and ownership
matrix are recorded in [algorithm-handoff.md](../contracts/algorithm-handoff.md).
An algorithm defect found during integration reopens Phase B and invalidates affected
evidence; it must not be patched as an unreviewed integration decision.

## Exit review invalidated during Phase C

The family-summary counterexample added during integration fails: a bundle whose
`families` artifact contains an invented family with 999 successes still returns
integrity PASS, provided its hash is consistent. The verifier checks the aggregate
summary but does not rederive family/seed/cost summaries required by T037. The prior
113-test result did not exercise this invariant. T037 and T044 are reopened; the
previous completion statement is superseded. Production integration cannot rely on
this verifier until the full reconciliation policy and counterexamples pass.
