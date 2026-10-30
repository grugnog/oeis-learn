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

## T037 repair and T044 reassessment

The fabricated-family failure is repaired. Verification now derives aggregate,
family, seed, cost, coverage and bound projections from strict per-attempt records
and a frozen invocation roster. Added counterexamples for each hash-valid false
projection, missing seed denominators, unequal families, duplicate ordinals,
unregistered rows/seeds, private attempt fields, invalid outcomes and timing types.
Partial-bundle integrity is independently exercised. Original stage references and
transitive qualification payload inventory are checked as part of reconciliation.

The combined feature/adapter and historical reference/followup/reuse/bundle selection
passes: **131 tests, 4 subtests**. This supersedes the recorded failing T037 regression.
T037 is repaired and T044's local algorithm exit gate is restored. The command is the
previous combined command plus `tests/integration/test_generalization_evidence.py`.
No native/container/HIP execution occurred. Phase C production integration and Phase D
qualification remain pending; this result establishes local policy/reference behavior.

## T035 resource-authority correction during integration

The first real resource projection exposed a boundary mismatch: `compute_resources`
returns effective CPU, Torch/native thread and learner limits, but the v1
qualification schema reused study budgets and rejected those fields. Dropping the
fields would make exact reuse matching incomplete. T035/T044 were reopened for this
counterexample and repaired within Phase B. `generalization-qualification/v2` now
binds the complete computed resource projection and concrete engine/security facts;
v1 producers fail closed. Study token/time budgets remain separate. The synthetic
fixtures were regenerated, with no claim that they represent hardware evidence.

Tests cover policy-to-authority serialization, every missing resource/security key,
every changed resource/security value, and v1 rejection. The combined feature and
historical selection passes **152 tests and 4 subtests**. The local T044 gate is
restored. No container/native/HIP qualification has run.

## T024 projection boundary correction

Schema checks on the prepared protected payloads exposed a second compositional
counterexample: `project()` returned a generic family-bearing record for every role,
which omitted training tokens and scorer strata. The Phase-B projection now emits
and validates the distinct training/scorer schemas, requiring a declared protected
holdout stratum for scorer payloads. The visible projection is still exactly opaque
id plus 20 indices/terms. Integration preserves that stratum and validates descriptor
family/grounding before materialization. This correction does not relax a schema or
supply hidden metadata to generation. The dedicated policy/preparation regression
selection passes (10 tests); the next combined result records the full exit review.
The combined selection after the projection correction passes **177 tests and
9 subtests**, restoring T024/T044's local gate. A subsequent preparation-only check
passes 6 tests after preserving parameter-target dispositions as protected metadata.


## Reopened T043/T044: replay overrun loses charged work

Integration exposed a failing counterexample in `test_search_overrun_retains_charged_attempt_evidence`.
A deterministic clock advances from 0 to 61 seconds inside the proposal callback,
which then raises a deadline/cleanup failure. `search_round` invokes
`SearchLedger.charge` only after the work. The policy rejects the over-limit
measurement, and the adapter returns an empty search ledger although one attempt
ran. The expected charged-attempt count is 1; actual is 0. No sleep, native process,
GPU or HIP command is involved in this reproducer.

T043 and T044 are reopened. Required Phase-B repair: define and test the accounting
transition for reserved/started attempts, measured elapsed work, timeout and cleanup
overrun, including interruption before completion. Keep the 8-attempt/60-second
admission bounds without discarding already performed work or disguising measured
costs. Reconcile the adapter against that policy and restore the complete gate.

This blocks continuation under plan.md's explicit rule: “A discovered algorithm
defect reopens B and affected gates. Do not implement a workaround in C/D and call
it integration.” The failing regression is retained and is not marked xfail/skipped.
C/D are incomplete; this is not a hardware limitation or a claim of qualification.


## T043/T044 replay accounting repair — local gate restored

The previously recorded 61-second counterexample now retains one charged attempt,
61 measured seconds, and an explicit 61-versus-60 reservation overrun. The repair
is in the Phase-B `SearchLedger`: separate reserve and finish transitions, immutable
v2 snapshots, exact nonnegative finite measurements, and admission limits applied
before new work. Settlement does not discard over-budget measured work.

The adapter reserves/publishes before callbacks, settles both failures and successful
work, refuses late audit/admission, and publishes interruption evidence before
propagating the original exception. Persistence failures prevent work from starting.
A pending snapshot survives interruption without enabling another reservation.
No automatic replay-search restart or durable persistence without an adapter callback
is claimed.

Independent checks cover the 61-second overrun, aggregate 14,401-second settlement,
256-task/eight-attempt limits, zero-duration observations, invalid measurements,
pending snapshot round-trip, interruption at three seconds, unexpected native
failure, late proposal rejection, failed persistence, and persistence consuming the
deadline. Focused checks: **18 passed**. Combined non-HIP feature/reference/contract,
CPU learner integration and historical regression checks: **200 passed, 9 subtests
passed, one GPU test deselected**. `git diff --check` passes. T043 and T044 are
restored. No native/container/HIP qualification or scientific run occurred.
The remaining C/D tasks retain their prior incomplete status.
