# Current Phase B status — 2026-10-31

T009–T044 are complete. The family reconciliation, fitted-source replay and retained
initializer authority defects documented below are repaired in their Phase B
policy owners, with independent counterexamples retained. Version 2 study/preparation
contracts require P4's verified common retained literal initializer and its original
selection/producer/ancestry proofs. Version 1 remains random-only. The current
non-HIP selection passes 285 tests and 9 subtests. See [local-checks.md](local-checks.md)
for exact commands and the separate environment-dependent historical failures.
All C integration is now implemented; installed-image and HIP validation remain open.
Earlier entries below record the evidence available at each historical increment.

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


## Reopened T035/T044: failed or unproven feature cases authorize research

While wiring the canonical acceptance consumer, a new independent regression showed
that `qualify_reuse(..., research=True)` checks only the aggregate feature status,
qualification identity, and the set of case names. It does not check the values of
those cases or require original per-case evidence. The three cases in
`test_research_gate_rejects_unproven_or_failed_feature_cases` all fail because reuse
is accepted: all five case values say FAIL; all five say PARTIAL; or cases are only
a list of names with no evidence. Each test uses synthetic qualification fixtures;
no hardware command is involved.

T035/T044 are reopened. The required Phase-B repair is a closed, versioned feature
acceptance authority binding every required case to intact original PASS evidence,
its qualification and case identity, and the case's mandatory artifacts. Reject
missing/unproven cases, failed/partial results, mismatched producer/qualification,
unknown fields/versions and substituted case payloads. Merely changing the producer
to print PASS or checking the aggregate flag again is insufficient. Define the
stable producer/consumer interface and independent negative cases before C creates
the feature-acceptance producer or enables research consumers.

This blocks T056–T059 and acceptance-dependent workflows; T044 explicitly gates all
C work. No new research endpoint was enabled as a workaround. The earlier replay
accounting repair remains passing. The new regressions are retained, not skipped or
xfail. Completed reporting and preparation/role-isolation integration remain
committed, while C/D as a whole remain incomplete.


## T035/T044 feature-acceptance repair — local gate restored

The three failing counterexamples are fixed. Research now requires a closed v2
feature manifest referencing original PASS canonical receipts for all five cases.
Each receipt binds its case and qualification, a closed result record with all
required checks, and all mandatory case artifacts. All original files and nested
public artifacts are verified; unproven lists, failed/partial/reused cases, unknown
fields/versions, changed bindings, false/missing checks, substituted/missing/tampered
artifacts and private descendants fail closed. `assemble_feature` applies the same
policy before an aggregate PASS can be produced.

The isolated consumer now projects the complete verified feature closure, including
original receipt inventory files. Positive tests verify all five synthetic original
cases, producer assembly, standalone copied closure, and consumer projection.
Negative tests independently enumerate the required case obligations and mutate
versions, bindings, statuses, fields, checks and payloads. Synthetic evidence is not
hardware qualification or independent attestation of an untrusted producer.

The combined non-HIP selection passes **242 tests and 9 subtests**, with one GPU
test deselected. The original replay-overrun and feature-gate regressions all pass;
`git diff --check` passes. T035 and T044 are restored. No HIP command, container
qualification or research campaign ran. Remaining C/D work is still incomplete.

### Replay fitter transformation repair

C replay wiring exposed a T039/T041 admission incompatibility: native fitting can replace literal coefficients with an explicit parameterized source before grounding. The previous admission API grounded the generated source itself, rejecting valid native transformations. The new independent regression first failed with an unsupported `transformed_source` argument. Version 2 admitted entries retain generated, transformed, and grounded sources separately. The policy verifies that transformation changes only literal leaves into parameters, then independently checks exact grounding and the existing 100-term oracle/native/reference agreement and shorter-target rule. Legacy v1 entries retain their original semantics. Restore revalidates the transformation, rather than trusting stored admission flags.

T044 was rerun after the repair: the combined generalization/reference/historical regression selection passed 246 tests plus 9 subtests, with the one marked GPU test deselected. This is CPU/fixture evidence, not native-container or HIP qualification. The dedicated v2 round/restore integration is also included in the follow-up regression run. This repair belongs to Phase B; C wiring resumes against the repaired interface.

### Open gate: frozen P4 initializer authority (2026-10-31)

T011/T044 are reopened. The independent counterexample
`test_p4_cannot_freeze_random_initialization_without_common_retained_authority`
constructs a P4 study with the correct paired arm names, literal target format,
4M-token/7200-second caps and 1M/2M/4M milestones, but only random initialization.
`freeze('study', ...)` accepts it. Both study and preparation schemas constrain
`model.initialization` to the constant `random`; neither can identify the common
retained initializer required by the P4 experiment contract. The low-level learner
can load a separate weights reference, but that is not frozen study authority.
Exposing it as an arbitrary runtime setting would weaken model/lineage binding.

The repair needs a versioned preparation/study initializer contract, with a stable
policy validating exact retained weight/checkpoint identity, mechanical-only
training lineage, compatible model/precision/literal representation, and the same
initializer for both paired arms. Keep random initialization valid for applicable
fresh studies and historical routes. Reject imported/unknown ancestry, incompatible
or changed weights and unregistered representation/replay interactions. Include
positive retained-initializer publication/resume cases and negative identity,
lineage and paired-arm counterexamples, then rerun T044 before C routing resumes.
Do not merely ban all P4 studies or relabel retained weights as random.

Latest combined regression: **249 passed, 1 failed, 1 GPU deselected, 9 subtests
passed**. The only failure is this new contract counterexample. This is a real
prerequisite failure, not a HIP failure. No HIP command ran. Remaining C/D is not
complete; the feature gate is not restored.

### Retained initializer repair — gate restored

Implemented versioned retained initialization and `initializers.verify_initializer`.
The original P4/random-initialization regression now passes. New independent tests
first failed for the absent policy; they now cover a valid P4 authority, v1 semantic
preservation, preparation v2, exact weight tampering, incompatible architecture or
precision, parameterized replay, forbidden ancestry, REUSED origins, changed selected
checkpoint, an unbound random parent, inconsistent selection measurements, and a
valid retained-parent chain whose ancestor weights are reverified.

The policy verifies the full selection decision and original producer inventory,
then follows the bounded mechanical-only parent chain. It never loads tensor state
or treats a standalone weight hash as lineage proof. T011/T044 are restored.
The combined local non-HIP selection passes **262 tests plus 9 subtests**, with one
marked GPU test deselected. No native-container or HIP qualification is claimed.
C/D implementation continues under the user's explicit instruction.

### Acceptance closure extension — 2026-10-30 (America/Los_Angeles)

The existing feature acceptance policy now exposes verification of a subset of
original PASS cases without granting feature qualification. A software-only
acceptance archive delegates to that policy and original qualification validation;
missing cases, stale aggregate status, tampering and partial workflow failures
cannot become PASS. The new independent counterexample first failed with the
missing verifier, then passed with complete/partial safe ZIP round trips. This
extends the Phase-B evidence policy; no scientific estimation algorithm was added
to routine integration. Research evidence still uses fixed-roster reconciliation.

## Reconstruction policy review — 2026-10-31

Reopened Phase B covers T071–T074: common row/family selection rosters;
explicit literal retention for unsupported control-dependent targets; study v3
execution binding and preparation v3 proposed-study binding; isolated stage,
paired-batch, refresh-mixture and acceptance policies; original uniform weights-only
provenance and historical pooled-loss checkpoint exposure. Counterexamples were
run before implementations (4, then 4, then 6 expected failures). The combined
focused policy/authority selection passes 30 tests. Routine production wiring
remains T075–T077. This is local policy evidence, not container/HIP qualification.

During reconstruction integration, Phase B was reopened for the preparation
resolution boundary: resolved sidecars must preserve all non-path execution choices.
The independent changed-batch counterexample failed before repair. Acceptance
counterexamples also require at least one real supported parameterized target and
finite parameter-update proof for **both** continued-SFT and replay arms; a nominal
PASS/resume result alone cannot establish those obligations. The replay false-PASS
counterexample was observed before repair. These checks remain in the same Phase B
policy boundary; C calls them and emits actual tensor proofs. The extra paired proof
is included in the unchanged 900-second case reservation (work ceiling 360 seconds).

The refresh input review also found that duplicate numeric draws could survive
per-cell quota selection and fail only at runtime. Phase B now claims each refresh
source once across its frozen training cells; duplicate draws remain charged and
are recorded as rejections. The three-draw/two-unique-source regression was observed
failing before repair. The combined preparation/input/schema selection passes 37
tests. Legacy schema branches retain their original positive-token/milestone rules;
only explicit v3 E0 authority permits zero learner exposure.

Acceptance launch convergence reopens B (T080/T081): verified legacy initialization
requires a new actual selection, original uniform provenance and producer inventory;
pending P4 initialization cannot authorize research or learning. The transition
preserves all choices except its declared initializer. Deterministic mechanical
fixtures enforce finite exclusions and paired target grounding; a separately labeled
neutral-expression diagnostic still requires native/reference full-horizon replay
admission. Fresh refresh/target fixtures use the existing 256-wide v1 constructor;
retained evaluation/P4 keep the original recipe and weights. This sizing choice
preserves the 32 GiB aggregate checkpoint ceiling without altering historical inputs.
Independent cases observed failures before implementation; 55 authority/contract
checks and 30 fixture/replay policy checks passed. Original imported weights remain
nonresumable. These checks are not HIP evidence.
