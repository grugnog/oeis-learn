# Phase A to Phase B algorithm handoff

Phase A implements T001–T008 only. Shapes, immutable boundary types, declared mounts and blocked dispatch are reviewable. They are not provenance checks, enforced container isolation, qualification reuse, learning, fitting or acceptance. Every prospective command currently returns BLOCKED/UNMEASURED with no artifacts or I/O; the portable host dispatcher remains unwired until Phase C.

## Stable contract surface

- `src/oeis_learn/generalization/interfaces.py`: frozen/slotted artifact, original qualification producer, weights-only/full-state checkpoint, command/result, preparation and selection types. Collections are tuples; status/exit mappings are immutable. An annotation is not runtime validation; Phase B validates values and provenance.
- `role_contracts.py`: immutable input/output mount declarations for preparation, learner, generator, fitter, selection, assessment and report. Each channel denotes a separate payload root. Phase B validates projections; Phase C enforces mounts and path confinement.
- `schemas/`: Draft 2020-12 schemas for the eight authority formats plus shared definitions and distinct visible/training/scorer/weights-only payloads. All authority objects and nested records reject unknown keys; typed count/assignment maps are explicitly permitted. Local references use the `https://oeis-learn.invalid/generalization/` namespace and must resolve offline from package resources.
- `pipeline_validation/generalization_cli.py`: immutable handler registry and dispatch stub for preflight, qualify, prepare, evaluate, train, replay, report, verify-bundle and acceptance. Unknown operations raise; known operations never return PASS until integrated and qualified.

Schema names are format authority: use `identity` consistently (not the conceptual `id` shorthand in the data-model table), `role` for cohort payload projection, and `full_state`/`weights_only` as distinct checkpoint kinds. `preparation` freezes policy before materialization; only successful Phase-B preparation can produce a new resolved study. Numeric budgets are finite positive integers; changing these shapes requires explicit contract/schema review.

Common fields are structurally validated only. In particular, hash-shaped identity strings and PASS/hip_available fields can be fabricated. Schema validity MUST NOT imply admission, checkpoint compatibility, an intact original PASS receipt, actual hardware execution or a valid selection decision. The test deliberately accepts a well-shaped fabricated identity to keep this distinction visible.

## Single-phase ownership and independent exit cases

| Phase-B tasks | Owner / future interface | Required independent cases and exit evidence |
| --- | --- | --- |
| T009–T012 | `contracts.py`, `budgets.py`; immutable types in `interfaces.py` | Verify actual canonical identities/bytes, confined regular-file references, forbidden ancestry, preparation transitions, integer/token budgets, pending batches and conservative interrupted debits. Unknown/tampered authority fails. |
| T013–T018 | `metrics.py`, `proposals.py`, `diagnostics.py`, `evidence.py` | Golden unequal-family/censoring fractions, unavailable weights, fixed denominators, charged duplicate/failed proposals and recipient-independent banks; oracle unions never become executable selections. |
| T019–T024 | `cohorts.py`, `coverage.py`, `exclusions.py`; `PreparationPolicy.prepare` | Mechanically generated active/zero-iteration loops and both branch paths; exact quota shortfalls; cross-role and future-shard duplicate/prefix/shift collisions; protected labels remain scorer-only. |
| T025–T032 | `refresh.py`, `targets.py`, `selection.py`, `state.py`, `checkpoints.py`, `experiments.py`; `SelectionPolicy.select` | Fixed descriptors vs legacy counter expansion; protected loop-carried constants, explicit sharing, nonlinear coefficients; below-95% eligibility and ties; interrupted versus uninterrupted state; assessment cannot choose progression. |
| T033–T038 | `qualification.py`, `recovery.py`, `bundles.py`, `resources.py` | Changed/unmapped source, image/device/runtime/security, malformed producer chain, fabricated receipt and payload, stale summaries, incomplete stage sets and reserved packaging budgets all fail conservatively. |
| T039–T043 | `replay_admission.py`, `replay_policy.py`, `replay_round.py` | Prefix matches that fail independent full audit, forbidden provenance, task-weight invariance under added alternatives, empty yield, deterministic two/task and global caps/eviction and exact restore. |
| T044 | Combined review and reference/differential tests | Every policy above implemented through reviewed interfaces and independent tests; no known algorithm deferred to integration. Record the exit in `validation/algorithm-review.md`. |

## Fixture provenance and use

`tests/fixtures/generalization/generate.py` imports only the standard library and no project policy. It deterministically produces generic affine programs from small coefficient combinations, a bounded accumulator loop with active/zero bounds, and a branch whose both sides execute over the registered indices. Expected terms/iteration counts are derived by elementary Python formulae. Existing independent reference execution checks those terms; it does not generate expectations.

`policy-expectations.json` contains manually derived rational row/macro bounds, a 20-term prefix match failing the 100-term audit, a duplicate-role group, and a pending-batch interruption ledger. Phase B must use these expected outcomes and expand counterexamples before implementing policies. `role-expectations.json` is independent of the production role declarations.

`authority.json` and its hashes/verification flags are synthetic shape fixtures. They are NOT signed/qualified, independently admitted or usable as real checkpoints, receipts or training artifacts. `inventory.json` hashes the five generated JSON payloads, excluding itself and the generator to avoid recursive identity. Regeneration must reproduce all bytes.

## Resource, compatibility and phase boundaries

No new runtime dependency is introduced; schemas use the project's existing jsonschema dependency and are included in wheel package data. Phase A does not register new host commands, modify historical recipes or alter the test-selection/source-closure rules assigned to Phase C. New tests must be registered in the qualification container in T060 before claiming container acceptance.

The implementation model for B receives the full plan, contracts, this handoff and independently computed fixtures. C may connect I/O and mounts only after T044. Any algorithm defect found during C/D reopens B and invalidates affected evidence. Normal real-HIP acceptance is still the single Phase-D invocation; no Phase-A result establishes HIP readiness.

## Phase-B exit review

The ownership matrix above is implemented at its listed module paths. T009–T044
are checked; local reference/contract and historical regression evidence is in
[algorithm-review.md](../validation/algorithm-review.md). Public dispatch remains
BLOCKED until Phase C registers adapters. No hardware qualification is implied.

Stable callable policy entry points:

| Owner | Entry points | Boundary obligations for adapters |
| --- | --- | --- |
| Shared | `freeze`, `validate`, `verify_artifact`, `resolve_study`, `Ledger.reserve/commit/restore` | Read-only confined authority roots; debit before work; persist pending batch atomically. |
| Evaluation | `summarize`, `paired_difference`, `seed_uncertainty`, `oracle_union`, `make_bank`, `apply_bank`, `portfolio`, `join_attempts`, `reconcile` | Native generation/fitting run under existing external deadlines; pass measured costs, not estimates. |
| Preparation | `observe`, `distribution`, `choose`, `ExclusionIndex`, `project`, `publish` | Bind observed coverage to actual rows; apply frozen exclusions to every future shard; separate role mounts. |
| Learning | `BigintSampler.descriptor`, `RefreshPlan.take/shard`, `pair`, `select`, `snapshot/restore`, `pin_plan`, `commit_plan`, `promotion` | Preserve unsupported target rows; native fitter unchanged; serialize actual tensor/RNG payloads and verify refs before load. |
| Execution | `consume`, `qualify_reuse`, `Workflow`, `restart`, `preflight`, `reserve_stage`, `bundle_plan`, `verify_bundle` | Inspect real source/image/device facts; receipts are integrity records, not cryptographic attestation; only trusted qualification producers issue them. |
| Replay | `admit`, `ReplayPool.add/sample/restore`, `SearchLedger.reserve/finish/snapshot/restore`, `round_plan` | Production callback must be qualified bounded native execution; restore verifier must redo admission against trusted training roster, never an identity callback. |

Internal versioned policy snapshots (`generalization-ledger/v1`,
`generalization-learner-state/v1`, `generalization-descriptor/v1`,
`generalization-admitted-entry/v1`, `generalization-replay-pool/v1`, and producer/device
receipts) are separate from the Phase-A external artifact manifests. Their methods
validate their internal transition contracts; Phase C serializes them as hashed
payloads referenced by those external manifests. A self-consistent hash never
proves an untrusted supplied origin or that hardware actually ran.

Quota selection uses deterministic bounded greedy feasibility. It may conservatively
report a shortfall even where a more expensive combinatorial search could succeed;
it never relaxes quotas. Numeric draws preserve the registered independent uniform
law; repeated variants consume draws and are rejected by the exclusion/admission
callback rather than silently recycled. Finite-window exclusion is not a semantic
proof. Small-seed bootstrap intervals are explicitly labeled by resampling unit.


Replay accounting handoff (T043 correction): reserve an attempt before invoking
proposal/native work; atomically persist the pending ledger and pool through the
`search_round(save=...)` callback before work starts. Settle measured elapsed time
without clipping it to admission limits. Overruns remain explicit and cannot admit
late candidates. Eight attempts/task, 60 seconds/task, 256 tasks and four aggregate
hours gate new reservations; already measured costs may exceed these ceilings.
`generalization-search-ledger/v2` preserves pending reservations and overrun records.
An unfinished reservation has unknown final cost, not a completed zero-cost attempt;
restoring it does not permit another reservation. Durable recovery must preserve it.
`generalization-replay-round-result/v2` carries the full ledger plus its legacy
per-task count/time projection. Interruptions propagate with `replay_evidence` and
are also sent to the save callback; without a save callback that evidence is only
in memory. C must wire atomic persistence to claim recovery across process loss.


T035 feature-acceptance handoff: `feature_acceptance.assemble_feature`,
`verify_feature` and `feature_artifacts` enforce the v2 aggregate and original
per-case receipt/result formats described in artifacts.md. The C acceptance
producer must persist actual per-case public results and original runner receipts,
then call `assemble_feature`; a five-name list or aggregate PASS is insufficient.
Consumers project the complete verified closure and revalidate it inside the role.
No hardware execution or arbitrary self-authored claim is certified by hashing.

Retained initialization handoff: `initializers.verify_initializer(root, study)`
verifies version-2 frozen initializer authority, original producer/selection and
all mechanical-only ancestors. It returns exact `weights`, their `root`, selected
`checkpoint`, initializer `authority`, and verified lineage documents. Invoke before
projection/training; mount only the returned weight bytes to the learner/generator.
Never substitute runtime initializer settings or project ancestor selection truth
into those roles. `resolve_study` invokes the policy before publication. Both P4
arms consume the same verified initializer; full-state resume keeps its existing
exact compatibility check. Contracts/artifacts.md specifies the local reference roots.
