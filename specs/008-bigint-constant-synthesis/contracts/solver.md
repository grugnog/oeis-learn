# Contract: Native Fitting and Independent Admission v1

Status: planned. All fitting/classification/substitution mathematics belongs to implementation Phase 2. Transport and caller routing integrate it later.

## Native request boundary

The new native worker uses 4-byte unsigned little-endian length-prefixed UTF-8 JSON, protocol `synthesis-worker/v1`, maximum 4 MiB/frame and at most 32 candidates per batch. Validate length before allocation. Batch by both bytes and count; the existing 1 MiB Python worker protocol remains unchanged. Stdout is protocol only; stderr is bounded diagnostics. Worker lifetime, RSS, deadline and kill/reap supervision stay outside native solving. The PyO3 interface is an optional thin transport facade, not another solver implementation. The binding/worker crate depends on the core and supplies retained-profile execution through a native trait; the core does not depend back on the binding. Bigint execution is implemented by the core. This avoids a crate cycle and introduces no Python trial callbacks.

Request fields: `request_id`, `operation` (`fit`, `execute`, `replay_unsat`), `worker_generation`, `contract_sha256`, `language_profile`, `resource_profile`, `solver_profile`, `candidates`, `visible_indices`, `visible_values`, `remaining_budget`, `seed`. Reject unknown fields/version, duplicate IDs, malformed exact integers, contract mismatch or over-limit input. No hidden values, evaluation metadata/family labels or privileged files enter fitting requests. The batch budget is aggregate; child requests cannot refund it.

Each candidate carries original source/hash, final codec profile, explicit parameter IDs or a requested deterministic abstraction policy, and an occurrence mapping when pre-parameterized. Default concrete abstraction selects at most eight mathematical literals by canonical source order after dependence analysis; it does not infer shared identity from equal values. Protect any literal flowing into control guards, trip counts, storage indices or unsupported scalar machinery. Parameter-tainted control is conservatively unsupported even when sampled values happen to choose the same branch.

An explicit signed-256 adapter supports checked whole-integer arithmetic whose semantics can be encoded exactly, with all intermediate guards. Old wrapping/scalar/bitwise/streaming cases outside that adapter return `unsupported`, never implicit bigint execution. Cross-profile translation, if requested, produces a distinct source/profile experiment and is verified independently.

## Solve procedure and limits

1. Validate/type-check complete source, explicit parameters and profile. Preserve original and transformed source separately.
2. At each visible actual index, symbolically execute supported bounded control. Guards/trip counts must be parameter-independent. Unrolling caps: 1,024 iterations/loop and 100,000 symbolic nodes across the request; exceeding a cap is resource exhaustion, not UNSAT.
3. Propagate exact affine forms `(constant, coefficient vector)` where proven by syntax/data dependence; multiplication is affine only if one operand is parameter-independent. Retain intermediate expressions needed for numeric/profile checks even if the final expression simplifies.
4. For full-rank affine systems use deterministic pivoted fraction-free elimination, exact rational back-substitution and original-equation checks. Nonintegral unique solutions and inconsistent rows need replayable contradiction records. Rank-deficient systems use bounded exact integer QF_LIA; never pick rational free variables heuristically and infer UNSAT.
5. For supported nonlinear parameter arithmetic use QF_NIA. No division/remainder/bitwise, arbitrary memory or parameter-dependent control. Do not read a model following UNKNOWN as if it were satisfying.
6. Substitute every candidate assignment, validate the final source, execute production WASM over all visible indices and retain at most eight provisional assignments. Independent reference verifies this bounded shortlist before public admission. Failed assignments may be skipped while budget remains; failures are not proofs that the coefficient domain is empty.

Initial profile: maximum eight parameters, coefficient capacity 4,096 magnitude bits, maximum eight assignments; native fitting/production budget 2 seconds/candidate plus a reserved independent-reference budget of 2 seconds/candidate. The parent portfolio has a 60-second visible-only wall budget/prompt, with all actual generation, solver and verification time counted. Unused reservation may be reused only under the frozen portfolio policy; neither a timeout retry nor an escalation gets a new parent allowance.

Unique affine solutions are solved directly, then checked against coefficient and numeric/resource limits. Ambiguous/nonlinear search escalates magnitude bounds 16, 64, 256, 1,024, 4,096 bits under the same budget. Report UNSAT only for completed scopes: failure at a smaller bound does not cover the maximum bound. If the final bound was not completed, outcome is resource-exhausted/unknown with earlier scoped evidence, not maximum-domain UNSAT.

Rank discovered assignments by `(nonzero coefficient count, summed absolute bit lengths, signed coefficient tuple in parameter-ID order)`. This is deterministic ranking of the discovered set, not guaranteed deterministic discovery under wall-clock scheduling or global optimality. Keep seeds, engine version, constraints and actual assignments to replay evidence. Uniqueness uses a blocking clause for the complete assignment within the same declared scope and remaining budget; states are `unique_in_scope`, `multiple`, `unchecked`, `incomplete`. Do not infer uniqueness from one model.

## Native versus public statuses

| Native result | Public admission result |
| --- | --- |
| `assignment_found` with production visible fit | Provisional until independent reference and final-source validation succeed |
| Provisional + full independent agreement | `verified_solution` with two execution identities and all visible-index evidence |
| Provisional + reference mismatch | `verification_failure`; retain counterexample, admit nothing, flag implementation defect |
| Provisional + reference/production limit | `resource_exhausted` with phase and measured coverage; admit nothing |
| Exact supported-domain contradiction / completed SMT UNSAT | `unsat_in_scope` with bounds, constraints and replay status |
| Unsupported semantics | `unsupported` with source location/operator/reason |
| Solver UNKNOWN | `unknown` with engine reason |
| Timeout/node/memory/fuel limit | `resource_exhausted` with reason, not UNSAT |

Python batches independent final checks; it never proposes or searches constants. The reference parses final source independently, not the production AST. The native worker cannot grant `verified_solution` on its own. Evidence replay must re-derive the constraint problem from recorded source/profile/visible inputs and compare its hash, not merely solve arbitrary attached SMT text.

An UNSAT replay record is scoped solver evidence, not a formal proof of the whole sequence or proof that no program can generate it. Optional solver proof text is not promoted without an independent certificate checker with declared assumptions. Historical records can remain `historical_unreplayed` if no supported replay exists; do not execute retired Python solvers to upgrade them.

## Response/evidence fields

Per candidate: request/candidate ID, original/transformed/final source hashes, abstraction occurrence map, parameter assignment exact strings, numeric/solver/runtime/reference identities, visible-prompt hash and actual indices, outcome/reason/phase, completed bounds, ambiguity status, complete scope, elapsed phase times, fuel/steps/memory when available, coverage, constraints hash and bounded replay artifact, final validation and reference results. Missing metrics are null plus a reason, never zero invented measurements.

Partial batch responses identify completed, failed and unstarted candidates. Cache keys include exact visible prompt/index bytes, source/parameterization, language/resource/solver/reference identities and policy. A cached timeout/unknown is not permanent UNSAT; a cached verified result cannot cross profiles or skip requested independent-evidence validation. Restarted workers have a new generation and no surviving arena/Z3 state.
