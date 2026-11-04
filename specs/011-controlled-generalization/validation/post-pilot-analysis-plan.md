# Post-pilot candidate audit — planned 2026-11-04

Input: the operator's `report(5).zip`, SHA-256
`b5c0ea49c89b62b819856054b2f4ae82672a8304a4351d18acb8a51f2f1dd494`,
from `fresh-p1-hourly-20261104T093555Z`. This is a selection-only diagnostic,
not another experiment, checkpoint choice, assessment opening or qualification.
Preserve all original scores and censoring. No training, HIP, corpus generation
or change to qualified production code is planned.

## Existing tools and reuse decision

| Tool | Use in this audit |
| --- | --- |
| `pipeline_validation.bundle.verify` | Verify every archived member and the sealed inventory before reading candidates. Add aggregate input bounds before invoking it. |
| `generalization.diagnostics.signature` | Independently parse and group generated structure with literal values removed. |
| `decoder.bigint_codec.decode_body` and independent `reference_parser.parse` | Check retained token/source agreement and distinguish grammar failure from semantic failure. |
| `synthesis.reference.evaluate` | Bounded independent execution of original and fitted programs against archived visible-only verification values. |
| `generalization.targets.ground` | Check that archived assignments reproduce the selected AST; inspect explicit-parameter feasibility without changing targets. |
| `fresh_pilot_report.py`, `pilot_clocks.py`, generation attempt/portfolio records | Reconcile original scores, attempt counts, completion and measured clocks; missing metrics remain missing. |
| `foundation_run pilot-diagnose collect/models/evaluate` | Existing collection expects historical producer/stage layout; model diagnostics require weights and a new runtime. Not directly applicable to this public generalization bundle. |
| `scripts/diagnose_bigint_run.py collect/replay` | Existing visible-only isolated CPU/native replay, with fresh output and bounded fitting. Its collector expects the historical tiny-run layout. Reuse this boundary if offline evidence identifies a specific unresolved native question; do not silently enlarge fitting budgets or rescore. |
| `fresh/token_diagnostic.py`, `fresh/checkpoint_diagnostic.py` | Teacher-forcing/cache and checkpoint tests require original weights/runtime; defer unless the retained token/source audit provides a concrete reason. |
| `pipeline_validation/replay.py`, `validation/replay_linear.py` | Backend/cache qualification probes, not candidate-fitting analysis; no new HIP probe warranted just by equal headline scores. |

## Single reopened Phase B: analysis policy and counterexamples (T166)

Implement one disposable, standard-library-only helper beside this plan. Reuse
the parsers, codec, grounding and independent interpreter above. Its contracts:

1. Read only a bounded, verified archive; reject duplicate members, mismatched
   candidate/source identities, inconsistent task/control rosters or contradictory
   visible prompts. Never extract or execute archived code.
2. Recompute each arm/control's success, completed and unresolved counts from
   original scoring rows. Join selected sources to original attempts and solver
   assignments, checking hashes and AST grounding. Never turn an incomplete fit
   into success, even if a partially searched assignment was independently verified.
3. Census all retained generated attempts: generation completion, independent
   parse, token/source agreement, literal versus parameter usage, structural
   diversity, fit phase/outcome/reason and partial-search cases. Compare paired
   task successes and source identities across arms, rather than totals alone.
4. Independently execute primary-mode originals and selected final programs on
   the archived 20 visible values under existing reference limits and a bounded
   total audit deadline. Missing prompts and local resource limits are explicit
   unavailable observations, never model failures. Bind every replay to task,
   arm, ordinal and source hash. The hidden continuation truth is not in this
   public archive: preserve recorded 100-term results, do not claim to rescore it.
5. Deterministically select multiple cases from each relevant category: shared
   successes, arm-exclusive successes, visible-fit/continuation failures,
   complete no-fit, timeouts/decode failures, and template wins over the model.
   Include original/transformed/final source and assignment where available;
   explicitly show null after-fit output when none was admitted. Inspect branch
   thresholds, unconstrained paths, repeated literals, fit restrictions and
   nonlinear/resource-limited cases. Sampling is diagnostic, not a new estimate.
6. Keep machine-readable census and case evidence in fresh output with a verified
   compact archive. Reject overwrite and write failure/partial status on bounded
   execution failure. Tests precede implementation: tampering, identity joins,
   differing outcomes with equal totals, malformed token streams, wrong grounding,
   prefix-only agreement and preserved incomplete status.

## Routine Phase C/D: execute, inspect and report (T167)

Run the helper on the supplied archive locally, inspect at least twelve balanced
before/after cases, and commit a concise analysis with exact counts, case IDs and
source snippets. Audit the relevant production paths for any discrepancy found;
reopen Phase B and regression tests before changing an algorithm. Report separate
evidence for syntax/decoding, fitting, selection and continuation. Explicit
parameter targets remain a possible P3 direction, not an inferred remedy for all
failures. Do not prescribe retraining until the diagnosis supports it.

Only if a concrete question cannot be resolved from the retained traces, supply
one bounded CPU/native replay invocation, reusing canonical execution and the
existing isolated replay machinery. No new hardware run merely to reconfirm
already captured facts. Commit and push the plan, Phase-B implementation, then
C/D evidence separately. Do not wait for CI.

Requirements: FR-033/034/035, SC-001/004, NFR-001/002/003/006/007/008. Existing
qualification remains unchanged; this diagnostic cannot satisfy or reopen a HIP
gate by itself.
