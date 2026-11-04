# Observable continuation evaluation and bank provenance — 2026-11-04

Status: implementation plan; T168–T170 are not completed. This follows the
[retained pilot audit](post-pilot-analysis.md). It does not change historical
P1 scores, train a model, open assessment, or authorize an automatic hardware run.

## Objective and limits

Separate demonstrated information loss from model, structure and fitting errors.
The previous55-task ambiguity group is supported by concrete witnesses; the
remaining201 tasks are **not** thereby proved observable. A new policy must retain
an unresolved category. Classification must depend on source-family semantics and
the fixed visible/scoring indices, never model outcomes or which arm wins.

A certificate means: within this exact registered family, any two supported
parameter assignments that agree on indices0..19 agree on20..99. It does not
identify the family from the prefix, prove a unique program across all grammars,
or guarantee that the model can synthesize it.

The current pilot has already been inspected. Applying a new rule to its retained
outcomes remains a post-hoc diagnostic, even if the rule is written before the
next command. Prospectively valid comparison requires freezing the rule, task
roster and decision criteria before inference on any new evaluation cohort.

## Existing implementation review

- Reuse `post_pilot_analysis.py` for verified archive reads, family-hash joins,
  explicit ambiguity witnesses and original outcome reconciliation.
- Reuse independent `reference_parser`, `reference.evaluate`, `targets.ground`,
  canonical family identities and existing sealed artifact/bundle helpers.
- The existing `branch_visibility` diagnostic is a syntactic screen, not an
  observability certificate. Its bounded witness search cannot certify absence.
- The source-only numeric diversity screen in `fresh_pilot_inputs.py` rejects
  near-fixed windows, but does not require every continuation-relevant parameter
  to affect the prefix. Do not silently change its frozen policy.
- Existing canonical checkpoint evaluation and scoring can reuse retained weights
  and compatible prepared inputs. No new learner or corpus is needed for an
  outcome projection or a later compatible evaluation-only comparison.
- The bank defect is in `generalization_evaluation.evaluate_retained`: fitted bank
  admissions lack `_original_source` before `bigint_diagnostics._seal`. The
  ordinary portfolio already preserves this field. Legacy v1 selected seals
  bind the final candidate, not the defective outer original-text field.

## Single reopened Phase B — policy and counterexamples (T168)

Implement a temporary analysis helper beside the existing validation helpers.
Freeze a versioned, sealed classification policy and original selection roster,
recording family identities, index sets, implementation/source identity and scope.
Keep model scores out of the classifier's input.

Classify each family into exactly one of:

1. **Certified within-family continuation.** Conservatively support straight-line
   integer polynomial expressions and the sampler's fixed index-only branches.
   Constants/parameters have degree0, index degree1, addition/subtraction take
   a conservative maximum degree, multiplication adds degrees, and local reads
   use validated path-specific definitions. Reject unsupported operations,
   uncontrolled locals, parameter-dependent guards and loops from certification.
   For each path reached by a continuation index, either its output is independent
   of free parameters, or its degree bound d has at least d+1 distinct visible
   indices on that same path. Polynomial identity then establishes within-family
   continuation equality. Record the paths, degree bounds and observed indices
   as the certificate. Conservative overestimation yields unresolved, not a false
   certificate. Only claim bounded runtime compatibility separately.
2. **Demonstrated ambiguity.** Retain an independently executed pair of legal
   assignments with the same visible prefix and a different continuation. Bind
   both programs and assignments to the same family and policy. Preserve existing
   constructed witnesses where compatible; never substitute actual hidden target
   coefficients or model outputs.
3. **Unresolved.** Everything else, including unsupported loops, insufficient
   path observations without a witness, cancelled expressions not proved by the
   conservative analysis, resource limits and missing evidence.

A witness and certificate for the same family is an implementation error and must
fail closed. A failed witness search is never certification. Polynomial
certificates are mathematical family properties; finite resource/error behavior
must not be relabeled as a successful executable result.

Before implementation, write and observe failing independent tests for:
a linear/quadratic family with enough points; a branch at19 versus20; a late
single-index equality branch; different independent coefficients across branches;
repeated/shared parameters; a parameter-independent hidden path; degree bounds
above available observations; unsupported loops/guards/locals; cancelled
parameters; an explicitly constructed same-prefix/different-tail pair; corrupted
identities; and outcome-blind membership invariant under swapping arm results.

Freeze the exact original256 selection IDs into the derived diagnostic protocol,
with exhaustive/disjoint categories and unchanged denominators inside each.
A classification protocol must not include actual assessment rows or later become
authority to leak family labels into the generator. Include family-level and
task-level counts; never present the certified subset as a replacement P1 headline.

## Routine Phase C — bank provenance repair (T169)

Write an integration regression through the real retained-evaluation bank path
and real sealer, using a fake fitter only at the native boundary. Return an
admitted candidate whose literal original differs from its parameterized
transformation and final fitted source. First reproduce the original/hash mismatch.

After a fit returns, bind each admission to the exact source passed to the fitter:
verify its `original_source_sha256`, preserve a copy with `_original_source`, and
allow disagreement to fail rather than being swallowed as an unsupported-source
result. Do not mutate shared cache records. Test wrong hashes, several admissions,
empty/failed/incomplete results and distinct recipient prompts. Preserve search
budgets, candidate ordering, selected-source semantics, costs and original
historical envelopes. Do not infer completion from the presence of an admission.

Update the relevant checked-in source/qualification inventory and review bindings
using existing repository tooling. Production source changes invalidate affected
old qualification reuse; do not relabel the old receipt as current. No hardware
run is needed merely to inspect or fix metadata, and this repair is not a claim of
fresh container/HIP qualification.

## Routine Phase C/D — execution and interpretation (T170)

Run the classification helper on the original verified report; write a fresh
protocol, certificates/witnesses, task-category mapping, original-score projection,
omission ledger and verified diagnostic zip. Keep existing audit records unchanged.
Report success/completion by category and arm, explicitly retaining unresolved
classification and unresolved evaluation as different quantities. Check all
selected identities and avoid oracle candidate unions.

Inspect representative certified, ambiguous and unresolved cases. Publish the
classification counts only after actual execution; do not promise that the entire
201-row residual qualifies. Document what would need a new evaluation identity
and what can reuse retained checkpoints. Do not automatically generate new data,
score assessment, retrain or alter the number of visible observations.

Run affected local tests and source-inventory checks; commit and push the plan,
the Phase-B policy, then routine integration/evidence groups separately. Do not
wait for CI. If native/HIP follow-up becomes necessary, bundle the required
canonical endpoints into one bounded operator invocation and explain why retained
evidence is insufficient. A diagnostic projection alone needs no HIP.

Requirements: FR-033/034/035, SC-001/004, NFR-001/002/003/004/006/007/008.

## Execution recovery

The workspace exec service initially disconnected and recovery timed out. The
plan was published through GitHub with tasks unchecked. Execution recovered
later in the same turn; implementation then resumed with local failing-test
observations and validation. No production repair was published during the outage.

The existing `discovery/polynomial_certificate.py` uses SymPy and a different
logical IR; it does not support the parameterized bigint branch contract or a
stdlib-only host. The temporary helper therefore uses conservative degree bounds
on the existing independent bigint parser, with the explicit root-count proof
above. It adds no production polynomial engine or dependency.
