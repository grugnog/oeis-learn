# Structural-only comparison: accepted scope and implementation plan

The operator accepted the structural-only comparison on2026-11-04 and stated that
several hours of further training are acceptable. This plan recommends the order
of work; it does not claim that the scoped runner is implemented or that the P3
training recipe/lineage has already been fully bound. It supersedes the scope-decision gate in
the quota audit. Preserve the failed256-row run and its scientific status.

## Recommendation and decision to answer

First compare the two completed25M-token static/refreshed checkpoints on fresh
structural instances, using the existing primary metric. Establish a usable
baseline before spending additional training compute. Training time is available,
but training cannot repair an infeasible evaluation cohort. Do not repeat P1 or
extend its token cap merely because both old headline totals were128/256.

Next train explicit-parameter targets (P3). The operator clarified that moving
away from literal numeric prediction was already the intended direction. Existing
literal checkpoints are baselines, not a recommendation to spend more training on
numeric prediction. The earlier audit supports removing avoidable numeric-target
work, but does not yet quantify the gain from training parameterized targets.
Keep structure-support expansion (P2), refresh/replay changes and fitter-budget
changes separate from this representation comparison.

## Code review: existing capabilities and required changes

| Area | Existing behavior | Required structural-only behavior |
| --- | --- | --- |
|`prospective_evaluation_policy.design/draw`|256 tasks; both strata; per-family counter-keyed draws|New explicit scope/identity; preserve exact structural draw stream and quotas |
|`prepare` and summary file inventory|Hard-coded parameter then structural|Prepare only declared structural cell; no parameter gate or artifacts |
|`prospective_evaluation.publish_inputs`|Reads both prepared cells|Consume exactly declared cells and reject extra/missing roster members |
|`summarize`|Requires nonempty certified parameter and structural cells to interpret primary|Structural completion only; explicit DEFERRED parameter endpoint, no zero-score substitution |
|`lineage`|Verifies16 original exclusion cells and retained paired weights|Keep all original exclusions, including parameter cells; evaluation scope is not exclusion scope |
|Qualification helper|Explicit original producer reuse with source/image/resource checks|Reuse compatible20261104T193226Z producer, fail closed if incompatible |
|`post_pilot_analysis.py`|Historical256-task roster/layout guard|Reuse validation/census/case-review functions via a separately bound prospective adapter; keep historical guard |
|P3 preparation/runtime|Paired concrete tasks, parameter proofs, protected-control fallback and arm-order checks exist|Later fresh research manifest/controller integration still needed; template presence is not a runnable research handoff |

## Frozen evaluation scope

-128 tasks,27 structural families,2,288 pre-exclusion candidate draws. Keep the
  exact existing structural seed stream (20261104 base plus structural offset),
  descriptor identities, capacity and numeric law[-1024,1024]. Do not change
  RefreshPlan identity inputs while filtering the design.
-68 certified tasks across18 families form the primary population. Report26
  ambiguous tasks/four families and34 unresolved tasks/five families separately.
  Within-family certification does not prove unique identification across families.
-Primary: family-macro equal-budget confirmed100-term success. Require at least
  95% completion in each arm on the68 certified tasks (at least65 completed).
  Publish counts, paired discordances, per-family results and censoring alongside
  the macro rate. This remains a single-training-seed exploratory screen.
-Use the exact retained latest checkpoint pair,20 visible/100 total terms,
  eight attempts,60-second matched portfolios, all seven controls and bank seeds.
  Freeze source/helper/image/checkpoint/roster bindings before inference.
-Keep assessment sealed except existing isolated exclusion-only custodianship.
  Parameter evaluation is DEFERRED; all16 original exclusion inputs remain required.
  No adaptive seed retry, quota relaxation, hidden-truth fitting or automatic winner.

## Implementation sequence

### One Phase B group: scope, admission and reporting policy

Write counterexamples first for structural draw byte-equivalence, exact27-family
counts, changed design identity, unchanged checkpoint/control policy, parameter
rows rejected, missing families rejected, complete structural result interpretable
with parameter explicitly deferred, and genuine primary incompletion preserved.
Thread an explicit frozen scope through policy and binding, not ad hoc detection
of missing parameter files. Keep the original all-strata design reproducible.

### Routine Phase C: temporary orchestration and analysis

Update the temporary wrapper to use declared evaluation cells throughout input
publication, scoring validation and export. Preserve failure ledgers, aggregate
availability, first-error reporting and bounded archive verification. Reuse
canonical CLI endpoints and original qualification. Reuse verified preparation
only if an exact existing artifact and compatible identity are available; the
failed run contains no successful structural preparation to resume.

Add a prospective archive-analysis adapter. Census all attempts and compare raw,
first-candidate rescue and matched-portfolio results, fitter outcomes, timeouts and
structural diversity. Deterministically inspect at least12 available case/arm
records spanning shared successes, arm-exclusive successes, visible-fit/continuation
failures, completed no-fit, timeouts/decode failures and template wins. Include
original, parameterized and fitted forms; record absent categories rather than
inventing cases. Reuse independent parser/codec/grounding checks. Hidden truth is
not reconstructed; replay visible-only data where exported and label missing data.

### Routine Phase D: tests and one operator invocation

Run affected policy/controller/export/analysis tests and real CLI construction
checks; verify unchanged production closure. Commit and push each implementation
group on the existing branch, without waiting for CI. Supply one command that
checks lineage and qualification, performs CPU/native structural admission, runs
both GPU generator/scorer pairs serially, and packages verified evidence. All
quota/exclusion/native gates must pass before GPU generation. For the retained-baseline step, no fresh learner,
image rebuild or new qualification is expected if the production closure is
unchanged and original compatible images/evidence remain available.

Retain generous4-hour preparation,6.5-hour generator/arm and30-hour campaign
ceilings with reporting reserves. Expected inference is approximately30minutes
per arm by halving the previous54.8/60.5-minute256-task measurements; allow roughly
1–2hours for the combined evaluation if preparation is straightforward. These
are extrapolations, not measured structural-only throughput or guarantees.
Structural admission against original data remains unmeasured.

## How to spend the next training budget

Parameterized-target training is the next learning experiment, not conditional
on first seeing another bad literal baseline. Freeze its recipe and new structural
selection cohort before looking at the baseline outcomes; the baseline can then
be measured in the same consolidated session without an intervening scope decision.

Prefer the existing static25M-token checkpoint as the literal control, provided an
explicit audit establishes matching initialization, concrete training shard/order,
model/optimizer recipe, token cap and execution semantics. Reuse is scientific
provenance, not relabeling its original study/receipt. Its paired parameterized
arm should start from the matching random initialization, not from the trained
literal checkpoint: otherwise representation and extra training are confounded.
Keep the refreshed checkpoint as a separately labeled reference, not the matched
control for a static parameterized arm.

Use the existing native paired-target preparation/runtime checks and the same
fitter for both representations. Preserve control constants and explicit sharing;
the existing fitter does not support turning every constant into a free parameter.
Keep literal fallback with explicit unsupported-row counts. No replay or support
expansion in this pair. If exact old-control reuse cannot be established, disclose
that before handoff; a newly trained literal arm is then a scientific control only,
not a preferred modeling direction. Do not silently accept an unmatched comparison.

Implement/audit retained structural scope first, then the P3 input/initializer and
control-reuse bindings. Aim for one final operator invocation: structural CPU/native
admission, compatible qualification reuse, retained baselines, parameterized learner,
parameterized evaluation and verified diagnostic export. Gate each dependency and
freeze the complete P3 plan before any baseline scores. This consolidated workflow
is proposed, not already supported by the current retained-only wrapper. If added
integration would risk reliability, keep the verified baseline artifact reusable
rather than rerunning it during a later training invocation.

Default to the existing25M-token/6-hour-per-arm P3 policy and4/8/16/25M milestones,
with hourly recovery checkpoints and frequent lightweight progress records. Prior
literal learners took147.4/159.5minutes each; one new parameterized arm suggests roughly2.5–3hours of training as an initial
planning allowance, plus preparation/evaluation. If a new literal control is
necessary, budget roughly5–6hours for the pair. Actual
parameterized throughput is unmeasured; keep the full6-hour-per-arm limits and
report partial runs honestly. Shorter targets change examples and updates per
token, so report all three and elapsed time; a token-matched result alone is not
an equal-data or equal-compute claim.

Before such training, freeze its selection use and comparison rules. The structural
cohort is selection/development data; do not later call it unseen confirmation.
If baseline outcomes are inspected before the P3 recipe is frozen or used to alter
it, disclose that adaptation explicitly. Stronger conclusions require preregistered additional
paired seeds and untouched evaluation evidence. No confirmation or assessment
run is automatically authorized by this planning recommendation.

## Local feasibility evidence (planning only)

A source/reference-only check of the unchanged structural reservoir drew2,288
candidates, retained2,281 unique sources and selected all128 required rows with
no family quota shortfall in24.96seconds. Seven duplicate sources were removed;
there were no reference rejections. This did not run native execution or original
corpus exclusions and does not establish workstation admission readiness.
