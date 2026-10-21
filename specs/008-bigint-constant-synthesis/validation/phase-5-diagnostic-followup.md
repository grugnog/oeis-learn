# Diverse32 failure diagnosis and evidence reliability

The reviewed T096 attempt remains complete with a negative outcome. Its
31/32 rescue successes meet the success threshold, but 31/32 complete rescue
observations do not meet the frozen coverage requirement. The incomplete
prompt has a successful raw selection and equal-budget selection. The bounded
bundle lacks its detailed fitting trace, so the cause is not established.

This follow-up implements the user-approved diagnostic plan within Spec 008.
Evidence semantics return to Phase 2 ownership (T043/T047/T058); persistence
and export changes cover T069/T072/T077. No solver subset, candidate ranking,
raw/rescue fallback, model, budget, threshold or experiment is changed. The
original completed run is never resumed or automatically retried. Changed
source requires new qualification identities before future experiments.

## Work and acceptance

- [ ] D001 Reproduce deadline and cross-arm completeness defects with focused
  tests; define separate generation, fitting and scoring evidence per arm.
- [ ] D002 Correct accounting without changing visible-only selection; retain
  compact, reason-bearing evidence for every prompt, including failed cases
  outside the detailed-example allowance.
- [ ] D003 Add a standard-library, read-only collector for the original run's
  sealed selections, permitted logs and retained fit evidence. Report missing
  records explicitly and produce a bounded archive without weights or private
  scoring truth.
- [ ] D004 Provide a bounded candidate replay path with a separate analysis
  identity, original resource limits, and immutable parent records. Supply
  portable container commands; decoding replay on HIP is conditional on the
  collected evidence and is not an automatic qualification rerun.
- [ ] D005 Run focused CPU regressions, document remaining hardware/data
  requirements, and leave CI unmonitored after pushing.

Completion means failures can be attributed to a stage and outcome, expired
scoring cannot count as complete, and unrelated controls cannot contaminate
another arm's coverage. It does not mean T096 passes. If diagnosis establishes
a wider solver/search limitation, propose a separate design and preregistered
experiment rather than tuning against this failed prompt's hidden outcomes.
