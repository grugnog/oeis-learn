# Research and design decisions

Design baseline: merged 011 at `efce05d64294cb013d7075802e44560ce1d45b05`.
Reviewed existing code, the 011 closeout and its reliability/experiment records.
No private run archive was newly inspected in this design pass; numerical history
below is attributed to the committed reconciliations, not presented as a new audit
of original tensors. Operator-side feasibility and control receipts remain gates.

## 1. Register a new experiment, preserve P3

**Decision:** introduce `PAC-screen/v1`, a new closed policy for the literal-fixed
A, parameterized-fixed B and parameterized-adaptive C screen. Do not turn P3 into
an arbitrary three-arm recipe. A may be an audited retained control, a predeclared
fresh control or an explicitly unmatched historical reference.

**Rationale:** `generalization/execution.py` fixes P3 to literal/parameterized
static arms and requires a selected incumbent for research. Fresh random B/C and
adaptive data order cannot be added as a wrapper flag while retaining that meaning.
The [011 closeout](../011-controlled-generalization/validation/merge-closeout.md)
records unimplemented T177–T181 and two completed literal pilot arms; equal scores
do not establish target-format or adaptive-policy effects.

**Alternatives:** silently expand P3 (breaks historical interpretation); compare
C directly with the old refreshed arm (changes representation and allocation);
require another literal learner unconditionally (unnecessary if control audit passes).

## 2. Preserve the structural population and make admission decisive

**Decision:** inherit the accepted 128-task structural allocation and all original
exclusions. Keep the parameter endpoint DEFERRED. Complete native admission against
actual original inputs before any research GPU work. Freeze the new learning
comparison before inspecting new structural baseline scores.

**Rationale:** the [quota audit](../011-controlled-generalization/validation/prospective-quota-audit-20261104.md)
found zero surviving candidates in five parameter-family quotas, not a native
execution timeout. Its independent algebra examples establish a low-diversity
window mechanism, not exhaustive original support saturation. The structural
source-only feasibility check is not evidence that original exclusions will pass.

**Alternatives:** more timeout for a completed shortage (cannot change survivors);
more draws/reseeding or broader coefficients (new scientific design); report only
surviving families (biased denominator).

## 3. Parameterization with existing safety boundaries

**Decision:** reuse `generalization/targets.py` paired source/codec/grounding
checks and native fitting. Preserve declared sharing and the existing whole-row
literal fallback when control dependencies make parameterization unsupported.
This first experiment does not implement a new partial-control parameterizer.

**Rationale:** fitting cannot repair wrong structure. `ground` limits assignments
to eight parameters and exact source equivalence; `learning_target` already keeps
unsupported rows. Replacing every integer blindly would change control flow or
create an underdetermined fit. Audit eligibility/fallback coverage before training.

**Alternatives:** drop unsupported rows (changes data); free every control literal
(unsafe and unsupported); redesign the grammar/fitter now (confounds the comparison).

## 4. Fixed finite training pool, separate bounded feedback

**Decision:** B/C share the original 32-family, 32,768-row concrete static pool.
Create a separately committed train-only probe panel from those same mechanical
families; do not remove or replace gradient rows. Probe overlap rules differ from
final evaluation and must be disclosed: exact source/output duplicates with gradient
rows are rejected, shared visible windows may remain within training support.
Final structural selection exclusions remain fully unchanged. A failed probe quota
is a preparation failure, not permission to substitute held-out families.

**Rationale:** 011 static/refreshed changed instances under a fixed mixture. The
older `curriculum/symple_bandit.py` exposes useful aggregate/cost/exploration
patterns but selects different units and explicitly is not an EXP3 implementation.
`PreparedBatches` currently strips family metadata and supports only static/refresh;
family provenance and a new adaptive stream are real integration work.

**Alternatives:** online generation/support expansion (extra confounds and native
cost); reserve probes by removing old training rows (invalidates retained A);
use final structural outcomes as rewards (leakage); require training probes to be
unseen under every final-evaluation window rule (unnecessary and potentially
infeasible for low-diversity training families).

## 5. Bounded progress allocation plus scheduled review

**Decision:** use a small explicit progress/retention policy with an exploration
floor, concentration cap and reserved review opportunities. It is a new heuristic
policy, not a claimed reproduction of adversarial-bandit theory. Freeze the
[algorithm contract](contracts/curriculum.md) before research. Measure loss progress
on independent train-only examples; use sparse bounded semantic probes for retention,
not generation/fitting on every batch. Review examples receive actual updates.

**Rationale:** Graves et al., [Automated Curriculum Learning, §§3.1–3.3](https://proceedings.mlr.press/v70/graves17a/graves17a.pdf),
study learning-progress rewards and distinguish signals measured on trained versus
other samples. This motivates an independent probe signal; their tasks and results
do not prove a gain for this model or justify our constants. Schaul et al.,
[Prioritized Experience Replay, §§3.3–3.4](https://arxiv.org/html/1511.05952v4),
describe noise/diversity and sampling-bias problems with greedy prioritization.
That is RL evidence, not a direct SFT guarantee. Here reweighting is the intervention,
so report it rather than importance-correcting it away while claiming the same
curriculum. Neither paper establishes the proposed spaced-review schedule.

**Alternatives:** highest loss only (can favor impossible/noisy tasks); same-batch
loss reduction only (memorization bias); full per-batch solver reward (unbounded
practical overhead); large algorithm sweep (no initial causal baseline).

## 6. Whole-state transactions and sparse full snapshots

**Decision:** extend the installed learner's existing state and persistence
transaction, not a parallel learner. Include every adaptive/probe/review decision
and pending work in the same complete snapshot as weights/optimizer/RNG. Keep
hourly recovery saves, initial/final and 4/8/16/25M milestone pins; cheap progress
is not a resumable snapshot. A later journal can debit lost work but cannot advance
an older model's scheduler. Clean pause/resume is exact; discarded crash work is
separately charged and may reduce useful updates under the same cap.

**Rationale:** `generalization_persistence.py` already accounts spent-but-lost
tokens. Replaying newer adaptive events into an older model would violate both
scientific identity and exact recovery. Old per-update full saves dominated runtime.
The qualified no-Lt process-start policy remains the default; a newer ROCm image
requires its own evidence, not an assumed fix.

**Alternatives:** scheduler-only journal restoration (inconsistent state); reset
adaptive state on resume (different experiment); full tensor saves per update
(known overhead); silent backend upgrade (invalid qualification).

## 7. Consolidate validation, not authority

**Decision:** a temporary, source-bound 012 wrapper serially invokes canonical
operations. It performs full CPU/native admission, actual representative calibration,
small real HIP acceptance, compatible reuse, optional already-authorized research
and verified normal/recovery export in one operator session. It does not implement
training, fitting or a second acceptance engine.

**Rationale:** [reliability-review.md](reliability-review.md) maps the concrete 011
failures to preventive tests. Tiny model throughput alone cannot predict production
initialization/save/export cost. Operational limits need measured headroom through
all nested roles; scientific per-task budgets stay fixed. Existing completed stages
remain reusable only by verified dependency identity.

**Alternatives:** one manual command per case (too much back and forth); repeat all
successful work after each failure (wasteful); skip failures because prior source
passed (unsound); blanket infinite timeouts (no containment).

## Research boundary

One seed, 25M target tokens per new arm and the mixture/cadence constants are
planning defaults, not measured optima or authorization to launch. No expected
accuracy gain is a software gate. Wider support, refresh/adaptive interaction,
review-only ablation, replay/search teaching, extra seeds and final assessment are
separate future decisions. No unresolved design question blocks task generation;
missing workstation inputs and measured runtime are explicit execution gates.
