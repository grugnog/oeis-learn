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

## 5. Published family bandit with bounded per-example review

**Decision (revised after operator review):** select the Graves et al. EXP3.S
fixed-share kernel, beta=0, and a bounded per-example Leitner adaptation. Replace
the custom weighted-priority formula, 0.20 probability cap, family-level intervals
and family-probe-driven promotions. Full implementation details and defaults are
in [curriculum.md](contracts/curriculum.md). The scientific recipe still requires
an explicit pre-execution freeze; a design document is not a completed experiment.

**Rationale:** separate family allocation from example retention. Use one chosen
family's contiguous training block and its own matched pre/post probe for reward;
a periodic trend spanning other updates cannot provide the same credit assignment.
Target recall grades the particular reviewed anchor. Common family panels measure
transfer/forgetting independently. Prototype numerical/state behavior on CPU before
actual-device work. The combined wrapper is an experiment, not a theorem-backed
optimal training policy.

**Alternatives:** the legacy named scheduler (current implementation explicitly
is not EXP3); RbF retention fitting (additional model/calibration assumptions);
MIR (hypothetical-update scoring and state/compute cost); importing human-memory
parameters (no validation for this learner); a bandit over every training example
(much larger action space and sparse feedback); highest loss or absolute loss slope
(can reward noise, stuck tasks or repeated forgetting rather than useful learning).

### Primary-source assessment and limits

| Source inspected | What supports this design | What it does not establish |
| --- | --- | --- |
| [Graves et al. 2017](https://proceedings.mlr.press/v70/graves17a/graves17a.pdf), Eq. 1, §§2.3–3.3 and experiments | Published fixed-share selection/update; learning-progress feedback; separate-sample measurement and reward normalization | Our finite reused probes, eight-update blocks, review interleaving or new constants are not a reproduction of its experiments; no optimal final-accuracy guarantee |
| [Auer et al. 2002](https://homepages.math.uic.edu/~lreyzin/papers/auer02.pdf), Fig. 6/§8 | Original adversarial-bandit and switching-comparator foundations; confirms the need to identify the exact variant | Its share equation is not interchangeable with Graves Eq. 1; regret bounds are not guarantees for this state-changing training wrapper |
| [Amiri et al. 2017](https://aclanthology.org/D17-1255.pdf), Algorithms 1–2 and Table 2 | Neural-training Leitner baseline has per-example promotion/demotion; RbF adds fitted retention kernels | Headline speedups process fewer examples per epoch, with lower reported average accuracy than full-data training; not fixed-token superiority or validation of our target-recall proxy |
| [Aljundi et al. 2019](https://papers.neurips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval.pdf), §3.1 | MIR scores anticipated interference on stored examples using a hypothetical update | No reason to assume its extra computation/integration pays off here; it is not an interval scheduler |

Read methods/results, not just abstracts. The 2026 SRT preprint was also discovered,
but its full text was unavailable through the available reader; it supplies no
algorithm choice or performance claim here. No claim of an exhaustive literature
survey or universal best algorithm is made.

### Adaptations explicitly owned by this experiment

- Kernel variant and beta=0 are named exactly; epsilon=0.25 preserves the project's
  exploration preference rather than copying the paper's epsilon=0.05. Eta=0.001
  is a starting default, not tuned evidence. Numerical counterexamples distinguish
  share variants and the first update's alpha=1/2.
- Use signed pre/post loss on the same gradient-independent finite probe pair.
  Divide by charged gradient tokens, not noisy wall time, to preserve deterministic
  replay. Report actual wall and diagnostic costs separately. Keep bounded
  historical quantile scaling with explicit cold-start/tie behavior and RNG state.
- No importance update for deterministic review or censored feedback. Review has
  its own service clock; missing observations do not advance the valid kernel clock.
- Three eight-update family blocks followed by eight review updates amortize
  observations while avoiding cross-family reward contamination. The 25% review
  opportunity fraction is measured in updates, not tokens or elapsed time.
- Four anchors per family carry their own five-level queue. Pre-update exact target
  recall is a cheap proxy; separate semantic/family tests prevent a memorization
  result being called structural generalization. Actual incidental exposure resets
  spacing, but does not invent a successful recall.
- The 128-item queue needs a 512-update service bound under this cadence, not the
  superseded 128-update family bound. Initial acquisition and terminal truncation
  are explicit. CPU enumeration must validate all cycle offsets before handoff.

### Attribution and alternative arms

Default C–B measures the combined bandit/review package. Optional D is a parameterized
bandit-only arm, disabled unless frozen before outcomes. C–D measures allocating
25% of update opportunities to scheduled review instead of further ordinary bandit
practice; it is not a free extra replay benefit. D requires a complete additional
training/evaluation budget and identical common monitoring, with extra policy-required
reward observations separately charged, and can be included
in the same serial operator command. No fourth arm is silently required for
software acceptance or selected after results. RbF/MIR and review-only ablations
remain later choices.

## 6. Whole-state transactions and sparse full snapshots

**Decision:** extend the installed learner's existing state and persistence
transaction, not a parallel learner. Include bandit logweights/action and valid-update counters, scaler reservoir/RNG,
selection-time probability and pending pre/block/post observations, plus every review decision
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
accuracy gain is a software gate. The optional bandit-only arm requires its own predeclared scope. Wider support,
refresh/adaptive interaction, review-only ablation, replay/search teaching, extra
seeds and final assessment are
separate future decisions. No unresolved design question blocks task generation;
missing workstation inputs and measured runtime are explicit execution gates.
