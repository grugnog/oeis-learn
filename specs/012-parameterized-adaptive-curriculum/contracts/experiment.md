# Experiment contract — proposed `PAC-screen/v1`

## Frozen comparison

| Arm | Targets | Sampling | Initialization and data |
| --- | --- | --- | --- |
| A | Literal | Original fixed stream | Verified retained static control, or a predeclared fresh matched control |
| B | Parameterized, counted safe literal fallback | Same fixed underlying stream as matched A | Fresh random initialization; original finite static pool |
| C | Same target view as B | EXP3.S family blocks plus bounded per-example review | Identical initial weights, model/optimizer and concrete pool as B |
| D (optional, off by default) | Same target view as B | Same EXP3.S kernel/reward; ordinary bandit blocks throughout, no review-directed gradients | Same initial weights and pool as B/C; separately frozen additional arm |

B–A measures representation and its consequences at fixed token budget. C–B
measures allocation plus review together. If D is predeclared, D–B measures bandit
allocation and C–D measures the effect of substituting scheduled review for a
declared fraction of ordinary bandit practice; no review-only effect is inferred
from C–B. The old refreshed arm may be evaluated
as a separately labeled reference, not a matched adaptive control. Reference-only
A permits C–B but no causal representation claim. If retained A fails compatibility,
the campaign manifest must already choose `fresh-control` or `reference-only`;
the wrapper may not silently select a new scientific arm. Default handoff favors
reference-only unless a fresh control is explicitly frozen/authorized.

Compatibility receipt binds initial weight hashes and seed, original concrete
row IDs/order, tokenizer/numeric/architecture/optimizer configuration, optimizer
step/learning-rate schedule semantics, token cap, checkpoints, backend and all
interventions. Diagnostics may differ from old A only if they provably do not
mutate model/optimizer/RNG/order, do not censor updates under wall limits, and are
excluded from any claimed matched-time comparison. Missing historical curves/costs
remain unavailable. New B/C receive common probes and aligned bounded observer opportunities; B
records diagnostics without adapting its sampler. Reward panels are conditioned on
each arm's selection; they are not the common evaluation panel. Common diagnostics
and shadow opportunities are defined in the curriculum contract; D has additional
policy-required reward observations, whose costs are explicitly separate. An optional D
adds its complete training, observation, checkpoint, generation and export reserves
to the same serial campaign; it cannot be activated after outcomes are inspected.

## Defaults to freeze before any new outcome

- One paired initialization seed, default original `20261016` when its initializer
  can be exactly reconstructed/verified; otherwise a predeclared new seed and A
  reference-only or fresh-control path. No trained literal weights initialize B/C.
- Policy `pac-exp3s-leitner/v1`: named Graves fixed-share kernel, epsilon=0.25,
  eta=0.001, beta=0; eight-update family blocks; three bandit blocks then eight
  review updates for C; D has ordinary bandit blocks throughout. Review state is
  per anchor, with five queues and 512–8,192-update intervals. These are explicit
  initial defaults, not tuned or claimed optimal.
- 25,000,000 target-token allowance per new arm. Use canonical exact batch charging;
  do not truncate a batch to fake exact equality. Report useful, reserved, discarded
  and unspent tokens separately, alongside examples, updates and wall cost.
- Same float32 model/optimizer recipe as the retained pilot, subject to explicit
  compatibility verification. New model architecture or precision is out of scope.
- 32 training families, 32,768 original static concrete rows; B/C identical eligible
  and fallback sets. Train-only probe seed/counter manifest is frozen separately.
- Full initial/final and 4/8/16/25M milestone pins; recovery approximately hourly.
  Deduplicate coincident saves. Cheap progress at most 60 seconds apart.
- Existing exact integer profile, native fitting and at most eight free parameters;
  protect existing control-dependent rows using literal fallback. Report eligible
  fractions by family so diluted parameterization is visible.
- Calibration determines generous finite operational envelopes before research,
  with explicit nested startup/save/cleanup/export reserves. The six-hour legacy
  learner validation ceiling must not remain as a hidden limit in this new recipe.
  Do not modify old recipe envelopes or scientific per-task limits implicitly.

## Structural scope inherited from 011

Read the accepted [structural plan](../../011-controlled-generalization/validation/structural-comparison-plan.md)
and its frozen allocation. Preserve 2,288 structural draws, seed 20261104 with the
original structural offset, counter keys and original reservoir capacity 8,192
used in identity derivation. Reducing the executed scope must not change draw IDs.
Preserve all 16 original exclusion inputs, including parameter/assessment cells;
only the custodian sees protected truth. Prepare exactly 128 tasks across 27
families: 68 certified across 18 families, 26 ambiguous and 34 unresolved.

Preserve 20 visible terms, 100 scoring terms, eight attempts, 60-second matched
portfolios, seven controls and bank seeds 20261101–20261108. Parameter evaluation
is DEFERRED. No assessment execution. Native agreement and every post-exclusion
family quota must pass before new structural inference or learners launch.

T177 scoped preparation and T178 publication/case analysis become 012 work;
T179 tests/handoff and T180 actual hardware evidence remain separate completion
gates. T181 becomes the control receipt above. T174 remains failed/superseded in
011. Completing 012 must not retrospectively check those historical boxes.

## Report and decision rule

Primary: certified structural family-macro success at the frozen inference budget,
with task counts and paired discordances. Require at least 65/68 certified tasks
completed per compared arm. Report ambiguous/unresolved strata and execution
censoring separately. Also show raw proposal validity, fitted finite agreement,
family training-probe retention, exposure curves and preparation/diagnostic/review/
checkpoint/end-to-end costs. Curves from train-only probes are not repeated final
structural benchmarks; full structural inference is final-only by default.

Select at least 12 available case/arm records deterministically across shared and
arm-exclusive successes, continuation failures, no-fit, timeout/decode failure and
simple-control wins. Retain raw/parameterized/fitted source, assignment and reason
where export permissions permit; missing categories are explicit. Reuse existing
post-pilot analysis with a versioned structural-scope adapter; historical 256-task
guards must still reject mismatched inputs.

Single-seed results are exploratory. Report paired uncertainty as descriptive;
task resampling is not uncertainty across training seeds. Positive accuracy or
speedup is not an acceptance gate. No automatic winner, assessment exposure,
additional seed/arm, altered quota, support expansion or policy tuning after scores.
Checkpoint/diagnostic overhead and useful/discarded exposure must be reported for
each arm; the optional comparison is equal allowance, not equal useful updates or
guaranteed equal wall time.


### 2026-11-05 structural feasibility amendment

The operator authorized repair after actual post-exclusion quota failure. New runs
use the explicitly versioned 128-task/23-family population (68 certified/15 families)
in [the amendment](../validation/structural-amendment.md). It freezes revised quotas from conservative source capacity
before model outcomes, preserving all 2,288 draws and every original exclusion.
Historical 27/18-family scope stays readable and failed; training, numeric law,
inference budgets and the 65/68 threshold remain unchanged. No actual new native/HIP
acceptance or research success is claimed by this implementation.
