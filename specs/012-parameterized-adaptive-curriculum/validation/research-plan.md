# First B/C/D research campaign

Recorded 2026-11-05 against PR #63 implementation `e2279dda55ed88997c54da43ea4ae6314648d2ad`.
The operator selected **B/C/D and one paired seed** during planning. The operator also permits adding a short rehearsal
before expensive learners; its concrete diagnostic configuration is proposed below. No device work, research or new qualification
has been performed by this planning/implementation pass. R0 software is now implemented; see [the current handoff](r0-implementation.md).
Actual R0 HIP evidence remains pending. This document is a planning record, not
a sealed execution manifest or a claim of current-source hardware readiness.

The actual `pac-012-bcd-r0-002` attempt failed CPU qualification on six T084
ownership regressions before HIP qualification/R0. The implementation and bounded
failure-reporting repairs, read-only recovery command and fresh `-003` retry are
documented in [the current repair handoff](r0-implementation.md#current-repair-and-retry-2026-11-06).
The failed scope stays failed; a new source-bound qualification is required.

## Questions and comparisons

| Arm | Targets | Practice | Purpose |
| --- | --- | --- | --- |
| B | Parameterized, existing safe literal fallback | Original fixed stream | Fresh fixed-allocation baseline |
| C | Identical target view/pool | Three eight-update EXP3.S blocks, then eight review opportunities | Adaptive allocation plus review |
| D | Identical target view/pool | Eight-update EXP3.S blocks throughout | Adaptive allocation without scheduled review |
| A reference | Historical literal weights | Already trained; evaluate only | Context, explicitly unmatched |

Primary comparisons: C–B (combined package), D–B (adaptive allocation/ordering
package), C–D (substitute scheduled review for 25% of ordinary bandit update
opportunities). D–B also changes blocking and within-family order; it is not an
isolated test of the EXP3.S equation versus an otherwise identical stochastic
scheduler. C–D is not a free addition of review. Opportunity fractions are in
updates, not tokens or time. Preserve and disclose policy-required observation
cost differences. No fresh A, second seed or hyperparameter sweep is selected.
Historical A does not support a causal parameterized-versus-literal conclusion.

## Shared production configuration

| Setting | Planned value / authority |
| --- | --- |
| Scientific policy | `PAC-screen/v1`, curriculum `pac-exp3s-leitner/v1` |
| Initialization | Seed `20261016`; identical verified fresh random initial tensor content for B/C/D |
| Model, codec, optimizer, learning rate | Exact original pilot recipe resolved from verified private originals; same across arms; include actual values and hashes in the frozen manifest |
| Precision/backend | Existing float32, pinned ROCm images and qualified no-Lt startup environment |
| Batch size | 32, existing canonical exact target-token charging |
| Learner allowance | 25,000,000 charged target tokens per arm; 75M total; no batch clipping to fake equality |
| Concrete training data | Original finite static 32-family / 32,768-row corpus; identical eligible and fallback rows |
| Targets | Existing independently grounded parameterized views; whole-row safe fallback; existing fitter and max eight parameters |
| Probe panel | Eight independent train-only probes/family, 256 total, frozen finite draw/exclusion commitments |
| Review anchors | Four existing rows/family, 128 total; acquisition and incidental practice recorded |
| Bandit | Graves Eq. 1, epsilon .25, eta .001, beta 0; selection-time probability and valid-update sharing clock |
| Reward | Same two independent probes before/after one contiguous eight-update block; signed gain per charged token, frozen past-only quantile/reservoir scaler |
| C review | Earliest due anchor; own pre-update target recall; real charged update; queues 0..4, intervals 512/1024/2048/4096/8192 committed updates |
| Monitoring | Common loss panel initial/final/every 256 updates; rotating four-family semantic panel, one attempt/probe, five-second scientific portfolio |
| Persistence | Hourly full recovery plus initial/final and 4/8/16/25M pins; frequent cheap progress; coincident saves deduplicated |

The exact private model dimensions/learning rate must be printed in the handoff
from the verified recipe; they are not guessed from diagnostic fixtures. The
historical public preparation reported all 32,768 rows parameterization-supported
and zero fallback, with 2,064,384 charged parameterized corpus tokens. Verify this
again under the new closure. For B, 25M is approximately 12.11 corpus passes under
those counts; adaptive arms do not have a comparable uniform epoch count.

## Short diagnostic rehearsal

The operator offered a tiny run to reduce reliability risk. Accept that approach,
but keep the rehearsal separate from scientific evidence and initialization.

Use a **temporary diagnostic harness**, composing existing canonical/installed
handlers and the same verified inputs/images. Do not add another learner or
change the registered 25M production recipe. The temporary runner now exposes `--rehearsal` and `--stop-after-rehearsal`,
with a canonical installed `pac-rehearse` stage and an independently verified
diagnostic report. The exact current commands are in [the handoff](r0-implementation.md).

Add **R0**, a diagnostic rehearsal, before **R1**, the three-arm research campaign.
R0 is not another comparison arm and supplies no learning-effect claim.

Recommended R0 scope:

- Run 256 committed updates for each of B/C/D with the production model, batch 32,
  32-family pool, target view, eight-update blocks and 32-update C cycle. Use the
  installed stop-at-update boundary; preserve the declared research cap in config,
  but label these intentionally stopped diagnostic learners PARTIAL, not completed
  research. Never reuse their weights or adaptive state as research initialization.
- Require finite changed parameters, identical initial weights, correct B order,
  actual D kernel feedback, actual C review/acquisition or incidental exposure,
  expected channel/update/token counters, initial/final snapshots and final load.
- Use existing reduced-cadence acceptance separately for acquired-anchor grading,
  exact resume and crash-debit proof; 256 production updates are below the first
  512-update review interval, so the rehearsal must not claim production spaced
  retention from acquisition alone. Report actual graded production reviews; zero
  at this length is not automatically a failure. If deeper review diagnosis is
  needed, predeclare a C-only extension up to 2,048 updates; report whether due
  grades actually occur, since incidental exposure legitimately resets spacing.
- Exercise normal evidence export and a deliberate interruption/resume with an
  owned diagnostic output. Validate final research analysis/export using the
  producer-shaped local fixtures and offline reconciler before device handoff.
  A tiny learner smoke test does not prove live full structural inference. Avoid
  turning four complete 128-task evaluations into a supposedly tiny rehearsal.
- Bind the temporary helper and explicit stop schedule, give it a finite generous
  envelope and package its outcomes automatically. No mocked gradient, silent
  CPU fallback, altered published kernel or hidden shortening of startup limits.

The smallest useful extension is B/D installed smoke coverage beside existing
C acceptance/calibration. Prefer that and existing report fixtures over a new
miniature experiment/schema framework. If a complete diagnostic path cannot be
composed without production changes, expose that scope explicitly, add focused
local counterexamples first, and requalify the affected closure once.

Provisional work allowance: approximately **30–60 minutes for the three 256-update
learners and the extra C equality trace together**, plus image build, admission, qualification, existing acceptance,
checkpoint and packaging costs. This is an arithmetic planning range, not a measured
rehearsal ETA: 1,024 actual optimizer calls at the old worst sampled .898s/update are about15 minutes;
real loss/recall observation, model load/checkpoint/proof and export add overhead.
R0 explicitly omits timed semantic generation for deterministic state comparison;
production calibration and R1 retain it, and the R0 report marks it UNMEASURED. Recalibrate actual counts/costs and
retain a much larger finite wall ceiling. C extension, if chosen, adds real work.

Prefer one composed invocation when that is straightforward. The operator permits
an additional experiment/run, so **two manual invocations are acceptable** if they
keep R0 simpler: first current-source admission, qualification, calibration/acceptance
and R0 with verified export; then R1 after inspecting that diagnostic report. Reuse
compatible producer evidence by full identity and do not repeat completed qualification
merely because the plan now selects research. Freeze R1 scientific choices before
any new structural outcomes. Automatic continuation is appropriate only if the
combined plan already authorizes R1 and every gate passes. Use the implemented [R0/R1 handoff](r0-implementation.md);
the historical acceptance command does not itself cover R0.

## Final evaluation and interpretation

Freeze scientific choices before baseline inference. Use the authorized
`pac-structural-scope/v2`: 128 tasks / 23 families; primary 68 certified / 15 families,
26 ambiguous / 3 families, 34 unresolved / 5 families. Preserve all 2,288 original
draw IDs and sixteen original exclusion cells. Keep 20 visible / 100 scoring
terms, eight attempts, 60-second matched portfolios, seven controls and bank seeds
20261101–20261108. Parameter evaluation remains DEFERRED and assessment unexecuted.

Evaluate the retained A reference and final B/C/D checkpoints using the same
frozen structural budget. No checkpoint selection on structural outcomes and no
intermediate full structural evaluations; 4/8/16M pins support later analysis,
not outcome-driven restarts or tuning during this campaign.

Primary endpoint: certified structural family-macro finite success, paired by
same task. Require at least 65/68 completed certified tasks in every compared arm.
Show all seven controls, task discordances, completion/censoring, raw proposals
versus fitted agreement and at least twelve available deterministic case/arm
records across successes and failure types. Task-paired uncertainty is descriptive;
one initialization seed supplies no across-seed uncertainty.

Secondary endpoints: complete family probe loss and finite semantic retention
curves; anchor acquisition/grade/incidental exposure and queue usage; family/channel
probability, examples, updates and charged/useful/discarded tokens; actual update,
combined observation, checkpoint, stage and end-to-end costs. Keep unmeasured
components null and overlapping timing components explicit. Equal token allowance
is not equal useful data, steps or wall cost.

Read results as follows, without an automatic winner:

- D better than B with C similar/worse than D: evidence favors the allocation/ordering
  package in this seed; review substitution has not demonstrated added value.
- C better than D, with coherent family retention evidence: review substitution is
  promising in this seed and merits a separately frozen replication.
- Improved recall/loss without structural improvement: investigate memorization,
  transfer and decoder/fitter limitations rather than calling this generalization.
- All arms weak or similar: inspect passing/failing raw/parameterized/fitted cases,
  structural validity and completion first. Do not automatically spend another day
  training or infer that all curricula are equivalent.
- Incomplete/censored comparison: identify that limitation and fix operational
  validity before interpreting accuracy differences. Preserve original evidence.

Do not change policy, data, budgets or arms after seeing results. A follow-up
confirmation seed or longer learner is a new explicit pre-execution decision;
reusing the selection cohort adaptively must be disclosed. Hidden assessment
stays sealed until separately authorized.

## Runtime, resources and recovery

Use token allowance as the scientific stopping rule. The operator's 6–12-hour
range is a useful expected work budget, **not a hard twelve-hour cutoff**. Raising
wall limits does not add training tokens. Keep finite relaxed nested ceilings,
>=24h preparation, >=12h learner floors raised by measured forecasts, prior longer
allowances and >=1h finalization. The campaign sums every selected stage/arm and
startup/save/hash/copy/cleanup/export reserve; no inherited global ceiling.

Recomputing the current forecast code using the old published samples and its
26,940-update upper planning count gives:

| Arm | Update estimate | Diagnostic estimate | Combined work estimate | Learner ceiling from those samples |
| --- | --- | --- | --- | --- |
| B | 6.72h | 4.92h | 11.64h | 66.47h |
| C | 6.72h | 4.92h | 11.64h | 66.47h |
| D | 6.72h | 4.92h | 11.64h | 66.46h |

Approximately **35 hours of learner work** for the three arms is a conservative
historical-sample planning figure. It excludes preparation, qualification, rehearsal,
retained-reference/final structural generation/scoring and packaging. Allow an
uninterrupted multi-day window; do not promise a full-campaign ETA until those
stages are calibrated. The diagnostics estimate charges full five-second semantic
budgets despite two old attempts completing in about .35s. Actual learned-model
behavior and changed diagnostic retention/export may change costs; current-source
calibration takes precedence. These figures neither guarantee a bound nor promise
that spending the full ceiling improves learning.

The current initial disk reservation for B/C/D is **256 GiB free with R0**, then enforce
the measured peak-live graph and preserve originals, snapshots, proofs and recovery
copies. GPU roles are serial; only bounded deterministic CPU work can parallelize.
If interrupted, resume the smallest compatible unfinished dependency in a fresh
owned sibling root, debit discarded work and keep useful exposure visible. No
resetting clocks/caps or restoring newer policy journals onto old model weights.

## Handoff and remaining preparation

1. The temporary bounded B/D (and production-cadence C) rehearsal is implemented,
   composed with existing qualification/acceptance. One separate R0 invocation is
   acceptable; repeated case-by-case manual loops are not needed. Record exact support/limitations and source/helper identities.
2. Publish one fresh output, exact command, scientific plan identity and original
   input roots. Existing campaign selection is `--research --bandit-only --seed
   20261016`, with no `--fresh-control`; include `--rehearsal` for the implemented R0 stage and optionally
   `--stop-after-rehearsal` to inspect it before resuming R1. Do not launch from this planning document automatically.
3. Rebuild and qualify the current post-conformance source/image scope, run measured
   readiness gates and preserve longer computed deadlines. Historical -004/T072
   PASS remains historical evidence, not current-source qualification.
4. Run the frozen serial campaign and independently verify normal/recovery exports.
   Complete conditional T073 only from actual research results and review the cases
   and limitations before choosing any next experiment.
