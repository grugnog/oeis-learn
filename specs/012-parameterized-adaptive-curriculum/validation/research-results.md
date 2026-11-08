# Completed R1 research review — 2026-11-06

T073 is complete for the supplied `pac-012-bcd-r1-002` report and subsequent
original-detail bundle. Software/feature qualification and R0 PASS retain their
original scope; scientific status remains INCOMPLETE, not a failure to perform
analysis. The earlier acceptance-only deferral is preserved below.

## Recorded results and interpretation

All 13 stages passed; the first five gates were reused. All B/C/D learners reached
their original 25M charged-token allowance without discarded exposure. Useful
tokens were B 24,998,929; C 24,997,760; D 24,998,016. Original A remains unmatched.

| Arm | Certified successes | Completed / 68 | Family-macro censoring bounds |
| --- | ---: | ---: | --- |
| B | 35 | 65 | 58.18–63.18% |
| C | 27 | 53 | 46.78–65.47% |
| D | 42 | 58 | 67.85–85.17% |

D–B has 14 D-only and five B-only successes among 57 jointly completed tasks.
Its family-macro difference bounds are +4.67 to +26.99 percentage points. These
are descriptive censoring bounds, not confidence intervals or training-seed
uncertainty. All three primary comparisons fail the predeclared per-arm 65/68
completion requirement. D–B changes allocation, blocking and within-family order;
it does not isolate EXP3.S or parameterization. C–D substitutes review opportunities.

D gains on three families (B/D successes 0/3, 0/4, 0/5) but loses on the product-
coefficient quadratic family (8/3). Nine of its fourteen wins over completed B
failures avoid B's unobserved exceptional branch at index 21 or 46. All five D
continuation mismatches also contain unobserved exceptional branches. The original
shortest-byte selector replaces a successful first fitted candidate with an
incorrect shorter candidate in four D and eleven C certified cases.

D correct/shuffled/absent/template/bank success counts are 42/41/49/55/46; their
completion counts are 58/64/63/68/64. Forty correct-input D successes also succeed
without model conditioning; the solver still sees the correct prefix. All three
D successes missed by the fixed templates are solved by the bank. The result
supports useful proposal coverage but does not establish a positive conditioning
advantage. B generates 1.85 distinct completed sources per eight-attempt certified
portfolio on average, D 5.85, C 6.69; diversity alone does not explain performance.

D's ten censored primary cases comprise three token-cap failures, two grammar dead
ends and five fitting timeout/deadline cases. No candidate was selected in these
ten; final scoring was not their failure point. All 68 certified raw controls in
each trained arm report invalid_source with unbound param.get in their first
proposal: these are not completed zero-accuracy outcomes.

Training-probe token losses finish at B .01060, C .10226 and D .21613 from identical
4.44605 initial loss. Latest rotating semantic panels contain B 151 / C 7 / D 40 successes
in 256 attempts each (227/231/239 completed); these span different last-eight-panel
update windows, not a simultaneous final test. C records 27 exact-recall successes
in 2,236 acquired-anchor grades. D visits all 32 training families; its five most
frequent families receive 20.96% of examples. No severe allocation collapse is shown.
Fresh-stage elapsed time totals 10.21 hours, of which B/C/D training stages total 8.04 hours;
reused gate timings are historical, not fresh elapsed work. Probe loss alone would
favor B despite D's stronger structural endpoint. No training/HPO change is adopted.

## Offline selector hypothesis audit

The new [audit contract and reproduction command](selector-audit.md) are separate
from the completed research policy. The [bounded summary](evidence/selector-audit-summary.json)
binds both supplied ZIP hashes, original report verification, policy and diagnostic
identity. The tool reproduces 384/384 original choices and observes all 3,851 original
eligible candidate sources, including sources omitted by the bounded archive.

| Arm | Certified choices changed | Known recoveries | Known regressions | Changed but unscored |
| --- | ---: | ---: | ---: | ---: |
| B | 23 | 0 | 0 | 20 |
| C | 20 | 10 | 0 | 10 |
| D | 27 | 0 | 0 | 27 |

All visible observations completed. B's other three changed choices have one
previously known success and two previously known failures. D's 27 changed exact
sources have no recorded complete continuation scores: no accuracy gain or lack
of regression is inferred. C's 10 recoveries are exact-source matches to original
successful selections/controls, not newly executed scores. The rule is unchanged
after this replay. These are exploratory findings; training-only probe validation
and a separately frozen fresh structural comparison remain unexecuted. No raw
private inputs, ZIPs, weights, prompt terms or native assignment payloads are
committed. The original 12-case report (A 10 / B 2 / C 0 / D 0) is supplemented here by the
all-task audit rather than treated as representative B/C/D case sampling.

## Historical acceptance-only record

### T073 at acceptance closeout — DEFERRED; scientific status UNMEASURED

The supplied acceptance plan explicitly records `research=false`, arms B/C,
seed 20261016, parameter DEFERRED and assessment NOT_EXECUTED. Its 25,000,000-token
allowance describes an optional future learner; it does not establish completed
research or authorize another campaign. No fresh A, bandit-only D, additional seed,
policy tuning or winner adoption was selected.

The implementation request closes Phase E against this uploaded evidence. It does
not silently rewrite the existing acceptance-only frozen plan into a multi-hour
research experiment. Existing software PASS/feature qualification is preserved,
and the subsequent actual HIP crash-debit summary now resolves T072.

## What is available

- Original-data/native admission for the amended 128-task/23-family population.
- 32,768 paired production targets and their representation counts/token totals.
- Production calibration: 64 updates, 122,464 charged tokens and six valid rewards.
- Reduced mechanical HIP acceptance: 16 updates, target/fallback exposure, graded
  review, clean pause/resume and a reloaded training-probe forward pass.
- Actual short-sample timing and conservative operational forecasts.

None is a final B/C research arm, a matched A comparison, a structural accuracy
result or a learning speedup. Acceptance/calibration weights cannot initialize
research; B/C require identical fresh random initialization.

## Exact missing evidence

No complete 25M-token B/C learner histories or final research checkpoints; no
retained structural baseline generation on this v2 roster; no per-arm structural
attempt/scoring exports establishing at least 65/68 certified completions; no
paired family-macro outcomes/discordances; no full production retention/exposure/
cost curves; no twelve available before/after case-arm records spanning the required
categories. Historical A is reference-only; matching evidence is not manufactured.
The public report's production exposure fields are null, not zero completed work.

The final loaded acceptance checkpoint has target recall false on both diagnostic
training probes. This is a real finite forward result after a tiny run, not a
conclusion about eventual structural performance. It supplies no passing/failing
research case population for deterministic case selection.

## Conditional next scope

T073 remains **unchecked / DEFERRED**, as tasks.md explicitly requires actual
completed research analysis before checking it. This deferral does not block
software acceptance and does not authorize a new campaign. Before optional
research: retain the verified T072 producer evidence, explicitly choose/freeze
the new research plan and any fresh A/D, preserve the exact v2 population and
original exclusions, and reserve complete serial learning/evaluation/export costs.

If such a run is later authorized and completed, use the existing scoped analysis
adapter for C–B allocation-plus-review attribution and separately disclosed costs.
Do not infer representation gain from an unmatched A, isolate review without D,
treat equal token allowances as equal examples/time, or turn one paired seed into
seed-level uncertainty. No extra HIP commands, training, baseline inference or
analysis of invented case rows were performed in this closeout.
