# Paired pilot candidate audit — 2026-11-04

The equal headline scores are not evidence of a general inability to emit valid
programs. The retained records show different model outputs that frequently
converge to the same fitted program, a substantial prefix-identifiability problem,
structural mistakes that fitting cannot repair, and a smaller but real solver
completion problem. One bank-control provenance defect is confirmed separately.

This implements [the prior evaluation plan](post-pilot-analysis-plan.md), T166–T167.
The [census](post-pilot-census.json) contains exact counts, checkpoint identities,
family bindings and constructed ambiguity witnesses. The [25-case review](post-pilot-cases.md)
contains original, parameterized and fitted programs, assignments, recorded scores,
independent reference results, and incomplete-generation token tails. Categories
overlap: these are25 case/arm reviews, not25 independent randomly sampled tasks.

## Evidence and scope

Input: `fresh-p1-hourly-20261104T093555Z`, uploaded as `report(5).zip`, SHA-256
`b5c0ea49c89b62b819856054b2f4ae82672a8304a4351d18acb8a51f2f1dd494`.
All576 manifest members were independently byte/hash verified. Analysis uses
retained candidate/token/native-fit/scoring records and current qualified source,
not conclusions from the generated report summary. Original files and all scores
remain unchanged. Assessment stays sealed; actual100-term target values are not
in this public archive. Recorded full-horizon outcomes are not newly rescored.

The helper uses the existing independent parser, codec, interpreter, grounding and
bundle verifier. It runs under `python3 -S`, with no third-party packages, container,
GPU, native solver, training or corpus preparation. Its PASS means the audit
completed, **not** that the original experiment or all provenance was clean.
The207 explicitly reported bank metadata findings remain findings.

## Validity and where the outcomes diverge

| Check | Static | Refreshed |
| --- | ---: | ---: |
| Primary generated attempts | 2,048 | 2,048 |
| Completed generation; independently parsed and token/source matched | 2,042 | 2,047 |
| Primary fit attempts returning verified solutions | 1,375 | 1,386 |
| Primary fit attempts reporting scoped unsatisfiability | 509 | 475 |
| Primary fit attempts exhausting resources | 158 | 186 |
| Primary decoded-source structures, ignoring literal values | 127 | 109 |
| Selected primary programs matching visible20 | 192 | 192 |
| Recorded primary100-term successes | 128 | 128 |
| Recorded primary continuation failures | 64 | 64 |
| No selected program, completed unsuccessful portfolio | 48 | 43 |
| No selected program, unresolved portfolio | 16 | 21 |

Across all four generated modes,16,360 completed sources passed independent
parsing and retained-token/source checks;24 generations were incomplete. All
29,280 archived native assignment records reproduced their final AST through
independent grounding. All1,512 selected outputs across the seven controls
independently matched their20 archived visible values, including the bank outputs
whose original-source metadata needed an explicit diagnostic recovery.

The final full execution sweep completed20 indices for2,042 static and2,046
refreshed originals. One simple refreshed candidate hit this audit's100ms local
allowance before producing a value; a separately retained [one-second reference
retry](post-pilot-reference-retry.json) completed in1.1ms and correctly reported a
visible mismatch. The original timeout remains in the census. Thus all4,089
completed primary originals were executable in this bounded independent check;
the seven incomplete generations remained unexecuted. This retry is diagnostic
CPU execution, not a change to the original native fitting budget or score.

Missing public prompt values affect56 static and50 refreshed task records. Their
sources are still executed at indices0..19; the helper explicitly reports that
expected values are unavailable and leaves `visible_match` null. It does not
infer a mismatch or reconstruct actual held-out coefficients. This is distinct
from incomplete generation, which has no executable original source.

Only **one of256 first proposals is byte-identical between arms**;119 have the
same literal-erased structure. Of190 tasks with selections in both arms,156 end
with the same fitted source. Their checkpoint weight hashes differ and the
archived evaluation/bank identities bind each to the correct retained checkpoint;
the initializer hashes agree. Success sets overlap on127 tasks, with one
static-only and one refreshed-only success. This is strong evidence against an
accidental identical-output explanation, although it is not a new model execution.

The first-candidate rescue scores differ substantially:115 versus92. Eight attempts
bring both arms to128. Fitting discards much of the difference in predicted numbers,
and a portfolio can compensate for a worse first proposal. Similar final totals
therefore conceal differences already visible in the same run.

## The benchmark itself contains unobservable continuation parameters

The public measurement plan identifies each **selection** task's typed family.
I matched these hashes to the frozen seed20261016 sampler catalog, independently
checked each family identity, and inspected its branch guard and parameter use.
No assessment row or actual hidden target value was accessed.

Sixty-one tasks have family parameters referenced only in a branch never taken
at indices0..19 but taken later. A reference-executed witness strengthens this
syntactic observation: for12 families, two constructed assignments, using2 and3,
produce identical first20 values and different later values. These families cover
**55 of256 tasks**. The remaining six syntactically flagged tasks belong to a
family whose unseen parameter cancels; they are correctly excluded from the
witness group. This is why counting parameter occurrences alone would mislead.

| Post-hoc diagnostic group | Tasks | Static success / unresolved | Refreshed success / unresolved |
| --- | ---: | ---: | ---: |
| Family admits verified same-prefix/different-continuation witness | 55 | 0 / 2 | 0 / 6 |
| No witness in this bounded check | 201 | 128 / 14 | 128 / 15 |

The witness group accounts for46 of64 static continuation failures and45 of64
refreshed failures. A model or fitter cannot uniquely recover an independently
sampled coefficient that has no effect on its input. A family-level witness does
not establish an exact accuracy ceiling for the filtered finite dataset, nor prove
every other task identifiable. The201-row subgroup is a post-hoc diagnosis, not a
replacement primary benchmark or a new claim of generalization performance.

The source-only diversity screen excluded near-fixed difference windows across
numeric variants, but did not require **every continuation-relevant parameter**
to be observable in the first20 terms. Prefixes can vary strongly with one parameter
while other parameters remain completely hidden. The observed limitation is thus
consistent with the implemented screen, rather than an implementation bypass.

## What the passing and failing programs actually do

Task IDs below are unambiguous prefixes; full IDs and complete source are in the
case review. All mathematical forms here are readings of actual program ASTs.

| Cases | Task prefix | Before/after observation and implication |
| --- | --- | --- |
| 8,20 | `019150104976` | Both emit the correct branch at9 with different coefficients. The solver produces `(n-50)*523` before9 and `780*n²+n` afterward in both arms. Both pass100 terms. Structure learning and fitting work. |
| 9,21 | `0315a16b4ea4` | Both emit the branch at11. Fitting gives `-137*n` before11 and `960*n+309` afterward. Another shared success from different original numbers. |
| 1 | `35b36586178b` | Static's eighth proposal fits a multiplicative loop to initial value-983, offset40 and trailing`2*n`; it passes. Refreshed does not solve this task. Nontrivial loops can work. |
| 14 | `8ff8f4f17677` | Refreshed's third proposal has redundant constants; fitting yields `n*(n+560)` and passes. Static does not. Redundancy does not universally block the fitter. |
| 4,17 | `060e7ac816d1` | Both select `-315*n²` through20 and zero afterward. The actual target **family** branches at47. The generated guard is wrong and later coefficients were unconstrained by the prefix. Fitting cannot change its protected guard. |
| 5,18 | `0df5dcb23185` | Different affine proposals become constant475. The target family has a special branch at31, with an unseen coefficient. Prefix agreement cannot identify that exception. |
| 10 | `12c7183a74d7` | Static's first source is `2*n²*(p0+p1-p2)` after abstraction, so its quadratic coefficient is necessarily even. The successful template is `525441*n²`, an odd coefficient. Scoped unsatisfiability is mathematically appropriate for this candidate. More time or placeholders for its existing literals cannot repair the fixed`n+n`. |
| 22 | `12c7183a74d7` | Refreshed's first source is affine except at19; it cannot reproduce this quadratic prefix. The same simple template passes. This is a shape error, not malformed syntax. |
| 11,23 | `1cfa1b159a04` | Both fit`2305*n` except for a fabricated special case at46, where their chosen program outputs0. The successful template has no exceptional branch. This is an avoidable structural extrapolation error on a family without hidden branch parameters. |
| 3,15 | `01fe6576b431` | Static emits an affine expression; refreshed emits an affine visible branch. The target family is an additive loop plus`n²`, with a nonzero quadratic term. Both original first attempts are valid but scoped-unsatisfiable. |
| 12 | `102962cd40a7` | A valid multiplicative-loop proposal reaches SMT and times out. The trace grants roughly1.29 seconds of search within the native and reference reserves. This is censored evidence, not a proof that the target is unsatisfiable. |
| 24,25 | `003fade2ad71`, `0af9c5e8439b` | Valid factored-polynomial attempts time out in SMT. Other attempts can finish unsatisfiable in the same unresolved portfolio. The review deliberately selects the incomplete **fit**, not just a completed generation record. |
| 6,7,19 | `126e41c5b686`, `3d59c54c06ae`, `4bb1da2f48c7` | Static's tails repeatedly open nested conditionals and reach grammar dead ends; refreshed reaches1024 tokens without EOS. They are correctly excluded from fitting. Case7 still has a valid selection from a different attempt. |

The counterexample suite also directly checks that a program matching20 terms
but switching afterward is not treated as a100-term success by this analysis.

## Confirmed implementation defect and bounded fitting limitations

**Bank provenance defect.** Every selected bank envelope,98 static plus109
refreshed, stores `transformed_source` in `original_source` while retaining the
hash of the actual original bank proposal. The cause is visible in
`pipeline_validation/generalization_evaluation.py`: bank admissions reach `_seal`
without `_original_source`, unlike the ordinary portfolio path in
`fresh/bigint_diagnostics.py`. `_seal` falls back to transformed text.

The archived `proposal-bank.json` uniquely recovers each original by hash. The
audit records this recovery separately, never rewrites originals, and still
validates final source hashes, grounding, selected seals and visible execution.
The legacy bank v1 seal covers the selected program, not this outer original-text
field. This is a real metadata defect needing a small future repair, but it does
not explain the128/256 primary tie. No qualified production code was changed in
this analysis. A production fix must retain the original proposal before sealing,
check its hash, and test a fitted proposal whose transformed source differs.

**Search completion.** Static/refreshed have79/95 primary attempts with a verified
assignment but an incomplete search. Source review confirms that independent
admission can retain such an assignment while the search remains incomplete.
Incomplete shortlists are not cached as complete. A selected independently
verified program may still receive a complete100-term score; a portfolio without
a selection retains upstream censoring. The audit preserves these distinct states.

Most missing primary results are timeouts:14 static and20 refreshed task-level
outcomes; the other2/1 are decoding failures. All eight scheduled attempts were
retained per prompt/mode. Enlarging the outer evaluation timeout would not enlarge
the per-candidate native search cap. A future solver comparison needs a separately
recorded search-budget intervention, not simply more wrapper time.

The literal abstraction policy protects control-dependent constants, abstracts at
most eight eligible occurrences, and does not force equal literal occurrences to
share a parameter. These restrictions are intentional. The samples above show
both successful nonlinear fitting and legitimate unrepairable structural errors.
They do not support blaming a broad native arithmetic or parser failure.

## Recommended next decision

1. **Keep this P1 result unchanged and exploratory.** One seed, the below95%
   primary completion rates, and the observability problem preclude a strong
   static-versus-refreshed conclusion. The audit subgroup is not a benchmark edit.
2. **Before more training, define an observable evaluation slice prospectively.**
   Require the evaluated continuation's free parameters to influence the available
   observations, or explicitly separate ambiguous extrapolation tasks. Preserve
   the original benchmark and assign any changed benchmark a new identity. Use
   retained checkpoints where compatible; do not regenerate training merely to
   diagnose this issue. Observing scattered later indices would change the task
   from prefix continuation and needs its own protocol.
3. **Keep explicit-parameter P3 as a controlled candidate.** It may reduce digit
   prediction and redundant numerical search, but cannot reveal unseen coefficients,
   change a protected branch guard, or repair a missing structural factor. Compare
   supported parameter targets against literal targets with the same fitting
   policy; retain literal fallback for control-dependent cases. Report examples
   and updates as well as tokens because parameterized sequences are shorter.
4. **Treat solver completion separately.** If pursued, replay a small fixed set
   of existing sources with available visible prompts under deliberately recorded
   native-budget changes. Do not regenerate outputs or retrain, and do not merge
   those diagnostic scores into the original results. The existing single-case
   replay helper fixes its fitting allowance at4 seconds; raising only its outer
   `--seconds` does not test a longer native search.
5. Fix bank-original metadata before relying on bank envelopes for subsequent
   provenance-sensitive work. It is not necessary to repeat this full pilot to
   establish the defect or its primary-control isolation.

No new operator HIP/native command is needed to substantiate the findings above.
Missing private target values limit a new independent100-term rescore, not the
completed token, grounding, visible execution and source-level ambiguity analysis.

## Reproduction and validation

From the repository root, with the uploaded archive saved under its attachment name:

```bash
python3 -S specs/011-controlled-generalization/validation/post_pilot_analysis.py \
  --bundle 'report(5).zip' \
  --output "run-state/generalization/post-pilot-audit-$(date -u +%Y%m%dT%H%M%SZ)"
```

This is optional reproduction, not a requested hardware run. Output must be fresh;
the helper writes `audit.json`, `cases.json`, `cases.md`, `summary.json` and a verified
zip. The default bounded allowance is600 seconds; local full audits took roughly
21–26 seconds. This timing is CPU analysis evidence, not a HIP performance claim.

103 relevant tests passed, including23 new audit counterexamples and existing
reporting, reference, target, proposal and diagnostic checks. The final
missing-prompt execution extension's23 audit tests passed again. Failure tests
cover missing/corrupt inputs, explicit failure archives, overwrite rejection,
source/token/assignment/provenance disagreement, conflicting prompts, incomplete
outcomes, cancelled parameters and unavailable witness execution. Production
`src`, `crates`, `scripts`, dependency locks, Dockerfile and tests remain identical
to qualified `bfd04a203d580c70c9aee389c3c02241865267eb`. No HIP was run and no CI wait
was performed.
