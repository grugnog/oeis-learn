# Completed follow-up: evaluation policy, checkpoint and training-screen results

The supplied `matched.zip`, `matched-drain.zip` and `study-20261016.zip` all have selected-scope PASS, and every recorded stage passed. Their 663 manifested payloads verify against byte counts and SHA-256 hashes; all three manifests have no omissions. Each invocation passed 819 CPU/native tests and 26 HIP tests. All nine generation/scoring cases used the same 296-row cohort, work identity and source identity, `sha256:98f8622a75b45aa4a2d9f6d8bec47a38e8852d719d1c7906480b7a5c4daa13ad`.

This is now usable target execution evidence for the repaired paths. It does not establish generalization across new seeds or a broader corpus. Every case completed all 1,184 portfolio cost clocks; scientific completion remains arm- and split-specific. No case completed every scientific measurement. Hidden final-cohort access remains sealed.

## Main conclusions

1. Native draining substantially reduces fitting censoring. Every one of the 28,387 attempts decoded successfully in both policy runs has exactly the same generated source hash. The large yield difference is downstream of those shared neural proposals.
2. The final long-run checkpoint is worse on development continuation than the earlier checkpoints, while it is better on raw training-panel continuation. This supports the earlier overfitting diagnosis under the repaired evaluation.
3. The retained selected checkpoint and 3-hour checkpoint are close on row-weighted yield. The 3-hour checkpoint is better when each of the 18 structural families receives equal weight. Checkpoint ranking depends on the declared objective.
4. The repaired template baseline is useful, and constant fitting contributes much more than raw generation alone. Input-absent generation plus fitting is also strong, so fitted yield must be interpreted alongside controls.
5. The fresh screens finished near their 25M-token caps. At this seed, family balancing is worse than uniform sampling, and cosine decay provides no clear benefit. A longer unchanged run is not the leading next experiment.

## Native draining makes the comparison informative

`S/C` below means verified full-horizon successes / resolved outcomes. Each denominator is 128. Unresolved outcomes remain in the denominator, rather than being dropped.

| Checkpoint | Policy | Parameter S/C | Structural S/C |
| --- | --- | ---: | ---: |
| 3-hour / 2 | Matched overlap | 28 / 39 | 46 / 49 |
| 3-hour / 2 | Matched drain | 74 / 123 | 118 / 127 |
| Final / 5 | Matched overlap | 30 / 44 | 40 / 52 |
| Final / 5 | Matched drain | 67 / 122 | 83 / 128 |
| Selected / 6 | Matched overlap | 27 / 35 | 45 / 50 |
| Selected / 6 | Matched drain | 76 / 126 | 119 / 127 |

The drain policy reaches the specified 95% completion floor for the primary equal-budget arm on both development splits at all three saved checkpoints. The overlap policy does not. Both policies are fully cost-accounted: a terminal incomplete attempt is still a terminal cost event. A successful invocation receipt therefore cannot stand in for scientific coverage.

The two complete invocations took 2.47 and 3.63 hours, including qualification. Draining costs about 1.47 times as much wall time in this comparison. The policy bundles smaller prompt groups and native-first scheduling. These data support using that combined policy for this workload; they do not isolate the contribution of each scheduling choice. Generation completion differs on 29 of the 28,416 possible paired attempts, while all 28,387 jointly decoded sources match exactly.

## Earlier checkpoint performance survives completion bounds

These are full-denominator success percentages under matched drain. Intervals assign each unresolved outcome to failure or success. They are finite-cohort identification bounds, not statistical confidence intervals.

| Checkpoint | Parameter yield bounds | Structural yield bounds | Structural family-macro bounds |
| --- | ---: | ---: | ---: |
| 3-hour / 2 | 57.81–61.72% | 92.19–92.97% | 91.83–92.39% |
| Final / 5 | 52.34–57.03% | 64.84–64.84% | 65.35–65.35% |
| Selected / 6 | 59.38–60.94% | 92.97–93.75% | 87.43–90.21% |

The earlier checkpoints' development bounds exceed the final checkpoint's bounds on this cohort, most strongly on structural development. Comparing 3-hour to final on 127 jointly resolved structural rows gives 38 wins and 2 losses; selected versus final gives 37 wins and 1 loss. These paired counts describe this fixed cohort and single execution, not independent-seed evidence.

Selected has one more observed structural success than 3-hour, while 3-hour has higher family-macro yield even after allowing for missing outcomes. The family sizes vary from 1 to 16 rows. For example, 3-hour solves the single row in family `0b3a5d4e9c71…`, whereas selected does not; the two-row family `11060c7765c1…` is 2/2 for 3-hour and 1 success with 1 unresolved for selected. Larger families can outweigh these differences in the row-weighted score. Full family identities and counts are in `010-followup-004/families.csv`.

Raw continuation on the 32-row training panel improves from 8/32 at 3-hour to 27/32 at final; selected is 9/32. Raw structural continuation goes from 31/128 to 23/128, with selected at 22/128. High canonical training accuracy and strong training-panel behavior do not select the best development checkpoint here.

## What the model and fitter contribute

For the 3-hour checkpoint under drain:

| Arm | Parameter successes / 128 | Structural successes / 128 |
| --- | ---: | ---: |
| Raw greedy | 8 | 31 |
| Greedy proposal plus fitting / rescue | 54 | 81 |
| Correct-input eight-attempt portfolio | 74 | 118 |
| Shuffled-input eight-attempt portfolio | 14 | 43 |
| Input-absent eight-attempt portfolio | 42 | 107 |
| Fixed eight-template portfolio | 32 | 51 |

Raw measurement is complete on both development splits for this checkpoint. Rescue completes 122/128 parameter and 125/128 structural measurements. The large raw-to-rescue gain shows why weak literal-target exactness should not be equated with unusable structure proposals. Additional attempts improve yield further.

The template scores are identical across all nine cases: 32/128 parameter, 51/128 structural, and one confirmed OEIS continuation success. Template development measurements are complete; the eight-row OEIS cohort has only five resolved template outcomes. All neural arms have zero observed OEIS continuation successes, with varying completion outside raw. This small exposed cohort cannot support a broad OEIS-generalization claim.

Shuffling hurts, so the input affects useful proposals. Nevertheless, input-absent generation plus fitting solves 107 structural rows at 3-hour and 104 at selected, compared with 118 and 119 for correct input. A strong learned program prior and fitting/search coverage can therefore account for much of the headline structural yield. These controls do not establish encoder collapse or identify a unique causal mechanism.

Completion is asymmetric. At 3-hour, the input-absent structural arm resolves 121/128 rows, but the parameter arm resolves only 94/128. For fresh uniform training, correct-input structural yield bounds (85.16–85.94%) overlap input-absent bounds (70.31–86.72%); a strong positive conditioning effect cannot be concluded for that model from total yield alone. Completed-pair comparisons are exported separately and must not hide these unresolved rows.

## Fresh 25M-token training screens

All three arms start with identical observed validation metrics, use seed 20261016, and finish in about 2.34–2.49 learner hours. Uniform and decay share every logged token/update boundary. Total study time, including qualification and three saved-weight evaluations, is 11.02 hours.

| Arm | Tokens processed | Updates | Selected checkpoint tokens | Best pooled development loss |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 24,999,970 | 25,764 | 17,885,681 | 0.29821 |
| Family balanced | 24,999,751 | 27,068 | 3,975,250 | 0.34732 |
| Cosine decay | 24,999,970 | 25,764 | 17,885,681 | 0.29877 |

Saved-weight evaluation uses each arm's best pooled-development-loss checkpoint, not its latest weights. Thus the family-balanced evaluation is of the checkpoint at approximately 4M tokens, despite a completed 25M-token training screen. The selected uniform and decay weights are from approximately 17.9M tokens. Full training trajectories are in `010-followup-004/training.csv`.

| Arm | Parameter successes / complete | Parameter yield bounds | Structural successes / complete | Structural yield bounds | Structural macro lower bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 68 / 125 | 53.13–55.47% | 109 / 127 | 85.16–85.94% | 75.84% |
| Family balanced | 50 / 118 | 39.06–46.88% | 84 / 123 | 65.63–69.53% | 57.80% |
| Cosine decay | 69 / 122 | 53.91–58.59% | 100 / 127 | 78.13–78.91% | 64.55% |

Uniform exceeds family balancing on both splits even after assigning every unresolved family-balanced outcome to success. Uniform exceeds decay on structural development, including its bounds, while their parameter-development intervals overlap. Uniform versus decay has 16 wins / 7 losses on 126 jointly resolved structural rows; versus family balancing it has 33 wins / 11 losses on 122 rows. These are exploratory one-seed comparisons.

Family balancing's pooled development loss rises from 0.34732 at its selected checkpoint to 0.54111 at the end. Uniform and decay also finish with worse pooled loss than their selected checkpoints (0.31273 and 0.32904). These trajectories provide no current reason to promote the implemented balancing or decay schedules. They do not rule out different sampling weights, schedules, seeds or selection criteria.

The historical saved checkpoints and fresh screens have different training histories and selected-token exposure. Their semantic numbers are useful reference points, not a controlled estimate of a training-policy effect between the old and new runs.

## Optional next steps

1. **Use the repaired evidence now.** Keep matched drain as the evaluation baseline and retain both the 3-hour and selected checkpoints. Declare row-weighted, family-macro and raw/assisted objectives before a future checkpoint decision. Keep the final checkpoint as an overfitting control. No further run is needed to repair this completed execution.
2. **Extract more before spending learner time.** Evaluate retained latest versus selected weights from the fresh screens, using identical prompt/attempt seeds and explicit weight hashes. Add per-family raw/rescue/portfolio summaries and a paired report of correct versus absent inputs. Existing public archives contain outcomes, not the private weight bytes needed for another generation pass.
3. **A limited pilot, if desired.** Add an arm selector so two additional uniform 25M-token seeds can run without repeating the weaker variants. Replicate primary equal-budget outcomes under drain, and instrument native queue/service/cleanup costs for the still-incomplete control arms. Keep the existing deadlines and unresolved accounting. Select a new sampling or LR policy only after seed confirmation; this screen does not nominate one.
4. **Scope the larger data change.** Build a new versioned corpus with structural development coverage for loops, later stages and larger compositions, plus declared family/size quotas. The present structural cohort still contains only stage A and no loops. Preserve this cohort for historical comparison; report the new challenge cohort separately. Data refresh and broader holdouts are a more informative next direction than extending this fixed corpus unchanged.
5. **Scope target and selection changes separately.** Compare explicit parameterized targets against literal targets with the repaired fitter. Compare pooled canonical-loss selection against a prospectively defined family-macro semantic objective, using a separate development budget. Those changes alter what is learned or selected and should be isolated rather than combined with corpus expansion in one experiment.

Two focused research prompts now have higher value:

- Find primary academic research on conditional neural program proposals combined with symbolic parameter fitting. Identify experiments that separate conditional information from a learned program prior and search coverage, using matched raw/rescue/portfolio, shuffled and absent-input controls. Propose diagnostics for the observed combination of weak raw exactness and strong assisted structural yield; do not infer an encoder failure from similarity or ablation scores alone.
- Find primary research on checkpoint selection and generalization in program synthesis under family imbalance and teacher-forcing/semantic-objective mismatch. Compare family-macro behavioral selection, parameterized targets, canonicalization and dynamic data generation. Require controlled ablations, explicit search budgets and structural holdouts that include loops; distinguish findings that transfer to this DSL from those that depend on a different execution language.

## Reproduction and limits

Run the read-only extractor against the three attached archive filenames:

```bash
python3 specs/010-pilot-generalization/validation/analyze_followup.py \
  --input-dir /path/to/archives \
  --output-dir /path/to/analysis
```

It validates archive inventories and hashes, PASS receipts, source/work and cohort agreement, full denominators, outcome invariants, report aggregates, cost-clock completion and paired initialization observations. It exports `audit.json`, `groups.csv`, `families.csv`, `paired.csv` and `training.csv`. It reads public generation/outcome evidence and never opens model weights or hidden truth. Private weights are excluded from these public bundles; their bytes cannot be independently rehashed here.

The full runtime source snapshot is unchanged by this analysis commit and still matches the target's source identity. No new training, generation, native or HIP measurement was performed locally. No runtime repair or CI wait was needed. Finite 100-term verification (20 visible terms, 80 withheld) is not an unbounded proof, independent-seed confirmation or coverage of the sealed final cohort.
