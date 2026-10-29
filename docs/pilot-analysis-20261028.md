*Evidence review - 28 October 2026*

# What the pilot learned, and what to do next

**Recommendation:** repair the semantic evaluation path and compare retained checkpoints before spending another full day on training. Then run a short, controlled data-distribution experiment. The current evidence supports strong training-set memorization, meaningful input conditioning, and weak canonical-target generalization. It also exposes an unusable template baseline and substantial evaluation censoring.

| Finding | Why it changes the decision |
| --- | --- |
| 92.8% exact training targets at the final snapshot | The learner can fit this corpus; a basic inability to learn tokens is not the main problem. |
| 0/128 parameter and 1/128 structural exact targets | Held-out canonical reconstruction remains weak. This is distinct from semantic success after fitting. |
| All 14,208 template attempts were unsupported | A reproduced whitespace-sensitive parameter-detection bug invalidates the baseline comparison. |
| 12 of 27 exact training-panel targets had unresolved raw scores | The semantic pipeline hides demonstrated target competence behind service deadlines. |
| Structural development: 128 rows, only 18 stage-A families | Current results do not test broad unseen loop/branch generalization. |

This is a review of one seed, one 104.5M-parameter model, and one frozen generic corpus. It does not establish a capacity limit, disprove delayed generalization, or demonstrate OEIS transfer. The eight exposed OEIS prompts are too small to stand in for the project objective.

### What was independently checked

The three ZIP inventories contain 790 payload files. Every byte count and SHA-256 matches; no payload is omitted or unlisted. The run decision matches the admission decision, its referenced authority/cost/qualification/compatibility files match, and the diagnostic import binds to the supplied run and admission manifests. All six diagnostic snapshots report 33,024 completed rows, with unchanged model digests during probing and no optimizer loaded. The exported rows independently match those counts. Private weight/data verification is a recorded producer observation; those private bytes are absent from the public archives and were not re-hashed here.

I recalculated the principal measures from exported rows, joined teacher-forcing and semantic evidence, reviewed source files at the handoff commit, and reproduced the template-policy defect using the actual codec. Twenty-eight source files match the diagnostic dependency hashes. No training, new model generation, hidden-truth evaluation, or repository change was performed.

## Learning is real; generalization stalls

| Scheduled h | Corpus passes | Train loss | Parameter loss | Structural loss |
| --- | --- | --- | --- | --- |
| 0 | 0.00 | 4.5687 | 4.5732 | 4.6292 |
| 1.5 | 6.89 | 0.1740 | 0.2977 | 0.3880 |
| 3 | 13.31 | 0.1400 | 0.2717 | 0.3009 |
| 6 | 27.55 | 0.0882 | 0.2891 | 0.3738 |
| 12 | 56.22 | 0.0146 | 0.5560 | 0.8192 |
| 24 | 114.11 | 0.0040 | 0.6719 | 0.9360 |

At 24 hours, training token accuracy is **99.871%**, with 30,402/32,768 exact canonical targets. Parameter-development accuracy is 90.542%, with zero exact targets; structural-development accuracy is 88.318%, with one exact target. A few wrong tokens are enough to prevent exact reconstruction, and alternative correct programs need not match the canonical target.

The later loss increase is widespread. From snapshot 2 to 5, per-example loss rises on **119/128 parameter** and **122/128 structural** rows, while it falls on 32,722/32,768 training rows. The ten largest contributors explain only 20.2% and 19.4% of the respective net held-out loss increase. This is not a handful of outliers.

Held-out entropy falls sharply: parameter 0.246 to 0.054 nats/token and structural 0.222 to 0.048 between snapshots 2 and 5. Accuracy barely changes, so worsening cross-entropy reflects increasingly confident mistakes. Formal calibration error cannot be recovered from the exported scalar summaries.

The full validation journal contains 234 observations (702 split rows). The lowest pooled development loss occurs at update 30,720 / 29,809,170 tokens / exactly 15 corpus passes. Parameter loss alone reaches its minimum later, at update 38,912. Scheduled hours, actual learner-role hours (23.80 total), training-component hours (21.52), and inclusive wall hours (26.06) are different clocks.

## The corpus tests a narrow distribution

| Split | Rows / families | A / B / C / D rows | Maximum visible-value bit length |
| --- | --- | --- | --- |
| Training | 32,768 / 2,146 | 1,643 / 18,248 / 10,516 / 2,361 | 23,760 at 0-64; 9,008 at 65-256 |
| Parameter dev | 128 / 128 | 3 / 68 / 0 / 57 | 128 at 0-64 |
| Structural dev | 128 / 18 | 128 / 0 / 0 / 0 | 128 at 0-64 |

Neither development set contains loops; structural development contains only stage-A expressions. Its mean target length is 29.5 tokens, versus 61.5 in parameter development and 60.6 in training. Higher structural semantic yield therefore does not imply that unseen structure is easier than parameter generalization.

Two loop families supply **8,794 training rows (26.8%)**; the five largest families supply 30.9%. All targets have distinct exported identities, but example count substantially overstates structural diversity. The training sampler weights rows, not families. The selector explicitly sorts structural candidates by non-affinity, stage, and ID before filling the quota: [pilot_selection.py](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/pipeline_validation/pilot_selection.py#L116). These are distribution facts; their causal contribution needs an ablation.

### The input is being used

| Final matched panel | Correct loss | Shuffled loss | Absent loss |
| --- | --- | --- | --- |
| Training panel (32) | 0.0048 | 1.5424 | 1.3031 |
| Parameter dev (128) | 0.6719 | 1.9106 | 1.8109 |
| Structural dev (128) | 0.9360 | 2.6689 | 2.6097 |

These comparisons use identical targets and paired examples, so they are cleaner conditioning tests than the original unequal-attempt generation arms. Shuffling also reduces final token accuracy by 7.87 percentage points on parameter development and 9.67 points on structural development. Conditioning helps even where exact target recovery is poor.

The final matched panel has 521 integer-category errors and 669 control/operator-category errors. Those categories do not include their token denominators, and the non-integer category is broad. The evidence does not support treating every remaining error as a constant-binding problem.

The pooled-memory cosine falls from 0.637 at initialization to 0.403 at the final snapshot. Memory RMS falls from 1.00 to 0.480, while raw feature variance falls from 0.326 to 0.090. Scale changes complicate variance interpretation; neither these scalars nor the conditioning tests support a claim of complete representation collapse. Spectra were not requested.

## Semantic results are useful but incomplete

| Split / snapshot | Raw success | 8-attempt success | 8-attempt unresolved |
| --- | --- | --- | --- |
| Training panel / 3 h | 4/32 | 18/32 | 11/32 |
| Training panel / 24 h | 15/32 | 27/32 | 3/32 |
| Parameter / 3 h | 4/128 | 45/128 | 69/128 |
| Parameter / 24 h | 6/128 | 41/128 | 57/128 |
| Structural / 3 h | 14/128 | 75/128 | 48/128 |
| Structural / 24 h | 9/128 | 62/128 | 51/128 |

The equal_budget arm combines generation, visible-only fitting and selection, followed by independent 100-term scoring. It is not pure neural decoding. The raw arm uses the first greedy candidate without parameter rescue. Report full-denominator verified yield and measurement completion together; conditional accuracy among completed rows is heavily selected by execution cost.

Across both generic development sets, equal-budget verified yield is 120/256 at 3 hours and 103/256 at 24 hours. With unresolved rows left unknown, the corresponding descriptive bounds are **46.9-92.6%** and **40.2-82.4%**. These overlap widely and are not confidence intervals. The completed-pair subset also loses more successes than it gains, but it cannot remove censoring bias.

Across all snapshots and arms, 4,191/10,656 outcomes are unresolved under the recorded contract, including 1,480 at initialization. Neural generation itself is usually complete after learning: final correct-mode generation completes 2,365/2,368 attempts. Only 402 attempt records reach COMMITTED; 1,966 are INCOMPLETE. A successful portfolio can still contain incomplete attempts.

No exposed OEIS prompt has a verified success. At the final snapshot all eight raw outcomes are measured failures; only four equal-budget OEIS outcomes are complete. Original correct/template portfolios get eight attempts, whereas shuffled/absent controls get one, preventing a clean generation-level conditioning effect estimate.

## Two concrete evaluation repairs come first

### A. Template parameter detection is whitespace-sensitive

The frozen template bank contains explicit parameter nodes. [_template_candidate](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/fresh/bigint_diagnostics.py#L146) round-trips each template through the codec. [decode_body](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/decoder/bigint_codec.py#L117) joins tokens with spaces, producing `( param.get p0 )`. The fitter dispatch checks for `(param.get ` and therefore selects `eligible-literals/v1` instead of `none`.

The native abstraction code rejects a program that already has parameters with `unsupported:mixed_abstraction`: [parameters.rs](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/crates/oeis_synthesis_core/src/parameters.rs#L7). I reproduced the wrong dispatch for all eight actual templates. The exported event summaries retain only the top-level unsupported outcome, so the precise native reason is inferred from this verified source path rather than read from each event.

**Small repair:** detect parameter nodes through the parser or codec tokens, with a single shared helper; avoid raw formatting-sensitive substring checks. Add a real codec-roundtrip regression and a native end-to-end affine-template fixture. Re-evaluate the unchanged eight-template bank before expanding it. The old zero baseline is not evidence that templates are ineffective. The same dispatch can affect any generated explicit-parameter source.

### B. Native service clocks are largely spent before useful work

For final correct-mode incomplete events with all relevant clock fields (1,893 events), median queue delay is **3.178 seconds**, median service span 0.00033 seconds, and median delay until the owner observes completion 1.573 seconds. For 402 committed events, queue delay is 0.382 seconds and service span 0.283 seconds. These are conditional descriptive summaries, not whole-run wall-time components.

The native deadline is set at admission to roughly four seconds, before the request is queued. [SynthesisClient.fit](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/synthesis/client.py#L275) then reserves up to two seconds for later admission, leaving little or no native-search allowance after a long queue wait. This explains why increasing the solver timeout alone is not a well-targeted first fix.

**Small experiment:** replay a frozen panel with existing weights, comparing smaller generation batches / bounded native queue depth against the current scheduler. Record admission, dispatch, search budget, execution, verification and owner-observation times. Keep the overall portfolio clock explicit; a new queue/service allocation is a new evaluation policy, not a retroactive repair of the old scores.

A particularly revealing join: 27/32 final training-panel targets are teacher-forced exact; 15 have successful raw scores and the other 12 have unresolved raw service deadlines. The remaining five non-exact targets account for all three measured raw failures plus two further unresolved rows. This rules out interpreting the 15/32 raw yield as the model's demonstrated decoding ceiling.

## What is established, and what is still a hypothesis

| Explanation | Assessment | Discriminating follow-up |
| --- | --- | --- |
| Finite-corpus memorization and overconfidence | Strongly supported as an observed pattern; not a unique mechanism. | Fresh family-balanced / refreshed-data SFT at a matched token budget. |
| Model ignores its encoder | Contradicted as a complete explanation by paired input interventions. | Matched free-running controls; training-time ablations if needed. |
| All errors are numeric literals | Not supported; both integer and non-integer errors are present. | Token-role denominators, first divergence, AST structure and literal metrics. |
| Learning failed because programs cannot decode | Not the dominant late bottleneck: 99.87% of final correct attempts terminate. | Separate parsing, visible execution, fitting and continuation scoring. |
| Representation collapse or numerical failure | No nonfinite failure or identical pooled representation is established. | Normalized spectra/geometry and branch interventions only if targeted evidence warrants them. |
| Unchanged longer training will eventually generalize | Possible but unproven; this run gives no reliable forecast. | A predeclared delayed-generalization test with bounded compute and comparison arms. |

### Model and optimization observations

All learned floating state tensors change relative to initialization; fixed positional buffers remain unchanged. Decoder logit RMS on the activation panel rises from 0.578 to 9.534. Larger logits are consistent with increasing confidence, but neither norms nor parameter deltas establish the cause of generalization failure.

The pilot uses dropout 0, a fixed 1e-4 learning rate, AdamW, and gradient clipping at 1.0. Its optimizer objective averages per-program token means, whereas headline losses are token-weighted. This is a weighting difference, not necessarily an error; report both before changing it. The analyzer exports both. See [supervised_step](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/rl/sft_trainer.py#L115) and [defaults](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/fresh/bigint_config.py#L14).

The public bundles cannot reconstruct logits, gradients, full activations, hidden values or weights. They can support row joins, family summaries, transitions, time-budget analysis and source-level defects. New forward passes or corrected semantic scoring require the retained workstation state and a separate bounded invocation.

The historical SYMPLE/GRPO system is a different learning regime. Its earlier limited success is not a controlled comparison with this pure supervised run. In particular, current deterioration cannot be attributed to GRPO advantage collapse: GRPO is not the training objective here.

## A short sequence of optional next steps

| Priority | Work | Decision it enables |
| --- | --- | --- |
| 1. Saved-model evaluation repair | Fix parameter dispatch; qualify template fixtures; isolate raw scoring from fitting clocks; compare queue/concurrency policies on a frozen panel. | Whether current weights already outperform functioning baselines under usable measurement coverage. |
| 2. Probe the retained best checkpoint | Resolve selected.json to its full private checkpoint identity, then compare it with snapshots 2 and 5. | Whether the best development-loss checkpoint is a better synthesis starting point than the final snapshot. |
| 3. One-factor short training screen | Family-balanced sampling versus row-uniform SFT; keep model, seed, token budget and evaluation policy matched. | Whether distribution concentration is a major contributor. |
| 4. A second small screen if justified | Fresh generic data or a regularization/LR-schedule change, each isolated from the first comparison. | Whether more diversity or reduced confidence improves semantic generalization. |
| 5. Longer training | Only for a promising, measured recipe or an explicitly bounded delayed-generalization question. | Whether gains persist beyond the early learning window. |

The retained selected pointer is `state-2162a0ecb868411c8e1848b1706f9a4a`, hash beginning `5016f434f0e1`. The validation history and checkpoint-selection logic indicate the pooled optimum at 15 passes; verify the private checkpoint metadata before assigning it that update. It is separate from all six scheduled diagnostic snapshots.

**Illustrative short-pilot design:** screen roughly 25-50 million training tokens per arm, concentrating checks near 5, 10, 15, 20 and 25 passes of the current corpus. This brackets the observed development-loss knee. Use identical initialization and paired seeds for the initial comparison; confirm a promising result with additional seeds. These budgets are design suggestions, not newly measured cost estimates.

For a larger fresh corpus, compare both equal training tokens and equal inclusive compute; a pass over a different corpus is not the same treatment. Save full state at the best validation criterion and at declared milestones. Keep the old cohort as a historical comparator and add a separately frozen, stage/family/size-stratified development cohort with loops and larger integers.

**Proposed promotion criteria:** at least 95% measurement completion in every primary split/arm; matched-attempt and matched-clock controls; improving verified continuation yield and family-macro performance on fresh development examples; confirmation across seeds; no dependence on the final cohort. These are proposed criteria, not claims that they were met.

The v3 recipe is deliberately closed. A changed sampler, target representation, regularizer or budget requires a new prospective profile and cost/admission binding. Extending the final checkpoint unchanged is a low-priority option; nothing here proves it cannot help, but it currently has weaker evidence than the cheaper discriminating experiments.

Another limited pilot could compare literal and parameterized targets using the existing paired-token and native-fitting machinery. First fix and qualify explicit-parameter dispatch. The pilot publisher currently hardcodes concrete targets, so this requires a scoped adapter/profile change rather than simply setting a flag. A new recurrence DSL or learned subprogram vocabulary is a substantially larger project.

## Larger changes that merit a new specification

### Option A - A generalization-focused data and evaluation contract

**Scope:** family quotas or caps; explicit stage/complexity/bit-length strata; matched parameter holdouts; disjoint structural families across every stage; bounded generation of fresh data; behavior/target-equivalence audits; row and family-macro metrics. Preserve the original cohort as a versioned historical benchmark. Expand development coverage before expanding final exposure.

**Success test:** under matched compute, fresh held-out semantic yield improves across multiple strata and seeds, including loops. Log preparation cost and rejection rates. Distinguish two programs agreeing on 100 indices from universal equivalence. This is the most directly supported larger direction because the present cohort imbalance is measured.

### Option B - Verified search, distillation and adaptive sampling

**Scope:** a training-only search loop; independently verified candidate replay with provenance; task-family sampling driven by measured learning progress; a nonzero exploration floor; forgetting-aware revisits; bounded per-task compute. Compare uniform SFT, static family balancing, adaptive SFT and search-distillation separately before considering RL.

**Success test:** adaptive work beats a static balanced baseline at equal inclusive cost, without using development/final hidden outcomes as rewards. Preserve successful alternative programs without flooding the buffer with easy or duplicate behaviors. Treat expert iteration/replay and GRPO as separate choices.

Do not copy the old SYMPLE implementation unchanged: its feedback routine appends all successes before all failures and derives a temporal slope from the resulting list. A 4-success/8-attempt batch can create an artificial negative within-window slope. Reuse the idea of learning progress, but define a timestamped batch-level statistic and validate its behavior. This historical code issue did not affect the present pilot.

### Option C - A synthesis representation designed for structure

**Scope:** compare literal targets with parameterized skeletons using the repaired native fitter; then consider compact AST actions, recurrence forms or learned reusable subprograms. Measure structural correctness, parameter identifiability and final execution separately. Account for inference/search cost as well as training loss.

**Success test:** higher held-out verified yield at a comparable total budget, with attribution to structure versus fitting. Any new DSL requires execution, canonicalization, codec, provenance and checkpoint contracts. This can be valuable, but the current scalar diagnostics do not justify a wholesale encoder replacement.

For all three options, attach exact baseline profiles, independent scoring boundaries, finite clocks, missingness rules, checkpoint lineage, abort conditions and a held-out decision rule. Prefer one specification centered on a testable hypothesis over combining a new encoder, DSL, RL objective and curriculum in one run.

## Research prompts grounded in this pilot

Use the seven inspected historical reviews as a hypothesis inventory, not as validated experimental evidence. They emphasize cold-start RL, SYMPLE, constant binding, collapse and encoder design; the current pilot instead uses supervised teacher forcing. For every prompt below, request original papers and authors' code, verify titles/versions/quoted numbers, and label speculation and transfer limitations. Search through 28 October 2026; do not assume an old review's citation is accurate.

### Prompt 1 - Synthetic data that teaches reusable algorithms

Design a literature-backed data strategy for OEIS Learn: a 104.5M encoder-decoder trained from scratch to infer exact integer programs from 20 visible terms, scored on a 100-term horizon. Its frozen 32,768-row corpus was repeated 114 times; two loop families supply 26.8% of training rows. Structural development has only 18 stage-A families, and neither development split contains loops. Review synthetic program generation, semantic deduplication, family balancing, compositional holdouts, online data refresh and inverse-problem ambiguity. Start with DeepCoder and Deep symbolic regression for recurrence prediction, and follow verified later work. Compare learning literal source programs, canonical representatives and sets of behaviorally valid alternatives. For each method, identify required supervision, generation cost, leakage risks, compute scale and whether evidence comes from scratch-trained models. Return five implementable experiments ranked by information gain, with a fair matched-token/compute baseline and expected failure signatures. Preserve generic-only training and sealed final-cohort evaluation.

### Prompt 2 - Overconfidence, target ambiguity and numeric binding

Explain which research can distinguish memorization, numerical binding difficulty, target non-uniqueness and autoregressive error accumulation in this setting. Training reaches 92.8% exact canonical targets; parameter/structural development reach 0/128 and 1/128. Development token accuracy stays near 90.5%/88.3% while entropy falls to 0.054/0.048 nats and loss rises. Matched shuffling substantially worsens loss, so input conditioning is active. Both integer and operator/control errors remain. Compare literal decoding, parameterized skeletons with exact integer fitting, alternative-target marginalization, sequence-level objectives, regularization and data refresh. Start with end-to-end symbolic regression work, and assess whether grokking findings on small arithmetic tasks justify any continuation experiment here. Do not infer a capacity bound or causal collapse from scalar norms. Return a minimal saved-model diagnostic panel, a short factorial design, required exports, and concrete conditions under which a longer run would be informative.

These two prompts have the highest immediate value. They address observed distribution and generalization problems without presupposing that RL, a larger model, or a new encoder is necessary.

## Research prompts and verified starting points

### Prompt 3 - Curriculum and replay without measurement bias

Develop a compute-aware curriculum for scratch-trained exact program synthesis. Compare row-uniform SFT, family-balanced SFT, learning-progress bandits, forgetting-aware replay and verified expert iteration. Begin with Graves et al.'s Automated Curriculum Learning and DreamCoder, then assess recent primary work. The legacy OEIS Learn SYMPLE scheduler uses EXP3.S-style weighting and adaptive rollouts, but the current pilot disables curriculum/replay. Separate empirical support for scheduling from support for RL. Handle noisy, delayed and censored execution outcomes; never treat a service timeout as observed incompetence. Avoid manufactured temporal slopes from success-sorted rollout batches. Specify sampler state, exploration floors, family quotas, training-only reward boundaries, replay eligibility and exact resume behavior. Return a staged experiment where the simplest static balanced baseline must be beaten before introducing adaptive allocation, and include all search and validation time in the budget.

### Prompt 4 - A strong non-neural baseline and a better search interface

Design fair synthesis baselines and model-search interfaces for exact integer sequences. The present template baseline is unusable because formatting-sensitive parameter detection sends explicit-parameter templates to literal abstraction. The learned portfolio uses one greedy plus seven stochastic attempts and visible-only parameter fitting. Review enumerative/SMT search, neural-guided search, sketch/skeleton synthesis, recurrence DSLs and learned abstractions, starting with DeepCoder and DreamCoder. Compare frozen templates, bounded enumeration, neural proposals and hybrid search using identical visible terms, independent continuation scoring and explicit wall-clock budgets. Distinguish grammar completion, visible agreement, parameter identification, held-out continuation and proof. Recommend a small baseline repair first, then a larger DSL/search specification only if it adds demonstrated coverage. Return code availability, exact-integer and loop support, verification requirements, expected CPU/GPU cost and an ablation that attributes any improvement to neural conditioning rather than fitter capacity.

### Verified research entry points

[D'Ascoli et al. (ICML 2022), Deep symbolic regression for recurrence prediction](https://proceedings.mlr.press/v162/d-ascoli22a.html)

[Balog et al. (ICLR 2017), DeepCoder: Learning to Write Programs](https://arxiv.org/abs/1611.01989)

[Ellis et al., DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/abs/2006.08381)

[Graves et al. (ICML 2017), Automated Curriculum Learning for Neural Networks](https://proceedings.mlr.press/v70/graves17a.html)

[Kamienny et al. (NeurIPS 2022), End-to-end symbolic regression with transformers](https://arxiv.org/abs/2204.10532)

[Power et al. (2022), Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)

These papers motivate investigations; their published achievements are not evidence that their methods will improve this pilot. This review verifies useful starting sources, not a comprehensive new literature survey.

## Reproduction and evidence map

The companion bundle includes **analyze_pilot.py**, 19 CSV tables, **audit.json**, source-hash checks, and the template-policy reproduction output. Run the stdlib analyzer against a directory containing the three original ZIPs:

`python analyze_pilot.py --input-dir /path/to/zips --output-dir /path/to/results`

| Question | Primary exported evidence / derived table |
| --- | --- |
| Manifest and admission consistency | All manifest.json files; run contract.json; admission decision/authority/cost/qualification; tables/audit.json |
| Teacher forcing, cohorts, matched controls | Diagnostic per-example.jsonl and matched-controls.json; teacher_forcing, strata, cohort, families, control_effects |
| Semantic yield and missingness | Run score-N/outcomes.json and generate-N/selections.json; semantic_outcomes, outcome_reasons, teacher_semantic_join |
| Scheduling and generation behavior | Run attempts.jsonl; generation, native_outcomes, native_timing |
| Checkpoint timing and forgetting | Run validation-history.json and checkpoint-identity.json; validation_history, paired_changes, semantic_transitions |
| Weight and activation observations | Diagnostic tensor-comparisons.json and activations.json; tensors, representations |

The analyzer verifies ZIP membership, all payload hashes, run/admission/import identities, immutable probe digests, cohort IDs and target identities, and recomputes aggregate losses against the supplied diagnostic report and original saved-snapshot losses. It does not rerun the project's CPU/HIP suite or execute the native fitter. The separate codec reproduction exercises the actual fetched codec; native mixed-abstraction rejection is established by source inspection.

### Useful additional extraction before another training run

**From existing public evidence:** family-clustered descriptive intervals and per-family success transitions; generated/selected AST diversity using available program bodies; prompt-level queue and scoring completion plots; final error concentration and exact-to-semantic disagreements. The new tables already provide the joins and denominators.

**From retained workstation inputs/weights:** inspect the selected checkpoint; export first-error position, token-role denominators, top-two logit margin and target log-probability; compare canonical and generated AST skeletons; run grammar-masked/unmasked greedy parity; audit multiple valid targets for a visible prefix; rescore raw candidates independently of fitting deadlines. Export scalar/structural summaries without leaking hidden evaluation truth into the learner.

Use a separately versioned development cohort for new comparisons; do not relabel the old results. Independent hidden continuation agreement remains finite-horizon evidence, not proof or novelty. No final-cohort access is needed for the next decisions.

## Source references and interpretation notes

**Attached evidence:** oeis-learn-analysis-handoff.txt; 009-learning-pilot-longrun-20261027T180510Z.zip; 009-longrun-admission-20261027T180510Z.zip; 009-pilot-diagnostics-20261028T234932Z.zip. The handoff supplies project history; the findings here are recalculated from the attached pilot and diagnostic evidence.

**Source binding:** run source `fcfbb37256bd...`; diagnostic source `931476c28a71...`. These are content identities, not Git SHAs. Code was retrieved at commit `b87102233e8716f9ddd0408e5b4436a2b7171868`; relevant file bytes match the diagnostic dependency closure. The import records the permitted diagnostic-only source changes.

### Key implementation references

[Training, optimizer, checkpoint selection and milestone evaluation](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/fresh/bigint_worker.py)

[Closed v3 recipe and prospective-change requirement](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/pipeline_validation/pilot_recipe.py)

[Family reservation and stage-ordered structural selection](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/pipeline_validation/pilot_selection.py)

[Template roundtrip, parameter policy and four-second native request](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/fresh/bigint_diagnostics.py)

[Token spacing in decode_body](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/decoder/bigint_codec.py)

[Native fitting and admission deadline reservation](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/synthesis/client.py)

[Mixed explicit-parameter / literal abstraction rejection](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/crates/oeis_synthesis_core/src/parameters.rs)

[Scalar row metrics, controls, tensors and activation panels](https://github.com/grugnog/oeis-learn/blob/b87102233e8716f9ddd0408e5b4436a2b7171868/src/oeis_learn/pipeline_validation/diagnostics_runtime.py)

### Historical research reviewed

[Historical research directory at f191229](https://github.com/grugnog/oeis-learn/tree/f19122917e1f4dacc59c041ee4b6728d4e99ff49/research): Cold-Start RL Code Generation Research; Constant Binding Program Synthesis; Dynamic Multi-Task Curriculum Scheduling and Exploration Allocation in Neuro-Symbolic Reinforcement Learning; Neural Code Generation Literature Review; OEIS Curriculum Learning Design; RL Program Synthesis Collapse Mitigation; Tri-Stream Encoder Architecture Design. The directory contains 15 reports; these seven were the targeted review set.

[Historical symple_bandit.py](https://github.com/grugnog/oeis-learn/blob/f19122917e1f4dacc59c041ee4b6728d4e99ff49/src/oeis_learn/curriculum/symple_bandit.py) was also inspected. Its successes-then-failures history construction motivates a careful reimplementation if adaptive scheduling is revived; it is not part of this pilot's active training path.

### Statistical interpretation

Losses are summed nonpadding cross-entropy divided by summed tokens unless explicitly labeled mean-per-example. Exact target agreement is the teacher-forced argmax match over the complete canonical target. Bounds for unresolved semantic outcomes are identification bounds for this finite cohort, not uncertainty intervals. Families and repeated arm/snapshot outcomes are correlated. No significance claim, universal capacity claim, or arithmetic theorem is inferred.

Recommended disposition: the supplied evidence supports closing the scoped diagnostic evidence review as complete, subject to normal project review; it does not close scientific adoption, final exposure, or broader Spec 009 gates. The next strategy decision should carry forward both the successful engineering qualification and the newly identified evaluation limitations.
