# Contract: Controlled Bootstrap and Encoder Evidence v1

All defaults below are proposed pre-registration values, not measured outcomes. Emit the resolved recipe with hashes before work. Changing a value after seeing results creates a distinct experiment. Algorithms for generation, family keys, grouping, ranking, curriculum and replay belong to Phase 2; later phases configure and exercise them.

## Data and provenance

Start from a pinned bulk OEIS snapshot for evaluation and independently generated generic programs for strict training. No per-sequence downloader or imported solution program enters the generic sampler. Generate connected expressions/assignments (A), interacting values/compositions (B), bounded loops with meaningful carried state (C), and conditions/longer compositions (D). New D guards use supported comparisons and parameter-independent control rather than copying old remainder-based guards into an unsupported language.

Use deterministic counter-based structure sampling and at least 16 parameter variants per retained family. Record source priors, constants/identity proportions and rejected/trapped/resource-excluded samples. At most 10% constant-output and 10% identity/offset-only examples in the transfer dataset; the tiny gate uses its separate explicit caps. No exception-dropping dead-code rewrite: only transformations qualified on the declared domain may shorten sources. Preserve concrete/parameterized paired records; if either exceeds target token limits exclude from both matched arms and report the count.

Initial bounded datasets: 32 diverse memorization prompts, then 512 training prompts, 128 parameter-development prompts and 128 structural-development prompts. At least four structure keys and 16 non-affine-in-index examples in the tiny set; at most eight offsets. Non-affine means behavior is not representable as a*n+b on the frozen 100-index oracle, a bounded criterion rather than a theorem about all indices. Reject an underfilled dataset with a precise shortfall report; never silently seal it as the requested full set.

For parameter-dev, reserve variants within training families while keeping detected identical behavior/prefix/shift groups out of training. For structural-dev, reserve entire family keys and all their parameter variants. Check observed behavior overlap separately from syntactic/family distinction; otherwise a differently written n+15 can count as false structural transfer. Structural normalization is a declared approximation; no claim of complete equivalence detection. OEIS dev/final cohorts undergo detectable overlap screening, with a fresh unexposed final cohort frozen before revised claims.

## Experiment sequence (not extra implementation phases)

| Study | Fixed controls | Changed factor / report |
| --- | --- | --- |
| Existing checkpoint rescue — T093 user-deferred2026-10-22 | Optional historical comparison, outside current Phase5 completion; no model conversion required | Read-only implementation/compatibility retained; no executed result claimed |
| Literal vs parameterized bootstrap | Same generic behaviors/structures/splits, small backbone, chosen input profile, optimization and seeds | Representation and fitting; raw/fitted scores, token/parameter counts and realized compute |
| Curriculum and replay | Winning or inconclusive representation decision explicitly frozen | Enable curriculum first, then verified replay as a separate recorded comparison, not an unlabelled joint intervention |
| Generic-only vs assisted | Same generic foundation checkpoint and downstream budgets | Reviewed mathematical demonstrations; shared foundation cost, lineage and transfer on excluded families/compositions |
| Encoder inputs | Same retained recipe/data/decoder and matched initialization policy | Four post-training ablations, followed by matched retraining of full hybrid and selected leave-one-group-out variants |

Three initial paired seeds: 20261016, 20261017, 20261018. Tiny/screen starting defaults are 3,000 maximum updates, 7,200 seconds total including diagnostics, batch eight, AdamW learning rate1e-4, tiny-gate validation every100 updates and transfer-screen validation every500 updates, and eight stale validations with minimum synthesis-score improvement one prompt. The 7,200-second value is an unmeasured initial qualification default, not a throughput requirement or evidence-based runtime target. Source-bound Phase 5 measurements may support a separately preregistered finite budget, cadence or portfolio for each scope before execution. Model d256, four encoder/four decoder layers, four heads, ff1024, FP32/eager. The earlier successful *joint* decoder candidate (token multiplier1, sinusoidal position gain16, encoder/decoder dropout0) is an explicit starting recipe, not an independently proven positional-encoding or regularization conclusion. Reuse common decoder initialization and record extra input-branch parameters; do not claim identical model sizes when they differ.

For the diverse32 memorization gate, periodic and end-of-run synthesis cover exactly the frozen32 prompts and its declared raw/rescue/correct/shuffled/absent/template controls. For periodic transfer-screen synthesis, freeze a separate stratified diagnostic panel of32 training,32 parameter-dev and32 structural-dev prompts before training; evaluate every member of that panel with one greedy candidate and its optional fit. Teacher-forced loss covers all development examples. Record panel IDs explicitly: this is not a hidden first32 slice or a substitute for reporting all128+128 development prompts at the predeclared end of screen. The final selected checkpoint receives the complete raw/rescue/equal-budget/conditioning/template evaluation; it is selected before hidden continuation scoring.

Qualification first estimates complete scheduled diagnostic and end-of-run cost using separate non-scoring fixtures with matching size/complexity. Every receipt identifies its scope (`diverse32` or `transfer-screen`), exact periodic prompt count, end-of-run prompt count, attempts, controls, cadence, source/image/profile and measured p95 inputs. A scope may start only if its projection fits within80% of its preregistered finite budget, leaving20% for variance/checkpoint/cleanup. The initial7,200-second default may be revised after measurement; record the rationale and new experiment identity before training. Never use a failed larger-screen projection to block a smaller diagnostic without calculating the smaller workload, and never use a smaller receipt to admit the larger screen. Preserve fixed denominators, controls and success thresholds; do not omit examples mid-run. No blanket Cartesian sweep is required. Post-training ablations are inexpensive sensitivity measurements; choose retraining comparisons before examining the new final cohort. Report active/total parameters, throughput, group-specific integer-size/token-length/complexity outcomes, and interaction limitations of leave-one-out attribution. A large weight or a zero-gated stream does not determine universal mathematical usefulness.

## Synthesis metrics, selection and promotion

Raw evaluation uses complete grammar-constrained generation with no fitting. Same-candidate rescue fits precisely those raw complete candidates. Equal-budget evaluation separately compares portfolios under a shared total cost: default 60 seconds per prompt including generation, fitting and visible verification; at most eight candidates, one greedy plus seven seeded stochastic, 1,024-token cap each. Beam search is opt-in with explicit budget reallocation, not additional free attempts. Decoder-only time and fitting/reference time are separately recorded. Hidden scoring has a separate fixed budget and cannot rerank candidates.

For every split report total eligible N, attempted, complete evidence count, success count/N, resource/unsupported/invalid counts and exact exclusions before freezing N. All incomplete evaluation cases remain unsuccessful in the fixed-denominator score and also appear as incomplete coverage. Training selection uses synthetic visible-fit scores on the fixed full diagnostic set, then complete teacher-forced loss and smaller update as deterministic ties. Hidden continuation scores are reporting-only and never select a checkpoint or advance curriculum. Do not choose a checkpoint evaluated on an easier completed subset.

The T096 diagnostic follow-up makes coverage evidence explicit per arm. Raw and
same-candidate rescue refer only to the first greedy attempt; the equal-budget
and conditioning/template arms retain their own scheduled portfolios. Record
generation, fitting/admission and scoring separately. A sealed selection has
complete scoring evidence only when production and reference both return all100
values; an expired scoring deadline cannot inherit generation completeness.
With no selected candidate, a completed arm may have a conclusive negative
outcome (for example unsupported fitting or scoped UNSAT), but incomplete
decoding, unstarted required attempts, unknown fitting and resource exhaustion
remain incomplete. Other controls cannot supply or remove this evidence.
Later unused attempts cannot invalidate an independently completed100-index
score for the already sealed selection. Legacy records without per-arm evidence
are explicitly unattributed; their historical assessments are not overwritten.
This clarifies evidence accounting without changing selection or promotion
thresholds. Compact stage receipts for every prompt precede sampled verbose
examples in exports; any bounded-detail omission remains explicit.

The Phase 5 scheduling follow-up groups up to eight visible prompts for each
compatible decoder call. Each prompt keeps the same greedy/seven stochastic
attempt order, per-attempt seed, grammar and token cap; fitting remains serial.
A group shares a slice of the original global deadline, with the60s per-prompt
ceiling retained. Cost estimates and role reservations charge measured full-group
latency once plus conservative singleton tails per round. They cannot omit
prompts, combine tails across rounds, or multiply budgets. New-source mixed-prompt
greedy/stochastic parity and actual HIP cost qualification are required; see
[the follow-up evidence and protocol](../validation/phase-5-generation.md).

The primary transfer metric is `continuation_accuracy_100`: the number of prompts whose **one sealed, visible-only selected final program** agrees with every exact value at all 100 declared source indices, divided by the frozen split size. Production execution and the independent reference must both complete and agree. Matching only the 20 visible terms is `visible_fit_accuracy_20` and does not count as transfer success; a continuation mismatch, missing independent evidence or resource failure contributes zero to the primary numerator. The 128-member synthetic development splits must have complete 100-index truth before sealing. OEIS examples without the required horizon are excluded with reasons before their cohort denominator is frozen.

The transfer gate uses `continuation_accuracy_100` from the **exact-fitting equal-budget portfolio**: at most one greedy plus seven stochastic candidates per prompt, selected using visible evidence within the shared 60-second budget. Correct, shuffled and absent conditioning use that same portfolio; the non-neural comparator uses its frozen bank of at most eight templates and the same fitter and budget. The literal/full-input parameter-development control uses this same scoring/portfolio rule. Raw and same-candidate-rescue scores remain separately reported and cannot replace this gate metric. Hidden scoring evaluates the sealed selection; it cannot search the candidate list for whichever program matches the continuation.

Memorization gate: at least 31/32 (95% rounded up) independently verified greedy-plus-fitting successes through the 100-index diagnostic horizon, reliable termination and complete coverage. Continue to report raw score, correct/shuffled/absent conditioning and template baseline; passing this gate alone does not permit long runs. This synthetic diagnostic is explicitly a memorization test and its oracle outcomes are not a substitute for a clean final cohort.

Transfer promotion gate for a candidate recipe, frozen before its short screen:

- At least 95% complete evaluation coverage on each fixed development split for every seed, and 100% of attempted admissions independently verified; no unexplained cache/production/reference disagreement.
- At least eight additional `continuation_accuracy_100` successes on structural-dev out of 128 versus **each** shuffled, absent-conditioning and non-neural baseline comparator in at least two of three seeds; mean paired advantage at least 6.25 percentage points against each comparator across all three seeds. Score on the fixed denominator, not only completed cases.
- Familiar-structure/new-parameter `continuation_accuracy_100` does not drop by more than four successes out of 128 relative to the declared literal/full-input control mean under the same fitted portfolio. Report any trade-off rather than hide it.
- Report all seed-level counts and paired differences; use a predeclared structure-family bootstrap (2,000 fixed-seed resamples) for descriptive 95% intervals. With three seeds, report seed spread separately; do not imply the interval establishes population-wide significance.
- Check the gate once at the predeclared end-of-screen checkpoint-selection outcome. Do not stop adaptively on the first favorable hidden-continuation result. A positive gate allows planning a longer experiment; it does not authorize automatic execution.

The non-neural baseline uses a frozen small bank of generic expression/loop templates generated independently of evaluation outputs, maximum eight per prompt under the same total budget and fitter. Its manifest exposes supplied structural priors. It cannot expand templates using hidden failures or family labels. Correct/shuffled/absent tests use the same fitted candidate policy and compute; shuffled mapping is a fixed derangement, absent input uses the explicit unconditioned marker and valid masks, not malformed zero-length padding.

Curriculum advancement uses full synthetic **visible-fit** evidence for current stage, threshold 80% for two consecutive scheduled validations with at least 95% coverage; continue sampling at least 25% earlier-stage examples after advancement. Replay is disabled for representation comparison. Later replay caps at 25% of each batch, admits only independent training-target successes, and maintains deterministic capacity/replacement IDs; neither dev/final successes nor their visible prompts enter updates.

## Final-cohort lifecycle

Freeze the OEIS final cohort's identity, membership, exact indices, horizon, overlap exclusions and private truth at preparation/preregistration. Its evaluation state begins `sealed`. Tiny gates, representation screens, curriculum/replay studies, assistance selection and encoder ablation/retraining use synthetic/OEIS development or explicitly exposed diagnostic cohorts. Their ordinary `train`, `learnability`, `diagnose` and development `evaluate` operations cannot launch final-cohort generation/scoring, mount its truth or emit its outcomes. A training run's finalization means completion of that run; it does not authorize final-cohort evaluation.

After all intended recipe, assistance and encoder decisions are recorded, create an immutable `final-evaluation/v1` plan containing the cohort hash; frozen recipe/portfolio/runtime identities; the complete arm/seed roster and checkpoint hashes already selected by visible-only training criteria; fixed denominators and controls; explicit per-arm and aggregate scoring budgets; and the decision-manifest hash. Reject a plan with incomplete decisions, missing budgets, changed checkpoints or a previously exposed cohort claimed as fresh. Validate this record before changing the cohort to `authorized`. This policy and its state transitions belong to Phase 2; production command/mount enforcement belongs to Phase 3.

Only an explicit `evaluate --split final --final-plan PATH` request registered in that plan may execute the final study. Durably mark the cohort `exposed` before the first authorized final-prompt evaluation and debit the shared final-study ledger before work. Failed/interrupted execution still consumes budget and freshness. Every registered arm remains in the report, including unstarted/timed-out arms. Resume may finish only the identical plan with its remaining allowance and sealed outputs; changing the plan or tuning after exposure requires a new final cohort for a fresh generalization claim. Final scoring occurs after the input studies and before the readiness report; it cannot retrospectively change fitted constants, selected candidates or checkpoints. Finite results remain finite evidence.

## Assistance and discovery

Assisted branch manifests must identify demonstrations and their verification/provenance, not merely a flag. They can teach generic mathematical techniques or named families, but exposed/duplicate evaluation groups are excluded and lineage remains assisted. Historical checkpoints are not automatically qualified; they require an explicit approved migration/manifest and cannot replace the shared generic control silently.

Retain up to eight verified alternative programs per sequence by the Phase 2 structural/cost retention rule, recording omissions. For each, distinguish changed coefficients, reused structure, new combinations and a candidate structural novelty requiring review. Finite continuation, bounded SMT evidence, conjecture, proof and novelty review remain separate fields. No automated theorem/novelty promotion is in scope.
