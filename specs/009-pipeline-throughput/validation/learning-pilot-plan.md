# Prepare a bounded long run that measures sequence learning

Recorded 2026-10-25 on `009-pipeline-throughput`, PR #60. This is the current prospective preparation plan, following the [accepted primary hardware soak](soak-operator-20261025T232558Z.md). The user now requests evaluation of sequence learning and estimates of what additional training might achieve. That authorizes preparation of the measurement paths; it does not launch a multi-day run or expose final data. Original Spec009 comparison tasks and historical008 results keep their meanings.

## Question and existing evidence

Determine whether the existing strict, stochastic generic-program training approach acquires sequence-to-program capability that transfers beyond its training examples, whether performance continues improving with additional training, and whether a subsequent longer run is justified. This pilot precedes the organic/stochastic-versus-curriculum decision. Keep architecture, FP32 mathematics, optimizer and learning rate fixed. Curriculum, synthesis replay, assisted data and architecture sweeps are separate future interventions.

Reuse completed hardware evidence:104,496,662 parameters, training batch16, inference batch16, four bounded native workers, Torch2 intra-op/1 inter-op, native one thread, one GPU owner, bounded overlap and ordinary observations. No repeated size/batch ladder or identical two-hour soak is needed. Independent Docker and original optimization-adoption comparisons remain separate gates; no claim of supported global defaults follows.

The public soak archive already exposes a data-adequacy warning, recorded in [learning-pilot-existing-evidence.json](learning-pilot-existing-evidence.json):

| Observation | First recorded leg endpoint | Second recorded leg endpoint |
| --- | ---: | ---: |
| Cumulative updates |7,424|14,646|
| Data-order epoch counter |232|458|
| Last training batch loss |0.0000353537|0.000000928706|
| Complete development teacher-forced loss,256 rows |1.73105|2.14846|

These are last-batch losses and two development observations, not a full training learning curve or independent continuation evaluation. The rising development loss is an overfitting warning; it does not prove the cause or establish sequence-learning failure. Visible-panel records have different/incomplete coverage and cannot be compared as a learning-success curve. The archive has no actual-model independently scored100-term continuations; its CPU scorer used public fixture programs. No missing continuation scores are invented. Engineering checkpoints remain engineering-only.

Training on the same512 examples for days would repeat an already heavily revisited corpus. Audit and freeze data adequacy before finalizing the long-run recipe. A larger mechanically generated generic corpus is the proposed primary arm, not an automatic favorable-result fallback or a curriculum change.

## Prospective recipe and finite budget

The preparation target is **one fresh strict pilot**, one fixed seed, the accepted104.5M/batch16 configuration, and a **48-hour inclusive invocation ceiling**. Reserve at most **24 cumulative learner-role hours**, including that role's validation, progress, checkpoint and drain work. This is a proposed finite profile, not a user-confirmed duration or an admitted command. The admission artifact will state the actual measured expected duration and cost. No efficient-learning assumption is used to justify the budget.

Proposed data target: **32,768 unique generic training examples**, full128 parameter-development and128 structural-development examples, a frozen32-row training diagnostic panel, and the existing compatible8-row exposed OEIS-development cohort. Freeze and hash that development cohort and its prior exposure history; do not access the final cohort. If the approved cohort is absent/incompatible, block that proposed scope rather than substitute final data. Use the existing mechanical generator and exact semantics, with declared family/assignment seeds. A32,768-example corpus is a preparation target whose generation/diversity/length/cost feasibility must be demonstrated before freezing it. Missing diversity or generation capacity blocks this profile; do not pad with duplicate examples, silently truncate it, expand generator semantics or tune its contents using development results.

Audit source/program hashes, exact duplicate/prefix/shift groups, family/structural separation, literal/parameter distributions, target lengths, unsupported/censored examples and exact100-term reference outcomes. Keep detectable groups on one split. Preserve strict generic provenance and exclude OEIS/LODA teachers/metadata. Training samples may contain only permitted training targets and visible conditioning. Development hidden continuations stay in a CPU-scoring view. Treat already exposed development cohorts as development, never as fresh finals. A changed corpus is disclosed; the old512-example soak is diagnostic history, not a matched causal control for the new corpus.

| Hard reservation | Seconds |
| --- | ---: |
| Cumulative learner-role time |86,400|
| All frozen checkpoint generation and independent scoring |28,800|
| Current affected-source qualification |3,600|
| Shared startup/data preparation/checkpoint publication outside learner roles |3,600|
| Cleanup/emergency reporting |900|
| Archive/export |900|
| Unallocated safety margin, never added to prompt allowances |48,600|
| **Inclusive maximum** |**172,800**|

Image builds and large corpus preparation must be completed and charged in their own finite preparation scope before this warm-image pilot profile can be admitted. Their costs remain visible in overall study preparation; they are not free work hidden outside a claimed end-to-end cost. If the measured warm preparation cannot fit3,600s, revise the prospective profile before execution. Never increase a running allowance.

The scope-correct conservative projected total must be at most**138,240s (80% of48h)**. Reservations are hard maxima, not measured projections; the table alone does not satisfy this gate. Measure each required component, retain incomplete outcomes and mark missing costs UNKNOWN. Use three complete exact-scope observations and the existing conservative observed-envelope policy; do not call three measurements a population p95. No two-hour engineering rate admits full scientific controls. If the forecast fails, produce a concrete revised prospective cadence/budget proposal; do not shrink denominators or launch automatically.

Freeze independent work ceilings for updates, unique examples, nonpadding tokens, validations, candidates, native/reference/scorer calls and evidence bytes from the exact roster. Current v2's3,000 updates/four-hour recipe and engineering's7,800s/128 cycles/49,152 updates/frontier/event limits are incompatible with this profile. Add an explicit versioned recipe **and prepared-input manifest/checker**; current input preparation/checking and split selection also fix512/128/128. Preserve their v1 readers and exact old meanings. Do not raise those historical limits globally or renew the existing engineering lease. Record unique versus repeated examples/tokens and effective epochs so repeated exposure is not mistaken for new information.

Use the measured90%-physical/ancestor/engine shared memory ceiling, actual available-memory admission and16 CPU equivalents. Keep minimal1Hz safety observations; no optional always-on profiler. Propose a separately identified **16GiB transient** and **32GiB durable study-state** budget, with at least the existing5GiB free-space floor and independently checked image/cache space. These are caps, not allocations or observed requirements. Derive a byte-level bound before admission. Reserve current/latest/selected/staging generations, six snapshot payloads if retained, optimizer/RNG state, inputs, native caches, public evidence and export copies. Prune only after atomic publication and reference checks. The accepted run reached7.546/8GiB, so simply extending its storage policy is unsafe.

## Fixed measurement schedule

Freeze model snapshots at cumulative learner-role milestones **0,1.5,3,6,12,24 hours**. Capture once at an update/drain boundary, record actual wall/token/update counters and any lateness, and do not fabricate an exact scheduled x-coordinate. No quality-driven extra checkpoints, favorable-seed replacements or automatic continuation past the endpoint. Record incomplete/unstarted milestones and all consumed reservations after interruption.

- At **all six snapshots**, evaluate the full128+128 generic development denominator, the same32-row training diagnostic panel and the frozen8-row exposed OEIS-development cohort (**296 prompts each**). Freeze panel membership and family/length strata before results. Full populations at every checkpoint provide comparable full-split learning curves; a smaller panel must not substitute silently.
- Total registered generation-evaluation rows: **1,776** across the six snapshots (**10,656** selected-program scoring outcomes across the six existing arms). Each uses the existing correct/shuffled/absent/template schedule and its raw/rescue/equal-budget selections. Count every arm and its actual attempt budget explicitly. Full checkpoint controls, native fitting, independent reference work and scoring belong in cost admission.
- Record full-split teacher-forced validation on a fixed token-based cadence and at all six snapshots. Aggregate bounded per-minute training bins rather than appending one unbounded record per update. Freeze the cadence after data-token counts are known and before timing/results.

Three distinct measurement levels are required:

1. **Optimization/representation diagnostics:** token-weighted training and separate parameter/structural development cross-entropy; token accuracy and canonical-target exact-match where available; unique/repeated examples and tokens, updates, effective epochs and wall time. Full-program exact match is diagnostic because different programs can represent the same sequence. Do not merge the two development splits or compare differently weighted losses without disclosure.
2. **Visible synthesis:** generation termination/legality, raw visible20 fit, fitted rescue and equal-budget visible fit, attempts, original deadlines, source/assignment/selection seals and completeness. Conclusive failures, resource exhaustion and unstarted work remain distinct.
3. **Actual sequence learning:** independent exact100-index production/reference scoring of the one visible-only sealed selected program per arm, on both generic development splits, the training panel and the exposed OEIS-development cohort. Report success/full denominator, complete coverage, paired changes from random initialization, and correct-versus-shuffled/absent/template results. Fitting a visible prefix, low loss or high GPU use alone cannot establish this result.

Load and hash the actual snapshot for every generation/evaluation role; public canonical programs remain labelled fixtures. Candidate generation and selection use only visible20 evidence. CPU scorers alone receive hidden100-index truth; scores never return to learning/checkpoint selection/curriculum/replay. Keep visible-only selected checkpoint and latest recovery checkpoint distinct, and report the fixed final endpoint regardless of whether an earlier checkpoint looked better. Evaluate all frozen snapshots rather than picking the best hidden-scored checkpoint.

Use existing encoder, decoder, sampling, fitter, portfolio/selection, independent evaluator and CheckpointStore. Preserve raw/rescue/equal-budget semantics and the original8/1/1/8 correct/shuffled/absent/template attempt roster. A one-attempt periodic hardware probe cannot replace this scientific evaluation. Generation groups retain60s/1,024-token and native four-second original clocks; independent scoring retains its separate declared deadline. No hidden-best-of-candidate scoring or budget multiplication through workers.

## Learning curves and estimates of further learning

Report curves versus **nonpadding training tokens, unique data exposure, cumulative updates and inclusive wall cost**, with the training/development gap and all failure/coverage denominators. Keep parameter-development, structural-development, training-panel and exposed OEIS-development results distinct. Eight exposed OEIS cases provide limited exploratory evidence; report their full denominator and exposure history rather than claiming broad fresh-OEIS generalization. This profile does not expand or access the final cohort.

Publish per-prompt outcomes and paired changes, stratified by family, length, magnitude, structural complexity and non-affine/offset-only status where those labels exist. Use the existing deterministic family-level resampling convention for descriptive uncertainty and show family counts. One seed gives no estimate of training-seed variability; correlated examples and repeated checkpoint inspection preclude treating all scores as independent fresh trials. No final transfer gate or population confidence claim follows from this single pilot.

The additional-training report must separate three kinds of bounds:

| Bound/estimate | Meaning |
| --- | --- |
| Observed capability | Independently verified successes on each frozen denominator at each actual checkpoint; report actual regressions too, not only a cumulative best |
| Incomplete-evaluation bounds | For a fixed checkpoint/arm, successes/N through (successes+unresolved cases)/N, retaining known complete negatives; distinguishes uncertain measurement from weak learning |
| Conditional further-training scenarios | Plateau, recent trend per token doubling, and a diminishing-gain scenario at2× and4× observed training tokens, with explicit assumptions and elapsed-cost estimates; these are not statistical confidence bounds or guaranteed capacity ceilings |

Require at least four comparable nonzero-exposure checkpoints, adequate complete coverage and frozen comparable populations before fitting a trend. Backtest on a held-out later checkpoint before publishing extrapolations. Disclose sensitivity to the fitted window and model, nonmonotonic behavior, ceiling clipping and family uncertainty. When slopes disagree, coverage changes, overfitting worsens or the backtest fails, report extrapolation **UNKNOWN/UNRELIABLE** instead of an optimistic asymptote. Keep the32-row training-panel curve distinct from full128-row development curves. Low loss or a flat short test cannot bound eventual sequence-learning capacity. A six-checkpoint full-denominator design deliberately costs more than an endpoint-only evaluation: it is required here to support comparable progress estimates. If costs cannot fit, revise the finite learner allowance prospectively rather than silently substituting a small panel. No finite run provides a guaranteed nontrivial upper bound on what an altered recipe or much more training could learn.

At the registered endpoint, report: demonstrated learning, continued/slowing/absent evidence of progress, overfitting/data limitations, cost per additional independently verified sequence, and whether further same-regime compute is supported or inconclusive. A result can recommend more same-regime data/compute, a separately designed curriculum study, or stopping; it never switches strategies automatically. Preserve original008:T096 NEGATIVE and all inherited thresholds. The new pilot is not a retrospective repair of it.

## One consolidated validation handoff before long execution

Finish all unblocked implementation and host fixtures before issuing a workstation command. Reuse compatible accepted hardware evidence with explicit closures and actual current-package checks. Do not repeat the full primary soak solely because host analysis/planning code changed.

One future preparation invocation should:

1. Build/inspect current affected images once; run required new CPU/HIP/provenance/recovery/leakage fixtures, sharing preparation and retaining explicit missing Docker capability.
2. Prepare/audit/hash the proposed full corpus and separate visible/hidden views under finite time/work/storage limits; block dependent work on a shortfall.
3. Run **30 minutes cumulative learner time**, including a15+15-minute complete-state restart, using the actual proposed corpus/model/optimizer and bounded journal/retention paths. A lost original debit, changed policy, orphan or hidden leak is fatal.
4. Exercise baseline/post-update full checkpoint generation/control/CPU-scoring paths and fixed-population learning metrics; collect three complete exact-scope calibration observations for all forecast components under predeclared finite caps. Derive the projected full1,776-row schedule from those identities/counts; disclose the uncertainty of extrapolating later, differently trained models.
5. Run analysis on complete, zero-success, all-invalid, missing-control, timeout, interrupted, nonmonotonic and known improving/overfitting fixtures. Seal a single archive with source/image/data/snapshot/job/budget/cost/learning reports, including failures.

The short test passes on **correct plumbing, completeness accounting, numerical updates, isolation, state recovery, bounded resources and reproducible reporting**. It has **no minimum loss reduction, learned-success or transfer threshold**. Zero continuation successes can be a valid plumbing result; censored/incomplete cost observations can still block long-run cost admission. Do not retry until learning looks favorable.

Thirty minutes describes learner work only. Full-corpus preparation, current qualification and three full-control cost observations can take hours; freeze and disclose their measured expected time and hard ceiling before handoff. Target one combined validation/archive loop, followed by one separately admitted long-run invocation. Do not hide a multi-hour evaluation inside a claimed30-minute end-to-end test. If a concrete defect requires another loop, repair it first and rerun only its affected scope.

The resulting readiness artifact must identify the exact command, admitted decision, immutable source/image/data/profile/seed, projected cost at most80% of the finite budget, stage/job/storage bounds, progress every60s, resume/recovery procedure and archive path. The runtime extension is now implemented; use the separate [implementation/preflight handoff](learning-pilot-implementation.md). Exact-scope target cost evidence and long-run admission still do not exist. This plan does not supply long-run authority; launch remains an explicit subsequent action and no background run starts here.

## Implementation packages and original-task alignment

The checklist in [tasks.md](../tasks.md#long-run-learning-pilot-preparation) is the execution ledger. Code must extend existing trainer/diagnostic/scorer/phase6 adapters; there is no second trainer, fitter or general scheduler.

| Package | Deliverable and existing owners | Original tasks affected |
| --- | --- | --- |
| L001–L002 | Freeze this plan; audit existing public evidence without rerunning hardware | T086 preparation; does not close its costed handoff |
| L003 | Complete missing host archive/failure tests before implementation changes | T011 |
| L004 | Corpus generator feasibility, group leakage/diversity audit, finite preparation and provenance manifests | T093/T095 preparation portions only; no research execution closure |
| L005 | New closed pilot decision/recipe/cost schema and prospective budgets; prior-learning-PASS not required for a study establishing learning, but runtime/provenance/device/cost gates remain | T093 preparation; preserve original diverse32/representation semantics |
| L006 | Existing worker continuation, cumulative time/work bounds, complete-state milestones, rolling logs/frontiers/checkpoints and interruption recovery | T064/T069/T110 affected-source review |
| L007 | Bounded per-split learning metrics, fixed panels/full generic and exposed OEIS-development checkpoints and hidden-isolated actual-checkpoint generation/scoring | T093/T095/T097 preparation portions; full studies remain unexecuted |
| L008 | Paired curves, coverage bounds and cautious backtested compute-extension analysis | T107 reporting support; does not close a readiness result before execution |
| L009 | Exact full-schedule cost projection, reservations, admission and failure/no-go dispositions | T086; no missing-cost waiver |
| L010 | Host/CPU/HIP synthetic counterexamples and30-minute no-learning-threshold functional test | T011/T110 affected closures |
| L011 | One consolidated preparation DAG/CLI/archive and current source/binding/packaging qualification | T086/T110 |
| L012 | All unblocked code/local checks, commit/push and one finite workstation handoff | T086 remains pending until measured target cost is reviewed |
| L013 | Review that single validation archive, preserve compatible evidence, freeze costed command and readiness/no-go | T086 closure only if its complete handoff is evidenced |
| L014 | Explicitly admitted fresh pilot execution and complete endpoint report, including negative/interrupted outcomes | New pilot result; original research tasks close only for their exact scope |
| L015 | Review learning/generalization/compute-extension evidence and decide whether to continue same-regime training or design curriculum separately | T107 support; no automatic next run |

Original T034/T041/T059/T060/T062/T073 remain the original optimization/comparison investigation; their missing confirmation is not silently filled by this pilot. Policy/exactness/clock/containment counterexamples relevant to the chosen experimental configuration must nevertheless pass before learning. T084 remains independent Docker portability. Historical-model rescue stays deferred. Representation seed pairs, curriculum/replay, assistance, encoder studies and final exposure remain separate planned decisions.


## Corpus preparation repair, 2026-10-26

The142043Z archive passed primary CPU/native/HIP/policy/operations qualification
but exhausted262,144 raw counters without a selected corpus. Its old selector did
not retain checkpoint failure details or reference rows in a reusable private
cache. Local reproduction found that global overlap components included unused
bridge candidates. The v2 selected-population guard in the contract removes that
unintended dependency while requiring pairwise disjoint selected windows, the
independent final overlap audit, the existing structural/family rules and exact
32,768/128/128 counts. This is a prospective disclosed selection correction;
the old negative remains negative. No learning outcome selects a row.

Preparation now preserves private raw-reference batches and public selection
progress/failures under the original finite deadline. Explicit compatible cache
reuse avoids resampling retained counters; native execution is always repeated
for every selected source. Complete codec validation uses one parse of the same
typed grammar, avoiding quadratic prefix replay during corpus publication. Local
feasibility and current target native agreement remain distinct gates. No short
preflight learning-success expectation, multi-day admission or original task
closure changes.


Local execution interruption during concurrent large-corpus probes exposed a
representation-memory concern (the interruption's cause is not proven). On the
same65,536 rows, sharing the fixed generic index vector and equal immutable
strings reduces measured process peak RSS from833,552KiB to202,116KiB, with the
same full serialized-row hash. Preparation applies this exact-byte compaction;
corpus semantics, resource limits, horizons and counts are unchanged. Finish
large local probes in separate phases and preserve completed reference batches.
Cache recovery additionally rejects unknown row fields/oracle identities and
visible integers exceeding the original input-bit limit.
