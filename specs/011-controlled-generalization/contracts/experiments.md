# Experiment recipes, progression and budgets

These are prospective contract definitions for implementation, not executed configurations. Committed prospective recipe templates live under `configs/foundation/pipeline_studies/`; runtime preparation emits fully resolved sealed JSON and artifacts only in operator workspace/output roots; every authority input is hashed independently of source closure. Unspecified hardware-dependent coverage counts are resolved by bounded preflight, never by observing learning results.

## Operator-authorized fresh start (2026-11-01 Pacific)

Execute P1-screen next, with fresh paired random initialization at seed20261016.
E0 and P0 remain optional deferred historical investigations; their completion is
not a prerequisite for this authorized sequence. Retain their artifacts and
limitations. Do not count historical seed16 as part of fresh P1 confirmation.
P1's 25M-token cap per arm, fixed support/mixture, selection-only advancement and
sealed assessment rules remain in force. A changed effective budget/design must
be frozen under a new identity before learning, never patched after outcomes.

The temporary fresh-start runner freezes a **60-hour campaign ceiling**, superseding
the default36-hour P1 screen envelope only for this registered invocation. It includes
one current-source acceptance session (395 minutes maximum), preparation and validation,
two6-hour learners, two6-hour generators and two6-hour scorers, with host/setup/cleanup
headroom and one hour reserved for final reporting/recovery. Image builds occur before
this clock starts. These are bounds, not expected durations. Per-arm25M/6-hour limits
and per-target inference budgets remain unchanged. Both roles use64GiB/four CPUs and
must consume qualification with those exact resources. Acceptance does not authorize
research by itself; the operator's explicit `fresh_pilot.py run` does.

The frozen pilot protocol selects latest available checkpoints at the two terminal
learner boundaries. Retained milestone states remain private and available for later
separately declared analysis. This screen does not perform adaptive checkpoint ranking,
publish a new replay initializer, open assessment, or automatically launch confirmation.
Partial/unequal exposure remains descriptive. The public measurement plan freezes
256 selection rows, seven policies and both arms before learning; each metric slot is
the outcome of the policy's selected program, not an oracle union of its proposals.
Original proposal/cost ledgers accompany these projections. A controller receipt makes
post-command failures part of the mandatory software ledger; missing/failed verification
produces a diagnostic recovery archive, never a completed-research claim.

## Named research stages

| Recipe | Arms and data | Fixed budget and checkpoint rule | Stop / advance |
| --- | --- | --- | --- |
| E0-evaluation-extension/v1 | Retained 3-hour and selected historical weights plus available uniform best/latest; matched controls and one shared absent-input bank. Additional family/decay weights require explicit selection. | No learner updates; historical 20/100 terms, 8 attempts, 60-second portfolio, matched-drain/v2. Snapshot roster/hash frozen before launch. | Missing artifacts mark unavailable. Primary cells need 95% completion and valid accounting; fix runtime/evidence failures before interpreting new training. |
| P0-uniform-replication/v1 | Historical uniform literal/static/no-replay recipe; new seeds 20261017 and 20261018; existing 20261016 is reference. | 25M target tokens and 6 learner hours/seed; preserve historical continuous pooled-loss selection for this replication. Report best/latest separately. | Scientific result is seed variation on exposed historical cohorts, not unseen confirmation. Failure to improve is valid. Regenerate missing/incompatible seed16 only in a separately listed run, or report two-seed result incomplete. |
| P1-refresh-screen/v1 | Fresh static vs refreshed-within-support, identical model/initialization, fixed descriptor roster and sampling law, literal targets/no replay; seed20261016. | 25M tokens and 6 learner hours/arm; pinned 4/8/16/25M checkpoints; separate selection supervision and sealed assessment. | Advance to seeds17/18 only after valid coverage/isolation, budget/accounting and >=95% primary selection completion. Do not choose continuation using assessment labels. No accuracy gate to complete a valid negative screen. |
| P1-refresh-confirm/v1 | Same frozen paired recipe at seeds20261017/20261018 plus seed16 under identical definitions. | Same per-arm limits; select/freeze all comparison cells before assessment. | Apply preregistered FR-032 decision; if screen assessment influenced any advancement, checkpoint/arm/incumbent decision or recipe change, create fresh assessment identity for confirmation. |
| P2-support/v1 | Baseline vs expanded structural support, static data, same admitted-example count (default32,768), matched coverage marginals where feasible; disclose residual distribution changes. | Same paired token/model/inference settings; one seed screen then separately registered three-seed confirmation. | Frozen selection-only baseline/advancement. Impossible matching → declared shortfall/new design, not silent substitution. |
| P3-targets/v1 | Literal vs explicit parameterized targets for paired concrete programs; identical fitting availability; replay off. | Same model/initialization, tokens and inference limits; record actual examples/updates and explicit parameter sharing. | Report unsupported rows and raw/fitted outcomes. Adoption uses preregistered confirmation, not best observed assessment. |
| P4-replay/v1 | From a common frozen literal initializer, continued SFT vs one search-and-distill round; no simultaneous representation change. | 4M additional target tokens/arm, checkpoints at1M/2M/4M (not the25M milestone list), 2 learner hours/arm; search <=256 training tasks,8 attempts/task,60 seconds/task and 4 aggregate hours; replay4096 total,2/task,25% batch. | Empty discoveries recorded without weakening admission. Added search cost explicit. Efficiency claim needs separately cost-matched control. |

P2/P3/P4 use the current selection-designated incumbent, with lineage and the entire changing-factor list frozen. No automatic combined winner recipe. Choosing an incumbent after assessment makes subsequent reuse of that assessment exploratory; it requires a fresh sealed cohort for new confirmatory claims.

All capabilities are built before delivery; running only P0/P1 initially is experimental discipline, not a reduced feature scope. Execution after qualification still follows normal operator run authorization; the validation script never automatically starts research.

## Resource ceilings

All values below are planning defaults and hard ceilings, not measured runtime estimates. Preflight may lower resources without pretending the profile is unchanged; revised effective contracts receive new identities.

| Scope | Ceiling / rule |
| --- | --- |
| Hardware acceptance session | 5 hours total; one original CPU/native qualification and one HIP qualification scope; all new-path training fixtures <=20,000 target tokens each, <=6 fixtures; <=8 synthetic tasks per inference control fixture. |
| Research learner | One learner; 6 hours/25M-token arm (P4 two hours/4M). Timeout yields partial, not cap-complete. |
| E0 / P0 / P1 campaign wall time | E0 <=24 hours; P0 <=36 hours for two new seeds plus evaluation; P1 screen <=36 hours, confirmation <=72 hours. Aggregate ceiling includes preparation/evaluation and stops partial if exhausted. |
| Preparation | <=1,000,000 candidate draws, <=16 shards of32,768 admitted rows, <=8 GiB shard storage and <=4 hours; historical static training count stays32,768. Prepare enough accepted token capacity for requested one-pass refresh; otherwise fail feasibility. |
| Coverage | 128 rows/cell for selection-parameter, selection-structural, assessment-parameter, assessment-structural. Freeze explicit per-stage/length/magnitude/control-flow quotas after feasibility inspection. At least two loop-active and two branch-active families in each structural cell; protect whole-family separation. No full cross-product promise. |
| Memory/CPU | Existing `resources.resolve` effective ceiling: min(90% physical,cgroup caps), admission with measured peak+headroom; <=16 effective CPU count; torch2 intraop/1 interop, native1 thread/worker, <=4 GiB native aggregate reservation unless separately qualified. |
| Storage | <=64 GiB per study root, including <=32 GiB private pinned checkpoints, <=8 GiB shards and <=4 GiB raw evidence/logs. Preflight measures required snapshot sizes; insufficient capacity blocks start rather than pruning required pins. |
| Sharing | <=512 MiB final bundle, bounded per-stage log tail <=100 lines and <=1 MiB. Oversized mandatory evidence fails packaging with an explicit partial bundle; never silently omit required payloads. |

Preparation-only feasibility can inspect synthetic sources/coverage, not learning results. Freeze generation seeds, descriptors, exclusions, exact quota counts and numeric/length bins before any learner sees data. Infeasible quotas halt and require a new registered manifest. Historical E0/P0 may remain useful even when the new challenge cannot yet be prepared, but do not substitute them for P1's benchmark.

## Acceptance-specific reservations

The acceptance manifest uses reduced synthetic fixture quotas, not the research preparation limits. Prebuild/freeze qualification and GPU images before the session. Reserve at most45 minutes for preflight/current inspection,120 minutes for CPU/native plus HIP qualification,10 minutes for preparation (<=10,000 draws and2,048 admitted fixture rows),95 minutes total for the five representative cases, and20 minutes for report/verification/recovery:290 minutes plus10 minutes emergency margin within the five-hour ceiling. Initial runtime validation allows600 seconds per case; the two initial qualification image inspections allow900 seconds each. The qualification runner reserves6600 seconds including300 seconds for cleanup/reporting, leaving600 seconds for post-run inspection and authority validation inside the7200-second qualification stage. Recovery image inspection allows300 seconds and packaging retains480 seconds inside a900-second report stage, followed by300 seconds for independent archive verification. Warm qualification-consumer inspections retain their120-second bounds. Image-inspection bounds include a30-second shared process/container cleanup reserve; other roles retain their existing cleanup limits. The retained-model replay/resume fixture is capped at35 minutes and the other four cases at15 minutes each; earlier completion does not authorize extra research work. Keep final packaging time reserved when terminating work. Failed reservation/feasibility checks stop before hardware work; these limits are not performance predictions.

After acceptance-008 exceeded its30-second retained-model learner deadline, generated P4 acceptance inputs allow180 seconds per learner invocation and180 seconds per current-policy evaluation. Token cap512, seed, original model/initializer, replay search30 seconds and research recipes are unchanged. Replay reserves four learners, two evaluations, search and two proofs. Twenty minutes move from the CPU qualification ceiling (55 to35 minutes; observed CPU suite336.07 seconds) to replay/resume. The HIP suite retains40 minutes, each cold image inspection15 minutes, and post-qualification inspection/validation10 minutes. No test is skipped or treated as optional. The five-hour aggregate ceiling,20-minute packaging reservation and10-minute emergency margin remain unchanged.


### Operational repair T106/T108 (2026-10-31)

Supersedes earlier acceptance-only timing allocations:335 minutes total, with preflight45, qualification120, preparation10, cases130, packaging20 and emergency margin10. Replay/resume reserves70 minutes; other cases retain15 minutes. Each replay learner gets600 seconds including input projection and up to30 seconds cleanup; current-policy evaluation remains180 seconds, each full-state proof gets180 seconds, search remains30 seconds. Four learners, two evaluations, search, two proofs,540 seconds inspection/validation and120 seconds I/O margin total3810 seconds within the4200-second case. The original model,512-token cap and single seed are unchanged. Acceptance-011 checkpoint writes reached26–45 seconds; the prior180-second learner envelope expired during persistence. The additional allowance is an operational margin, not a throughput claim.

Checkpoint preflight uses `serial-checkpoint-graph/v1`: initial/latest plus milestone pins for each run, acceptance-only paused/resumed runs, one live resume projection, an atomic generation, proof latest states and separate weight-copy allowances. Input projections have one verified copy per channel; proof inputs include latest state plus initial weights, not entire checkpoint stores. Acceptance replay policy evaluation does not publish unused initializer/selected-state copies; research evaluation retains publication behavior. Existing26GiB P4/32GiB aggregate checkpoint ceilings remain in force. Any actual peak outside the declared accounting still fails under the runtime disk watchdog.


### Acceptance-013 evaluation envelope amendment (T111/T112)

Supersedes the T106 acceptance-only allocation above:375 minutes total, with
preflight45, qualification120, preparation10, cases170, packaging20 and margin10.
Retained-eval gets25 minutes, replay/resume100 minutes, other cases15 minutes each.
The retained two-checkpoint evaluation scope gets720 seconds; each paired policy
evaluation gets1200 seconds. For one policy this allocates600 seconds to generation,
minus up to30 seconds container cleanup and30 seconds process startup/exit, leaving
about540 seconds including model loading for the unchanged portfolios. This allows
headroom for eight60-second prompt/control portfolios even without batching benefits.
The retained checkpoint scopes are also enlarged so charged startup/exit does not
reduce their previously passing effective work allowance.

Replay/resume worst-case reservations are4*600 learners +2*1200 evaluations +30
search +2*180 proofs +540 inspection/validation +120 I/O =5850 seconds, within6000.
Retained evaluation720 +540 inspection/validation fits1500. Learner tokens512,
single seed, model, portfolio attempts, per-target60-second limits, proof checks and
storage caps are unchanged. Existing research recipes/budgets are unchanged.


### Acceptance-014 small-model persistence amendment (T113)

Supersedes small-model timings above: refresh and target-format learners get180
seconds each instead of30; each acceptance proof gets180 instead of60. Both cases
reserve1500 seconds. Feasibility includes two learners and two proofs:
2*180+2*180+540 inspection/validation =1260 <=1500. Case reservations total190
minutes; serial total395 minutes (6h35), including20 packaging and10 emergency
margin. Replay/retained evaluation envelopes remain as in T112. All model, token,
seed, portfolio, storage and research budgets are unchanged. This addresses the
measured acceptance-014 checkpoint-write timeout without relaxing success gates.


### Fresh research preparation and persistence (T133/T134)

Research preparation uses its existing four-hour limit instead of the acceptance
600-second limit. Research custodian validation permits3600 seconds, including
input projection; acceptance retains its300/600-second validation envelopes.
A training shard may contain32,768 distinct variants; cohort JSON readers permit
16MiB to hold its source/group/variant identities. Exact fixed-family quotas are
expressed as `family:<verified-family-identity>` and derived only by preparation.
Quota ordering and cross-role finite-window decisions retain their original meaning.

Full state is still durable before each update debit is executed and after each
completed update. Byte-identical serialized state components may share verified
hard links across generation-local paths. Generations remain independently copyable;
logical storage reservations do not assume deduplication. This changes persistence
I/O, not numerical updates, checkpoint frequency, milestones or consumed budgets.
Existing source/image qualification does not authorize these production changes.


### Registered fresh P1 design (T135)

The first fresh screen fixes32 training families from the first4096 generic
sampler descriptors at seed20261016: A4/B6/C2/D20, each with at least three
parameters. Each32,768-row shard admits1024 unique sources/family; twelve shards
provide393,216 distinct examples and a27,267,072-token encoding floor. Static uses
the first shard repeatedly; refreshed consumes the fixed twelve-shard stream once.
Actual exclusions/native admission/token capacity remain mandatory; shortfalls
block training without silently changing family proportions, support or quotas.

Prepare selection and assessment first. Each has128 parameter and128 structural
rows, with source/variant/group/finite-window isolation. Structural roles use
disjoint generic families and require two loop-active and two branch-active families,
explicit stage/magnitude quotas and32 rows in each length bin. Source-only feasibility
found counted AST sizes1–18, so this prospective benchmark freezes `long >=16`,
`short <16` rather than the infeasible default32. Cell plans carry the boundary;
historical profiles retain their old definitions. Both structural rosters reserve
large-magnitude loop support before choosing the remaining generic descriptors.

The temporary protocol evaluates each arm's latest available checkpoint using only
selection supervision, four matched controls, raw/rescue diagnostics and one shared
absent-input bank/checkpoint. Partial or unequal training exposure remains descriptive
and cannot support a matched25M-cap claim. No assessment is exposed or initializer
promoted, and no confirmation starts automatically. Full-state cadence remains
unchanged; six learner hours/arm is a ceiling, not a promise of reaching25M tokens.
The complete single-session ceiling is60 hours including any fresh acceptance,
preparation/validation, both learners, generation/scoring, export and recovery.
The operator authorized a longer ceiling where useful; this is an explicit new
campaign budget, not a runtime estimate or an extension of per-portfolio60 seconds.


### Superseding feasible P1 design (T139/T140, 2026-11-02 Pacific)

The v1 roster is retired for future runs: actual first-shard admission was20,634
of32,768, including11 empty families. The new `fresh-pilot-design/v2` retains
A4/B6/C2/D20 and every quota/isolation rule, but freezes coefficients independently
uniform over[-1024,1024] and screens descriptors on16 independent source variants.
No19-difference window (a20-term segment up to translation) may occur in more than
two screen variants. This explicitly conditions the descriptor distribution on
numeric variation; it is not a comparison to the historical[-16,16] distribution.
Both research arms share exactly the new roster, law and quotas. The screening
rule is only a candidate filter, never proof of full admission or semantic novelty.

Before any hardware stage, the temporary helper checks all four protected cohorts
and all12 training shards using the canonical reference admission/quota policy.
Every selected training source is unique across shards; actual encoded loss-target
tokens must cover25M. The report is FEASIBLE, never native/HIP PASS. Canonical native
preparation must still pass and select the same ordered rows before learning.
Native disagreement or any changed selection blocks the campaign.

CPU reference generation uses at most four ordered workers,64 in-flight rows,
explicit deadlines and bounded termination. Native preparation uses up to three
reference workers plus its existing single native client within four qualified CPUs.
One reference execution supplies values/coverage; exact cross-role exclusions run
before native work. Once only training cells remain, retaining training source
hashes suffices for uniqueness: same-role windows never exclude one another.
Protected windows remain exact tuples. GPU work remains serial.

The research preparation allowance is now six hours (including reference/native
work), matching the existing maximum role envelope; acceptance stays ten minutes.
This is an operational ceiling, not a measured native ETA. Full CPU feasibility has
a separate six-hour ceiling. Current-source qualification must be regenerated.


T140 native transport review: surviving candidates use existing `execute` requests
in batches of at most eight, with response IDs reordered/validated and every100-term
result compared independently. Missing/duplicate/foreign results cannot become
admitted rows. The existing four-second allowance per candidate is reserved within
one <=32-second batch and the enclosing preparation deadline. There is no hidden
retry and no extra native worker/process reservation. This amortizes transport
without changing the fitting portfolios or learner workload.


### Gated single-invocation revision (T141)

The v2 runner supersedes the60-hour ceiling with72 hours after builds, including
a separate6-hour source feasibility allowance,6-hour paired smoke allowance and
6-hour native preparation envelope plus validation/cleanup headroom. Both learners
remain25M/6 hours; generation/scoring remain6 hours/arm; finalization keeps one hour.
Failed source feasibility starts no hardware, failed native preparation starts no
smoke learner, and failed smoke starts no full research learner. The wrapper never
skips qualification based on a reference-only result.

The same-model smoke takes one already native-admitted example/family from each of
the first two shards. Batch size stays16. Its target-token cap is first-shard tokens
plus the larger of the next static/refreshed16-row batch costs (<=16,384), ensuring
three updates and actual fresh-row consumption. Learners get900 seconds each.
Canonical finite-weight proofs verify changed finite parameters. One deterministic
selection prompt/stratum exercises all seven existing controls with1800-second
generator/scorer envelopes. All outcomes must be recorded, with valid upstream and
scoring records; negative or correctly censored proposals are allowed. No accuracy
threshold, checkpoint ranking, initializer publication or assessment scoring occurs.
The smoke is separate software evidence and cannot contribute to research results.
