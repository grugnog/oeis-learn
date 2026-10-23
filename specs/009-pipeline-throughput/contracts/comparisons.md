# Contract: finite workload and adoption comparisons v1

**Status**: Prospective defaults. Freeze the resolved machine-readable manifest before candidate measurements. Historical profile results motivate this design but are not confirmation samples.

## Workload manifest and calibration

Every workload lists exact source/policy/weights/input hashes, prompt and attempt IDs, expected work, seed roster, order, cache state, instrumentation, deadlines, resource envelope and required output hashes/denominators. Use the existing completed generic bigint checkpoint as read-only diagnostic weights; no T093 model conversion or T096 resume. All disposable learning measurements restore the same full diagnostic state between arms and cannot write to the parent run.

Phase 1 performs one baseline cold run, one unscored warmup and three baseline-only calibration repetitions of the declared short fixture set. Freeze the derived finite duration reservations, workload strata and five-pair order in a child comparison contract **before observing candidate performance**. Calibration can reveal that the scope cannot fit its cap; report BLOCKED and propose a separately identified scope revision. Do not silently reduce counts, extend deadlines or search for easier prompts after a candidate fails.

A calibration contract may contain the fixed deterministic rule for deriving bounds, so one combined command can calibrate, seal the child contract and continue without an extra manual launch. The rule is: for each measured complete stage, reserve the larger of twice the maximum observed elapsed time or its declared minimum; round up to whole seconds. Never exceed the phase/role ceilings. Missing or incomplete baseline measurements cannot produce admission. This conservative envelope is not a statistical p95 estimate from three observations.

### Required components

| ID | Frozen work | Interpretation |
| --- | --- | --- |
| `cache-replay` | Batch 1 and 8, supplied token streams at 1/32/128/513/1024 positions; mixed lengths, inactive/finished/reordered rows, supported positional/cache modes | Forced replay measures cache work, not generated-program success. Batch16 is a separately named later policy case |
| `generation32` | Existing diverse32 generic training prompts, stable groups of eight; correct 8 attempts, shuffled 1, absent 1, template 8 per prompt; greedy first, then existing per-attempt stochastic seeds; 1,024-token cap | 576 scheduled attempt records, matching the reviewed reduced diagnostic scope; full bookkeeping for every arm, including unstarted attempts |
| `native-corpus` | Frozen existing arithmetic/fitting fixtures plus visible candidates from the baseline: small/large integers, easy/slow/unsupported fits, batch1 and larger batches; raw, solve, production and independent-reference outcomes separate | No model needed; include legal 64/256/512/1024/4096-bit inputs and supported construction of up-to-16,384-bit intermediates. Expected resource exclusions remain labelled |
| `updates-validation` | 500 disposable updates on the same 128 training rows, update batch8; 256 teacher-forced validation rows; identical initialization/optimizer/RNG/data order in each arm | Actual finite HIP parameter updates; this does not train a reusable model or qualify learnability |
| `operations-cost` | One full diagnostic checkpoint write/load, bounded interruption/recovery, selected-summary/export and role launch/cleanup | Required cost components, separate from kernel time; checkpoint contains all state but is labelled diagnostic |
| `complete-control-cost` | Three fitted neural conditioning portfolios with eight attempts each plus eight templates for all32 prompts; complete scheduled panel/end-screen shapes for proposed research | 1,024 scheduled attempt records in the32-prompt control scope. Needed for cost admission; reduced576-attempt diagnostics cannot substitute for a full transfer schedule |

For each measured scalar, retain short/long output, mixed length, easy/difficult fit, small/large integer, batch1/batched and cold/warm strata. Freeze membership from baseline data/fixtures. If free generation supplies no long output or difficult fit, report that coverage gap and use separately labelled replay/fit fixtures; do not pretend forced inputs establish model-generated coverage. A full research forecast remains blocked if a material production stratum has no representative measurement.

## Intervention sequence and bounded candidate set

1. **Decoder bookkeeping**: baseline and candidate at batch8, same attention backend, precision, grammar, weights, seeds and v1 scheduling. Free-generation token/source equivalence is mandatory. Because v1 selection uses wall cost, any early whole-pipeline v1 selection differences are reported as timing-sensitive diagnostics, not transparent end-to-end equivalence. Default adoption waits for the common v2 policy baseline below.
2. **Policy-only**: serial v1 versus serial v2 on identical complete candidate records; no scheduler change. Freeze v2 comparison authority and record differences without choosing it by hidden scores. Qualify its independent admission and deterministic representative semantics.
3. **Native slots**: compare 1 versus 2 versus 4 using the same v2 policy, fixed candidate stream, global cache limits, matching cache states and workload. Select a candidate by the preregistered rule: fastest eligible median, then fewer workers on ties within 5%; no eight-worker expansion in this manifest.
4. **Overlap**: fixed decoder batch8 and selected worker count; serial versus bounded overlap at identical policy/limits. Start with one decoder-batch lookahead. Test two-batch lookahead as one additional named candidate, not every possible queue size.
5. **Batching**: with selected bounded overlap settings, compare batch8 with batch16 under the new generation/scheduling contract. Batch1 is a required regression diagnostic. Changing training batch size is excluded. Larger batching is not described as numerically or temporally equivalent without evidence.
6. **Memory/cache capacity**: the workstation default may use the user-authorized90% aggregate memory ceiling. Retain16 GiB only as a historical control. Compare16/64/workstation90 envelopes with identical work and settings where baseline calibration indicates pressure, then compare the declared larger result-cache candidate separately. If batch16 improves throughput and calibration predicts batch32 fits the authorized envelope with at least10% of that envelope unreserved, one predeclared batch32 exploratory branch is allowed; otherwise report it unneeded/unavailable. All branches and their cost reservations are included before execution. Do not expand model/update batch size or queues merely to consume RAM. Additional cache/batch hypotheses require a new manifest, not an open-ended search.
7. **Confirmation**: choose at most one combined configuration prospectively from calibration/exploratory outcomes and freeze it. Compare that candidate with the corresponding serial v2 baseline in fresh confirmation runs. Both use the same arithmetic, weights, seeds, source/policy versions and telemetry mode; component controls preserve attribution. Do not choose among multiple confirmation winners.

Exploratory timing is one unscored warmup plus two paired runs per named candidate, under the owning phase cap; it is not adoption evidence. Failure of a correctness gate excludes a candidate. Nonnegative/noisy exploratory gains may be carried forward only if the frozen selection rule supports it; no candidate may bypass exactness/resource prerequisites. Candidate selection uses visible diagnostic performance, not hidden continuation outcomes. If no candidate is supportable, retain the baseline and publish no-adoption.

## Fixed completed work

Run every manifest-listed attempt, with identical input tokens for cache replay and identical prompt/seed inputs for free generation. Completion means the scheduled operation reached its declared terminal outcome, which can be conclusive invalid/unsupported/non-match; timeout, unknown, cancellation and unstarted operations are incomplete work and cannot support latency-equivalence claims.

For fixed-work free-generation diagnostics, allow up to 180 seconds per group and a 180-second per-prompt ceiling within the frozen scope; each native visible service retains its original maximum four-second allowance and reference cap. A source that intrinsically exceeds native limits remains a separately reported stress/coverage case. Fixed-work inference uses a separately named diagnostic policy to avoid silently inheriting the production60-second decoder clamp. No such receipt qualifies production60-second success. Calibration must establish a complete fixed-work confirmation subset before candidate measurement; excluded stress records remain in a separate fixed-denominator stress report, with exclusions frozen and explained.

The primary adoption workload combines `generation32` on the frozen complete subset, the required native/independent scoring work, `updates-validation` and `operations-cost`. Export counts as required work; image build is excluded from this repeated warmed pipeline metric but included in full invocation/cold-start reports. Record the fixed subset size alongside the original32 denominator and the full-workload fixed-budget results; gains cannot be advertised as unconditional full32 speedups if fixed-work calibration excludes cases.

No internal timing subtraction may remove an unfavorable validation/update/export stage from the primary elapsed interval. Stage fractions are explanatory only. Forced cache replay and a native-only speedup cannot qualify whole-pipeline adoption.

## Fixed-budget assessment

Use the full32-prompt denominator and original60-second per-prompt ceiling, at most eight attempts and1,024 tokens under the explicit group/global allocation contract. Freeze group start order and each group's maximum allowance; a group may not borrow saved time unless that redistribution policy was prospectively declared as a separate comparison. Include generation, waiting, native fit and visible reference inside that allowance. Hidden continuation scoring has an independently fixed allowance and cannot select candidates.

Report verified successes per total end-to-end minute, successes/32, complete evidence/32, all scheduled/started/completed attempts, failure reasons and stratified distributions. Fixed-budget throughput cannot stand in for equal-work latency. Require no lower median full-denominator success or completeness count than the matched baseline across the five pairs; any worse paired case is disclosed. A significant or unexplained coverage loss prevents adoption even if the fixed-work subset is faster. All admissions must independently verify.

## Confirmation rule and uncertainty

Use one warmup excluded from headline results, then exactly five independent reset paired repetitions per required cold/warm confirmation scope. Pair order is fixed `AB, BA, AB, BA, AB`; seed/workload membership is identical within each pair and registered across pairs. Reset weights, RNG, caches and artifacts according to the declared cold/warm policy. Never reuse a favorable exploratory pair as confirmation.

Primary measure: complete required pipeline elapsed time. Define pair gain `1 - candidate_seconds / baseline_seconds`. Adoption requires:

- all five pairs complete the same frozen work with matching candidate/selection/admission outcomes under v2;
- each of the five gains is at least 0.10;
- every paired mandatory stratum latency ratio is at most 1.05, with identical work;
- all correctness, resource, current-source/device and fixed-budget coverage gates pass.

Report median, minimum and maximum paired gain, absolute durations and stratum counts. This conservative all-pairs rule is the uncertainty rule; do not advertise population confidence from five workstation repetitions. If medians look favorable but the rule fails because results vary, report INCONCLUSIVE; a consistent shortfall is NO_ADOPTION. No adaptive repetitions, outlier dropping, retries that retain only favorable pairs or threshold changes are allowed. A future revised comparison requires a new predeclared identity and keeps the original result.

## Numerical and behavioral parity

Retain existing CPU cached/full logits tolerance `atol=1e-5, rtol=1e-4` and HIP bigint tolerance `atol=2e-5, rtol=2e-4`; disclose every compared tensor shape/dtype/device. Tolerant logits alone are insufficient: complete greedy/stochastic tokens, decoded sources, prompt/RNG mapping and admission outcomes must match for equivalent-work optimization. A difference blocks equivalence until fixed or explicitly treated as a separately qualified policy change. Keep finite-logit failure detection and inactive-row state preservation.

Test positions through1,024, heterogeneous lengths, repeated/reordered indices, EOS at capacity, all-inactive rows, weight/prompt/codec/dtype/mode invalidation and supported nonproduction positional/backend combinations. Require actual HIP for device claims and finite forward/backward parameter changes; CPU fallback or an allocated GPU tensor is not sufficient.

## Instrumentation and projection

Telemetry is1 Hz, with CPU/GPU/memory/queue counters and missing-sensor reasons. Before confirmation, one ordinary versus telemetry paired complete-work calibration checks observation overhead; if elapsed overhead exceeds5%, report the limitation and use ordinary runs for primary timing with a separate observational run, under a predeclared measurement policy. Both candidate and baseline use the same mode. Detailed per-token/byte/growth probes are off by default.

When investigating a concrete cause, allow at most one update trace and one eight-token decode trace, each40 MiB, within128 MiB total trace allowance. Forced long replay can reach the desired offset without recording all earlier steps. Trace time is excluded from confirmation. A trace failure does not invalidate ordinary timing when tracing is optional; a missing mandatory actual-device update does.

Do not synchronize the GPU around every overlapped stage in ordinary mode. Use monotonic host intervals and bounded device events collected at existing drain boundaries; report causal attribution limits and compute union intervals for overlapped work. Retain an explicitly named synchronized diagnostic mode so removal of legacy timing barriers can be assessed separately from decoder/scheduler changes.

For a new learning scope, project every scheduled component: updates, teacher-forced validations, all diagnostic/control attempts, native/independent scoring, checkpoint/export, startup and cleanup. Use measured representative workloads with conservative observed envelopes; label incomplete components unknown and block admission. The total must fit within80% of the new scope's preregistered finite allowance. A32-prompt or reduced-control receipt cannot admit the128+128 full transfer screen; changing weights/length distributions and extrapolation limitations remain explicit.
