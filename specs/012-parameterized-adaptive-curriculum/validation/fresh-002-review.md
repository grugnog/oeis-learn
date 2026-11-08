# Fresh structural comparison: key learnings

Reviewed 2026-11-07. Report, scoring and generation producer hashes/seals agree;
all 84 paired summaries reproduce. The enlarged reservoir admits the complete
fixed cohort. The result remains scientifically **INCOMPLETE**: D completes
64/68 certified cases against the unchanged 65/68 floor. No scores, checkpoints,
qualification, inference limits or frozen experiment choices have changed.

## Scientific signal

| Certified arm / selector | Successes / 68 | Completed / 68 |
|---|---:|---:|
| B / original or coverage | 37 | 66 |
| D / original | 50 | 64 |
| D / coverage | 53 | 64 |
| Generic templates / either selector | 55 | 68 |

D's certified family-macro advantage over B survives worst-case censoring on
this fixed cohort: +11.67 to +20.06 percentage points under the original selector,
+14.85 to +23.23 under coverage. These are descriptive censoring bounds, not
confidence intervals or across-training-seed uncertainty. Gains concentrate in
three families; unresolved-category outcomes favor B.

Coverage adds three observed D successes with no paired regressions, but its
family-macro effect bounds cross zero. The eight fixed generic templates are a
real visible-fit baseline. D coverage solves six completed cases they miss;
templates solve five completed cases D misses, plus three where D is censored.
This suggests complementary candidates, not an established overall learned
baseline win. Both B and D already use placeholders and fitting; their training
allocation differs.

## Throughput findings

Shared sampling time is counted once per batch. Concurrent fitting intervals
are merged rather than summed as wall time.

| Measured scope | B | D |
|---|---:|---:|
| Four-control portfolio span | 641.5 s | 1,082.8 s |
| Neural decode calls, once per batch | 450.4 s | 954.7 s |
| Fitting service interval union, all four controls | 185.2 s | 122.0 s |
| Batches reaching the 1,024-token cap / neural batches | 0 / 768 | 41 / 768 |
| Time in those cap-reaching batches | 0 s | 393.3 s |
| Full-horizon scoring, both selectors/all seven controls | 5.19 s | 5.29 s |

D spends 88% of the measured portfolio span in decoding, versus B's 70%.
Only 5.3% of D's neural batches hit the token cap, but they account for 41.2%
of its decode time. Typical batches take about 0.65 seconds; cap-reaching
batches take about 9.6 seconds. One unfinished row holds its batch open.

The configured batch size is 16, but the frozen drain policy caps actual
neural batches at the four native-worker slots and waits for native work before
more decoding. All observed neural batches contain four rows. Larger hardware
capacity alone therefore does not increase this evaluation's batch size or
restore decode/fitting overlap. Changing that schedule requires an explicit
qualified comparison under unchanged deadline/accounting rules.

Native dispatch queues are typically about two milliseconds. Runtime reuse is
active; the missing primary fitting cases time out inside SMT at about 1.4
seconds of search. Scoring and ranking costs are small. These findings identify
sampling tails and scheduling as throughput targets, while SMT remains a
completion issue. They do not establish GPU saturation, memory-bandwidth limits,
thermal throttling or complete campaign wall time: decode includes host grammar
and sampling work, and model load, bank work, role startup and export lie outside
the measured four-control portfolio span.

## Concrete reliability findings

D's four incomplete certified portfolios comprise two fitting-timeout cases,
one token-cap case and one grammar-dead-end case. Both selectors have the same
upstream failures. B's two incomplete cases are the same fitting-timeout cases.

The two fitting cases repeat two factorized-quadratic query forms with no
integer solution. The existing small-modulus checks are inconclusive; an
independent modulo-11 enumeration rejects them using only visible inputs.
This motivates bounded algebraic reduction or stronger necessary checks, with
replayable UNSAT evidence. Simply extending the modulus list can exceed the
existing residue-assignment cap, so it is not yet a validated implementation.

Replaying every exported grammar-dead-end prefix places it at nesting depth 32.
The mask can admit an operator whose required child cannot fit within that
limit. A small synthetic prefix reproduces this without model weights. A
grammar-completion feasibility repair is a concrete next engineering target;
it must preserve the accepted language and receive new qualification. The
token-cap sequences show continued program growth, without long numeric-literal
tails. They do not establish that increasing the token budget improves results.

## Original training costs

The already supplied training producers also verify against their original
workflow. Each arm consumes approximately 25 million charged tokens.

| Arm | Learner hours | Charged tokens / learner second | Update timer / learner time | Checkpoint timer / learner time |
|---|---:|---:|---:|---:|
| B | 2.69 | 2,586 | 70.7% | 0.11% |
| C | 2.57 | 2,699 | 66.6% | 0.13% |
| D | 2.63 | 2,644 | 65.7% | 0.11% |

Checkpoint writes are not the dominant measured training cost. The remaining
29–34% is outside the update timer; the available counters do not separate all
observation, data, journal and orchestration costs. Charged-token throughput is
not a useful-token or accelerator-utilization measurement.

## Recommended next step

Prioritize grammar feasibility and bounded fitting reliability, using existing
training-only probes and fixed candidate diagnostics before another research
run. Preserve the original candidate inputs, seeds and service allowances in
any replay, and retain this result as observed INCOMPLETE evidence. Investigate
larger batches/overlap separately; do not silently change the frozen schedule.

D remains the more promising trained arm for certified structure. Reliability
and a prospective comparison against the strong template baseline come before
HPO. Any combined learned/template portfolio needs a frozen visible-only
selection rule and total budget; complementary observed successes are not an
achievable accuracy estimate for such a selector. Future timing work should
capture the complete existing role/attempt/clock evidence together with any
newly measured hardware counters, avoiding repeated small data requests.
