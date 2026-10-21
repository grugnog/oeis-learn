# Complete-screen cost preflight — T095

Status: current-source measured HIP preflight **FAILS**. Prepared inputs pass
checking with512/128/128 plus tiny32 and native admission. All12 fixed-state
batching groups pass parity/deadline checks. The complete-screen estimate is
**47,749.33s** (13h16m) against **5,760s** (96min),8.29x the allowed cost.
The registered7200s budget and20% reserve are unchanged. T095 remains open;
no measured recipe is admitted and learnability has not run. See
[reviewed results and next decision](phase-5-generation.md#reviewed-current-source-hip-run--2026-10-21)
and [the exact receipt](phase-5-batched-hip-review.json).

The helper's `--prepare` invokes the existing pinned bulk fetch, prepare and check
commands with declared workspace caps. Alternatively pass `--inputs PATH` for
an already prepared set. Public checking must verify512 training,128 parameter
development,128 structural development, native admissions, matched views,
exclusions and diverse32. The host sends only a public shape/hash manifest to
the non-scoring measurement role; no scoring truth or final prompts are mounted.

The performance screen projects every scheduled panel and full end-of-screen
control using `fresh.cost_policy.screen_cost`, the unchanged recipe cadence,
3000-update/7200s cap and20% reserve. No denominator, cadence or success threshold
is relaxed. Cost admission requires actual HIP, current image/profile/fixture
hashes, complete supported fit measurements and the matching prepared shape.
A forecast above80% produces no admitted recipe and blocks training. Only a
successful preflight writes `performance/measured-recipe.yaml` before a run.
Changing declared settings requires a new registered recipe, not a budget refund.
This is a conservative sampled cost estimate, not evidence of learned accuracy.

New receipts include measured batch8 latency, singleton tail costs and fixed
calibration/model identities. The coordinator and preflight share batch-cost
accounting; complete prompt counts and per-round tails remain charged. Old
singleton receipts retain their original calculation. Failed mixed-prompt
greedy/stochastic parity, deadline checks or an obsolete measurement protocol
cannot produce an admitted recipe, even if a scalar forecast is cheap.

