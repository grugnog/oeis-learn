# Curriculum contract — proposed `pac-exp3s-leitner/v1`

This design replaces the unimplemented `pac-progress-review/v1` draft. The
selected kernel is Graves et al. (2017), Eq. 1 with beta=0; the surrounding block,
measurement and bounded review choices below are explicit project adaptations.
No scientific run is frozen by this document. Freeze the complete manifest before
execution; no inherited theoretical guarantee is asserted for the combined policy.

## 1. Roles, finite data and clocks

Retain the original 32-family gradient pool and private hash-bound family sidecar.
Prepare eight separate training probes per family from a finite reservoir of at
most 256 draws/family, with seed/counters frozen before research. Require every
quota and native/reference agreement. Reject exact canonical-source/full-output
duplicates with gradient rows and detectable final-evaluation collisions inside
the custodian. Count permitted shared visible windows within training support.
These probes are never gradient examples or unseen structural evaluation. Reuse
across rounds is adaptive validation; do not claim fresh-IID or unbiased population
progress. Probe infeasibility blocks research without automatic reseeding.

Choose four distinct anchors per family from existing gradient rows by frozen
hash order over available target/length strata: 128 bounded review items. Record
missing strata. An anchor remains eligible for ordinary practice; its actual
exposure must be recorded even when not selected by the review queue.

For C and optional D, ordinary rows use a seeded Fisher–Yates permutation per
family, consumed without replacement then reshuffled at wrap. Derive independent
per-family sampler streams from the frozen initializer seed, family identity and
versioned stream label, identically across C/D; keep their state separate from
bandit/scaler/diagnostic RNG. Freeze exact implementation/runtime and retain RNG,
permutation epoch and cursor. B keeps its original fixed stream. Review filler
uses that family's sampler but skips the designated anchor and already included
row IDs, advancing the cursor for skips without claiming training exposure. Require
enough distinct rows for the declared batch size. Record other anchors incidentally
trained in filler. This changes selection/order, never target eligibility or pool.

Separate counters: committed optimizer updates, bandit action/block IDs, valid
kernel updates, completed review services, diagnostic events and charged target
tokens. Review intervals use committed optimizer updates; kernel sharing uses only
valid kernel updates. Wall-clock jitter does not select families or change queues.

## 2. Published bandit kernel and exact defaults

For K>=2, normalized positive weights q and selected family a:

- Selection: `p_i = (1-epsilon)*q_i + epsilon/K`.
- Observed signed reward r: `rhat_i = r*1[i=a]/p_a`.
- Reward step: `v_i = q_i * exp(eta*rhat_i)`.
- Sharing: `z_i = (1-alpha)*v_i + alpha*sum(v_j for j != i)/(K-1)`;
  normalize z to obtain the next q.

This is the Graves fixed-share variant, not the different Auer Fig. 6 share
formula. Initialize uniform q and valid-update count u=0; set `alpha=1/(u+2)`
for the next valid reward, then increment u. Thus first alpha is 1/2, not 1.
Use stable log-domain arithmetic, checkpointed sampling RNG, and deterministic
family order. Reject nonfinite inputs and unsupported K=1 rather than silently
substituting a different policy.

Defaults: epsilon=0.25, eta=0.001, beta=0. Epsilon deliberately preserves the
project's 25% exploration mixture on bandit draws; it differs from the paper's
experimental epsilon=0.05. These constants are not claimed optimal. For K=32,
`p_min=1/128` and `p_max=97/128`. Do not retain the old arbitrary 0.20 cap or
custom progress/uncertainty priority; either changes this kernel. Report actual
finite-window concentration separately from these probability bounds.

Store the probability at selection. Never divide by a later probability, the
review/bandit mixture fraction or a without-replacement probability. Use one
family draw with replacement per block. Duplicate feedback IDs cannot update
weights twice. A deliberately observed zero still invokes sharing; missing
feedback invokes neither reward nor sharing, and does not increment u.

## 3. Credit assignment and bounded reward observation

A bandit action is a contiguous block of eight ordinary optimizer updates on the
selected family. Choose a rotating two-example panel from that family's eight
training probes before the block; evaluate exactly that panel before and after,
in eval/no-gradient mode with training RNG and model mode preserved. Record
non-padding target-token loss sums/counts and paired panel IDs. No review, another
family's update or external optimizer mutation may intervene between the two
observations. Rotation is deterministic and checkpointed.

Raw gain is `g = mean_loss_before - mean_loss_after`. Preserve its sign: falling
performance is not positive reward. Work-adjusted gain is `x=g/charged_block_tokens`.
The denominator is positive actual gradient-token cost, a deterministic work proxy;
it is not elapsed time or total hardware cost. Record all actual compute/I/O costs
separately. No highest-loss reward, absolute slope bonus or semantic-success bonus.

Use the paper's past-reward 20th/80th-percentile scaling idea, fully specified here:
keep a bounded 4,096-value uniform reservoir of valid prior x values with separate
checkpointed RNG and total-seen count. Quantiles linearly interpolate sorted values
at index `(n-1)*fraction`. Before adding the current x, clip/map it using those
quantiles using `r = 2*clip((x-q20)/(q80-q20), 0, 1)-1`. If fewer than five prior values exist or the quantiles coincide,
use r=0 with an explicit scaler-warmup/degenerate marker, then add the valid x.
Reservoir update: the nth value is appended while n<=4096; otherwise draw integer
j uniformly from [0,n-1] and replace slot j only if j<4096. Freeze float format,
quantile rule, ordering and reservoir identity. The scaled reward is relative:
a positive r does not necessarily mean positive raw learning gain; export both.

Finish observation and one kernel update before selecting another bandit block.
All-family panel trends spanning other updates are monitoring only. This measures
local within-family improvement and may miss cross-family transfer or favor
relearning; the common family panel and final structural evaluation detect those
limitations. No claim of globally optimal curriculum follows from the kernel.

On a missing pre/post observation, record CENSORED and retain actual gradient
updates/debits; do not fabricate r=0 or add a value to the scaling reservoir. Keep
q/u unchanged while action count advances. Two consecutive censored bandit rewards,
or required reward-budget exhaustion, pause further adaptive training and export
INCOMPLETE evidence. Do not silently continue an indefinitely frozen policy.
Clean pause/resume continues a pending observation/block before a new decision.
A terminal token-limited partial block receives no complete-block reward; record
its actual updates, tokens and truncation. Do not repeat already committed updates.

## 4. Bounded per-example Leitner adaptation

C repeats a 32-update cycle: three eight-update bandit blocks, then one eight-update
review block. Each review update may service a different earliest-due anchor;
select by `(due_update, anchor_id)`. Put the designated anchor once in the batch
and fill remaining positions from its existing family using the declared ordinary
row sampler. Do not pad the batch with duplicate copies merely to amplify review.
Review gradients are charged normally. No due item means a recorded uniform-family
fallback update, not a fictional review or bandit reward. Review/fallback updates
never update EXP3.S weights, its valid-update clock or reward reservoir.

Thus 75% of full-cycle updates are bandit-directed and 25% are review opportunities,
not necessarily 25% due reviews or tokens. The uniform component of bandit draws
alone supplies 18.75% of full-cycle update mass, not 25% of the entire campaign.
There is no separate exploration lane. Observe actual tokens/time by channel.

Each anchor has `acquired`, queue q=0..4, last committed exposure, due update,
last grade/event and one pending obligation. Initial never-seen anchors are
acquisition-due at update0. Queue intervals are `512 * 2^q` updates: 512–8192.
First exposure acquires the anchor at q=0 without retention success/promotion.

For an acquired designated anchor, observe pre-update teacher-forced argmax
agreement with every authoritative target token (excluding BOS/padding, including
EOS) under the frozen eval/codec policy. Complete agreement is **target recall**;
other finite complete predictions are a target-recall failure. This is not a test
that every semantically equivalent program would fail, and not family mastery.
No full generation/fitting is required for this grade. After its successful real
optimizer update, a valid success promotes by one queue, a valid failure demotes
to q=0, and a missing/censored grade leaves q unchanged. Set last exposure to the
committed update and due to exposure+interval. Grade and queue change cannot survive
a failed optimizer update. Incidental ordinary exposure also resets the exposure/
due clock after commit, but supplies no invented grade or promotion.

The queue contains at most 128 obligations. Every 512 consecutive completed updates
contains 128 review slots, so one service per slot gives a conservative due-delay
bound of 512 updates under earliest-deadline ordering; a newly inserted obligation
cannot overtake an earlier overdue item. Incidental exposure can discharge/reset
an obligation because the anchor was actually practiced. The maximum interval plus
service delay is 8,704 updates while training continues. This bounds exposure or
service, not necessarily a graded review when incidental exposure keeps recurring.
Terminal truncation and pending acquisitions must be reported. Validate all 32
cycle offsets, simultaneous due times, repeated failure and incidental rescheduling.
No claim of the old family-only 128-update delay remains.

## 5. Comparable diagnostics and cost gates

Retain a fixed common eight-probe-per-family loss panel at initial/final and every
256 committed updates. Sparse semantic diagnosis uses a rotating four-family panel
at that cadence, one attempt/probe with the existing proposed five-second portfolio
inside a generous process envelope. These reports distinguish transfer, target
recall, finite semantic success and censored execution. They never promote anchors
or retrospectively update bandit weights.

B/C and optional D share definitions and the same common monitoring panels/cadence.
Policy-required observation counts differ and must be separately charged/disclosed. C's selected-family reward panels are policy-conditioned; they are
not paired final evaluation. B uses a separately seeded uniform-family pre/post observer at each of C's three
nominal bandit-block positions in every 32-update cycle, without changing B's fixed
gradient stream. In the fourth block, B has up to eight shadow anchor observations
in frozen round-robin order; C grades actual acquired due anchors when available.
B has no scheduled anchor gradients or queue-directed changes. These are aligned
bounded opportunities, not guaranteed equal realized grade counts or cost. D has
four bandit reward blocks per cycle and no scheduled review grades. Its additional
reward observations are an intrinsic policy cost; do not claim identical total
diagnostic intervention or compute across B/C/D. Freeze observer schedules and
charge their actual work; historical A receives no invented missing curves.

Preflight forecasts the **complete** cost of two probe evaluations per bandit
block, pre-update anchor grades, common loss/semantic panels, RNG/mode restoration,
publication and cleanup on representative production shapes. Default diagnostic
reservation is 20% of forecast update duration plus measured initial/final overhead;
it must be increased and frozen before research if representative measurements
require more. This is an explicit work allocation, not a short process timeout.
Optional monitoring can be censored at its own ceiling; required reward observation
cannot be silently removed. Two full semantic rotations without a completed result
for a family make its retention evidence incomplete. Two consecutive review blocks
with acquired due items but no valid grades pause with incomplete review evidence.
Software readiness and scientific interpretation stay separate.

## 6. Independent conformance and failure cases

Use published equations and hand-calculated expectations, not production output as
its own oracle. For K=3, epsilon=1/4, uniform q, eta=log(2)/3, choose arm1:

| Observation | Expected next probabilities |
| --- | --- |
| r=1, first valid update alpha=1/2 | (35/96, 61/192, 61/192) |
| r=-1, same initial state | (37/120, 83/240, 83/240) |

For q=(0.6,0.3,0.1), observed r=0 and alpha=1/4, next q is
(0.5,0.3125,0.1875). A missing reward must leave the original q unchanged.
Also test extreme logweights, denominator source, fixed-share indexing, reward
signs, scaler warmup/ties/outliers/reservoir restore, delayed/duplicate events,
forbidden-role feedback, interrupted pre/block/post observation, clipped terminal
exposure, anchor acquisition/incidental practice and full checkpoint equality.

References and evidence limitations: [research.md](../research.md). No optional
RbF fitting, MIR hypothetical updates, FSRS parameters or legacy heuristic mixing
belongs in this first policy.
