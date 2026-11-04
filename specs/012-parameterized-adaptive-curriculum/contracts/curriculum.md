# Curriculum policy contract — proposed `pac-progress-review/v1`

This is a frozen initial design, not an optimal policy or an implemented API.
Changes before research require a new policy identity; changes after looking at
outcomes are a new experiment. FR-010–020 are the authority.

## Units and role boundaries

One decision selects a family and a batch from its existing gradient rows. Batch
size, tokenizer, padding, optimizer and token charging remain canonical. Family
IDs come from a private hash-bound sidecar retained during preparation; they are
not inferred from hidden structural labels or decoded targets at runtime.

Prepare eight probe examples per training family from a deterministic finite
reservoir of at most 256 draws/family, using the original mechanical grammar and
numeric law. Freeze draw seed/counters with the experiment, separate from gradient
RNG. Require all 32 quotas and native/reference agreement before research. Probes
never enter gradients. Reject exact canonical-source or full authoritative-output
duplicates with gradient rows and detectable final-evaluation collisions inside
the custodian. Shared visible windows within the training role may remain; count
and disclose these overlaps. Do not describe these probes as unseen structural
generalization. Infeasibility blocks the proposed panel; no reseeding or selecting
held-out structural families is automatic.

Choose four review anchors per family from existing gradient rows by a frozen
hash order, covering available declared target/length strata. Record unavailable
strata. Practice may repeat anchors; transfer measurements use the separate panel.
No inferred OEIS family names, external teachers or generated-model demonstrations.

## Decision channels and bounds

Use repeating four-update slots: exploration, adaptive, review, adaptive.

- Exploration samples uniformly over all 32 families using checkpointed seeded
  RNG, independently of loss and model success. Its conditional family probability
  is 1/32; the separate review queue supplies the finite opportunity bound.
- Adaptive slots sample probabilities obtained by normalizing nonnegative priorities
  and deterministically redistributing excess above 0.20. All-zero priorities use
  uniform probabilities. Reject a cap below `1 / family_count`.
- Review slots serve earliest due family, ties by frozen family ID, cycling that
  family's anchors. If none is due, sample uniformly and record fallback.
  Each family has at most one outstanding review obligation; new obligations cannot
  overtake existing earlier deadlines.

The scheduled mixture is exactly 25% exploration, 25% review opportunity and 50%
adaptive slots over full blocks, not necessarily 25% actual due-review updates.
Log the slot's conditional selection probability, reason and actual exposure;
never report the mixture as the conditional probability of a deterministic review.
The adaptive cap is 0.20; a conservative per-family expected mass bound over a
four-slot block is `0.25 + 0.50*0.20 + 0.25/32 = 0.3578125`, allowing one forced
review. Expose actual finite-window concentration as well; this is not a hard cap
on every random realization. Initial readiness rejects empty families/anchors and
an unserviceable due queue.

## Measurements and priority

Every 256 committed updates, and at initial/final eligible boundaries, measure
mean target-token negative log likelihood on the same eight probes per family,
without gradients. Preserve model mode and separate/restore all training RNG;
measurement must not change weights, optimizer, learning-rate schedule or batch
order. Record counts/sums, not means without denominators. Normalization is by
actual non-padding target tokens; A/B loss is not a cross-representation accuracy
metric. B/C definitions and opportunities are identical.

For each family keep at most eight completed panel observations and an EMA with
coefficient 0.25. With positive previous loss `l_prev`, calculate clipped relative
change `d = clip((l_prev - l_now)/max(l_prev, 1e-6), -1, 1)` using matched panel
identities. Define progress `g=max(d,0)` and regression `r=max(-d,0)`. Neither high
absolute loss nor a solver timeout alone supplies progress. Fewer than two valid
observations contributes zero progress/regression and maximum uncertainty.

Every 256 updates, a rotating four-family panel runs one attempt on one probe per
family, with a five-second per-probe portfolio including fitting, inside a reserved
stage envelope. Rotate over the fixed panel deterministically. Record exact valid
execution, finite success, failure and censoring separately. This is a cheap
training diagnostic, not the eight-attempt/60-second structural benchmark. A family
semantic success rate uses aggregate complete observations with actual counts;
no Bernoulli confidence claim treats correlated attempts as independent.

Priority before normalization is:

`(g + r + 0.25*u + 0.25*s) / c`

where `u=1/sqrt(1+n)` for the number `n` of valid panel observations, `s` is clipped
updates since last valid measurement divided by 2,048, and `c` is the family's frozen calibrated
update cost per charged token relative to the median, clipped to `[0.5,2]`.
Absent costs use 1 with an explicit unavailable marker. Freeze rounded cost ratios
in the preflight identity. Live wall-clock measurements are reported but must not
change decisions: timing jitter must not break exact clean pause/resume. This
limits cheap-family preference. Maintain loss regression
and semantic forgetting as separate evidence; a drop in the latest completed
semantic success fraction adds at most 0.25 to the numerator. No missing/censored
result supplies that drop. Cold start uses uniform adaptive probabilities until
valid relative changes exist. Cost, clipping and tie rules are identity-bound.

## Review intervals

Initial family interval is 256 committed updates; minimum 128 and maximum 4,096.
A due review is an actual anchor batch with normal optimizer/token debit. Serving
it schedules the next due update using the current interval. The independent
semantic observation changes the interval: two consecutive completed successes
with no intervening completed failure double it up to the maximum; a completed
failure halves it to the minimum and resets the success streak. A diagnostic
failure/timeout does not count as a semantic failure or success. Intervals are
measured in learner updates, not workstation time. Loss-only improvement is not
semantic mastery and cannot lengthen intervals.

With 32 families and one slot every four updates, an already due family must be
served within 128 additional completed updates. A newly generated obligation may
not preempt an overdue one. The minimum interval prevents sustained demand above
reserved capacity; enforce this invariant and test synchronized due times. The
maximum opportunity gap is 4,224 updates; a run that ends earlier reports remaining
obligations as unserved due to the terminal budget, not a forgotten success.
Observation-driven shortening may pull a due time forward once per completed
semantic observation, while preserving earliest-deadline queue fairness.

## Diagnostic limits and incomplete evidence

Freeze a maximum diagnostic wall-time reservation per arm at preflight; default
20% of the forecast learner-update duration plus measured initialization/final
panel costs. Every invocation also has its own finite bound. Both B/C use the same
reservation and cadence. At the ceiling, skip future optional measurements with
explicit censored events; keep safe exploration/review and last valid bounded
priorities. Two consecutive failed full loss panels or exhaustion before all
families receive a post-initial observation makes adaptive evidence INCOMPLETE.
No automatic cap increase, hidden extra solver attempts or fallback mastery.
Final export separates operational PASS from scientific interpretability.

## Required independent counterexamples

Cold starts and equal priorities; improving versus hard-stuck families; noisy loss;
memorized anchors but regressing probes; costly versus cheap families; missing and
censored panels; simultaneous due queues; interval bounds and terminal truncation;
conditional versus mixture probability; cap redistribution; nonfinite observations;
aggregate counts; duplicate event IDs; observations from a forbidden role; restored
RNG/order; exact token cost of review. Use hand-calculated outcomes and independent
fixtures, not production helpers to compute their own expected answers.
