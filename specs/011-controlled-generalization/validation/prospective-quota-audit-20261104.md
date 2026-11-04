# Prospective quota audit and proposed scope amendment

Status: T176 evidence reconciled. Proposed amendment below is **not applied**;
the existing256-row run command still fails closed. Do not repeat it unchanged.

## Original evidence

Reviewed `report(6).zip`, SHA256
`c5aef988cef2260e010dd07f31c321b1a1f4281ec728faa02793c5f566b9816d`.
Canonical verification passed:547 payload members,4,829,355 expanded bytes.
This read-only export retains the original FAIL/UNMEASURED workflow. It supplies
the previously missing quota/progress projection, not a new preparation run.
The original ledger reference is
`prepare/cohort/cells/parameter/ledger.json`,744 bytes,
`sha256:4b1d8561f794ec0aca103383ead70f0a90733a83b49164d5102f448d796a0140`.

Five parameter-family quotas are completely unfilled. The deterministic chooser
gives an unfilled family positive priority over already satisfied families. Since
quotas sum to128, these full deficits imply zero admitted candidates for these
families, not just unlucky tie-breaking. All1,681 submitted parameter candidates
were admitted by native validation, with zero rejections. Structural preparation
was not reached; its feasibility remains unmeasured.

Family prefixes below uniquely identify full IDs in the archived design. Each
listed20-term segment depends on only one integer coefficient k in[-1024,1024].

| Family prefix | Required / deficit | Category | Segment indices | Segment formula |
| --- | --- | --- | --- | --- |
|073880ed|4 /4|ambiguous|49–68|(2k+1)n|
|7988b229|3 /3|ambiguous|0–19|(k−1)n|
|99c9a504|7 /7|certified|9–28|kn²+n|
|b44aad9a|3 /3|ambiguous|0–19|(k−1)n|
|faaf9c87|8 /8|ambiguous|0–19|kn²|

Total deficit25:seven certified and18 ambiguous tasks. Each segment has exactly
2,049 possible outputs at those fixed indices. Increasing the number of parameters
elsewhere in a program does not increase this segment's support. The required
exclusion rejects any exact shared20-term window, even if the full100-term
sequences differ. This is consistent with the frozen leakage policy.

Added15 synthetic counterexamples (five families times coefficients−1024,0,1024).
For each, change a parameter outside the bottleneck segment. Independently written
algebraic expectations match both programs' segment; their full sequences differ;
the actual exclusion code reports only `finite_window`. These are generated
fixtures, not original protected examples or evidence of model quality.

## What remains uncertain

The original version did not write per-family exclusion-rule counts. This export
cannot recover counts that never existed. It proves zero survivors from the fixed
reservoir, not that all possible fresh assignments are exhausted, nor which rule
removed each original candidate. The source mechanism makes finite-window support
saturation a plausible explanation; it is not a measured exhaustive-coverage claim.
Do not claim that a larger reservoir cannot ever help. It is simply unsupported
as a reliable fix. More time alone cannot repair a completed finite draw shortage.

The original training design contains12,288 selected examples per training family
across12 shards, but their coefficients are neither necessarily uniform after
exclusion/selection nor independent. Do not infer complete2,049-value coverage
from that count. No assessment payload was read in this analysis.

## Recommended next comparison (requires a scope decision)

Run the existing primary **structural** comparison independently of parameter
cohort feasibility:128 rows across27 unchanged families, comprising68 certified,
26 ambiguous and34 classification-unresolved tasks. Preserve the structural
seed/draw stream, exact family quotas, coefficients, exclusions, retained weights,
eight-attempt/60-second portfolios, controls, bank seeds and completion threshold.
The primary remains certified structural family-macro equal-budget success.
The parameter secondary endpoint must be explicitly DEFERRED, not zero, successful,
or silently replaced by a103-row survivor-only score.

This changes the proposed selection population from256 to128; it does not establish
that structural preparation will pass. Freeze a new design identity and disclosure
before inference. Check all structural quotas/exclusions/native agreement before
either GPU generator launches. Preserve every original training/selection/assessment
exclusion input and keep assessment access inside the custodian. Reuse the existing
compatible qualification via explicit original path, with normal source/image and
resource checks; no image rebuild is justified by this audit.

If chosen, implementation should stay in temporary validation helpers: one Phase B
group for explicit scoped design/denominators and counterexamples, then routine C/D
binding, reporting and a single consolidated command. Tests must reject accidental
parameter inference, missing structural families, changed checkpoint/control budgets
and interpretation of the absent secondary endpoint. Expected generation work is
roughly half the previous256-task comparison (about30minutes per arm), but that is
an extrapolation; retain generous existing limits and make no success guarantee.

Alternative: retain the full256-row population and perform a bounded CPU-custodian
support audit against original exclusions before choosing a reservoir or numeric-law
amendment. This retains the intended secondary comparison but adds analysis/run work.
Broader coefficients would measure numeric extrapolation, a different experiment.
Weakening20-term exclusions or silently dropping only failed families is not proposed.

Validation:62 prospective policy/controller tests passed, including the15 new
independent-formula collision cases. Archive verification and diff checks passed.

No scientific policy, production source, images, seeds or budgets changed in this
audit. The implementation decision remains open; no HIP was run or requested.
