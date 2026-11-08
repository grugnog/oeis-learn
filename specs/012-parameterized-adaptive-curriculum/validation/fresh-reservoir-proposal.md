# Verified all-window support and proposed reservoir amendment

Status: **adopted for implementation by the operator on 2026-11-07**.
The evidence and proposal below remain the decision record.
`pac-selector-fresh-001` remains BLOCKED. The separately versioned implementation,
qualification gates and exact commands are in [the amendment handoff](reservoir-amendment.md).

## Verified diagnostic result

The uploaded `pac-selector-support-002/report.zip` has SHA-256
`2cb2ae97ddc4098903e639ef53d6db699c28081568180bd1a73e152540a0db55`.
Its canonical result seal verifies:
`sha256:231c3273b96a7cc513cbc7b9f8939575153896cdf712b18601d6f823675e35b1`.
The reported helper digest
`sha256:33c1423edc1b05c10a7c3b6ebc114ae6e7038ff3f5bb525b5bc79d0c7080764b`
matches the reviewed code. The original frozen-plan identity and byte hash are
unchanged. All 18 input-cell records are present; each cumulative decrease and
sum of newly blocked assignments independently reconciles to the final counts.
The read-only census completes in 35.837 seconds, with no training or model
evaluation and no qualification/cohort authority.

| Family prefix | Category | Window-free assignments | Distinct window-free outputs | Single-draw window survival |
|---|---|---:|---:|---:|
| `3f7f434b8e62` | certified | 6 / 2,049 | 6 | 0.293% |
| `5c475cecf9da` | certified | 98,334 / 4,198,401 | 98,334 | 2.342% |
| `83a8423db5b0` | ambiguous | 63,406 / 4,198,401 | 63,406 | 1.510% |
| `ee89096d39c3` | certified | 82,537 / 4,198,401 | 416 | 1.966% |

Thus all four families retain support under the finite-window rule. This rules
out full-domain window saturation as the explanation for the observed zero
availability. It does not prove that the remaining assignments pass every other
admission check, nor that the actual fixed draw prefix contains enough of them.
The first family is especially scarce: only six numeric values survive.

The n-(p0+p1) family has 82,537 parameter assignments but only 416 distinct
output sequences. Assignment counts use the original independent-uniform
parameter law and its triangular sum multiplicity; the output-class fraction
416/4,097 is not its per-draw survival probability. The frozen admission rule
requires distinct source programs, not a new within-cohort semantic equivalence
rule; no such rule is introduced by this proposal.

The observed PAC cohort further reduces the first family's window-free support
from 63 assignments to six, after the original cells and training refreshes have
been scanned. This is expected freshness conditioning rather than model outcome
feedback. Incremental attribution depends on the declared input scan order.

## Why the original reservoir was inadequate

As a planning model, suppose each draw is an independent uniform assignment and
only the window exclusion matters. For a domain of T assignments, M surviving
assignments and n draws, the probability of at least two distinct survivors is

\[
1-\left(\frac{T-M}{T}\right)^n
-M\left[\left(\frac{T-M+1}{T}\right)^n
-\left(\frac{T-M}{T}\right)^n\right].
\]

For a quota of one it is simply 1-((T-M)/T)^n. Source uniqueness matters for
the six-assignment family, so treating any two successful draws as distinct
would overstate its chance of filling the quota.

| Family prefix | Required tasks | Window-only probability of filling quota in 64 draws |
|---|---:|---:|
| `3f7f434b8e62` | 2 | 1.290% |
| `5c475cecf9da` | 1 | 78.059% |
| `83a8423db5b0` | 2 | 25.184% |
| `ee89096d39c3` | 2 | 35.920% |

Under an additional independent-family assumption, the joint planning chance is
about 0.091%. These are calculations under a random-draw model, not confidence
intervals or guarantees for the actual deterministic seeded stream. They ignore
other admission failures and cross-family source collisions, so they are
optimistic about complete admission. They explain why repeating a small draw
is not an adequate response to the failure.

## Proposed separately frozen amendment

Increase only four fixed draw-prefix lengths:

| Family prefix | Existing draws | Proposed draws | Window-only quota probability |
|---|---:|---:|---:|
| `3f7f434b8e62` | 64 | 4,096 | 99.976% |
| `5c475cecf9da` | 64 | 512 | 99.999% |
| `83a8423db5b0` | 64 | 512 | 99.634% |
| `ee89096d39c3` | 64 | 512 | 99.957% |

All other draw counts remain as frozen. Total draws become
2,288 - 4*64 + 4,096 + 3*512 = **7,664**, below the existing 8,192-identity
reservoir bound. Under the same optimistic planning assumptions, the four-family
joint chance is about 99.567%; this remains an estimate, not a promised outcome.

The proposed change retains the original numeric law, random seed and stream
construction/capacity. Existing prefixes must reproduce byte-for-byte; longer
reservoirs extend those prefixes, never choose a new seed or retry until scores
look favorable. Existing positive availability in other families is retained in
their unchanged prefixes. The failed admission produced no model scores, so
this proposal is based on mechanical feasibility rather than measured success.

Preserve the original 128-task allocation, 23 positive-quota families, category
counts 68/26/34, all exclusion inputs/rules, selector, checkpoints, controls,
visible/horizon/inference budgets, and certified 65/68 completion floor. There
is no retraining, HPO, quota transfer or family removal. Fixed admission and
subsequent paired B/D evaluation remain separate; a further deficit stays BLOCKED.

Implementation requires an explicit versioned amendment and a new frozen output
directory, suggested `pac-selector-fresh-002`. It must verify the old blocked
parent and successful probe bindings, pin the exact draw-count diff, reject
unlisted changes, and prove prefix identity in tests before any new execution.
Any source/image compatibility or probe reuse must be explicitly justified;
existing checks must not be bypassed by editing the old source hash or plan.
New preparation resource measurements must confirm the enlarged index remains
within the existing memory/work envelopes before model evaluation.

**Implementation boundary:** operator approval authorizes this specific larger
reservoir, not mutation or rerun of the old plan. Activation still requires the
new source-bound amendment command and fresh qualification documented in the
handoff; no new cohort/model outcome follows from approval alone.
