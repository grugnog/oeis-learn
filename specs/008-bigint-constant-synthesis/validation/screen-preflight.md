# Scope-identified cost preflight — T095

Status: **T095 complete; diverse32 admitted for T096 only**. The reviewed
`008-phase5-scoped-20261021T115212Z` run passes all nine HIP tests and twelve
fixed-state generation parity/deadline groups. Prepared inputs retain complete
512/128/128 counts, tiny32, native admission and the same split/prepared hashes.
Both independently scoped receipts and their executable reservations verify;
see [the current review](phase-5-scoped-hip-review.json).

| Scope | Measured projection | 80% admission limit | Reserved role ceilings | Decision |
| --- | ---: | ---: | ---: | --- |
| diverse32 |7,926.648s |11,520s |10,082s of14,400s |PASS; one T096 attempt |
| transfer-screen |29,637.220s |5,760s |37,102s of7,200s |BLOCKED |

The original complete-screen estimate of47,749.33s remains intact in
[its historical receipt](phase-5-batched-hip-review.json). It does not authorize
or reject the separately calculated diverse32 workload.

The archive's24 file hashes, sealed report/receipt/recipe hashes, source/image/
profile/prepared-shape/schedule identities and current recipe configuration all
verify. Recomputing the p95 inputs and each scope's projection/role ceilings
reproduces the receipts. The reviewed source is
`sha256:9c97818583e9050e0752417c343546f19ef6462e1b85c0b85cb1c0c819962542`;
the reviewed image is
`sha256:33c356249558cb31197ad4285db6223c13a423548661bb624f3e12889af9099f`.
The measured recipe identity is
`sha256:560df5bd317e8d85573e491a152d071100d0d323bb03ccb9a2dd54741401aeed`.
This review admits the preregistered finite budget and exact diverse32 schedule,
not a transfer run or learned-accuracy claim. T096 remains unexecuted.

## Decision

The original7,200-second value was an unmeasured finite qualification default,
not a throughput promise. Phase 5 will continue before a broad optimization
detour. T095 now requires two separately identified projections:

1. **diverse32** — exact32-prompt periodic and end-of-run work, its declared raw/rescue/correct/shuffled/absent/template controls, actual validation cadence, fitting/reference/hidden scoring, training and checkpoints;
2. **transfer-screen** — the96-prompt periodic panel, complete128+128 development evaluation and every registered final control.

Each receipt must bind source, image, profile, prepared-shape hash, prompt
counts, attempts, controls, cadence, measured p95 inputs, reserve and budget.
The diverse32 receipt can admit only T096. It cannot qualify the transfer screen,
and the failed transfer-screen receipt cannot substitute for calculating the
smaller diagnostic. A projection must fit within80% of its own preregistered
finite budget.

The scope-specific implementation is available in PR #59. Cost policy and
coordinator reservations now share the same workload functions. The new
source-bound HIP preflight and receipt review are complete. All32 denominator
members, controls, termination requirements and the31/32 threshold remain fixed.

## Preregistration — 2026-10-21

`configs/foundation/bigint_smoke_phase5.yaml` registers a 14,400-second finite
diverse32 budget, including diagnostics and debited role reservations, with the
same 3000-update maximum, 100-update validation cadence and seed20261016. This is
an explicit resource revision to the original smoke recipe, preserved unchanged
at 7,200s. Recipe validation permits at most 14,400s; no unbounded training option
is introduced. The 80% measured admission limit is 11,520s. Profile, batch size,
learning rate, portfolios, patience, denominators and success thresholds remain
unchanged. See [the machine-readable registration](phase-5-scoped-preregistration.json).

| Scheduled work | diverse32 | transfer-screen |
| --- | --- | --- |
| Periodic synthesis prompts |32 training |32 training +32 parameter +32 structural |
| Validation cadence / maximum calls |100 /30 |500 /6 |
| Full teacher-forcing development prompts per validation |256 |256 |
| End-of-run prompts / scorer groups |32 / one32 |256 / two128 |
| Neural decoding rounds at end |24:8 attempts ×3 conditioning modes |24 |
| Fitting rounds per end prompt |32:24 neural +8 template |32 |
| Raw greedy checks per end prompt |3 |3 |
| Independently scored100-index outcomes per end prompt |6 |6 |
| Checkpoints per validation role |2 |2 |
| Registered budget |14,400s |7,200s |

Periodic synthesis uses one correct-memory greedy attempt, its fitting/reference
check and visible raw20 check. Templates do not decode. Each validation still
teacher-forces all 128+128 development rows. The measured teacher-forcing latency
is normalized per example. A measured production+reference 100-index check
conservatively bounds a visible20 check. The forecast includes a partial final
validation when updates are not divisible by cadence. Each role reserves
`ceil(measured_work / 0.8) + 5s`, and admission also checks that the sum of all
role ceilings fits the complete budget. Early stopping never refunds a debit.

Applying the previously reviewed p95 inputs to this explicit schedule gives
7,999.085s for diverse32 and 10,183s of reserved role ceilings. This calculation
motivates the four-hour preregistration; it is **an inference, not a new HIP
receipt**. The separately corrected transfer-screen estimate is 29,723.688s,
with 37,206s reserved, and still fails its unchanged 7,200s budget. The original
47,749.33s receipt remains intact at its original source/schedule identity.

The next preflight writes `performance/cost-diverse32.json` and
`performance/cost-transfer-screen.json`, even when only diverse32 passes. Each
receipt seals the source/image/profile/shape/schedule, exact workload, p95s,
reserve, budget and resolved recipe. Only passing scopes get a
`measured-SCOPE-recipe.yaml`. Old unscoped receipts remain readable but cannot
authorize new scoped training. Changing the source, image, prepared shape,
profile, schedule, controls or budget requires a new receipt.

## Data and measurement rules

The helper's `--prepare` invokes the pinned bulk fetch, prepare and check commands
with declared workspace caps. Alternatively pass `--inputs PATH` for an already
prepared set. Public checking verifies512 training,128 parameter development,
128 structural development, native admissions, matched views, exclusions and
diverse32. The host sends only a public shape/hash manifest to the non-scoring
measurement role; no scoring truth or final prompts are mounted.

Cost admission requires actual HIP, current image/profile/fixture hashes,
complete supported fit measurements and the matching prepared shape. Only a
successful scope-specific preflight writes an admissible measured recipe.
Changing declared settings creates a new experiment identity; it is not a budget
refund. These conservative sampled estimates are resource evidence, not evidence
of learned accuracy.
