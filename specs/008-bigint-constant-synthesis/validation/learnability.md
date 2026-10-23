# Diverse32 memorization gate — T096

Status: **T096 complete with a negative result; promotion remains BLOCKED**.
The single admitted seed20261016 attempt finished in
`008-phase5-diverse32-20261021T160815Z`; see
[the reviewed result](phase-5-diverse32-review.json). The preflight remains
reviewed at [its original identity](phase-5-scoped-hip-review.json).

## Reviewed result — 2026-10-21

All15 outer archive file hashes and16 nested bundle hashes verify, with no
manifest omissions and complete core export. The source/image/recipe/preflight,
prepared split, contract, selected-checkpoint identity and ledger match.
Independent aggregation of all32 per-prompt observations reproduces every
control count and the frozen gate decision.

| Arm | Correct100-index continuations | Complete evidence |
| --- | ---: | ---: |
| Raw greedy |30/32 |30/32 |
| Same-candidate greedy plus fitting (rescue) |31/32 |31/32 |
| Equal-budget fitted portfolio |32/32 |32/32 |
| Shuffled memory |4/32 |4/32 |
| Absent memory |3/32 |3/32 |
| Non-neural template |0/32 |0/32 |

Greedy termination passes32/32. Rescue reaches the31/32 correctness threshold,
but its31/32 complete evidence fails the separately required32/32 completeness
criterion. Equal-budget success cannot replace that registered rescue criterion.
The public command completed successfully; the outer BLOCKED result is a
research outcome, not a process/build failure. Final access stayed sealed.

Training ended at1800 completed updates under the registered stopping policy.
The selected checkpoint manifest is
`sha256:b1c3213b3db3610630194a0076f67e075383d4aa7e03efa825a7648135797b62`;
latest and selected manifests remain distinct. The ledger debits7,730s:
18 training roles ×196s,4,134s generation and68s scoring, leaving6,670s of the
14,400s cap. These are accounted reservations, not measured wall time; all20
roles completed and there are no refunds. The aggregate generation elapsed
field sums per-prompt/conditioning work and is not role wall time.

The one incomplete rescue observation is prompt
`sha256:02aa8138af680017900e08e7c01148a4128583e1d0778195d05f4119c337b6b9`,
family `sha256:ebc51632e11b5c7f6e6f490188c6aa6994e7c677da65a8df3d7da8295cc25c53`.
Its raw and equal-budget observations succeed. It falls outside the eight
retained detailed examples, so this bounded bundle alone cannot identify
whether its rescue gap came from selection, fitting, execution or a limit.
The weights are intentionally excluded; their checkpoint identity is retained.
Diagnosing that gap requires the original role outputs or a separately bounded
analysis of the existing checkpoint, not another training attempt.

The registered attempt is consumed and the run is finalized. Do not resume it
or start another seed to change this outcome. A negative result completes the
task while retaining the promotion block. T095 remains closed despite the
producer's stale pending list; the transfer-screen cost block and other Phase5
qualification work remain independent.

## Registered recipe and archived procedure

The commands below document the completed registration. They are not a request
to repeat measurement or learning for this consumed attempt.

`configs/foundation/bigint_smoke_phase5.yaml` registers 14,400 seconds including
diagnostics/reservations. The model/optimizer starting recipe remains d256,
4+4 layers, batch8, LR1e-4, FP32/eager, seed20261016, at most 3000 updates,
100-update validation, visible-only selection and the frozen patience rule.
Training and periodic synthesis use exactly the prepared diverse32; full
teacher-forcing validation still covers both 128-row development sets. Its
composition, offsets, non-affine members and complete 512/128/128 preparation
are checked through the public CLI. Runtime fixtures cannot replace those inputs.

First measure without launching learning:

```bash
phase5_preflight="tmp-validation/008-phase5-scoped-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --performance-only \
  --cost-scope diverse32 --config configs/foundation/bigint_smoke_phase5.yaml \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "$phase5_preflight"
```

Review/share that output ZIP and both scope receipts. After the diverse32
receipt and budget are reviewed, run exactly one registered attempt:

```bash
python3 scripts/run_bigint_diverse32.py --engine podman \
  --reviewed-preflight "$phase5_preflight/performance/benchmark.json" \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "tmp-validation/008-phase5-diverse32-$(date -u +%Y%m%dT%H%M%SZ)"
```

For the reviewed run, set
`phase5_preflight=tmp-validation/008-phase5-scoped-20261021T115212Z`.
There is no need to repeat measurement for that unchanged source/image/data.
Keep `phase5_preflight` set to the reviewed directory if starting a new shell.
Substitute Docker consistently if appropriate. Both commands always write a
bounded ZIP on success/failure after creating their output directory. The
learning runner reuses the receipt's immutable image and costs; it never
remeasures, changes the seed or starts the transfer screen. It validates the
report/receipt/recipe hashes and current source/image/shape/profile/schedule,
then runs public `check`, `learnability` and `pack`.

A durable workspace claim under `run-state/phase5-diverse32/` prevents a second
fresh attempt for the same registration. An interruption preserves that claim
and the original run ledger. Inspect its archived summary first; use public
`foundation_run.py --engine podman resume --run ORIGINAL_RUN` to continue with
the original image/recipe and remaining allowance. Never delete the claim to
manufacture a new attempt or allowance. Public `pack --run ORIGINAL_RUN --output
NEW.zip` can export the resumed result. The outer utility does not automatically
retry or resume a learning failure.

The public run records raw/rescue, correct/shuffled/absent and non-neural
controls, independently scored 100-index continuations, coverage, greedy
termination, realized ledger cost and the selected real checkpoint. The
unchanged gate requires at least 31/32 rescue successes, 32/32 complete evidence
and 32/32 greedy termination. The wrapper returns PASS/0 or negative
BLOCKED/2 and retains the useful bundle; execution failure is FAIL/1. A completed
negative result completes T096 as a research outcome while blocking promotion.
T096 is now checked off for that reviewed negative outcome. Other Phase 5 gates,
including T093 existing matched-checkpoint rescue, remain separate; passing
this diagnostic never admits Phase 6 or the sealed final cohort.
