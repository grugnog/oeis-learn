# Diverse32 memorization gate — T096

Status: **ready for the one registered T096 attempt**. The reviewed current-source
preflight `008-phase5-scoped-20261021T115212Z` passes diverse32 cost admission;
see [the review receipt](phase-5-scoped-hip-review.json). Learning has not run.
The larger transfer screen remains independently blocked by cost.

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
T096 is left unchecked until that real result is reviewed. Other Phase 5 gates,
including T093 existing matched-checkpoint rescue, remain separate; passing
this diagnostic never admits Phase 6 or the sealed final cohort.
