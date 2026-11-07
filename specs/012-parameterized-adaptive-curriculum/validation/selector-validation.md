# Workstation selector validation

The offline audit is complete using the two uploaded ZIPs. They do not contain
model weights, the private recipe, or the complete training probe panel. This
follow-up therefore runs on the workstation. It uses the completed campaign
`run-state/generalization/pac-012-bcd-r1-002`; the checkout directory can retain
its existing `oies-learn` spelling.

## Experimental design

1. **Training-only validation:** reuse the original latest B and D checkpoints
   and all 256 disjoint training probes (eight per family, 32 families). No
   optimizer restore, gradient updates, checkpoint selection or hyperparameter
   search. B is the fixed-stream parameterized control; D is bandit-only.
2. Generate each checkpoint/control candidate pool **once** with the existing
   matched/drained eight-attempt, 60-second policy. Raw and rescue retain their
   one-proposal diagnostic definition. The same eight bank draws are reused
   across recipients. Keep all seven controls: raw, rescue, equal_budget,
   shuffled, absent, template and bank.
3. Compare the original selector against the fixed `visible-branch-coverage/v1`
   rule from the exploratory audit. Each receives the same complete eligible
   pool for its own control. No cross-control candidate borrowing. Retain every
   distinct admitted envelope before archive truncation. Ranking sees only
   candidate source/hash and the recipient's first 20 terms. The exact scorer
   receives frozen selections and all 100 target terms in a separate role.
4. Branch observation is a **separately charged additional selection cost**,
   after shared generation finishes. This is a paired selection experiment,
   not a claim of equal total inference cost. Preserve the original selection
   and mark the alternative incomplete if any eligible trace is unavailable.
5. The probe technical gate requires complete trace observations, complete
   generation accounting, and at least 244/256 completed primary scores for
   each checkpoint/selector. It does not choose a winner from probe accuracy.
   Run `freeze` separately after reviewing the probe report. The selector and
   checkpoint roster are unchanged at freeze; rule changes require a new source
   identity, qualification and probe validation.
6. **Fresh structural comparison:** separately commit a plan with draw seed
   20261108 (the shared draw implementation adds the existing structural offset
   of one), the original amended 23-family quotas and numeric law, 2,288 draws,
   bank seeds 20261101–20261108, exact checkpoints, source, immutable images,
   selector identity, probe result and role budgets. Exclude all 16 original
   corpus cells, the previously observed PAC structural cohort, and all training
   probe windows. Assessment cells are custodian-only exclusion inputs; never
   model/scorer inputs. No assessment evaluation occurs.
7. Admit the fixed 128-task allocation only after exact reference/native
   agreement: 68 certified / 26 ambiguous / 34 unresolved. A quota shortfall is
   BLOCKED with family deficits; native incompleteness is failure. Neither
   permits automatic reseeding, quota relaxation, or replacement after scoring.
   A frozen plan has one fixed execution directory and cannot be rerun into a
   different destination.
8. Primary comparison is certified structural family-macro success, with the
   unchanged 65/68 completion floor per checkpoint/selector. Report paired
   recoveries/regressions on jointly completed cases, full-denominator censoring
   bounds, every control/category, and B-to-D differences under each selector.
   Bounds are not confidence intervals. One training seed remains descriptive;
   passing this screen does not automatically adopt the selector or justify HPO.

The fresh population uses the same family allocation but additionally conditions
on disjointness from the observed cohort. It is a new prospective selection
screen, not a rerun or replacement of the original estimate. Limited family
capacity can legitimately block this design; the availability report is evidence
for an explicitly revised design, not permission to select an easier cohort.

## Workstation commands

After checking out this follow-up branch, from the repository root, run:

```bash
python3 -S scripts/foundation_run.py --engine docker pac-selector-validation probes \
  --campaign run-state/generalization/pac-012-bcd-r1-002 \
  --output run-state/generalization/pac-selector-probes-001
```

Use `--engine podman` instead when that is the workstation engine. The command
verifies original inventory hashes, builds the current qualification/GPU images,
runs the existing CPU/native and actual-HIP qualification, then evaluates B and D.
It does not use old source qualification for the changed reference observer.
Outputs are `summary.md`, `result.json`, `report.zip`, role results and private
replay artifacts under the chosen output directory. Operational ceilings are
conservative bounds enclosing all possible candidates and scoring, not duration
forecasts. Scientific per-portfolio and interpreter caps remain fixed.

Optional input-only verification (no build, engine or GPU):

```bash
python3 -S scripts/foundation_run.py pac-selector-validation probes \
  --campaign run-state/generalization/pac-012-bcd-r1-002 \
  --output run-state/generalization/pac-selector-preflight-001 --preflight-only
```

Once the probe report is reviewed, freeze the structural comparison separately:

```bash
python3 -S scripts/foundation_run.py --engine docker pac-selector-validation freeze \
  --probes run-state/generalization/pac-selector-probes-001 \
  --output run-state/generalization/pac-selector-fresh-001
```

Then execute that exact frozen plan:

```bash
python3 -S scripts/foundation_run.py --engine docker pac-selector-validation structural \
  --plan run-state/generalization/pac-selector-fresh-001/plan.json
```

The structural report is written beneath `pac-selector-fresh-001/execution/`.
Share each phase's `report.zip` for analysis. No full model upload is needed.
Fresh output paths are required; failed runs and original campaign files remain
unchanged. Errors print their actual exception and original role-result path.

## Validation boundary

Local tests exercise visible-only ranking, exact shared candidate pools,
selection seals, canonical 100-term scoring with synthetic execution, training
role rejection, original inventory/weight tampering, freeze gates and unchanged
historical draws. Host CLI help works with `python3 -S`. These checks do not
constitute workstation HIP execution or new scientific results.
