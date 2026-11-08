# Review of the repaired training-only probe run

Reviewed 2026-11-07 from `pac-selector-probes-003/report.zip`.
The result is PASS / COMPLETE_DESCRIPTIVE / `freeze_eligible: true`.
This passes the predeclared technical screen; it does not select a winning arm
or establish a useful coverage-selector effect.

## Evidence and checks

- Archive SHA-256: `0e935667e08a82d3a6809e129a574a9bcad8c20c9ed6acfc11249331cdc19e62`.
- Plan: `sha256:2afe0dc9a92a4351349d84cce2ba17542bcaed3095cf6d0f96338b97df09b42c`.
- Result: `sha256:63b53533a276cda197a2b7384f0f0387714f6f1d165864e70125f77bd6281584`.
- Source: `sha256:d29306dbcc384b15ab054757c4da3d28ada9ca1883c74085f6dc7e477d661040`, matching the reviewed checkout.

The plan/result and both generation-result seals verify. The result binds the
plan, and the selector matches the frozen `visible-branch-coverage/v1` policy.
Original input references, checkpoints, model counts/profile, generation policy,
attempt counts and the 256-task roster are unchanged from `probes-001`.
There are 32 families with eight probes each. All per-arm/control summaries and
all cross-arm/control comparisons recompute exactly from exported task outcomes.
The eligibility calculation independently reproduces the 244/256 completion
floor, both generation PASS states and zero incomplete selector traces.

No training tokens were consumed and assessment was not executed. The successful
host path requires current CPU/native and actual-HIP qualification before probe
generation. This ZIP contains the report/plan, not those qualification receipts;
the qualification claim follows from the guarded execution path rather than a
separate review of underlying test transcripts. The workstation freeze command
also verifies the original local scoring producers, beyond the public report.

## Primary results

Every denominator is 256. Success counts include only completed scores.

| Arm | Original success | Coverage success | Completed, both selectors | Joint recoveries / regressions |
|---|---:|---:|---:|---:|
| B | 182 | 182 | 255 | 0 / 0 |
| D | 83 | 84 | 244 | 1 / 0 |

D's completion rises from 235 to 244. The nine newly completed case IDs exactly
match the nine cases in the fixed-candidate native-fit replay; all nine become
completed failures, with no selected candidate. There are no primary success
changes between the original and repaired implementations for either selector.
The coverage selector's one D recovery is the same case as before:
`sha256:1093a24a4b7cefba983f9102e20815349aaed0ea65e5765b3f046e5724cd61c7`.

One already-completed B coverage selection and one already-completed D coverage
selection have different candidate hashes than the old run, without changing
success/completion. Thus the cross-run comparison is not an assertion of
byte-identical candidate pools. Within this run the paired selectors share each
control's admitted pool. The original primary selections have unchanged hashes.
All 12 remaining incomplete D primary cases, and B's one incomplete case, have
no selected candidate. Current per-attempt failure mechanisms are not exported
in this summary ZIP.

B substantially outperforms D on these training probes. The family-macro
censoring bounds are 71.09–71.48% for B, 32.42–37.11% for original D and
32.81–37.50% for coverage D. The coverage effect is one extra success out of 256
(0.39 percentage points), with conservative D effect bounds of -4.30 to +5.08
percentage points. These bounds account for unknown censored outcomes, not
sampling uncertainty; they are not confidence intervals. Absence of observed
regressions on 244 jointly completed D tasks is not a guarantee on the other 12.

Training-probe performance does not replace the original structural comparison:
the populations differ, and the protocol explicitly uses probes as a technical
screen rather than choosing an arm by their accuracy. The original descriptive
structural signal favoring D still needs the separately frozen fresh comparison.

## All seven controls

The two selectors have identical success/completion counts in every control
except the one-success D primary recovery shown below.

| Control | B success / complete | D original success / complete | D coverage success / complete |
|---|---:|---:|---:|
| raw | 0 / 0 | 0 / 0 | 0 / 0 |
| rescue | 180 / 254 | 71 / 256 | 71 / 256 |
| equal_budget | 182 / 255 | 83 / 244 | 84 / 244 |
| shuffled | 26 / 256 | 58 / 231 | 58 / 231 |
| absent | 61 / 256 | 70 / 234 | 70 / 234 |
| template | 72 / 256 | 72 / 256 | 72 / 256 |
| bank | 64 / 256 | 72 / 256 | 72 / 256 |

The raw control is entirely incomplete, not a measured 0% completed-case accuracy.
D shuffled/absent remain below 95% completion; the declared advancement gate
applies to the primary equal-budget comparison, and these other controls retain
their censoring. No control's success count changes from `probes-001`.

The selector changes 109 B and 88 D primary choices. Across all seven controls,
recorded additional selector-observation time is 2.977 seconds for B and 2.429
seconds for D (previously 2.954 and 2.471). The ranking overhead is small in this
run, but this public ZIP contains no generation wall-time or utilization trace
from which to infer an end-to-end speedup. The earlier 156x fixed-fit result is
not a campaign-throughput estimate.

## Next step

Proceed with the existing prospective structural design, without changing the
selector, checkpoints, quotas, seed or budgets. First freeze separately on the
same `012-pac-qualification-diagnostics` checkout:

```bash
cd /home/owen/workspace/oies-learn
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation freeze \
  --probes run-state/generalization/pac-selector-probes-003 \
  --output run-state/generalization/pac-selector-fresh-001
```

After that command reports FROZEN, execute its exact plan:

```bash
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation structural \
  --plan run-state/generalization/pac-selector-fresh-001/plan.json
```

The result is `run-state/generalization/pac-selector-fresh-001/execution/report.zip`.
The draw excludes the old structural cohort, original corpus cells and training
probe windows. The primary endpoint remains certified structural family-macro
success, with at least 65/68 complete per compared checkpoint/selector, plus all
controls and all categories. A family-quota deficit is BLOCKED and does not
authorize a new seed or relaxed quotas. Preserve the original failed/incomplete
runs. No HPO, arm combination or additional training is part of this step.
