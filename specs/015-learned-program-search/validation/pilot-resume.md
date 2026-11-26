# Pilot qualification and recovery

Use controller `04e0fd9c30ce64d835c91fb06e5014799bceccee` from the detached round027 checkout. The [operator quickstart](../quickstart.md#run-the-eleven-hour-generator-pilot) gives the qualification-only command and the guarded start/resume command. The physical qualification receipt for this controller is pending.

The controller authenticates the saved registration, its source tree and its parent receipt, then runs the worker named in that registration. A new registration uses the controller revision as its worker. On recovery, the two SHAs may differ. The qualification collector prints both after verifying the copied archive. A diagnostic includes the worker SHA when the controller has resolved it.

Repeat the same pinned command after an interruption. It reuses sealed choices and the original worker, preserves charges, and runs the independent scorer on those choices. If an endpoint-sized search was split, qualification scores the recovered work and measures a separate cold process on exposed inputs. A missing measurement or a deficit returns a diagnostic ZIP.

If all 160 learner slots have empty choices, the receipt records 160 empty outcomes. A separate scoring check replays one previously solved, exposed trial-05 program across 160 slots and records its own scores and timings under `score_control`. Those replays do not count toward the learner’s outcomes.

## Verify a copied qualification

Start beside the returned `round028-pilot-qualification.zip` on a computer with Git and Python. This command reads the copy with the pinned source and writes the result to `round028-qualification-verification.json`.

```bash
(
  set -euo pipefail
  rev=04e0fd9c30ce64d835c91fb06e5014799bceccee
  report="$(pwd -P)/round028-pilot-qualification.zip"
  reader=$(mktemp -d)
  trap 'rm -rf -- "$reader"' EXIT
  git -C "$reader" init --quiet
  git -C "$reader" fetch --quiet --depth=1 https://github.com/grugnog/oeis-learn.git "$rev"
  git -C "$reader" checkout --quiet --detach "$rev"
  python3 -S -B "$reader/scripts/foundation_run.py" --engine podman discovery round028 \
    --action verify --archive "$report" > round028-qualification-verification.json
  python3 - "$rev" <<'PY'
import json
import sys

with open("round028-qualification-verification.json") as f:
    receipt = json.load(f)
if (receipt["status"] != "VERIFIED" or receipt["scope"] != "archive_only_pilot_qualification"
        or receipt["controller"] != sys.argv[1] or "workspace_space" not in receipt):
    raise SystemExit("The qualification receipt does not identify this release.")
print("Controller", receipt["controller"])
print("Registered worker", receipt["worker"])
scale = receipt["scale"]
print("Minimum case allowance (µs)", scale["minimum_case_allowance_us"])
print("Slowest case and its margin (µs)", scale["slowest_case_us"], scale["slowest_case_margin_us"])
print("Independent scoring", scale["selected_choices"], "learner choices,",
      scale.get("score_control", {}).get("checked_choices", 0), "exposed control choices")
print("Available at collection", receipt["workspace_space"]["free_bytes"], "bytes,",
      receipt["workspace_space"]["free_inodes"], "inodes")
print("Research headroom (µs)", receipt["research_headroom_us"])
PY
)
```

T056 and T060 require the physical return. T046 and T047 track broader workstation acceptance. T048 tracks the subsequent scientific result.

## Compare a copied pilot with its pre-score qualification

Keep the qualification ZIP collected while the scientific status was `PILOT_UNMEASURED`. From the checkout of the controller that produced the Phase 9 qualification, run the public verifier on both copies. It matches the frozen family allocation and the initialization, ordered training stream, codec and learning policy for every reported checkpoint.

```bash
python3 -S -B scripts/foundation_run.py --engine podman discovery round028 \
  --action verify --archive /path/to/round028-pilot.zip \
  --qualification-archive /path/to/pre-score/round028-pilot-qualification.zip
```
