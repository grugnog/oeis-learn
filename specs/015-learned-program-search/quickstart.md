# Round028 workstation commands

[The verified historical audit](validation/round028-audit.md) selected the matched generator. All 128 cases were accounted for. On the 64 routing cases, 51 had a witnessed proposal failure, seven had a selected solve, and six remained unresolved.

## Verified training rehearsal

The [returned and verified training rehearsal](validation/round028-generator-rehearsal.md) projects 1 h 44 min for both 4M arms. Both original checkpoints passed full-state recovery.

The [returned CPU collection](validation/round028-panel-results.md) reserved all 576 cases. Historical costs project 6 h 26 min for a pilot and an earned repeat, leaving 1 h 34 min for differences in candidate costs and operating overhead.

## Measure candidate evaluation time

Use implementation `bf9d223501fa3bd6281b6c767c3d963c7587afab` on the original workstation. This collection measures how quickly both restored learners and the incumbent can search and check programs on the same eight previously exposed cases. It publishes the revised time budget and preserves the fresh panels for the scientific comparison. [Measurement and spending rules](validation/round028-candidate-timing.md)

```bash
(
  set -euo pipefail
  cd "$HOME/workspace/oies-learn"
  rev=bf9d223501fa3bd6281b6c767c3d963c7587afab
  git fetch --quiet --no-tags origin refs/heads/015-learned-program-search
  git merge-base --is-ancestor "$rev" FETCH_HEAD
  git show "${rev}:specs/015-learned-program-search/validation/launch-round028.sh" |
    bash -s -- "$rev" collect-timing
)
```

Return the ZIP at the printed path. The verified artifact is `round028-timing.zip`. If collection stops early, the same command produces `round028-timing-diagnostics.zip` with the available measurements and failure locations.

## Original CPU supply and historical costing procedure

Use implementation `8954cbd0644273bed4a8cf413df7573519c86d10` on the original workstation. The collector reuses the verified training rehearsal, prices the historical 128-case search and scorer work, and checks the fresh-family supply for both seeds. It can extend the original source order and run isolated native CPU preparation in the same invocation.

```bash
(
  set -euo pipefail
  cd "$HOME/workspace/oies-learn"
  rev=8954cbd0644273bed4a8cf413df7573519c86d10
  git fetch --quiet --no-tags origin refs/heads/015-learned-program-search
  git merge-base --is-ancestor "$rev" FETCH_HEAD
  git show "${rev}:specs/015-learned-program-search/validation/launch-round028.sh" |
    bash -s -- "$rev" collect-panels
)
```

Return the ZIP at the printed path. A verified result is named `round028-panels.zip`, including when one or more rosters are short. An operation that cannot reach verification produces `round028-panel-diagnostics.zip`. The status and costs in either file cover CPU preparation. The scientific pilot awaits its prospective data and device admission. [Supply and costing details](validation/round028-panel-supply.md)

## Original receipt recovery procedure

The [returned collection](validation/round028-rehearsal-collection.md) contains the verified historical audit and exit status zero. The first collector could silently discard the final rehearsal JSON. A build-progress line before the result reproduces the loss. The following block recovered the recorded CPU and HIP results from the original producer. Its verified return is recorded above.

```bash
(
  set -euo pipefail
  cd "$HOME/workspace/oies-learn"
  collector=48529c5e4b52aa34183bde475428a4dce95e2e5c
  producer=46594ef8ba9e4098b1f5759b28635a4537d6d399
  git fetch --quiet --no-tags origin refs/heads/015-learned-program-search
  git merge-base --is-ancestor "$collector" FETCH_HEAD
  git merge-base --is-ancestor "$producer" "$collector"
  git show "${collector}:specs/015-learned-program-search/validation/launch-round028.sh" |
    bash -s -- "$collector" collect-receipts "$producer"
)
```

Attach the one ZIP printed at the end, including when a receipt is missing. A complete export is named `round028-rehearsal.zip`. It contains the original historical audit and a verified aggregate with the native gradient, per-arm token charges and timing, checkpoint closure, and the forecast for two 4M arms. A diagnostic export is named `round028-receipt-diagnostics.zip`. It includes the first verification gap and available original console figures. Both exports keep case data and checkpoints on the workstation.

The original producer ran CPU native and gradient checks first. Its HIP rehearsal was bounded to two ordered updates per arm and a cold checkpoint restore. The training projection takes the slower observed update per charged token for each arm and reserves four of the eight hours for evaluation and replication. Fresh panels, matched evaluation timings and campaign-bound acceptance are still required before the research comparison.

## Original audit command

The audit is the first decision in round028. The implementation pin is `f864f238c8a741924ef29707fecc0dcd0688fffc` on `015-learned-program-search`. The workstation's physical checkout must be detached at `77667e67176ae32fa16457d52565bae1dbe41afc` with its original `run-state` tree. The launcher verifies the pin, extracts the candidate alongside that state, and checks the cached Git files on reuse.

Paste this whole block into Bash on Owen's NixOS workstation. It fetches the feature, runs metadata preflight and the CPU audit, verifies a publication against the private originals, verifies a copy through the archive-only CLI, and records final status. The command prints the absolute path of one ZIP to return, including when required original evidence is missing. A concurrent owner returns promptly. Repeating this block reuses the committed audit and its charges.

```bash
(
  set -euo pipefail
  cd "$HOME/workspace/oies-learn"
  rev=f864f238c8a741924ef29707fecc0dcd0688fffc
  git fetch --quiet --no-tags origin refs/heads/015-learned-program-search
  git merge-base --is-ancestor "$rev" FETCH_HEAD
  git show "${rev}:specs/015-learned-program-search/validation/launch-round028.sh" |
    bash -s -- "$rev" collect-audit
)
```

The collector writes detailed output for each action to its own local directory under `run-state/discovery-tools/round028-cpu-*/`. Return **`round028-workstation.zip`** from the path printed at the end. It includes `summary.json` with exit codes, prerequisite availability and diagnostic code locations. When the copied audit verifies successfully, the ZIP also contains `round028.zip`. Private case rows, program bytes, raw command output and error messages stay on the workstation.

## Read the result

| Evidence | Where to look |
| --- | --- |
| Workstation could open the historical study | `summary.json` gives the donor revision, prerequisite files and audit exit status. A missing frozen campaign gives 0/128 authenticated cases. |
| Historical outcomes and coverage | Inside `round028.zip`, `audit.json` contains the frozen and companion counts for six 1M measurements and trial 00 at 2M, the D/trial-05 paired counts by stratum, unfinished fits, and the four 25M checkpoint dispositions. Each historical stratum retains its original denominator of 32. |
| What could justify spending | `projection` contains the feasible-set histogram for the 64 primary cases. `decision` gives the lower and upper bounds on proposal minus selection opportunities. `coverage` gives bank gaps, exact witnesses and measured training support. A potentially decisive CPU replay with unmeasured funding has a null action, a blocker, a case count and an original service-time estimate. |
| Independent checks | `verify-private` reopens the bound inputs and recomputes the audit. `verify-copied` recomputes the decision from the aggregate alone. The owner keeps private records under `run-state/discovery/015-research-round-028/`. |

A successful historical audit can identify an experiment or settle a retain decision. The audit command publishes that decision. The selected generator rehearsal uses its own receipts. Prospective panel freeze, evaluation timing and campaign acceptance remain open in [tasks.md](tasks.md). A blocked or invalid audit supplies a diagnostic and authorizes zero GPU work. The original Optuna decisions, donor images and charges belong to round027.

## Verify the returned ZIP on a different computer

Start in the directory containing the returned `round028-workstation.zip`. This command extracts its verified report when present, fetches the exact implementation into a temporary directory, and runs the standalone verifier. It needs Git, Bash, tar, Python and access to the public repository.

```bash
(
  set -euo pipefail
  rev=f864f238c8a741924ef29707fecc0dcd0688fffc
  result=$(pwd -P)/round028-workstation.zip
  temp=$(mktemp -d "$(pwd -P)/round028-verify.XXXXXXXX")
  trap 'rm -rf -- "$temp"' EXIT
  python3 - "$result" "$temp/round028.zip" <<'PY'
import json
from pathlib import Path
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as bundle:
    summary = json.loads(bundle.read("summary.json"))
    if not summary["includes_verified_audit"]:
        raise SystemExit("Workstation diagnostic only. See summary.json inside the returned ZIP.")
    Path(sys.argv[2]).write_bytes(bundle.read("round028.zip"))
PY
  git init --quiet "$temp/repo"
  git -C "$temp/repo" fetch --quiet --depth=1 https://github.com/grugnog/oeis-learn.git "$rev"
  mkdir "$temp/source"
  git -C "$temp/repo" archive "$rev" | tar -xf - -C "$temp/source"
  python3 -B "$temp/source/scripts/foundation_run.py" --engine podman discovery round028 \
    --action verify --archive "$temp/round028.zip"
)
```

The specification, research handoff, evidence index and outstanding acceptance criteria are in [research.md](research.md), [contracts/experiment.md](contracts/experiment.md), [contracts/operations.md](contracts/operations.md) and [tasks.md](tasks.md).
