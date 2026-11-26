# Round028 workstation commands

[The verified historical audit](validation/round028-audit.md) selected the matched generator. All 128 cases were accounted for. On the 64 routing cases, 51 had a witnessed proposal failure, seven had a selected solve, and six remained unresolved.

## Run or resume the matched generator rehearsal

Run this block on the original detached donor. It reopens the published audit, uses only its registered strict training producer for a fresh candidate measurement, and returns one ZIP. The CPU stage rechecks both programs in a training pair on the installed native evaluator and measures the auxiliary gradient on freshly initialized model parameters. A finite nonzero gradient admits a bounded HIP rehearsal. Both arms start from identical weights, see identical ordered batches, save full optimizer and RNG state and replay their next complete update after a cold restore.

```bash
(
  set -euo pipefail
  cd "$HOME/workspace/oies-learn"
  rev=46594ef8ba9e4098b1f5759b28635a4537d6d399
  git fetch --quiet --no-tags origin refs/heads/015-learned-program-search
  git merge-base --is-ancestor "$rev" FETCH_HEAD
  git show "${rev}:specs/015-learned-program-search/validation/launch-round028.sh" |
    bash -s -- "$rev" collect-rehearsal
)
```

Return the printed `round028-workstation.zip`. Its `summary.json` contains the original audit reference, training support after excluding fixed probes, the native gradient, per-arm charges, update and restore times, device memory and telemetry coverage, GPU lease time and an eight-hour shift forecast. A measured projection uses the slower of the two observed updates per arm and reserves four hours for evaluation and replication. The forecast covers training only. Missing counters have an explicit coverage measure. A blocked stage records its phase and available code locations. Raw training programs and checkpoints stay in the workstation's private directory.

The paired 4M research comparison still requires fresh development, retention and replication panels, matched evaluation timings and campaign-bound acceptance. The rehearsal uses historical training data and carries no prospective scores. The pinned launcher and ZIP failure path were exercised on a data-free detached donor. The CPU test suite exercised real parameter changes and checkpoint continuation. The physical native and HIP measurements come from the workstation ZIP.

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

A successful historical audit can identify an experiment or settle a retain decision. This release stops at that decision. Native replay dispatch, the prospective pilot executors, and physical qualification remain open in [tasks.md](tasks.md). A blocked or invalid audit supplies a diagnostic and authorizes zero GPU work. The original Optuna decisions, donor images and charges belong to round027.

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
