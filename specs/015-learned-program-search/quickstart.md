# Round028 validation and workstation use

## Available at the planning revision

No workstation data collection is needed to review this design. To read the plan on Owen's workstation while keeping the qualified donor checked out, run the following from Bash.

```bash
(
  set -euo pipefail
  cd ~/workspace/oies-learn
  git fetch origin refs/heads/015-learned-program-search:refs/remotes/origin/015-learned-program-search
  rev=$(git rev-parse --verify 'refs/remotes/origin/015-learned-program-search^{commit}')
  git show "${rev}:specs/015-learned-program-search/plan.md"
)
```

In a development checkout of the feature branch, the repository can validate the planning artifact set now.

```bash
.specify/scripts/bash/check-prerequisites.sh --json --require-spec
```

The result names this feature directory and lists research, data model, contracts and quickstart. Existing regression modules for the integration points are `tests/unit/test_selector_audit.py`, `test_progressive_metrics.py`, `test_progressive_decisions.py`, `test_progressive_budgets.py` and `test_progressive_freeze_progress.py` in the same directory. Run these with pytest in the project's qualified environment. Part of the freeze suite constructs a real PyTorch model. These tests cover inherited behavior.

## After the implementation is published

The following is the acceptance target for the future launcher. The planning revision contains documentation only. The presence check below stops before creating a local revision pin when the reviewed launcher is absent. Use the published implementation commit identified in its release instructions. The first block captures the fetched branch tip after checking it matches that reviewed commit.

```bash
(
  set -euo pipefail
  cd ~/workspace/oies-learn
  test "$(git rev-parse HEAD)" = 77667e67176ae32fa16457d52565bae1dbe41afc
  git fetch origin refs/heads/015-learned-program-search:refs/remotes/origin/015-learned-program-search
  pin=refs/round028/implementation
  launcher=specs/015-learned-program-search/validation/launch-round028.sh
  if ! git show-ref --verify --quiet "$pin"; then
    candidate=$(git rev-parse --verify 'refs/remotes/origin/015-learned-program-search^{commit}')
    git cat-file -e "${candidate}:${launcher}"
    test "$candidate" = "${ROUND028_IMPLEMENTATION_COMMIT:?set this to the published full implementation commit}"
    git update-ref "$pin" "$candidate" ''
  fi
  rev=$(git rev-parse --verify "${pin}^{commit}")
  git cat-file -e "${rev}:${launcher}"
  git show "${rev}:${launcher}" | bash -s -- "$rev" preflight
)
```

Expected preflight output includes the two source identities, immutable image IDs, actual free and required bytes/inodes, any deficit, the usable donor closure and the first affected stage. Preflight can produce a metadata receipt without running a model or exposing case outcomes. A blocked resource has a numerical explanation and the reservation it prevents.

Once admitted, the following is the single start **and resume** command. It can be repeated after an interruption. The recorded Git ref keeps subsequent branch updates from changing an active study.

```bash
(
  set -euo pipefail
  cd ~/workspace/oies-learn
  rev=$(git rev-parse --verify 'refs/round028/implementation^{commit}')
  launcher=specs/015-learned-program-search/validation/launch-round028.sh
  git show "${rev}:${launcher}" | bash -s -- "$rev" run
)
```

For a status check, use the following while the run is active. Replace the final `status` with `audit`, `rehearse` or `verify` to request just that operation. The same `run` action consumes its verified receipts.

```bash
(
  set -euo pipefail
  cd ~/workspace/oies-learn
  rev=$(git rev-parse --verify 'refs/round028/implementation^{commit}')
  launcher=specs/015-learned-program-search/validation/launch-round028.sh
  git show "${rev}:${launcher}" | bash -s -- "$rev" status
)
```

## Evidence to check on the first implementation

| Exercise | Expected result |
| --- | --- |
| Audit the original 128 | Seven measurements and all four 25M dispositions are accounted for. The paired D/trial 05 rows, six unfinished composition fits, original versus companion verdicts and missing prerequisites are traceable locally. CPU replay occurs only when registered. |
| Compare choices on a known bank | Ordinary, learned and shuffled arms receive identical sources and true-prefix admission. Permuting candidate order or replacing a member is detected. The independent scorer reproduces the one sealed selection. A correct unselected alternative contributes only to headroom. |
| Exercise each spending branch with controlled evidence | Produce fit fault, selection, proposal, ambiguous and no-opportunity cases. Each maps to the registered action. Where missing cases could change the sign, only the authorized replay or inconclusive branch runs. Show that 30/32 measured at an intermediate look may reach its exploratory endpoint. |
| Rehearse on the production workstation | Run the actual CPU/native and HIP checks, a full-state update/restore, a valid slow portfolio, production-sized collection through archive verification, genuine disposable shortage and two competing launchers. Check measured headroom, receipt reuse, charge reconciliation and diagnostic usefulness. |
| Restart after a lost worker or a partial publication | Repeat the same start command. The next stage identifies reused evidence, original costs, remaining legal opportunity and the next incomplete unit. A changed producer is rejected before its dependents run. |
| Copy the published share and verify it with the donor unmounted | The independent archive reader reproduces counts, routing, paired effects, uncertainty, verdict and spending action. Private sequences, programs and per-case traces remain local. Compare the scope of this receipt with the workstation's exact-program replay. |

Session evidence is under `run-state/discovery/015-research-round-028`. The final public artifact is `share/round028.zip`. The normal entry point stops after a verified terminal report. A recommendation for 25M confirmation requires a separate prospective budget.
