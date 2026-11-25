# Reproduce the cleanup analysis and comparisons

Run from a full clone. The portable probes need Git and Python 3.11+. Use the repository’s existing Python test environment or qualification image for pytest. The planning run used Python 3.12.14, pytest 9.1.1 and Optuna 4.5.0 in a disposable review environment. The optional import and unused-name tools have no role in a campaign install.

## 1. Establish the two revisions

For the first comparison, use the audited PR #72 baseline below. For an individual cleanup group, set `before` to that group’s parent and `after` to its commit. Put observations and fixture writes outside the two source trees.

```bash
repo=$(git rev-parse --show-toplevel)
review_tmp=$(mktemp -d)
before=a2618bbd1ff908a1d720f47268edf4234852c5d8
after=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" worktree add --detach "$review_tmp/before" "$before"
git -C "$repo" worktree add --detach "$review_tmp/after" "$after"
git -C "$repo" diff --name-status "$before" "$after"
git -C "$repo" diff --check "$before" "$after"
cd "$review_tmp/before"
```

Run the applicable probes again from `"$review_tmp/after"` with fresh output directories. Compare consumers, statuses, costs, verdicts and source-map keys alongside the diff. After preserving the comparison evidence, `git worktree remove` on each clean temporary checkout releases its registration.

## 2. Cross-check code and data consumers

A disposable environment with `grimp==3.20` and `vulture==2.16` reproduces the static probes. For example, `uv venv "$review_tmp/tools"` followed by `uv pip install --python "$review_tmp/tools/bin/python" grimp==3.20 vulture==2.16` provisions the two analysis tools. Run this block in each checkout to reproduce the graph and syntax counts.

```bash
PYTHONPATH=src "$review_tmp/tools/bin/python" - <<'PY'
import ast
import os
import subprocess
from collections import Counter
from pathlib import Path
import grimp

paths = [Path(os.fsdecode(p)) for p in subprocess.check_output(
    ['git', 'ls-files', '-z', '--', '*.py']).split(b'\0') if p]
calls = Counter()
for path in paths:
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {'run_path', 'run_module', 'import_module'}:
                calls[node.func.attr] += 1
print('Parsed Python files and literal attribute call sites', len(paths), calls)
graph = grimp.build_graph('oeis_learn', cache_dir=None)
package = 'oeis_learn.pipeline_validation.'
print('Package modules and import edges', len(graph.modules), graph.count_imports())
print('Worker importers', graph.find_modules_that_directly_import(package + 'progressive_role'))
print('Preparation chain', graph.find_shortest_chain(package + 'progressive_workstation',
                                                     package + 'progressive_research_launch'))
PY
"$review_tmp/tools/bin/python" -m vulture src/oeis_learn scripts --min-confidence 100 --sort-by-size
```

On the baseline, expect 1,175 parsed files, 558 package modules, 3,291 edges and an empty worker-importer set. Vulture exits with findings. Inspect its reported names and read their callers/decorators. These Git queries expose three different consumers that the import graph misses.

```bash
git grep -nF 'runpy.run_module' -- src/oeis_learn/pipeline_validation/progressive_roles.py
git grep -nE 'qualify_bigint_phase[25]|qualify_progressive_foundation' -- .github/workflows/foundation.yml src/oeis_learn/pipeline_validation/progressive_handoff.py
git grep -nE 'run_hpo_sweep\.py|hpo_search_space\.yaml|hpo_sweep_results|real_data_e2e_report\.md' "$before" -- .
git grep -nw _populate_canonical_defaults "$before" -- .
python3 -BS scripts/inventory_synthesis_retirement.py --check
```

For an individual file, search its basename, relative pathname, SHA-256, directory readers and package patterns. Inspect the hits. For the HPO reports, the hits in the old script are output destinations. For the launcher itself there are historical hash bindings. [Candidate interpretations](research.md)

## 3. Compare identities and historical retrieval

```bash
PYTHONPATH=src python3 -BS - <<'PY'
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.pipeline_validation.identity import digest
snapshot = collect_snapshot('.')
print(len(snapshot), digest(snapshot))
for path in ('scripts/run_hpo_sweep.py', 'configs/hpo_search_space.yaml',
             'reports/hpo_sweep_results.json', 'rustfmt.toml'):
    print(path, snapshot.get(path, 'outside this source map'))
PY
validation=specs/013-progressive-mathematical-discovery/validation
git cat-file -e '77667e67^{commit}'
git show "9030a2bc:$validation/continue-round027-publication.py" > "$review_tmp/resume.py"
git show f09d5471:src/oeis_learn/pipeline_validation/progressive_publication.py | sha256sum
git archive -o "$review_tmp/reporter.tar" a2618bbd "$validation/report-round027.py" src
tar -tf "$review_tmp/reporter.tar" | rg 'report-round027\.py|generalization/progressive/verify\.py'
```

The baseline map has 1,332 entries. The HPO launcher is a key. The internal publication adapter digest is `effe3ec15ce90935f99a90450fa04c03de5ee467e61b94c926d837903812d3ce`. The reporter and package resolve from the same revision. Use a clone containing the cited historical commits for the Git retrieval probes.

To reproduce the wheel finding, export `a2618bbd` to a new directory with `git archive`. Build an sdist and direct wheel there with `python setup.py sdist bdist_wheel` in an isolated builder. Extract the sdist to a second directory and build a wheel there with `python setup.py bdist_wheel`. Compare `oeis_learn/experiments/qualification-sources.json` from the two ZIPs, by both key and value. The local baseline had a 1,332/1,331 split for the spec 005 contract and identical values for every shared key. Repeat with the repository’s actual release build for a packaging-sensitive group. [Release wheel comparison](../../scripts/foundation/test_wheel.sh)

For a donor-sensitive change, export revisions `7332f0be1f76c5bd735b682765f3b531a1f3ed59` (round009) and `77667e67176ae32fa16457d52565bae1dbe41afc` (qualified round027) separately. From the original round009 tree serialize its own `collect_snapshot('.')` to a file outside either tree. From each proposed tree call `progressive_calibration_equivalence.validate(original_map, collect_snapshot('.'))` with its own package on `PYTHONPATH`. The positive qualified control accepts 89 changes. Removing only the HPO launcher key from a copy of the qualified map triggers a rejection naming the launcher. Compare the unmodified reporter tree separately. [Donor evidence](research.md)

## 4. Run bounded operator and regression checks

These host commands parse the real entry points, create no research session, and display a waiting round from a disposable root.

```bash
python3 -BS scripts/foundation_run.py --engine docker discovery --help
python3 -BS scripts/foundation_run.py --engine podman data fetch --help
python3 -BS scripts/discovery_hip_handoff.py --engine podman \
  --session run-state/014-review --dry-run --run-after-prepare prepare-research
monitor_root=$(mktemp -d)
mkdir -p "$monitor_root/run-state/discovery/013-research-round-027"
python3 -BS "$validation/research-progress.py" --root "$monitor_root" --round 27 --once
find "$monitor_root" -type f
```

Expect the 12 discovery commands, a corpus `--config` option, a dry-run description with launches disabled, and pending freeze/acceptance with no files created in the empty round. For interval mode on a rich copy, record the initial paths and hashes, run `--interval 0.1` through several displays, stop it, and compare the paths, hashes and journal counts. Include a waiting and an interrupted case, an active learner heartbeat and a recoverable checkpoint. Keep the log outside the monitor root.

The following four independently runnable selections passed on the planning baseline, **107 distinct cases** in total. Use the test interpreter from the existing project environment, with the following shell function defined in the checkout under test. The explicit spec 013 recovery file is outside default test discovery.

```bash
review_tests() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider "$@"
}
# Candidate, consumer and packaging controls — 23 passed.
review_tests tests/unit/test_elite_buffer_replay.py tests/unit/test_run_manager.py \
  tests/unit/test_foundation_symbolic.py::test_legacy_report_is_not_a_proof \
  tests/unit/test_foundation_symbolic.py::test_reporting_replays_scope_and_marks_legacy_labels \
  tests/contract/test_progressive_packaging.py::test_diagnostic_templates_ship_identically_and_all_feature_tests_are_registered \
  tests/contract/test_progressive_diagnostic_blueprints.py

# Fresh gates, recovery, screening, confirmation and separate assessment — 40 passed.
review_tests tests/contract/test_progressive_cli.py \
  tests/unit/test_progressive_research_inputs.py::test_complete_original_registration_and_private_exposure_import \
  tests/unit/test_progressive_research_inputs.py::test_missing_state_sidecar_blocks_but_other_arm_and_exposures_are_checked \
  tests/unit/test_progressive_research_census.py::test_automatic_training_needs_completed_preparation_and_acceptance \
  tests/unit/test_progressive_research_census.py::test_automatic_training_rejects_changed_acceptance_bytes \
  tests/unit/test_progressive_freeze.py::test_complete_diagnostic_freeze_binds_original_producer_and_task_roster \
  "$validation/test_round027_publication_recovery.py" tests/unit/test_progressive_round025_successor.py \
  tests/unit/test_progressive_dag_decisions.py tests/unit/test_progressive_trial_specifications.py \
  tests/unit/test_progressive_streamlined.py::test_linked_confirmations_preserve_tokens_and_fit_original_total \
  tests/unit/test_progressive_streamlined.py::test_only_sizing_intermediate_evaluations_are_removed \
  tests/unit/test_progressive_streamlined.py::test_old_design_and_roster_unchanged \
  tests/contract/test_progressive_assessment.py::test_assessment_exposure_is_one_way_and_blocks_adaptive_resume

# Publication, independent verification, study inputs and journal limits — 41 passed.
review_tests tests/contract/test_progressive_bundle.py tests/integration/test_progressive_reporting.py \
  tests/unit/test_progressive_study_policy.py \
  tests/unit/test_progressive_budgets.py::test_large_journal_allowance_is_frozen_and_legacy_default_stays_bounded

# Actual pinned sampler and saved trial history — 3 passed.
review_tests tests/unit/test_progressive_trials.py
```

To observe function execution on real fixture paths, profile the reporting parametrizations and the interrupted-publication fixture separately. These profiles repeat cases counted above.

```bash
PYTHONPATH=src:. python -m cProfile -o "$review_tmp/report.pstats" -m pytest -q -p no:cacheprovider \
  tests/integration/test_progressive_reporting.py::test_run_report_verify_collects_owned_scores_and_resource_journal
PYTHONPATH=src:. python -m cProfile -o "$review_tmp/recovery.pstats" -m pytest -q -p no:cacheprovider \
  "$validation/test_round027_publication_recovery.py::test_interrupted_formal_publication"
python - "$review_tmp/report.pstats" "$review_tmp/recovery.pstats" <<'PY'
import pstats, sys
for filename in sys.argv[1:]:
    pstats.Stats(filename).strip_dirs().print_callers('collect$|plan_remaining$|finish$|boundary$|verify$')
PY
```

## 5. Close the affected evidence gaps

- **Fresh and installed.** Use the [foundation CI partitions](../../.github/workflows/foundation.yml) on the pinned qualification image. Its runner names 132 test paths/nodes in four partitions (32, 34, 34, 32). Its merge checks their complete inventory. Exercise the directly built and sdist-rebuilt wheels outside the source tree, the public `oeis-learn` console and `pipeline --workspace`, the Docker/Podman host commands and relevant native crates. The Torch model-construction and continuation tests, actual first worker dispatch and physical HIP qualification require the corresponding environment. Use the [existing operator quickstart](../013-progressive-mathematical-discovery/quickstart.md).
- **Accepted-run rehearsal.** Use a disposable copy with the required 47 outputs, four checkpoints, original producers, full journal and pinned source. Record input hashes, task ownership and original charges. Exercise replay, the approved continuation and a repeated attempt on isolated state. Compare with the recovery contract, including the separately selected assessment case. The small interruption fixture above verifies the commit boundary and next task. The original-data checks remain outstanding.
- **Full-size publication.** Supply the combined 49 MB calibration input, four large projections and roughly 290,000 budget events on a disposable fixture or a copy of a terminal invocation. Use `discovery report --run <run> --output <fresh-output>` and then `discovery verify-bundle --archive <report.zip> --output <fresh-verification>` through the portable foundation runner in a separate workspace. Paths are workspace-relative and the bundle is the verifier’s input. Compare totals, scope, original byte identities, private-data exclusion and refusal of altered evidence. The measured four-by-5 MiB projection round trip and separate 49 MiB policy lookup cover only part of this test. [Acceptance contract](contracts/cleanup-acceptance.md)
- **Disposition.** Compare each affected result on the group’s parent and candidate. Record any outstanding physical/private check or pre-existing failure under that group. Revert a failing trial in its disposable checkout or withhold its commit. Track the actual files, bytes and symbols removed by accepted groups. [Record format](data-model.md)
