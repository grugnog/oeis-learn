# Reproduce the cleanup analysis and comparisons

Run from a full clone. The portable probes need Git and Python 3.11+. Use the repository’s existing Python test environment or qualification image for pytest. The planning run used Python 3.12.14, pytest 9.1.1 and Optuna 4.5.0 in a disposable review environment. The optional import and unused-name tools have no role in a campaign install.

## 1. Establish the two revisions

Use the audited PR #72 revision below to reproduce planning. The implementation baseline `a3f9808e` includes the screening audit from `5489aed2`. Run section 2 there for T003 and on the combined tip for T034. [Implementation inventory](implementation.md) For an individual cleanup group, set `before` to that group’s parent and `after` to its commit. Put observations and fixture writes outside the two source trees.

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

On the baseline, expect 1,175 parsed files, 558 package modules, 3,291 edges and an empty worker-importer set. Vulture exits with findings. Inspect its reported names and read their callers/decorators. Run the following queries from the root of each clean worktree. Record its HEAD alongside the hits so a reference introduced by the candidate is visible.

```bash
git rev-parse HEAD
git grep -nF 'runpy.run_module' -- src/oeis_learn/pipeline_validation/progressive_roles.py
git grep -nE 'qualify_bigint_phase[25]|qualify_progressive_foundation' -- .github/workflows/foundation.yml src/oeis_learn/pipeline_validation/progressive_handoff.py
git grep -nE 'run_hpo_sweep\.py|hpo_search_space\.yaml|hpo_sweep_results|real_data_e2e_report\.md' -- .
git grep -nw _populate_canonical_defaults -- .
python3 -BS scripts/inventory_synthesis_retirement.py --check
```

For an individual file, search its basename, relative pathname, SHA-256, directory readers and package patterns. Record a `git grep` exit of 1 as zero matches for that checkout. Inspect the hits. For the HPO reports, the hits in the old script are output destinations. For the launcher itself there are historical hash bindings. [Candidate interpretations](research.md)

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

### Test each source-changing group on the passing qualified control

Use the exact parent and tip of **one** group for `before` and `after`. The validator checks changed paths in sorted order and raises on the first rejection. The reporter already rejects `budgets.py`, so an identical error there cannot clear another source change. The following runs the original donor collector and the qualified validator on disposable trees. Use Python 3.12 for the historical AST pins. Each run keeps its patch, digest and results under a fresh directory.

```bash
(
  set -e -o pipefail
  gate=$(mktemp -d "$review_tmp/donor-gate.XXXXXX")
  donor=7332f0be1f76c5bd735b682765f3b531a1f3ed59
  qualified=77667e67176ae32fa16457d52565bae1dbe41afc
  mkdir "$gate/donor" "$gate/control" "$gate/group" "$gate/parent" "$gate/tip"
  git -C "$repo" archive "$donor" | tar -x -C "$gate/donor"
  git -C "$repo" archive "$qualified" | tar -x -C "$gate/control"
  git -C "$repo" archive "$qualified" | tar -x -C "$gate/group"
  git -C "$repo" archive "$before" | tar -x -C "$gate/parent"
  git -C "$repo" archive "$after" | tar -x -C "$gate/tip"
  git -C "$gate/group" init --quiet
  test "$(git -C "$gate/group" rev-parse --show-toplevel)" = "$(cd "$gate/group" && pwd -P)"
  git -C "$repo" diff --binary --no-ext-diff "$before" "$after" > "$gate/group.patch"
  {
    git -C "$repo" rev-parse "$before" "$after"
    git -C "$repo" diff --name-status "$before" "$after"
    sha256sum "$gate/group.patch"
  } | tee "$gate/change.log"
  PYTHONPATH="$gate/donor/src" python3 -BS - "$gate/donor" > "$gate/original.json" <<'PY'
import json
import sys
from oeis_learn.experiments.source_snapshot import collect_snapshot

print(json.dumps(collect_snapshot(sys.argv[1])))
PY

  donor_check() {
    PYTHONPATH="$gate/control/src" python3 -BS - "$gate" "$1" <<'PY'
import json
import sys
from pathlib import Path
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.pipeline_validation.progressive_calibration_equivalence import validate

gate, label = Path(sys.argv[1]), sys.argv[2]
root = gate / label
control = collect_snapshot(gate / 'control')
candidate = collect_snapshot(root)

def changes(before, after):
    return {path: (before.get(path), after.get(path))
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)}

if label == 'group':
    expected = changes(collect_snapshot(gate / 'parent'), collect_snapshot(gate / 'tip'))
    actual = changes(control, candidate)
    if not expected:
        sys.exit('No qualification-source change in the proposed group. Review its other consumers.')
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    if missing or extra:
        sys.exit(f'Source-map delta mismatch: missing={missing}, unexpected={extra}')
    for path, (old, new) in expected.items():
        observed_old, observed_new = actual[path]
        print(path, 'parent/tip', old, new, 'qualified/patched', observed_old, observed_new)
        if (old is None, new is None) != (observed_old is None, observed_new is None):
            sys.exit(f'Addition, deletion or modification differs for {path}')
        if old == observed_old and new != observed_new:
            sys.exit(f'Patched bytes differ from the group tip for {path}')
print(label, digest(candidate))
original = json.loads((gate / 'original.json').read_bytes())
try:
    accepted = validate(original, candidate, package_root=root / 'src/oeis_learn')
except ValueError as error:
    sys.exit(f'REJECT {error}')
print('PASS', len(accepted))
PY
  }
  donor_check control 2>&1 | tee "$gate/control.log"
  git -C "$gate/group" apply --check "$gate/group.patch"
  git -C "$gate/group" apply "$gate/group.patch"
  donor_check group 2>&1 | tee "$gate/group.log"
)
```

Require the clean control to pass with 89 accepted changes. A nested temporary directory can otherwise make Git skip a patch with exit 0. The group has its own Git root. The frozen collector derives the expected keys independently from the group’s parent and tip, including installed-resource aliases. The gate requires the qualified copy to show exactly those changes before invoking the validator. It also compares the resulting file hash wherever the starting hashes match. A report-only group outside the map follows its consumer and provenance checks. Record the group revisions, patch digest, changed keys and hashes, and its own pass or exact rejected path. Compare the reporter parent and candidate separately, exercising their own collector or validator if the group affects either. Hold a rejection or untestable transition. Removing only the retired-teacher method rejected `src/oeis_learn/rl/elite_buffer.py` on the qualified control. [Four-way donor result](research.md#source-identity-and-packaging-controls)

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

The following four independently runnable selections passed in the planning environment with Optuna 4.5.0, **107 distinct cases** in total. An independent review of `83684256` passed 104 cases. Its environment lacked Optuna, so the three sampler cases could not be collected there. Use the test interpreter from the existing project environment, with the following shell function defined in the checkout under test. The explicit spec 013 recovery file is outside default test discovery.

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

### Exercise the first dispatch and linked confirmation

Use the same two review fixtures in both checkouts. The source, tests and worker subprocess must come from the revision under review. From section 1, keep the absolute `repo` and `review_tmp` paths and the candidate revision in `after`. Run in the project test environment with its CPU Torch dependencies. For the first review of this change, use its PR tip for the fixture files. For later deletion groups, use the version of the fixtures under review.

```bash
(
  set -eu
  probes=$(mktemp -d "$review_tmp/cleanup-probes.XXXXXX")
  for name in test_fresh_dispatch.py test_linked_confirmation.py; do
    git -C "$repo" show "$after:specs/014-repository-cleanup/validation/$name" > "$probes/$name"
  done
  for tree in before after; do
    (
      cd "$review_tmp/$tree"
      git rev-parse HEAD
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python -m pytest -q -p no:cacheprovider --rootdir="$PWD" --basetemp="$probes/$tree-state" \
        "$probes/test_fresh_dispatch.py" "$probes/test_linked_confirmation.py"
    )
  done
  echo "Review evidence in $probes"
)
```

**Fresh start.** Six cases create their own synthetic qualification and acceptance inputs. The fixture uses the real freeze, workstation acceptance gate, foundation command parser, scheduler, projection and CPU census worker. It supplies fixture hardware facts and substitutes a local process for the container transport. The frozen roster has one task. Each case writes `dispatch-observation.json` beneath the corresponding pytest state directory. The accepted case records the frozen and projected input hashes and the path and digest of the worker’s `OUTCOME_BLIND_CENSUS` receipt. Missing or changed acceptance and a mismatched source, image or device must each produce zero dispatches. The small cohort can legitimately report a scientific deficit.

**Linked confirmation.** The fixture compiles the two same-seed research rows with targets of 12.5M and 25M cumulative tokens. Its CPU learner runs two updates per segment. An owned pause exposes the child’s inherited state before its first update. Require exact weights, AdamW state, RNG and scheduler/data state, the same prior tokens and update count, and the child’s full new token allowance. Reopen the child, advance two more updates and verify the parent’s remaining allowance and the child’s new charges. A weights-only initializer with a parent link must be rejected. The existing roster checks still cover the full targets and earlier designs. [First dispatch](validation/test_fresh_dispatch.py) · [Linked confirmation](validation/test_linked_confirmation.py)

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

### Check trial 00 before further training

T037 covers the screening link from `trial-00-1000000-train` to `trial-00-2000000-train`. T024 covers the later 12.5M→25M confirmation. Keep the original six 1M checkpoints, frozen campaign, study and controller transitions, their ledger, and generation/scoring receipts and producer files for the development and both retention panels. Compare the screening audit and its test with their original blobs and run their bounded cases from each checkout.

```bash
validation=specs/013-progressive-mathematical-discovery/validation
git diff --exit-code 5489aed2 HEAD -- "$validation/audit-round027-screening.py" "$validation/test_round027_screening_audit.py"
python -m pytest -q "$validation/test_round027_screening_audit.py"
git rev-parse --verify '77667e67^{commit}'
```

For the original check, copy the selected campaign, study, accounting, controller records and learning state with their referenced files to an isolated workspace. Authenticate the ordinal-zero proposal and selected finite parameter choice. Bind both frozen task IDs and the parent’s committed checkpoint to that choice, seed and compatibility. Verify every checkpoint file against its manifest and retain the original ledger prefix, token and update counts, spent charges and data position. On the copy, stop the child immediately after full-state restore and before its first update. Compare weights, optimizer buffers and step, scheduler, RNG streams, sampler/controller position and budget cuts with the saved parent. Check the ledger prefix and zero new training tokens. Record any maintenance-only accounting separately. Keep the live ledger, study decisions and qualified checkout intact. A bounded CPU result and a private original restore receive separate entries in the review record. The screening audit’s exploratory recoding grants no authority to advance the frozen trial.

## 5. Close the affected evidence gaps

- **Fresh and installed.** Use the [foundation CI partitions](../../.github/workflows/foundation.yml) on the pinned qualification image. Its runner names 132 test paths/nodes in four partitions (32, 34, 34, 32). Its merge checks their complete inventory. Exercise the directly built and sdist-rebuilt wheels outside the source tree, the public `oeis-learn` console and `pipeline --workspace`, Docker/Podman host commands and relevant native crates. The first-dispatch and linked-confirmation fixtures above provide CPU software evidence. A change affecting actual-device qualification needs the corresponding pinned workstation evidence. [Operator quickstart](../013-progressive-mathematical-discovery/quickstart.md)

### Stop the accepted-run rehearsal at the handoff

Record two results. SC-003 software accepts either an original copy or a representative saved state. On the same case, authenticate 47 distinct completed outputs and all files in four full-state checkpoints against their sealed references. Preserve the ledger prefix, commit the expected comparison once, observe `trial-00-propose` next and repeat on that copy with the engine blocked and the handoff intercepted both times. Require zero reruns, new research invocations, trial reservations, worker calls or GPU calls, and count the original charges once. Record the case identity, counts, charge key and before/after observations. T024–T025 must also pass for SC-003 software to close.

The **private round027 replay** additionally requires the original pinned helper, qualified checkout and genuine terminal publisher receipts. Steps 1–3 exercise that path. A representative pass can close the software observations while the private result stays open. A group that affects recovery of those originals needs the private pass before integration.

1. Create a disposable worktree at `77667e67` under `"$review_tmp/accepted-copy"`. Restore the sealed replay, session, producers, checkpoints, control and original ledger there using independent writable files. Keep the live workspace and its engine socket and GPU devices outside the rehearsal. Record the ledger prefix and original charges, task ownership, and hashes for all 47 saved outputs and all files in the four full-state checkpoints. Check these against the pinned replay on the copy. Save observations outside it.
2. For the helper path, start with the stopped comparison and a snapshot that has the genuine terminal publisher receipts and matching reservation, before the comparison output and transition were committed. Load the helper extracted in section 3 into the guarded process below. Its real `recover` and `finish` validate the evidence, reconcile the elapsed-only `diversity-comparison/publish/97` charge, record `INCOMPLETE / RETAIN_BASELINE` and check `RunStore.next_task()`. The helper then calls the workstation with `run-research`. Intercept that exact call before it can dispatch. A missing journal tail takes an earlier branch that constructs a host and can probe the GPU even for publication. Install all guards before entering the helper.

```bash
(
  set -e
  cd "$review_tmp/accepted-copy"
  python3 -IBS - "$review_tmp/resume.py" <<'PY'
import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

root, helper = Path.cwd(), Path(sys.argv[1])
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'ca54fae0a5812f93d99b8704a47c8482c8938eec6761538541ba72d7a38548ba'
sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
import foundation_local
from oeis_learn.pipeline_validation import progressive_roles, progressive_workstation

class HandoffReached(Exception):
    pass

def stop_handoff(request_root, args):
    assert request_root == root and args[-1] == 'run-research'
    session_path = 'run-state/discovery/013-research-round-027'
    assert args[args.index('--session') + 1] == session_path
    session = root / session_path
    assert not (session / 'run-research/execution-0001').exists()
    steps = json.loads((session / 'journal.json').read_bytes())['steps']
    assert not any(key.startswith('run-research/invocation-0001') for key in steps)
    raise HandoffReached

with (
    patch.object(progressive_workstation, 'main', side_effect=stop_handoff) as handoff,
    patch.object(foundation_local, 'Host', side_effect=AssertionError('engine access')) as host,
    patch.object(progressive_roles, 'invoke', side_effect=AssertionError('worker dispatch')) as worker,
):
    sys.argv = [str(helper), '--continue']
    try:
        runpy.run_path(str(helper), run_name='__main__')
    except HandoffReached:
        pass
    else:
        raise AssertionError('approved recovery did not reach the handoff')
    handoff.assert_called_once()
    host.assert_not_called()
    worker.assert_not_called()
PY
)
```

3. Reopen the copied store and assert `trial-00-propose` is next. Repeat the guarded invocation on the same copy. Require one comparison output and boundary across both attempts, one settled publication charge, the original ledger prefix intact, and the same 47 outputs and four checkpoints. Require zero new research invocations (`execution-0001` and its session journal step), trial reservations or screening dispatches. Confirm the live inputs still match their original hashes.
4. With only the stopped pre-publication state, authenticate its 47 outputs and four checkpoints. On a separate diagnostic copy, reconstruct the verified comparison table, call `check_table`, exercise CPU publication and use `finish` to commit and check the next task. A representative case can supply SC-003 software evidence once it also satisfies both guarded attempts and every observation above. A smaller interruption fixture or a count-only check supplies partial evidence. Record the private replay as open and keep synthetic receipts outside the accepted state.
5. Exercise dispatch on a tiny independent fixture. Project one ordinal-zero search request into the `search` channel, invoke the CPU `propose-trial` worker once and require a durable proposal in its fresh fixture journal. Stop there. Check screening, linked confirmation and separate final assessment with the fixtures in section 4. [Pinned helper](../013-progressive-mathematical-discovery/validation/continue-round027-publication.py) · [CPU commit fixture](../013-progressive-mathematical-discovery/validation/test_round027_publication_recovery.py) · [Proposal role](../../src/oeis_learn/pipeline_validation/progressive_role.py)

### Read a previously published ZIP with the candidate verifier

Use the already-published incomplete round027 ZIP from the diagnostics. In the independent review of `83684256`, the reviewed verifier reproduced the **entire saved receipt**. The ZIP’s SHA-256 before and after verification was `d24c9ad72551e927cb63b2b77b18f31ea585733b767990d14c4eddf764626c9e`. Both `status` and scientific `decision` were `INCOMPLETE`, the complete missing list was `["primary_comparison"]`, and the archive contained 352 accounting rows. Record the producing revision, report identity and saved receipt outside the reader workspace. For the accounting comparison, retain the original per-task/per-arm rows and totals, including caps and spend, elapsed time, and useful/discarded tokens.

In a fresh workspace expose only a read-only copy of that ZIP and the installed candidate verifier. Keep the campaign, producers, writer checkout and expected results inaccessible. Confirm the ZIP hash, run `oeis_learn.generalization.progressive.verify.verify` on it and capture the receipt outside the workspace. The CLI equivalent is `discovery verify-bundle --archive <report.zip> --output <fresh-verification>`. An incomplete report can yield exit 2 and a valid diagnostic receipt. Compare the full returned receipt with the saved one. After verification, decode the authenticated `tables/accounting-*.json` parts and compare every row and total against the independent baseline. Recheck the ZIP hash. Repeat this archive-only check for any candidate that changes verification or accounting. [Verifier](../../src/oeis_learn/generalization/progressive/verify.py) · [Original and fixture results](research.md#publication-review-probes)

### Publish and verify at the observed round027 scale

Exercise the candidate **writer and reader together** in a separate case. The independent review located exact byte counts and SHA-256s for all four generator/score pairs in the saved producer receipts. These provide fixture targets offline. For each frozen seed (`20261016`, `20261017`) and arm (static/baseline, refreshed/candidate), follow the registration to its two committed role results. The original task IDs take the form `diversity-{arm}-{seed}-primary-generate` and `diversity-{arm}-{seed}-primary-score` using `static` or `refreshed`. Check each result against its reference in the sealed replay, then record `metrics.candidates.bytes` and `metrics.candidates.sha256` from the generator and `metrics.scores.bytes` and `metrics.scores.sha256` from the matching scorer. The resulting four rows give all eight original lengths and digests. [Frozen roster](../../src/oeis_learn/generalization/progressive/research_roster.py) · [Producer receipt checks](../../src/oeis_learn/pipeline_validation/progressive_study.py)

Set `bundle` to the round027 diagnostics ZIP and `replay_json` to the saved `replay.json` from the comparison replay. The ZIP entries start at `run-research/`. Extract the result receipts beneath the session directory so the paths in the sealed replay resolve from `receipt_root`.

```bash
receipt_root=$(mktemp -d)
session_dir="$receipt_root/run-state/discovery/013-research-round-027"
mkdir -p "$session_dir"
unzip -q "$bundle" 'run-research/execution-0000/task-*/role/result.json' -d "$session_dir"
```

In the same shell, run the following command to authenticate the eight result receipts and print the four arm/seed rows. Keep the resulting table with the full-size fixture.

```bash
PYTHONPATH="$repo/src" python3 -BS - "$receipt_root" "$replay_json" <<'PY'
import csv
import json
import sys
from pathlib import Path
from oeis_learn.generalization.progressive.contracts import seal, verify_refs

root = Path(sys.argv[1])
replay = json.loads(Path(sys.argv[2]).read_bytes())
if seal(replay) != replay or replay['identity'] != 'sha256:4d67a539c4f58ba4ec6b9d4002d15b1ba85575b9eee8bd5d0496d211916f20d9':
    sys.exit('Expected the authenticated round027 replay')

def measure(task, operation, metric):
    cut = replay['outputs'][task]
    receipt = json.loads(verify_refs(root, [cut['result']])[0])
    if seal(receipt) != receipt or cut['operation'] != operation or receipt['operation'] != operation:
        sys.exit(f'Producer receipt differs for {task}')
    ref = receipt['metrics'][metric]
    return ref['bytes'], ref['sha256']

rows = []
for seed in (20261016, 20261017):
    for arm in ('static', 'refreshed'):
        task = f'diversity-{arm}-{seed}-primary'
        generator = measure(task + '-generate', 'generate', 'candidates')
        score = measure(task + '-score', 'score', 'scores')
        rows.append((arm, seed, *generator, *score))
writer = csv.writer(sys.stdout, delimiter='\t')
writer.writerow(('arm', 'seed', 'generator_bytes', 'generator_sha256', 'score_bytes', 'score_sha256'))
writer.writerows(rows)
PY
```

The original generators spanned approximately **45.5–52.2 MB each**, before scores. For full-size publication, use a copied terminal invocation or a generated fixture with representative payloads and compressibility. Check original files against all eight receipt identities when using original evidence. For a generated fixture, use the receipt lengths as targets and record the generated bytes and their own digests.

Include roughly 49 MB of calibration and roughly 290,000 journal events in the same fixture. Record actual calibration bytes, journal bytes and event count. Collect and publish with the round027 reporter’s frozen cap, `min(MAX_BYTES, campaign["payload"]["storage"]["package_bytes"])`, or pass that cap explicitly to `build(..., max_bytes=cap)` after collection. Keep the starting fixture read-only. Use a fresh output and an archive-only verifier for each row on both revisions.

| Full-scale case | Change on the disposable copy | Required result |
| --- | --- | --- |
| Intact baseline | Retain the complete registered generator/score pairs and selected comparison. | Publication and independent verification reproduce the baseline claim scope, accounting and complete missing-evidence list. Start with a comparison whose producer binding is exercised by the verifier. |
| Incomplete comparison | Retain the original bindings and supply the terminal state with an incomplete comparison at the same scale. | Record matching baseline integrity/status, scientific verdict and missing list. The comparison stays incomplete and its costs still reconcile. |
| Missing required producer | Omit a named generator or score using the reporting path for unavailable evidence. Keep its identity, arm, seed and affected comparison in the review record. | The independently verified report identifies the omission or missing comparison. Its dependent claim stays unverified. Check the surviving accounting against the parent. |
| Altered required producer | In the bound public comparison payload, replace one byte with an equal-length, valid change to the generator or score. Preserve the frozen original digest. Encode the changed payload through the writer so the outer ZIP is readable. | Require the offline verifier to reject the mismatched original producer bytes. Record the rejected binding and error. The valid baseline for the same producer must pass its integrity check. |

Retain the calibration, journal and unaffected pairs across all four cases. Record the single intended change, original and surviving byte counts, encoded-entry sum, final ZIP size, and the receipt or rejection for each. Inspect expanded public exports for the private marker. T033 closes when the full-size baseline and all three negative cases match their expected results. [Reporter cap](../013-progressive-mathematical-discovery/validation/report-round027.py) · [Producer bindings](../../src/oeis_learn/generalization/progressive/public_comparisons.py) · [Altered-binding example](../../tests/contract/test_progressive_bundle.py)

| Measurement | How to collect and use it |
| --- | --- |
| Original producer bytes | Record each of the eight file lengths before the inner transport encoding. Keep generator and score totals separate. |
| Encoded package bytes | Sum `ZipInfo.file_size` over **all** entries, including `manifest.json`. This reproduces the writer’s sum of encoded entries before ZIP compression and the verifier’s inventory-size check. Record the earlier input-table reference byte sum as a separate precheck. |
| Final ZIP bytes | Record the closed file’s `stat().st_size` and SHA-256. This measures storage and transport. The writer compares its package budget against the encoded-entry sum above. |

Exercise the package boundary on a disposable fixture with encoded-entry sum `N` greater than the input-table sum and within the absolute limit. With identical inputs, build and verify at `max_bytes=N`. At `N-1`, require the writer’s **packaged metadata** rejection before a ZIP is published and independently verify that the reader rejects the saved archive at that cap. Record all three measurements. Use a fixture that compresses to fewer than `N-1` bytes to expose a mistaken on-disk comparison. The four-by-5 MiB, space-padded projection test and separate 49 MiB policy test remain small regression controls. The observed sizes set acceptance-fixture targets only. [Writer](../../src/oeis_learn/generalization/progressive/report.py) · [Bundle tests](../../tests/contract/test_progressive_bundle.py)

**Disposition.** Compare each affected result on the group’s parent and candidate. Record any outstanding physical/private check or pre-existing failure under that group. Withhold a failing group and track actual files, bytes and symbols removed by accepted groups. [Record format](data-model.md)
