# Verified reconstruction — 2026-10-31

The earlier conversational T071–T079 completion report and commit references are
withdrawn. This is a new reconstruction from the checked-in specification and plan,
not a recovery of those claimed commits or the unavailable original task wording.

## Git provenance

Verified baseline: `6e63421e2d8ef137d01d1d701f9fb0205b012059`, fetched from
`origin/011-controlled-generalization`. Work is on `011-reconstruct-convergence` in
an isolated worktree; the older checkout's uncommitted changes were preserved.

| Local bundle commit | Work |
| --- | --- |
| `1cdc6e6` | Explicit reconstruction task definitions |
| `bfda21b` | Common selection rosters; safe unsupported learning targets |
| `bc789e1` | Frozen execution binding and named experiment policies |
| `153d099` | Original uniform provenance and explicit weights-only descriptors |
| `e7df926` | Production preparation/runtime/selection integration |
| `f5a7c5d` | Input assembly, examples and representative serial acceptance |
| `23febbb` | Charged duplicate refresh rejection and legacy schema compatibility |
| `94eeaac` | Registered regressions and actual validation/environment evidence |

The final delivery also includes validation/documentation commits. Verify the
bundle's advertised branch and commit list with `git bundle list-heads` and
`git log 6e63421..011-reconstruct-convergence`; do not use conversational hashes
as a substitute for Git objects. The initial delivery was local; the subsequent
GitHub connector publication is recorded below. No HIP run has been executed.

## Actual local validation

Existing Python 3.12 interpreter and the existing workspace dependency directory
were used; no host packages, services or drivers were installed. The restored
virtual environment lacks its interpreter launcher, so these checks used:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:../phase-a-venv/lib/python3.12/site-packages
python -m pytest tests/unit/test_generalization* tests/contract/test_generalization* \
  tests/integration/test_generalization* -q -p no:cacheprovider -m 'not gpu'
```

The original baseline selection passed **217 tests, 1 GPU deselected**. The final
broader run included **242 generalization tests, all passing**. Its exact selection
was the union of these globs, with the unavailable collection-only module removed:

```python
patterns = [
    'tests/unit/test_generalization*', 'tests/contract/test_generalization*',
    'tests/integration/test_generalization*', 'tests/unit/test_bigint*',
    'tests/contract/test_bigint*', 'tests/integration/test_bigint*',
    'tests/unit/test_pipeline*', 'tests/unit/test_pilot_followup*',
    'tests/unit/test_bootstrap_bundle.py',
]
files = sorted({p for pattern in patterns for p in glob.glob(pattern)}
               - {'tests/unit/test_bigint_ci_regressions.py'})
pytest.main(files + ['-q', '-p', 'no:cacheprovider', '-m', 'not gpu and not docker'])
```

Final broader result: **963 passed, 13 failed, 35 skipped, 27 deselected, 77 subtests
passed** in 235.91 seconds. Each of the 13 failures was inspected:

- Ten failed because `wasmtime` is unavailable: three bigint-build cases, CPU bigint
  workflow, engine-policy source inventory, model-sizing subprocess, two soak state
  cases and two pilot-runtime resume cases.
- Two failed because `/opt/oeis-build-info/synthesis-worker` is unavailable:
  generated-sizing native path and public scoring-cost sentinel admission.
- One requires the image's `/work` directory/mode, absent on this host.
- Initial unrestricted collection also failed in `test_bigint_ci_regressions.py`
  because of the same missing `wasmtime` import. It was explicitly omitted from the
  subsequent run; no tests were modified to hide the missing environment.

Source distribution creation and installed `build_py` succeeded. The installed
qualification snapshot was independently checked against current source: **892
matching entries**, including new policies, helper and owning tests. The sdist
contains the added modules, tests, schemas and example blueprints. The host assembler
runs under `python -S`, with no third-party imports. Git whitespace checks pass.
The existing current-dispositions implementation hash was refreshed using the
package's actual canonical serializer; no historical disposition claims changed.

The serial fixture exercised assembly, dry-run, all five case routes, original
qualification consumption, report/archive verification and stale-proof rejection.
Its dry-run has ten serial stages and exactly one qualification; it launches no
engine. CPU tensor tests compare actual updates and resumed state, but their native,
container and qualification boundaries are fixtures. They do not qualify HIP.

## Uploaded uniform evidence

The new read-only audit was also run against `study-20261016.zip` and
`uniform-import-metadata.zip`. Recipe byte hash, canonical recipe identity,
prepared-corpus identity, original uniform producer and loss history agree.
Historical best-loss exposure is **17,885,681 tokens / 18,433 updates**; latest is
**24,999,970 tokens / 25,764 updates**. User-side weight hash matches were reported
previously; the private tensors are not in this workspace. The importer checks
those actual hashes again on the operator machine before copying bytes.

## Remaining gate and next work

T063–T068 remain open for pinned-image/native/container/operator/HIP evidence.
No HIP command or research campaign ran. Complete original qualification evidence
is not replaced by any of the local checks above.

The uniform import supplies E0 weights and provenance. It deliberately does not
create full-state resume or a selection-designated P4 initializer. A runnable
four-case manifest still requires real reduced mechanical inputs, inspected image
IDs and the P4 initializer's original selection/producer closure. The uploaded
metadata alone does not supply those artifacts.

If no qualifying initializer already exists, the next implementation is an
acceptance-only bootstrap dependency within the same serial invocation. It must
reuse the original qualification and an existing bounded training case, select
using frozen selection supervision, publish actual original initializer evidence,
and resolve P4 through a new explicit authority transition. It must not relabel
historical pooled-loss selection, fabricate provenance, add hidden learner work,
or require a second normal HIP invocation. That bootstrap is **not implemented**
by this reconstruction and requires a Phase-B contract/policy review before wiring.

## Transfer the verified reconstruction

The incremental `generalization-reconstruction.bundle` contains this branch's Git
objects after the verified baseline. Download it into your repository root, then:

```bash
git fetch origin 011-controlled-generalization
git bundle verify ./generalization-reconstruction.bundle
git fetch ./generalization-reconstruction.bundle 011-reconstruct-convergence
git switch -c 011-reconstruct-convergence FETCH_HEAD
```

These commands create a new branch without resetting existing work. If Git reports
local changes that prevent switching, preserve them before retrying. The bundle
requires baseline `6e63421e2d8ef137d01d1d701f9fb0205b012059`; verification explicitly
checks that prerequisite. The quickstart documents original uniform import and
input assembly. The pending initializer/bootstrap gate above still applies.

## Publication to the existing PR

Published to existing branch `011-controlled-generalization`, PR #62, through the
GitHub connector. Each of the nine uploaded commit trees exactly matches its
corresponding local commit tree. The connector assigns new commit metadata and
therefore new commit IDs. No replacement branch or PR is needed.

| Local commit | GitHub commit |
| --- | --- |
| `1cdc6e653723d488c551567371b7a984bf7aafa2` | `6dcf13fba29e42efccc5add88d5b69bcd1e2aa73` |
| `bfda21b2667b2b031be345c822b364b1ea7834d0` | `cfda978f28afe8cad44dc304a82bf057ccf0565d` |
| `bc789e1a3629d42bba2d645b3451a5679a00223a` | `af790e0e7278dee350e0665ed7446be5fa0b46f8` |
| `153d09969d1c7600e88c7e9ecbf82c9363b673e6` | `f391c3297f8319727b8064c8b956c6ba9eee21bd` |
| `e7df926267d975a7fe1ca877cd2cc60d9bde5a0d` | `84d344e57d59124721f51d7f264894d275cb8b38` |
| `f5a7c5dda16e80c1ea99cb8dc0336adb7004ffc6` | `25ef3f5e0f901a76ff23832ae5c8f0132f426a80` |
| `23febbbbd0171b708d851065b5432042521c01db` | `ad1c464c5f8de880a1db0e3a4c4b331d304abc25` |
| `94eeaac84f0365c5902b25e421d612f8e4784de7` | `e1d1c697a8bcb942d8368da6592a164e7da10ba6` |
| `c5d0b7f78fa8f567f63752d374ae3f1eb0c63f77` | `95e727a0c168d2b5177ec885db9e52cf4ede8b5b` |

Use the existing branch directly:

```bash
git fetch origin 011-controlled-generalization
git switch 011-controlled-generalization
git merge --ff-only origin/011-controlled-generalization
```

The Git bundle is an alternative containing the original local commit IDs; do not
merge both histories. CI results must be checked on the published head. The
initializer/bootstrap and real-input gates above still apply.
