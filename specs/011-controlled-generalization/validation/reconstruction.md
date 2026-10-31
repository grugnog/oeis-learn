# Verified reconstruction — 2026-10-31

The earlier conversational T071–T079 completion report and commit references are
withdrawn. This is a new reconstruction from the checked-in specification and plan,
not a recovery of those claimed commits or the unavailable original task wording.

## Git provenance

Verified baseline: `6e63421e2d8ef137d01d1d701f9fb0205b012059`, fetched from
`origin/011-controlled-generalization`. Work is on `011-reconstruct-convergence` in
an isolated worktree; the older checkout's uncommitted changes were preserved.

| Actual commit | Work |
| --- | --- |
| `1cdc6e6` | Explicit reconstruction task definitions |
| `bfda21b` | Common selection rosters; safe unsupported learning targets |
| `bc789e1` | Frozen execution binding and named experiment policies |
| `153d099` | Original uniform provenance and explicit weights-only descriptors |
| `e7df926` | Production preparation/runtime/selection integration |
| `f5a7c5d` | Input assembly, examples and representative serial acceptance |
| `23febbb` | Charged duplicate refresh rejection and legacy schema compatibility |

The final delivery also includes validation/documentation commits. Verify the
bundle's advertised branch and commit list with `git bundle list-heads` and
`git log 6e63421..011-reconstruct-convergence`; do not use conversational hashes
as a substitute for Git objects. No changes were pushed and no remote CI run was
started by this reconstruction.

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
