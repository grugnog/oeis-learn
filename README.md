# OEIS Learn

`oeis-learn` explores synthesis of exact integer-sequence programs in WebAssembly
Text (WAT). Its foundation workflow separates generic training data, twenty
visible terms, private evaluation truth, executable evidence and scoped proofs.
The tri-stream encoder, Transformer decoder, independent arithmetic oracle,
contained execution, solver repairs, learning/search and program-derived discovery
are implemented. Workstation, real release-source and bounded experimental
acceptance remain partial; no corpus coverage or speedup is claimed.

Start with the [spec 007 quickstart](specs/007-experiment-foundation/quickstart.md).
[Foundation operations](docs/foundation.md) explains profiles, roles, continuation,
proof limits and measurement decisions. The [handoff](specs/007-experiment-foundation/validation/handoff.md)
records which gates have executable evidence and which still require acceptance.

## Setup and CPU diagnostics

Python dependencies and developer tools are locked in `uv.lock`:

```bash
uv sync --locked
uv run oeis-learn foundation --help
OMP_NUM_THREADS=2 uv run python scripts/verify_phase11.py --output runs/007-cpu-review
```

Use a new output directory for each quickstart. The script freezes a small fixture
cohort, builds a verified generic pool, pauses/resumes a three-update CPU smoke,
evaluates actual weights, executes census/measurement diagnostics and archives
results. It requires Linux boot-identity access. Failures and skipped dependent
slices are recorded; a CPU diagnostic cannot qualify GPU training or release.

## Native and installed-wheel validation

The root `Dockerfile` is the sole image build: `qualification` for CPU/native CI,
`gpu` (the default) for portable ROCm training. The host wrapper needs only
standard-library Python and your normal Docker/Podman connection. See
[fresh training](docs/fresh-training.md) for reusable inputs, validation and long runs.
Install native build dependencies through these images:

```bash
docker build --target qualification -t oeis-007-qualification .
docker run --rm oeis-007-qualification python -m pytest -q tests/unit/test_foundation_handoff.py
```

[Foundation CI](.github/workflows/foundation.yml) builds the actual Rust extension,
executes G7/G8/G9, installs the wheel outside the checkout with `PYTHONPATH` unset,
and runs all six installed acceptance cases plus Docker role tests. GPU tests run
through `python3 scripts/foundation_run.py --engine podman validate --suite gpu`;
hardware skips remain pending evidence.

## Historical interfaces

The original ingestion, training, synthesis and discovery interfaces remain for
historical experiments. Legacy weights, imported teachers and old readiness labels
cannot qualify the strict foundation track. Conversion preserves history and does
not manufacture missing optimizer, RNG, provenance or containment evidence.
See [foundation migration and recovery](docs/foundation.md#migration-and-recovery)
before reusing old artifacts. Long-run or LODA comparisons are separate work.

## Spec 008 fitting migration

New fitting uses the pinned native service and independent final-source
verification. Use `python3 scripts/foundation_run.py --engine podman` with
`configs/foundation/bigint_*.yaml`; substitute Docker when preferred. The
`solve-constants` image-side diagnostic accepts `bigint-body/v1` parameter sources
or the qualified bare checked-i256 concrete subset, and emits explicit outcomes.
It no longer solves scalar/bit-vector `const_?` or recurrence skeletons. These
unsupported forms are not reinterpreted under bigint semantics.

| Retired surface | Supported replacement |
| --- | --- |
| Python constant/grounding/Dixon/QFNIA/modular-filter/recurrence fitters | `oeis_learn.synthesis.fitting.fit_program` → one native client → independent admission |
| Literal-neighbour `fresh.search.grounded_variants` | Phase 3 native fitting portfolio |
| Old grounding replay | New native scoped replay; old JSON stays `historical_unreplayed` |
| Legacy teacher ingestion | Inspect historical JSON or use independently verified training-archive admission |
| run008/009/010 scripts and exclusive configs | Portable foundation commands and versioned bigint recipes |
| `enable_z3_qfnia`, Dixon/filter switches and scalar fitting methods | Explicit `native-fitting/v1` settings; old choices are unsupported |

Retained signed-256 execution, codecs and checkpoint readers keep their original
numerical meaning. SymPy remains for active independent certificate/discovery
checks. Loop-certificate obligations now use the same pinned native Z3 dependency
through a bounded certificate checker, with no Python solver package or model
interface. Historical results/specifications remain unchanged; obsolete
executables can be reproduced from commit
`b96b92ff680a1509d3cede89e9abf0ddb65d2665`.

Check the source audit with `python3 scripts/inventory_synthesis_retirement.py --check`. The clean installed-image audit and replacement regressions run with:

```bash
python3 scripts/foundation_run.py --engine podman validate \
  --suite bigint-retirement --output tmp-validation/008-phase4-retirement-001
```

See [the Phase 4 evidence](specs/008-bigint-constant-synthesis/validation/phase-4.md)
for the current pass/pending gates. Retirement is not permission to launch a
research screen or change a checkpoint's profile.
