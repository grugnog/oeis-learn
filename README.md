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
`gpu` for the existing ROCm workstation workflow, and the default `smolvm` target
for development. Install native build dependencies through these images:

```bash
docker build --target qualification -t oeis-007-qualification .
docker run --rm oeis-007-qualification python -m pytest -q tests/unit/test_foundation_handoff.py
```

[Foundation CI](.github/workflows/foundation.yml) builds the actual Rust extension,
executes G7/G8/G9, installs the wheel outside the checkout with `PYTHONPATH` unset,
and runs all six installed acceptance cases plus Docker role tests. GPU tests run
separately using `lab-gpu exec --`; hardware skips remain pending evidence.

## Historical interfaces

The original ingestion, training, synthesis and discovery interfaces remain for
historical experiments. Legacy weights, imported teachers and old readiness labels
cannot qualify the strict foundation track. Conversion preserves history and does
not manufacture missing optimizer, RNG, provenance or containment evidence.
See [foundation migration and recovery](docs/foundation.md#migration-and-recovery)
before reusing old artifacts. Long-run or LODA comparisons are separate work.
