#!/usr/bin/env bash
set -euo pipefail
cd /opt/oeis-build
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --no-build-isolation .
cd crates/oeis_wasm_evaluator
# Cargo.lock is currently ignored by the repository. Keep a generated lock in
# the image; use the checked-in lock if the project adds one.
export CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback
if [[ ! -f Cargo.lock ]]; then cargo generate-lockfile; fi
maturin build --release --locked --interpreter "$VIRTUAL_ENV/bin/python" --out /tmp/oeis-wheels
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps /tmp/oeis-wheels/*.whl
mkdir -p /opt/oeis-build-info
cp Cargo.lock /opt/oeis-build-info/Cargo.lock
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > /opt/oeis-build-info/python.txt
rustc --version > /opt/oeis-build-info/rust.txt
python -c 'import torch, oeis_wasm_evaluator; from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder; print(torch.__version__, torch.version.hip)'
