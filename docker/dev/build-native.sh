#!/usr/bin/env bash
set -euo pipefail
cd /opt/oeis-build
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --no-build-isolation .
export CARGO_TARGET_DIR=/opt/oeis-native-target
cd crates/oeis_wasm_evaluator
# The committed lock is part of the native source/build identity.
test -f Cargo.lock
cargo fmt -- --check
maturin build --release --locked --interpreter "$VIRTUAL_ENV/bin/python" --out /tmp/oeis-wheels
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps /tmp/oeis-wheels/*.whl
mkdir -p /opt/oeis-build-info
cp Cargo.lock /opt/oeis-build-info/Cargo.lock
cp ../oeis_bigint_runtime/Cargo.lock /opt/oeis-build-info/guest.Cargo.lock
cp ../oeis_synthesis_core/Cargo.lock /opt/oeis-build-info/core.Cargo.lock
cd /opt/oeis-build
python scripts/inspect_bigint_build.py --output /opt/oeis-build-info/bigint-build.json
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > /opt/oeis-build-info/python.txt
rustc --version > /opt/oeis-build-info/rust.txt
python -c 'import torch, oeis_wasm_evaluator; from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder; print(torch.__version__, torch.version.hip)'

# COPY preserves private checkout permissions, and native/package installation
# can create new root-only files after install.sh. Every runtime role uses an
# arbitrary non-root UID, so normalize the final image in both build targets.
# Keep ownership and write permissions unchanged; X only adds directory traversal
# and preserves executability for files that were already executable.
chmod -R a+rX /opt/oeis-build /opt/oeis-build-info "$VIRTUAL_ENV"
