#!/usr/bin/env bash
set -euo pipefail
cd /opt/oeis-build
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --no-build-isolation .
# Compiled before the full checkout COPY; native_identity below rejects a stale
# wheel against the exact current Rust/build inputs rather than rebuilding here.
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps /opt/oeis-wheels/*.whl
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
