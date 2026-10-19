#!/usr/bin/env bash
# Compile only the native source closure; Python packaging happens later.
set -euo pipefail
export CARGO_TARGET_DIR=/opt/oeis-native-target
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
cd /opt/oeis-build/crates/oeis_wasm_evaluator
test -f Cargo.lock
cargo fmt -- --check
# This wheel uses the pinned image's libc++ runtime; no manylinux claim.
mkdir -p /opt/oeis-wheels
maturin build --release --locked --compatibility linux \
  --interpreter "$PYO3_PYTHON" --out /opt/oeis-wheels
mkdir -p /opt/oeis-build-info
# The bridge links the same core and retained adapter without Python extension mode.
cargo build --release --locked --no-default-features --example synthesis_core_test_driver
cp /opt/oeis-native-target/release/examples/synthesis_core_test_driver \
  /opt/oeis-build-info/synthesis-core-test-driver
cargo build --release --locked --no-default-features --bin synthesis_worker
cp /opt/oeis-native-target/release/synthesis_worker /opt/oeis-build-info/synthesis-worker
cp Cargo.lock /opt/oeis-build-info/Cargo.lock
cp ../oeis_bigint_runtime/Cargo.lock /opt/oeis-build-info/guest.Cargo.lock
cp ../oeis_synthesis_core/Cargo.lock /opt/oeis-build-info/core.Cargo.lock
