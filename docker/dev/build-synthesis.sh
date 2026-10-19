#!/usr/bin/env bash
# Shared pinned dependency smoke builds, cached once for CPU and GPU images.
set -euo pipefail
cd /opt/oeis-build
export CARGO_TARGET_DIR=/opt/oeis-native-target
rustup target add wasm32-unknown-unknown --toolchain 1.90.0
cargo build --release --locked --manifest-path crates/oeis_bigint_runtime/Cargo.toml \
  --target wasm32-unknown-unknown
cargo run --release --locked --manifest-path crates/oeis_synthesis_core/Cargo.toml \
  --example dependency_smoke > /opt/oeis-build-info/dependency-smoke.txt
cargo test --release --locked --manifest-path crates/oeis_synthesis_core/Cargo.toml \
  > /opt/oeis-build-info/core-tests.txt
cp /opt/oeis-native-target/wasm32-unknown-unknown/release/oeis_bigint_runtime.wasm \
  /opt/oeis-build-info/bigint-dependency-smoke.wasm
rustc --version > /opt/oeis-build-info/rust.txt
# Model-free Phase 2 diagnostic, packaged beside the exact guest it executes.
cargo test --release --locked --manifest-path crates/oeis_bigint_runtime/Cargo.toml \
  > /opt/oeis-build-info/guest-tests.txt
cargo build --release --locked --manifest-path crates/oeis_synthesis_core/Cargo.toml \
  --example runtime_probe
cp /opt/oeis-native-target/release/examples/runtime_probe /opt/oeis-build-info/runtime-probe
