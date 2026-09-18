# oeis-learn Docker Foundations

## Building

```bash
# Simple foundation image (python:3.11-slim, CPU, no GPU deps)
docker build -f docker/foundation/Dockerfile.simple -t oeis-learn:simple .
```

The image installs a Rust toolchain (via rustup) and compiles the native PyO3 WASM
evaluator (`crates/oeis_wasm_evaluator`) with maturin, so
`oeis_learn.sandbox.runner` uses the native Cranelift-JIT evaluator rather than the
pure-Python wasmtime fallback. Because the sandbox DNS allowlist blocks the sparse
crates.io index (`index.crates.io`), cargo is configured to use the git-based index
on github.com via `CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git`.

## Running the minimal training smoke-test

The container's default CMD runs `oeis-learn train` with the full config.
To run the focused smoke-test instead:

```bash
docker run --rm oeis-learn:simple python scripts/test_minimal_train.py
```

### What the smoke-test does

1. Seeds a 2-sequence DuckDB database (perfect squares, Fibonacci)
2. Builds a miniature `TriStreamEncoder` + `WatTransformerDecoder` (64-dim)
3. Runs encoder forward passes and prints latent stats
4. Performs a single-step encoder training (MSE to random target)
5. Runs a decoder sampling step through `WatProgramSampler`
6. Verifies the native WASM evaluator (builds a small WAT program and checks output)
7. Cleans up and reports success
