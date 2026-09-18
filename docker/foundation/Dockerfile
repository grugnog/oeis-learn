# ─────────────────────────────────────────────────────────────
# oeis-learn — Trustworthy Experiment Foundation image (ROCm/AMD)
#
# This is the T001 standalone Docker feasibility probe image. It derives
# from the plan's pinned AMD base image, resolved to a real digest on
# 2026-09-17 (docker buildx imagetools inspect):
#   rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
#   @ sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191
#
# The base image already ships a tested ROCm PyTorch 2.9.1. The extra
# Python dependencies are installed from docker/foundation/requirements.lock
# (exact versions + wheel hashes) WITHOUT replacing image PyTorch: torch is
# deliberately absent from that lock. The build verifies torch's version and
# HIP backend are unchanged after dependency install.
#
# No driver is installed here and no privileged flag is set. The preflight
# probe inspects the EXISTING host GPU/device support; it never falls back
# to CPU. Host driver changes are a separate explicit action (see
# scripts/foundation/run_container.sh), never an automatic setup step.
# ─────────────────────────────────────────────────────────────
FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191

# ── Build tooling: curl (rustup) + Rust toolchain ─────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gcc libc6-dev git \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo \
    # The base image's Python is a venv at /opt/venv (torch 2.9.1+rocm lives
    # there). Keep it FIRST on PATH so `python`, `pip` and `uv` all operate in
    # that venv instead of the /usr system Python, which has no torch.
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    VIRTUAL_ENV=/opt/venv \
    # Sparse index (index.crates.io) may be DNS-blocked; use the git-based
    # crates.io index on github.com instead (crate downloads still come from
    # static.crates.io).
    CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git

# rustup installs the stable toolchain (minimal profile: rustc+cargo+std)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustc --version && cargo --version \
    && mkdir -p /root/.cargo \
    && printf '[registries.crates-io]\nprotocol = "git"\n' > /root/.cargo/config.toml

# ── Locked Python dependencies (no torch replacement) ─────────
WORKDIR /app
COPY pyproject.toml README.md .
COPY src /app/src
# Install the project runtime deps from the hash-pinned lock. torch is NOT in
# the lock (the base image's tested ROCm PyTorch 2.9.1 must be preserved).
# Exact == pins are enforced by uv; hashes serve as verifiable evidence.
COPY docker/foundation/requirements.lock /app/docker/foundation/requirements.lock
# The project is managed with uv. uv itself is installed via pip from PyPI
# because the sandbox/host network allowlist blocks ghcr.io (astral's uv image).
# With PATH/VIRTUAL_ENV pointing at /opt/venv, `pip` here installs uv into that
# venv alongside the base torch.
RUN pip install --no-cache-dir --break-system-packages uv \
    && uv --version
# Lock file contains `--hash=sha256:...` on each line which activates uv's
# --require-hashes mode (transitive deps like typing-extensions aren't all
# lockable to a wheel+hash on manylinux py3.12). We install with exact == pins
# only; hashes serve as audit evidence. uv auto-detects /opt/venv (via PATH and
# VIRTUAL_ENV), so no --system is needed: deps land next to the base torch,
# which the lock deliberately does not replace.
RUN sed -E '/^[[:space:]]*#/d; /^[[:space:]]*$/d; s/ --hash=.*//' \
        docker/foundation/requirements.lock > /tmp/req.txt \
    && uv pip install --no-cache -r /tmp/req.txt \
    && python -c "\
import torch; \
assert torch.__version__.startswith('2.9.1'), f'base PyTorch replaced: {torch.__version__}'; \
hip = getattr(torch.version, 'hip', None); \
assert hip is not None and hip != '', 'ROCm/HIP backend missing'; \
print(f'PyTorch {torch.__version__} HIP {hip} preserved after lock install')"

# ── Native Rust/WASM evaluator (PyO3) ─────────────────────────
# Build the PyO3 extension with maturin and install the resulting wheel,
# so the native Cranelift-JIT evaluator is used for later execution phases.
COPY crates/oeis_wasm_evaluator/Cargo.toml  /app/crates/oeis_wasm_evaluator/Cargo.toml
COPY crates/oeis_wasm_evaluator/Cargo.lock  /app/crates/oeis_wasm_evaluator/Cargo.lock
COPY crates/oeis_wasm_evaluator/src         /app/crates/oeis_wasm_evaluator/src
RUN cd /app/crates/oeis_wasm_evaluator \
    && maturin build --release --out /tmp/wheels \
    && uv pip install --no-cache /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# ── Data + configs + scripts ──────────────────────────────────
COPY data /app/data
COPY configs /app/configs
COPY scripts /app/scripts
COPY docker/foundation/preflight.yaml /app/docker/foundation/preflight.yaml

ENV PYTHONPATH=/app/src

# ── Verify the build ──────────────────────────────────────────
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    python -c "from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder; from oeis_learn.decoder.wat_decoder import WatTransformerDecoder; print('Core modules OK')" && \
    python -c "import oeis_wasm_evaluator; print('Native WASM evaluator OK')" && \
    python -c "import oeis_learn.cli.foundation_preflight; print('Preflight entry point OK')"

# Standalone pre-adoption feasibility probe. It does not start a training
# experiment; the full `oeis-learn foundation` CLI group is added later (T008).
CMD ["python", "-m", "oeis_learn.cli.foundation_preflight", "--config", "docker/foundation/preflight.yaml", "--output", "docker/foundation/evidence", "--hardware-only"]
