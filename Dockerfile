# Build from the repository root. Default target remains the Pi/SmolVM image.
ARG PI_BASE_IMAGE=sbx/pi-image:20260913-1e007109ec462d446025df6838fae797c36b70a5
ARG GPU_BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191

FROM ${GPU_BASE_IMAGE} AS gpu
USER root
ENV PATH=/opt/venv/bin:/opt/cargo/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
COPY docker/dev /opt/oeis-setup
COPY pyproject.toml uv.lock README.md /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh gpu
COPY src /opt/oeis-build/src
COPY crates/oeis_wasm_evaluator /opt/oeis-build/crates/oeis_wasm_evaluator
RUN bash /opt/oeis-setup/build-native.sh
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
# The broker clears entrypoint and supplies argv; no automatic training.
CMD ["bash"]

FROM ${PI_BASE_IMAGE} AS smolvm
USER root
RUN uv tool install specify-cli && \
    git config --global --add safe.directory /workspace
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    UV_PYTHON_INSTALL_DIR=/opt/oeis-python
COPY docker/dev /opt/oeis-setup
COPY pyproject.toml uv.lock README.md /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh cpu
ENV VIRTUAL_ENV=/opt/oeis-venv \
    PATH=/opt/oeis-venv/bin:/opt/cargo/bin:${PATH}
COPY src /opt/oeis-build/src
COPY crates/oeis_wasm_evaluator /opt/oeis-build/crates/oeis_wasm_evaluator
RUN bash /opt/oeis-setup/build-native.sh
ENV PYTHONPATH=/workspace/src
COPY --chown=root:root --chmod=555 system/lab-gpu* /usr/local/bin/
RUN usermod -u 1000 -g 100 agent
USER agent
RUN pi install npm:@juicesharp/rpiv-ask-user-question && pi install npm:pi-goal-list-loop-audit
COPY --chown=agent:agent --chmod=500 system/.ssh /home/agent/.ssh
COPY --chown=agent:agent system/llama-swap.js /home/agent/.pi/agent/extensions/llama-swap.js
WORKDIR /workspace
# Inherit the Pi base entrypoint/CMD; Smolfile supplies its long-lived command.
