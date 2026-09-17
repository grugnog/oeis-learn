# Build from the repository root. Override PI_BASE_IMAGE for a newer Pi image.
ARG PI_BASE_IMAGE=sbx/pi-image:20260913-1e007109ec462d446025df6838fae797c36b70a5
FROM ${PI_BASE_IMAGE}
# Preserve the base image's user, entrypoint and command. Pi auto-loads these.
COPY --chown=root:root pi/llama-swap.js /root/.pi/agent/extensions/llama-swap.js
COPY --chown=agent:agent pi/llama-swap.js /home/agent/.pi/agent/extensions/llama-swap.js
RUN uv tool install specify-cli && \
    git config --global --add safe.directory /workspace
