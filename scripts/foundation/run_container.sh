#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# oeis-learn — run the standalone foundation preflight in the
# ROCm container (T001). Exposes ONLY the existing /dev/kfd and the
# selected render node, using existing numeric device group IDs. No
# privileged mode, host networking/IPC, driver installation,
# architecture-spoofing override or broad host mount. The probe
# inspects EXISTING host GPU/device support; if the host cannot
# support the image it fails `hardware_not_ready` (never CPU).
#
# Container runtime detection: prefers podman (including podman-docker),
# falls back to docker. Podman-specific GPU flags are used when running
# native podman (CAP_SYS_ADMIN, `/dev/kfd` device, etc.).
#
# Usage:
#   scripts/foundation/run_container.sh [--image IMAGE_TAG] [--render DEV] [--runtime podman|docker]
# Default IMAGE_TAG: oeis-learn:foundation, RENDER=/dev/dri/renderD128
# Default runtime: auto-detect (prefer podman, fallback: docker)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="oeis-learn:foundation"
RENDER_NODE="/dev/dri/renderD128"
FORCE_RUNTIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE_TAG="$2"; shift 2;;
        --render) RENDER_NODE="$2"; shift 2;;
        --runtime) FORCE_RUNTIME="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

KFD="/dev/kfd"
if [ ! -e "$KFD" ]; then
    echo "hardware_not_ready: /dev/kfd absent on host (no AMD GPU device support). No CPU fallback; no driver install; no privileged container." >&2
    exit 3
fi
if [ ! -e "$RENDER_NODE" ]; then
    echo "hardware_not_ready: render node ${RENDER_NODE} absent on host. No CPU fallback." >&2
    exit 3
fi

# Existing numeric device group IDs (do not invent/create groups).
KFD_GID="$(stat -c '%g' "$KFD" 2>/dev/null || echo 0)"
RENDER_GID="$(stat -c '%g' "$RENDER_NODE" 2>/dev/null || echo 0)"

# ── Detect container runtime ─────────────────────────────────
detect_runtime() {
    if [ -n "$FORCE_RUNTIME" ]; then
        if [ "$FORCE_RUNTIME" = "podman" ] || [ "$FORCE_RUNTIME" = "docker" ]; then
            echo "$FORCE_RUNTIME"
            return
        fi
        echo "ERROR: --runtime must be 'podman' or 'docker', got '${FORCE_RUNTIME}'" >&2
        exit 1
    fi
    if command -v podman >/dev/null 2>&1 && podman --version >/dev/null 2>&1; then
        echo "podman"
        return
    fi
    if command -v docker >/dev/null 2>&1 && docker --version >/dev/null 2>&1; then
        echo "docker"
        return
    fi
    echo "ERROR: no container runtime found (tried podman, docker)" >&2
    exit 1
}

RUNTIME="$(detect_runtime)"
echo "==> Running foundation preflight in ${IMAGE_TAG} via ${RUNTIME}"
echo "    kfd=${KFD} (gid ${KFD_GID})  render=${RENDER_NODE} (gid ${RENDER_GID})"
"$RUNTIME" --version 2>/dev/null | head -1

OUT_DIR="docker/foundation/evidence"
mkdir -p "$OUT_DIR"
SRC="${REPO_ROOT}/docker/foundation"
CONTAINER_EVIDENCE="/app/docker/foundation/evidence"

START="$(date +%s)"
set +e   # preserve container exit code despite +e
if [ "$RUNTIME" = "podman" ]; then
    # Podman: use --device for KFD + render, --group-add for group access,
    # --cap-add=SYS_ADMIN for ROCm driver (HIP runtime requires /dev/kfd remapping).
    # --security-opt=no-new-privileges keeps enforcement tight.
    "$RUNTIME" run --rm \
        --device "$KFD" \
        --device "$RENDER_NODE" \
        --group-add "${KFD_GID}" \
        --group-add "${RENDER_GID}" \
        --cap-add=SYS_ADMIN \
        --security-opt=no-new-privileges \
        -v "$(pwd)/$OUT_DIR":${CONTAINER_EVIDENCE} \
        "$IMAGE_TAG" \
        python -m oeis_learn.cli.foundation_preflight \
            --config docker/foundation/preflight.yaml \
            --output docker/foundation/evidence \
            --hardware-only
else
    # Docker: same device exposure; SYS_ADMIN is explicit so ROCm KFD
    # works but no implicit privileged mode.
    "$RUNTIME" run --rm \
        --device "$KFD" \
        --device "$RENDER_NODE" \
        --group-add "${KFD_GID}" \
        --group-add "${RENDER_GID}" \
        --cap-add SYS_ADMIN \
        --security-opt=no-new-privileges \
        -v "$(pwd)/$OUT_DIR":${CONTAINER_EVIDENCE} \
        "$IMAGE_TAG" \
        python -m oeis_learn.cli.foundation_preflight \
            --config docker/foundation/preflight.yaml \
            --output docker/foundation/evidence \
            --hardware-only
fi
RC=$?
set -e

END="$(date +%s)"
echo "==> preflight container finished in $((END - START))s with exit code ${RC}"

# Exit 3 (hardware_not_ready) is an expected preflight outcome, not an
# internal error; preserve it for the evidence record.
exit "$RC"
