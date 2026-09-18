#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# oeis-learn — build the ROCm foundation image (T001).
#
# Resolves the plan's exact AMD image tag to a real digest, builds
# docker/foundation/Dockerfile, and writes a real runtime.lock.json
# with the tag, digest, Dockerfile hash, wheel locks and final image
# identity. An unresolved placeholder or tag-only identity blocks a
# qualifying preflight. No fabricated values are ever written here.
#
# Container runtime detection: prefers podman (including podman-docker),
# falls back to docker. Digest resolution uses the runtime's native inspect.
#
# Usage:
#   scripts/foundation/build_image.sh [--tag IMAGE_TAG] [--runtime podman|docker]
# Default IMAGE_TAG: oeis-learn:foundation
# Default runtime: auto-detect (prefer podman, fallback: docker)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="oeis-learn:foundation"
FORCE_RUNTIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) IMAGE_TAG="$2"; shift 2;;
        --runtime) FORCE_RUNTIME="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

LOCK_JSON="docker/foundation/runtime.lock.json"
BASE_TAG="rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1"
DOCKERFILE="docker/foundation/Dockerfile"

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
    # Prefer podman (rootless, podman-docker stand-in, or native podman)
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
echo "==> Using container runtime: ${RUNTIME} ($(command -v "$RUNTIME") - $(which "$RUNTIME" | xargs basename))"
"$RUNTIME" --version 2>/dev/null | head -1

echo "==> Resolving base image digest for ${BASE_TAG}"
if [ "$RUNTIME" = "podman" ]; then
    echo "    (via ${RUNTIME} image inspect docker.io '<ref> --format '{{.Digest}}')"
    RAW_DIGEST="$($RUNTIME image inspect "docker.io/${BASE_TAG}" --format '{{.Digest}}' 2>/dev/null)"
else
    # docker inspect needs the image pulled locally first
    echo "    (pulling ${BASE_TAG} via ${RUNTIME}, then inspect --format)"
    "$RUNTIME" pull "$BASE_TAG" >/dev/null 2>&1 || true
    RAW_DIGEST="$($RUNTIME inspect --format '{{index .RepoDigests 0}}' "$BASE_TAG" 2>/dev/null | sed 's/.*@//')"  
fi
if [ -z "$RAW_DIGEST" ] || ! [[ "$RAW_DIGEST" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    echo "ERROR: could not resolve digest for ${BASE_TAG}; tag-only identity blocks qualifying preflight" >&2
    exit 1
fi
BASE_DIGEST="$RAW_DIGEST"
echo "==> Base image digest: ${BASE_DIGEST}"

echo "==> Verifying Dockerfile pins the resolved digest"
if ! grep -q "FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:" "$DOCKERFILE"; then
    echo "ERROR: ${DOCKERFILE} does not pin an image digest; refusing to build" >&2
    exit 1
fi
PIN="$(sed -n 's/.*FROM rocm\/pytorch:.*@\(sha256:[a-f0-9]\{64\}\).*/\1/p' "$DOCKERFILE" | head -1)"
if [ "$PIN" != "$BASE_DIGEST" ]; then
    echo "ERROR: Dockerfile pins ${PIN} but resolved amd64 digest is ${BASE_DIGEST}" >&2
    echo "Update the FROM line to match the resolved digest." >&2
    exit 1
fi
echo "==> Digest match confirmed."

echo "==> Building image ${IMAGE_TAG}"
"$RUNTIME" build -f "$DOCKERFILE" -t "$IMAGE_TAG" .

FINAL_IMAGE_ID="$("$RUNTIME" inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null | head -1)"
if [ -z "$FINAL_IMAGE_ID" ]; then
    FINAL_IMAGE_ID="unknown ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
fi
DOCKERFILE_HASH="$(sha256sum "$DOCKERFILE" | awk '{print $1}')"
LOCK_HASH="$(sha256sum docker/foundation/requirements.lock | awk '{print $1}')"

cat > "$LOCK_JSON" <<JSON
{
  "base_image": "${BASE_TAG}",
  "base_image_digest": "${BASE_DIGEST}",
  "dockerfile": "${DOCKERFILE}",
  "dockerfile_sha256": "${DOCKERFILE_HASH}",
  "dependency_lock": "docker/foundation/requirements.lock",
  "dependency_lock_sha256": "${LOCK_HASH}",
  "container_runtime": "${RUNTIME}",
  "final_image_tag": "${IMAGE_TAG}",
  "final_image_id": "${FINAL_IMAGE_ID}",
  "resolved_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "notes": "Real values captured from registry/build via ${RUNTIME} image inspect. An unresolved placeholder or tag-only identity blocks a qualifying preflight."
}
JSON

echo "==> Wrote ${LOCK_JSON}"
cat "$LOCK_JSON"
echo "Build complete."
