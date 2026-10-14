"""Portable image recipes and standard-library host entrypoints."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_public_default_gpu_and_cpu_targets():
    recipe = (ROOT / "Dockerfile").read_text()
    stages = [l for l in recipe.splitlines() if l.startswith("FROM ")]
    assert stages[-1].endswith("AS gpu")
    assert any(l.endswith("AS qualification") for l in stages)
    assert "PI_BASE_IMAGE" not in recipe and "system/" not in recipe
    assert "sha256:" in recipe


def test_native_and_python_policy():
    setup = (ROOT / "docker/dev/install.sh").read_text()
    native = (ROOT / "docker/dev/build-native.sh").read_text()
    assert "uv python install 3.12.12" in setup
    assert "--prune torch" in setup and "--require-hashes --no-deps" in setup
    assert "cmp /tmp/oeis-torch-before /tmp/oeis-torch-after" in setup
    assert "maturin build --release --locked" in native


def test_runtime_state_is_excluded_from_images():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    for name in ("run-state/", "container-cache/", "system/", "**/.ssh/"):
        assert name in ignored


def test_host_help_needs_only_standard_library():
    for name in ("foundation_run.py", "foundation_local.py"):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "scripts" / name), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--engine" in result.stdout
