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
    wheel = (ROOT / "docker/dev/build-native-wheel.sh").read_text()
    native = (ROOT / "docker/dev/build-native.sh").read_text()
    assert "uv python install 3.12.12" in setup
    assert "--prune torch" in setup and "--require-hashes --no-deps" in setup
    assert "cmp /tmp/oeis-torch-before /tmp/oeis-torch-after" in setup
    assert "maturin build --release --locked" in wheel
    assert "--no-deps /opt/oeis-wheels/*.whl" in native
    assert "python scripts/inspect_bigint_build.py" in native


def test_runtime_state_is_excluded_from_images():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    for name in ("run-state/", "container-cache/", "system/", "**/.ssh/", "data/*", "downloads/", "**/*.duckdb"):
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


def test_only_registered_benchmark_fixtures_enter_image_context():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    fixtures = ("symbolic_definitions_v1.json", "trustworthy_synthesis_v1.json")
    # Parent traversal is explicit; arbitrary siblings, caches and future datasets
    # must not become image inputs just because benchmarks need their directory.
    exceptions = [line for line in ignored if line.startswith("!data/")]
    assert exceptions == ["!data/benchmarks/", *["!data/benchmarks/" + name for name in fixtures]]
    assert ignored.index("data/*") < ignored.index("!data/benchmarks/")
    assert ignored.index("!data/benchmarks/") < ignored.index("data/benchmarks/*")
    for name in fixtures:
        assert ignored.index("data/benchmarks/*") < ignored.index("!data/benchmarks/" + name)
        assert (ROOT / "data/benchmarks" / name).is_file()
    # COPY is resolved against Docker's actual filtered context in both builds.
    # It fails before installation/qualification if either fixture is omitted.
    recipe = (ROOT / "Dockerfile").read_text()
    copy = "COPY " + " ".join("data/benchmarks/" + name for name in fixtures) + " /opt/oeis-build/data/benchmarks/"
    assert recipe.count(copy) == 2
