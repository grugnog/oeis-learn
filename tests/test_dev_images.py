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


def test_only_registered_data_fixtures_enter_image_context():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    fixtures = {
        "benchmarks": ("symbolic_definitions_v1.json", "trustworthy_synthesis_v1.json"),
        "sample_oeis": ("names", "stripped"),
    }
    expected_exceptions = []
    expected_files = []
    recipe = (ROOT / "Dockerfile").read_text()
    for directory, names in fixtures.items():
        parent = "data/" + directory + "/"
        paths = [parent + name for name in names]
        expected_files.extend(paths)
        expected_exceptions += ["!" + parent, *["!" + path for path in paths]]
        assert ignored.index("data/*") < ignored.index("!" + parent)
        assert ignored.index("!" + parent) < ignored.index(parent + "*")
        for path in paths:
            assert ignored.index(parent + "*") < ignored.index("!" + path)
            assert (ROOT / path).is_file()
        # Docker resolves these explicit inputs against its actual filtered context
        # in both targets, failing the build immediately for an omitted fixture.
        assert recipe.count("COPY " + " ".join(paths) + " /opt/oeis-build/" + parent) == 2
    assert [line for line in ignored if line.startswith("!data/")] == expected_exceptions
    # Check all tracked data in a checkout, and all copied data in an image.
    # Container build contexts deliberately exclude Git metadata.
    assert image_data_inventory(ROOT) == sorted(expected_files)


def image_data_inventory(root):
    if (root / ".git").exists():
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "data"], cwd=root
        ).decode().split("\0")
        return sorted(p for p in tracked if p and p != "data/.gitkeep")
    return sorted(
        p.relative_to(root).as_posix()
        for p in (root / "data").rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() != "data/.gitkeep"
    )


def test_image_fixture_inventory_without_git_metadata(tmp_path, monkeypatch):
    import shutil
    import pytest
    for path in ["Dockerfile", ".dockerignore", *image_data_inventory(ROOT)]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    assert not (tmp_path / ".git").exists()
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    def no_git(*args, **kwargs):
        raise AssertionError("image validation must not require Git metadata")
    monkeypatch.setattr(subprocess, "check_output", no_git)
    test_only_registered_data_fixtures_enter_image_context()
    # The Git-free route still rejects unexpected data and missing fixtures.
    extra = tmp_path / "data/unregistered-corpus.txt"
    extra.write_text("must not enter an image")
    with pytest.raises(AssertionError):
        test_only_registered_data_fixtures_enter_image_context()
    extra.unlink()
    (tmp_path / "data/sample_oeis/stripped").unlink()
    with pytest.raises(AssertionError):
        test_only_registered_data_fixtures_enter_image_context()


def test_curated_sample_is_complete_at_image_fixture_paths(tmp_path):
    import shutil
    from oeis_learn.data.real_data_loader import RealOeisDataLoader
    sample = tmp_path / "opt/oeis-build/data/sample_oeis"
    sample.mkdir(parents=True)
    for name in ("names", "stripped"):
        shutil.copyfile(ROOT / "data/sample_oeis" / name, sample / name)
    records = RealOeisDataLoader(sample_dir=str(sample)).load_local_benchmark_records()
    assert len(records) >= 20
    assert {r.curriculum_stage for r in records} == {1, 2, 3, 4, 5}
    assert {"A000045", "A000032", "A000217", "A000290", "A000079", "A000142",
            "A000108", "A000040", "A000005", "A003188"} <= {r.oeis_id for r in records}
