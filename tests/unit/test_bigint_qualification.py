"""Qualification reports must be collectable outside the container UID."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("exit_code", [0, 1])
def test_reports_exclude_private_pytest_scratch_even_on_failure(tmp_path, monkeypatch, exit_code):
    script = Path(__file__).resolve().parents[2] / "scripts/qualify_bigint_phase2.py"
    spec = importlib.util.spec_from_file_location("phase2_qualification", script)
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fixture",
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(device_count=lambda: 0),
        ),
    )
    monkeypatch.setattr(qualification, "ROOT", tmp_path / "source")
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/opt/oeis-build-info/bigint-build.json":
            return "{}"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    output = tmp_path / "evidence"
    scratch_paths = []

    def run(argv, *, check):
        scratch = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--basetemp=")))
        assert not scratch.is_relative_to(output)
        scratch.mkdir(mode=0o700)
        (scratch / "private-fixture.json").write_text("test scratch, not evidence")
        scratch_paths.append(scratch)
        xml = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
        xml.write_text(
            f'<testsuites><testsuite tests="1" failures="{exit_code}" errors="0" skipped="0"/></testsuites>'
        )
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    assert qualification.inside(output, gpu=False) == exit_code
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == ("PASS" if exit_code == 0 else "FAIL")
    assert report["tests"]["failures"] == exit_code
    assert scratch_paths and all(not path.parent.exists() for path in scratch_paths)
    assert sorted(path.name for path in output.iterdir()) == ["summary.json", "tests.xml"]
