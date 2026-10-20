"""Qualification cannot promote skipped evidence or leak prepared/private payloads."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def helper():
    path = Path(__file__).resolve().parents[2] / "scripts/qualify_bigint_phase5.py"
    spec = importlib.util.spec_from_file_location("phase5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["failure", "error", "skipped", "empty"])
def test_zero_exit_cannot_promote_empty_failed_or_skipped_junit(tmp_path, failure):
    path = tmp_path / "results.xml"
    case = f"<testcase><{failure}/></testcase>" if failure != "empty" else ""
    path.write_text("<testsuites><testsuite>" + case + "</testsuite></testsuites>")
    with pytest.raises(ValueError):
        helper().junit(path)


def test_bounded_archive_reserves_summary_and_omits_private_runtime_and_weights(tmp_path):
    module = helper()
    root = tmp_path / "results"
    root.mkdir()
    (root / "summary.json").write_text('{"status":"FAIL"}')
    for name in (
        "private/truth.json",
        "inputs/values.json",
        "roles/scorer/input/truth.json",
        "learnability-run/roles/scorer/result.json",
        ".runtime/home/compiler.json",
        "model.pt",
    ):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("secret")
    (root / "stderr.log").write_bytes(b"bounded diagnostics\n" * 100000)
    (root / "link.json").symlink_to(root / "private/truth.json")
    output = module.archive(root, cap=2 << 20)
    with zipfile.ZipFile(output) as z:
        assert set(z.namelist()) == {"summary.json", "archive-manifest.json"}
        manifest = json.loads(z.read("archive-manifest.json"))
        assert manifest["omissions"][0]["path"] == "stderr.log"
        assert manifest["omissions"][0]["sha256"] == module.sha(root / "stderr.log")


def test_every_required_suite_node_exists_and_host_help_needs_only_stdlib():
    module = helper()
    for suite in module.GROUPS.values():
        assert suite and all((module.ROOT / name.split("::")[0]).is_file() for name in suite)
    result = subprocess.run(
        [sys.executable, "-S", str(module.ROOT / "scripts/qualify_bigint_phase5.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "--gpu" in result.stdout
