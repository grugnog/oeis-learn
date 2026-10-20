"""Qualification cannot promote skipped evidence or leak prepared/private payloads."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

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


def test_supplied_old_image_cannot_qualify_a_new_checkout(tmp_path, monkeypatch):
    module = helper()
    output = tmp_path / "mismatch"

    class Host:
        def __init__(self, args):
            self.policy = dict(runtime="fixture")
            self.engine = SimpleNamespace(name="docker")

        def ensure(self):
            return "sha256:" + "a" * 64

    monkeypatch.setitem(
        sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=lambda s: s)
    )
    launched = []

    def stage(host, image, path, argv, **kwargs):
        launched.append(path.name)
        module.write(path / "summary.json", dict(status="PASS", source_sha256="sha256:" + "0" * 64))

    monkeypatch.setattr(module, "container_stage", stage)
    monkeypatch.setattr(sys, "argv", ["qualify", "--output", str(output)])
    assert module.main() == 1
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "FAIL" and "image source differs" in report["error"]
    assert launched == ["arithmetic"] and output.with_suffix(".zip").is_file()


@pytest.mark.parametrize(
    "extra", [[], ["--gpu"], ["--gpu", "--inputs", "fixture", "--learnability"]]
)
def test_targeted_measurement_rejects_missing_inputs_or_training(extra, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--performance-only", "--output", "unused", *extra]
    )
    with pytest.raises(SystemExit) as error:
        helper().main()
    assert error.value.code == 2


@pytest.mark.parametrize("eligible,code", [(False, 2), (True, 0)])
def test_targeted_measurement_never_runs_training_or_claims_full_qualification(
    tmp_path, monkeypatch, eligible, code
):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.synthesis.contracts import identity
    import oeis_learn.container_validation as validation

    module = helper()
    output = tmp_path / "targeted"
    snapshot = {"fixture": "immutable"}
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: snapshot)

    class Host:
        def __init__(self, args):
            self.policy = {"runtime": {}}
            self.engine = SimpleNamespace(name="podman")

        def ensure(self):
            return "sha256:" + "a" * 64

    monkeypatch.setitem(
        sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=lambda s: s)
    )
    launches, stages, validations = [], [], []
    monkeypatch.setattr(module, "launch", lambda argv, *a, **kw: launches.append(argv))
    monkeypatch.setattr(module, "shape_manifest", lambda inputs: {"fixture": "shapes"})

    def validate(host, name, path):
        validations.append(name)
        module.write(path / "validation.json", {"status": "PASS"})

    def stage(host, image, path, argv, **kw):
        stages.append(path.name)
        module.write(
            path / "benchmark.json",
            {
                "source_sha256": identity(snapshot),
                "preflight_eligible": eligible,
                "slowdown_investigations": [],
            },
        )

    monkeypatch.setattr(validation, "validate", validate)
    monkeypatch.setattr(module, "container_stage", stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--performance-only",
            "--gpu",
            "--inputs",
            "fixture-inputs",
            "--output",
            str(output),
        ],
    )
    assert module.main() == code
    report = json.loads((output / "summary.json").read_text())
    assert report["scope"] == "performance-only"
    assert report["status"] == ("PASS" if eligible else "BLOCKED")
    assert report["qualified"] is False
    assert {"T089", "T090", "T096", "T098"} <= report["pending"].keys()
    assert stages == ["performance"] and validations == ["bigint-gpu"]
    assert len(launches) == 1 and "check" in launches[0]
    assert output.with_suffix(".zip").is_file()
