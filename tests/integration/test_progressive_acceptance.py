"""One installed full-feature rehearsal, with explicit CPU/HIP evidence boundaries."""

import copy
import xml.etree.ElementTree as ET

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
from oeis_learn.pipeline_validation import progressive_acceptance as acceptance


def junit(path, nodes):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for node in nodes:
        module, name = node.split("::")
        ET.SubElement(suite, "testcase", classname=module[:-3].replace("/", "."), name=name, time="0.1")
    ET.ElementTree(root).write(path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "skipped", "failure", "error"])
def test_integrated_receipt_requires_the_exact_successful_roster(tmp_path, change):
    nodes = acceptance.suite_nodes()
    report = tmp_path / "junit.xml"
    junit(report, nodes)
    tree = ET.parse(report)
    suite = tree.getroot()[0]
    if change == "missing":
        suite.remove(suite[0])
    elif change == "duplicate":
        suite.append(copy.deepcopy(suite[0]))
    elif change == "extra":
        ET.SubElement(suite, "testcase", classname="unregistered", name="test_other")
    else:
        ET.SubElement(suite[0], change)
    tree.write(report)
    result = acceptance.reconcile_suite(report)
    assert result["status"] == "FAIL"
    assert result["missing"] or result["invalid"] or any(c["status"] != "PASS" for c in result["cases"])


def test_short_suite_covers_all_stages_without_counting_scientific_success(tmp_path):
    nodes = acceptance.suite_nodes()
    assert len(nodes) == len(set(nodes))
    junit(tmp_path / "junit.xml", nodes)
    result = acceptance.reconcile_suite(tmp_path / "junit.xml")
    assert result["status"] == "PASS"
    assert set(result["stages"]) == {
        "capability", "diversity", "tuning", "scaling", "oeis", "components", "source_pilot", "evidence"
    }
    assert all(c["status"] == "PASS" for c in result["cases"])


def test_actual_installed_all_stage_suite_and_bounded_export(tmp_path, fitting_client):
    """Real nested suite, not a fabricated JUnit or a supplied PASS receipt."""
    from pathlib import Path
    import zipfile
    import json
    import os

    result = acceptance.run_suite(Path(__file__).resolve().parents[2], tmp_path / "suite")
    if destination := os.environ.get("OEIS_PROGRESSIVE_ACCEPTANCE_EXPORT"):
        acceptance.retain_suite(tmp_path / "suite", destination)
    assert result["status"] == "COMPLETE", result
    assert not result["qualified"]
    assert result["hardware_status"] == "PENDING"
    assert result["scientific_status"] == "UNMEASURED"
    assert result["installed"]["source"] == result["source"]
    assert result["actual_elapsed_us"] > 0
    with zipfile.ZipFile(tmp_path / "suite/diagnostics.zip") as archive:
        assert set(archive.namelist()) == {"acceptance.json", "cases.json"}
        cases = json.loads(archive.read("cases.json"))
        assert cases["status"] == "PASS"
        assert len(cases["cases"]) == len(acceptance.suite_nodes())


def hardware_fixture(root):
    """Boundary-only original-byte fixture; never a real HIP qualification claim."""
    from oeis_learn.pipeline_validation.progressive_qualification import matrix
    from tests.unit.test_progressive_training_policy import profile

    request = matrix([profile(0.1)], source=digest("source"), image=digest("image"),
                     engine="docker", input_lengths=(20,), microbatches=(7,), hardware={"fixture": True})
    case = request["cases"][0]
    checks = acceptance.HARDWARE_CHECKS
    producer = seal(dict(version="progressive-hip-case/v1", source=request["source"],
        scope=digest([request[k] for k in ("source", "image", "engine", "hardware")] + [case]),
        device="actual_HIP", status="PASS", requested=list(checks), executed=list(checks),
        measurements=dict.fromkeys(checks, True) | dict(parameters=1, peak_allocated_bytes=1,
            cleanup_allocated_bytes=0, baseline_allocated_bytes=0, elapsed_us=10)))
    atomic_write(root / "original.json", producer)
    return request, producer, reference(root, "original.json")


@pytest.mark.parametrize("field", ["source", "image", "engine", "hardware", "input", "dropout"])
def test_hardware_reuse_requires_exact_current_scope(tmp_path, field):
    request, producer, ref = hardware_fixture(tmp_path)
    original = acceptance.hardware_plan(tmp_path, request, [ref])
    assert not original["delta"] and original["reused"][0]["producer"] == ref
    assert original["original_execution_us"] == 10
    changed = copy.deepcopy(request)
    if field in {"source", "image"}:
        changed[field] = digest("changed")
    elif field == "engine":
        changed[field] = "podman"
    elif field == "hardware":
        changed[field] = {"fixture": False}
    else:
        case = changed["cases"][0]
        if field == "input":
            case["input_length"] = 4
        else:
            from oeis_learn.generalization.progressive.models import resolve_profile
            p = case["profile"]
            case["profile"] = resolve_profile(p["base_recipe"], shape=p["shape"], seed=p["model"]["seed"], dropout=0.2)
        changed["cases"] = [seal(case)]
    changed = seal(changed)
    plan = acceptance.hardware_plan(tmp_path, changed, [ref])
    assert plan["delta"] == changed["cases"] and not plan["reused"]
    assert plan["original_execution_us"] == 0


@pytest.mark.parametrize("change", ["missing", "tampered", "cpu", "omitted", "measurement", "duplicate", "duplicate_keys"])
def test_missing_or_invalid_hardware_producers_never_supply_reuse(tmp_path, change):
    request, producer, ref = hardware_fixture(tmp_path)
    if change == "missing":
        (tmp_path / "original.json").unlink()
    elif change == "tampered":
        (tmp_path / "original.json").unlink()
        atomic_write(tmp_path / "original.json", dict(producer, status="FAIL"))
    elif change == "duplicate_keys":
        path = tmp_path / "original.json"
        path.write_text(path.read_text().replace('"status": "PASS"', '"status": "FAIL", "status": "PASS"'))
        ref = reference(tmp_path, "original.json")
    elif change != "duplicate":
        if change == "cpu":
            producer["device"] = "cpu"
        elif change == "omitted":
            producer["executed"].pop()
        else:
            producer["measurements"]["parameter_change"] = False
        (tmp_path / "original.json").unlink()
        atomic_write(tmp_path / "original.json", seal(producer))
        ref = reference(tmp_path, "original.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        acceptance.hardware_plan(tmp_path, request, [ref, ref] if change == "duplicate" else [ref])


@pytest.mark.parametrize("reuse", [False, True])
def test_bounded_delta_executes_only_missing_cases_and_keeps_original_bytes(tmp_path, monkeypatch, reuse):
    """Orchestration boundary only: fake device/executor cannot qualify a campaign."""
    from pathlib import Path

    request, producer, ref = hardware_fixture(tmp_path)
    monkeypatch.setattr(acceptance, "_current_hardware", lambda request: None)
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv[-2])
        assert 0 < kwargs["seconds"] <= 30
        atomic_write(Path(argv[-1]) / "producer.json", producer)
        return 0

    monkeypatch.setattr(acceptance, "_bounded_process", execute)
    output = tmp_path / "batch"
    result = acceptance.run_hardware_delta(tmp_path, request, [ref] if reuse else [], output, seconds=30)
    assert result["status"] == "PASS" and not result["qualified"]
    assert result["campaign_device_status"] == result["sustained_status"] == "PENDING"
    assert calls == ([] if reuse else [request["cases"][0]["identity"]])
    assert not result["omitted"] and result["executed"] == calls
    [original] = result["producers"]
    assert (output / original["path"]).read_bytes() == (tmp_path / "original.json").read_bytes()
    assert result["original_execution_us"] == (10 if reuse else 0)


def test_failed_delta_retains_omissions_and_cannot_relabel_pass(tmp_path, monkeypatch):
    request, _, _ = hardware_fixture(tmp_path)
    monkeypatch.setattr(acceptance, "_current_hardware", lambda request: None)
    monkeypatch.setattr(acceptance, "_bounded_process", lambda *args, **kwargs: 1)
    result = acceptance.run_hardware_delta(tmp_path, request, [], tmp_path / "failed", seconds=1)
    assert result["status"] == "FAIL" and result["first_error"] == "ValueError"
    assert result["omitted"] == [c["identity"] for c in request["cases"]]
    assert not result["executed"] and not result["producers"]


def test_installed_failure_still_exports_bounded_public_diagnostics(tmp_path, monkeypatch):
    from pathlib import Path
    import zipfile

    monkeypatch.setattr(acceptance, "_bounded_process", lambda *args, **kwargs: 1)
    result = acceptance.run_suite(Path(__file__).resolve().parents[2], tmp_path / "failed")
    assert result["status"] == "FAILED" and not result["qualified"]
    assert result["first_error"] == "ValueError" and result["installed"] is None
    with zipfile.ZipFile(tmp_path / "failed/diagnostics.zip") as archive:
        assert set(archive.namelist()) == {"acceptance.json", "cases.json"}
    retained = tmp_path / "retained.zip"
    acceptance.retain_suite(tmp_path / "failed", retained)
    assert retained.read_bytes() == (tmp_path / "failed/diagnostics.zip").read_bytes()
    with pytest.raises(FileExistsError):
        acceptance.retain_suite(tmp_path / "failed", retained)
    with zipfile.ZipFile(tmp_path / "failed/diagnostics.zip", "a") as archive:
        archive.writestr("private-truth.json", "not a public projection")
    with pytest.raises(ValueError, match="public acceptance"):
        acceptance.retain_suite(tmp_path / "failed", tmp_path / "forbidden.zip")


@pytest.mark.parametrize("duration", ["-1", "nan", "inf", "1e308", "invalid"])
def test_invalid_case_duration_never_becomes_acceptance(tmp_path, duration):
    report = tmp_path / "junit.xml"
    junit(report, acceptance.suite_nodes())
    tree = ET.parse(report)
    tree.getroot()[0][0].set("time", duration)
    tree.write(report)
    assert acceptance.reconcile_suite(report)["status"] == "FAIL"


def test_unattributed_collection_failure_never_becomes_acceptance(tmp_path):
    report = tmp_path / "junit.xml"
    junit(report, acceptance.suite_nodes())
    tree = ET.parse(report)
    ET.SubElement(tree.getroot()[0], "error")
    tree.write(report)
    assert acceptance.reconcile_suite(report)["status"] == "FAIL"


def test_real_acceptance_subprocess_timeout_is_bounded(tmp_path):
    import os
    import sys
    import time

    began = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline"):
        acceptance._bounded_process([sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path, env=dict(os.environ), log=tmp_path / "timeout.log", seconds=0.1, maximum=1024)
    assert time.monotonic() - began < 5


def test_current_engine_must_match_even_when_capability_bytes_match(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_qualification as qualification
    from oeis_learn.experiments import source_snapshot

    request, _, _ = hardware_fixture(tmp_path)
    request = seal(dict(request, source=digest({"current": "source"})))
    monkeypatch.setattr(qualification, "require_hip", lambda: None)
    monkeypatch.setattr(qualification, "hardware_facts", lambda: request["hardware"])
    monkeypatch.setattr(source_snapshot, "source_snapshot", lambda: {"current": "source"})
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", request["image"])
    monkeypatch.setenv("OEIS_QUALIFICATION_ENGINE", "podman")
    with pytest.raises(ValueError, match="engine"):
        acceptance._current_hardware(request)
    monkeypatch.setenv("OEIS_QUALIFICATION_ENGINE", "docker")
    acceptance._current_hardware(request)
