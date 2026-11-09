"""Mandatory native CI must not silently accept skipped or absent test coverage."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "progressive_gate", ROOT / "scripts/qualify_progressive_foundation.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def junit(path):
    suite = ET.Element("testsuite")
    for node in gate.TESTS:
        module = node.split("::")[0].removesuffix(".py").replace("/", ".")
        ET.SubElement(suite, "testcase", classname=module, name="case")
    ET.SubElement(
        suite, "testcase", classname="tests.unit.test_progressive_coverage", name=gate.NATIVE_CASE
    )
    ET.SubElement(
        suite,
        "testcase",
        classname="tests.contract.test_foundation_packaging",
        name=gate.WHEEL_CASE,
    )
    ET.ElementTree(suite).write(path)
    return suite


def test_acceptance_has_explicit_software_and_future_specification_scopes(tmp_path):
    path = tmp_path / "results.xml"
    junit(path)
    assert gate.reconcile(path)["skipped"] == 0
    assert "tests/unit/test_progressive_coverage.py" in gate.TESTS
    assert "tests/contract/test_progressive_authority.py" in gate.TESTS
    assert set(gate.DEFERRED_SPECIFICATIONS).isdisjoint(gate.TESTS)
    assert "tests/unit/test_progressive_metrics.py" in gate.TESTS
    assert "tests/unit/test_progressive_decisions.py" in gate.TESTS
    assert "tests/unit/test_progressive_optimizer.py" in gate.TESTS
    assert "tests/unit/test_progressive_trials.py" in gate.TESTS
    assert "tests/unit/test_progressive_models.py" in gate.TESTS
    assert "tests/unit/test_progressive_optimizer_specifications.py" in gate.TESTS
    assert "tests/unit/test_progressive_training_policy.py" in gate.TESTS
    assert "tests/unit/test_progressive_polynomial_certificate.py" in gate.TESTS
    assert "tests/integration/test_progressive_claims.py" in gate.TESTS
    assert len(gate.DEFERRED_SPECIFICATIONS) == 0


@pytest.mark.parametrize(
    "mutation", ["skipped", "failure", "error", "missing_module", "missing_native", "empty"]
)
def test_mandatory_acceptance_rejects_missing_failed_or_skipped_cases(tmp_path, mutation):
    path = tmp_path / "results.xml"
    suite = junit(path)
    if mutation in {"skipped", "failure", "error"}:
        ET.SubElement(suite[-2], mutation)
    elif mutation == "missing_module":
        suite.remove(suite[0])
    elif mutation == "missing_native":
        suite.remove(suite[-2])
    else:
        suite.clear()
    ET.ElementTree(suite).write(path)
    with pytest.raises(ValueError):
        gate.reconcile(path)


def test_runner_requires_real_native_mode_before_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("OEIS_REQUIRE_NATIVE", raising=False)
    monkeypatch.delenv("OEIS_REQUIRE_BIGINT_BUILD", raising=False)
    assert gate.run(tmp_path) == 1
    import json

    result = json.loads((tmp_path / "progressive-foundation.json").read_text())
    assert result["status"] == "FAIL" and result["qualified"] is False
    assert "mandatory native image mode" in result["error"]["message"]


def test_passing_test_roster_without_retained_integrated_package_fails_ci(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.setenv("OEIS_REQUIRE_NATIVE", "1")
    monkeypatch.setenv("OEIS_REQUIRE_BIGINT_BUILD", "1")

    def run(argv, **kwargs):
        assert kwargs["env"]["OEIS_PROGRESSIVE_ACCEPTANCE_EXPORT"] == str(tmp_path / "progressive-acceptance.zip")
        junit(tmp_path / "progressive-foundation.xml")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.run(tmp_path) == 1
    result = json.loads((tmp_path / "progressive-foundation.json").read_text())
    assert result["status"] == "FAIL" and result["counts"]["tests"] > 0
    assert result["error"]["type"] in {"ValueError", "FileNotFoundError"}
