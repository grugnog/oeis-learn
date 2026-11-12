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


def junit(path, requested=gate.TESTS):
    suite = ET.Element("testsuite")
    for node in requested:
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
    assert "tests/unit/test_progressive_qualification_cleanup.py" in gate.TESTS


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
        assert kwargs["timeout"] == gate.PYTEST_TIMEOUT_SECONDS == 3000
        assert "-vv" in argv and "--tb=short" in argv
        assert kwargs["env"]["OEIS_PROGRESSIVE_ACCEPTANCE_EXPORT"] == str(tmp_path / "progressive-acceptance.zip")
        junit(tmp_path / "progressive-foundation.xml")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.run(tmp_path) == 1
    result = json.loads((tmp_path / "progressive-foundation.json").read_text())
    assert result["status"] == "FAIL" and result["counts"]["tests"] > 0
    assert result["error"]["type"] in {"ValueError", "FileNotFoundError"}



def test_partitions_are_disjoint_complete_and_keep_expensive_modules_separate():
    groups = [gate.partition_tests(name) for name in gate.PARTITIONS]
    assert set().union(*map(set, groups)) == set(gate.TESTS)
    assert sum(map(len, groups)) == len(gate.TESTS) == len(set(gate.TESTS))
    for module, owner in gate.ANCHORS.items():
        assert module in gate.partition_tests(owner)
    with pytest.raises(ValueError):
        gate.partition_tests('unknown')


def partition_receipts(root, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('OEIS_REQUIRE_NATIVE', '1')
    monkeypatch.setenv('OEIS_REQUIRE_BIGINT_BUILD', '1')
    monkeypatch.setattr(gate, 'source_snapshot', lambda: {'fixture': 'current-source'})
    for name in gate.PARTITIONS:
        folder = root / name
        requested = gate.partition_tests(name)
        def execute(argv, *, requested=requested, folder=folder, **kwargs):
            assert argv[-len(requested):] == list(requested)
            junit(folder / 'progressive-foundation.xml', requested)
            if 'tests/integration/test_progressive_acceptance.py' in requested:
                (folder / 'progressive-acceptance.zip').write_bytes(b'original export fixture')
            return SimpleNamespace(returncode=0)
        monkeypatch.setattr(gate.subprocess, 'run', execute)
        assert gate.run(folder, partition=name) == 0
    return root


def test_partitioned_receipts_reconcile_complete_mandatory_inventory(tmp_path, monkeypatch):
    import json
    root = partition_receipts(tmp_path, monkeypatch)
    assert gate.merge(root, root / 'summary') == 0
    report = json.loads((root / 'summary/progressive-foundation.json').read_text())
    assert report['status'] == 'PASS' and report['qualified'] is False
    assert report['requested'] == list(gate.TESTS)
    assert report['counts']['skipped'] == 0
    with pytest.raises(ValueError, match='fresh merged'):
        gate.merge(root, root / 'summary')


@pytest.mark.parametrize('damage', ['missing', 'changed_junit', 'changed_export', 'stale_source', 'wrong_membership', 'failed', 'skipped'])
def test_merge_rejects_incomplete_or_foreign_partitions(tmp_path, monkeypatch, damage):
    import json
    root = partition_receipts(tmp_path, monkeypatch)
    path = root / 'lifecycle/progressive-foundation.json'
    report = json.loads(path.read_text())
    if damage == 'missing':
        path.unlink()
    elif damage == 'changed_junit':
        (root / 'lifecycle/progressive-foundation.xml').write_text('<testsuites/>')
    elif damage == 'changed_export':
        (root / 'lifecycle/progressive-acceptance.zip').write_bytes(b'changed')
    else:
        if damage == 'stale_source': report['source'] = 'sha256:' + '0' * 64
        if damage == 'wrong_membership': report['requested'] = report['requested'][1:]
        if damage == 'failed': report['status'] = 'FAIL'
        if damage == 'skipped':
            xml = root / 'lifecycle/progressive-foundation.xml'
            suite = ET.parse(xml)
            ET.SubElement(suite.find('.//testcase'), 'skipped')
            suite.write(xml)
            report['junit'] = gate.reference(root / 'lifecycle', xml)
        path.write_text(json.dumps(gate.seal(report)))
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.merge(root, root / 'summary')
