"""Standalone acceptance module copied outside checkout by test_wheel.sh."""

import importlib.util
import json
import os
from pathlib import Path
import pytest


def test_packaged_resources_and_actual_execution():
    import oeis_learn
    from oeis_learn.sandbox.preamble import load_preamble_wat
    from oeis_learn.experiments.source_snapshot import source_snapshot, qualification_identity
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
    from oeis_learn.sandbox.pipeline import Runtime, Limits

    assert "(module" in load_preamble_wat()
    assert qualification_identity("G10") == digest(dict(gate="G10", sources=source_snapshot()))
    if os.getenv("OEIS_INSTALLED_WHEEL"):
        assert "site-packages" in str(Path(oeis_learn.__file__))
        assert not any(p.endswith("/src") for p in __import__("sys").path)
        assert "src/oeis_learn/sandbox/preamble.wat" in source_snapshot()
    result = Runtime().evaluate("i256.const -1", range(20), Limits())
    assert result.outcome is None and result.outputs == ["-1"] * 20 == result.reference_outputs


def test_public_measure_dry_run_and_actual(tmp_path, capsys):
    from oeis_learn.cli.main import cli
    from oeis_learn.evaluation.experiment_runner import fixture_manifest
    from oeis_learn.experiments.artifacts import canonical_bytes

    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_bytes(fixture_manifest()))
    common = ["foundation", "measure", "--manifest", str(manifest), "--json"]
    assert cli([*common, "--dry-run", "--output", str(tmp_path / "dry")]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "dry_run" and not dry["executed"]
    assert cli([*common, "--output", str(tmp_path / "actual")]) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual["status"] == "diagnostic_complete" and not actual["qualified"]
    assert actual["measurements"]["sample_shortfall"] == 124
    invalid = json.loads(manifest.read_text())
    invalid["final_truth"] = "private.json"
    manifest.write_text(json.dumps(invalid))
    assert cli([*common, "--output", str(tmp_path / "forbidden")]) == 2
    assert not (tmp_path / "forbidden/measurement.json").exists()


def test_census_public_invalid_input(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    source = tmp_path / "records.json"
    source.write_text(json.dumps([dict(terms=[0] * 120, offset=-2, tags="fini,tabl")]))
    assert (
        cli(
            [
                "foundation",
                "census",
                "--source",
                str(source),
                "--output",
                str(tmp_path / "census"),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["horizons"]["horizon_120"]["sequences_evaluated"] == 1
    assert report["source_metadata"]["offsets"] == {"-2": 1}


@pytest.mark.skipif(
    importlib.util.find_spec("oeis_wasm_evaluator") is None,
    reason="CPU Docker native build required",
)
def test_installed_native_has_no_fallback():
    from oeis_learn.sandbox.native_runtime import native_identity
    from oeis_learn.sandbox.pipeline import Runtime

    identity = native_identity()
    assert identity["backend"] == "rust"
    result = Runtime(backend="rust").evaluate("i256.const -1", range(20))
    assert result.outcome is None and result.outputs == ["-1"] * 20


@pytest.fixture(scope="module")
def qualified_cli_evidence():
    evidence = os.getenv("OEIS_QUALIFICATION_EVIDENCE")
    if not evidence:
        if os.getenv("OEIS_REQUIRE_NATIVE") == "1":
            pytest.fail("mandatory native acceptance requires actual G7/G8/G9 evidence")
        pytest.skip("actual native G7/G8/G9 artifacts required")
    from oeis_learn.evaluation.cli_acceptance import verify_learning_cli

    verify_learning_cli(evidence)
    return Path(evidence)


def test_installed_train_resume_search(qualified_cli_evidence):
    from oeis_learn.experiments.artifacts import load_json

    phase = qualified_cli_evidence / "learning"
    report = load_json((phase / "cli-acceptance.json").read_bytes())
    assert report == dict(
        status="PASS",
        qualification="cpu_diagnostic",
        updates=2,
        search_attempts=1,
        hardware_activation="rejected",
    )
    assert (phase / "training/qualified-00000001.json").is_file()


def test_installed_discover_and_fresh_certificate(qualified_cli_evidence):
    from oeis_learn.evaluation.cli_acceptance import verify_discovery_cli
    from oeis_learn.experiments.artifacts import load_json

    verify_discovery_cli(qualified_cli_evidence)
    report = load_json((qualified_cli_evidence / "discovery/cli-acceptance.json").read_bytes())
    assert report["status"] == "PASS" and report["stale_evidence"] == "rejected"
