"""Representative CLI contract: actual weights, exact groups and final sealing."""

import json
from oeis_learn.cli.main import cli
from oeis_learn.experiments.artifacts import load_json, compute_canonical_digest as digest
from oeis_learn.evaluation.foundation_cohort import write_json, load_cohort
from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file


def test_development_cli_and_invalid_inputs_never_produce_scores(tmp_path, capsys):
    checkpoint = make_checkpoint(tmp_path / "run")
    cohort = make_cohort(tmp_path / "data")
    protocol = protocol_file(tmp_path, attempts=1)
    args = [
        "foundation",
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--cohort",
        str(cohort),
        "--protocol",
        str(protocol),
        "--split",
        "development",
        "--output",
        str(tmp_path / "out"),
        "--json",
    ]
    assert cli(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["N"] == report["success_any"] == report["success_top1"] == 1
    assert report["qualified"] is False
    args[3] = str(tmp_path / "missing.json")
    assert cli(args) == 2
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["qualified"] is False and "success_any" not in invalid
    args[3] = str(checkpoint)
    manifest = load_cohort(cohort)
    (cohort / manifest["groups"][0]["truth"]["path"]).unlink()
    assert cli(args) == 2
    assert "success_any" not in json.loads(capsys.readouterr().out)


def test_freeze_finalize_and_final_cli(tmp_path, capsys):
    cohort = tmp_path / "cohort"
    assert (
        cli(
            [
                "foundation",
                "freeze-cohort",
                "--source",
                "tests/fixtures/foundation/source",
                "--config",
                "tests/fixtures/foundation/cohort.yaml",
                "--output",
                str(cohort),
                "--json",
            ]
        )
        == 0
    )
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["census"]["total_records"] == 5 and frozen["census"]["groups"] == 2
    checkpoint = make_checkpoint(tmp_path / "run")
    protocol = protocol_file(tmp_path, attempts=1)
    stop = tmp_path / "stop.json"
    write_json(
        tmp_path,
        "stop.json",
        {
            "decision": "stop",
            "reason": "software fixture",
            "checkpoint_sha256": load_json(checkpoint.read_bytes())["blob_sha256"],
            "protocol_sha256": digest(load_json(protocol.read_bytes())),
            "cohort_id": frozen["cohort_id"],
        },
    )
    lock = tmp_path / "lock.json"
    assert (
        cli(
            [
                "foundation",
                "finalize",
                "--run-dir",
                str(checkpoint.parent),
                "--checkpoint",
                str(checkpoint),
                "--protocol",
                str(protocol),
                "--cohort",
                str(cohort),
                "--stopping-record",
                str(stop),
                "--output",
                str(lock),
                "--json",
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    args = [
        "foundation",
        "evaluate",
        "--finalization",
        str(lock),
        "--split",
        "final",
        "--output",
        decision["evaluation_output"],
        "--json",
    ]
    assert cli(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["N"] == 1
        and result["finalization_id"] == decision["finalization_id"]
        and len(result["seal_ids"]) == 1
    )
    assert cli(args + ["--checkpoint", str(checkpoint)]) == 2
    assert "success_any" not in json.loads(capsys.readouterr().out)


def test_explicit_gpu_never_falls_back(monkeypatch, tmp_path, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        cli(
            [
                "foundation",
                "evaluate",
                "--checkpoint",
                "missing",
                "--cohort",
                "missing",
                "--protocol",
                "missing",
                "--split",
                "development",
                "--device",
                "cuda",
                "--output",
                str(tmp_path),
                "--json",
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out)["status"] == "hardware_unavailable"


def test_reference_canary_report_cannot_claim_checkpoint_or_model_evidence(monkeypatch):
    from oeis_learn.cli import evaluate_canaries as canaries

    monkeypatch.setattr(canaries, "CANARY_METADATA", [])
    result = canaries.run_canary_evaluation(checkpoint_path="nonexistent.pt")
    assert (
        result["checkpoint_evaluated"] is None
        and result["checkpoint_requested"] == "nonexistent.pt"
    )
    assert result["purpose"] == "reference_kernel_diagnostic"
    assert not result["qualified"] and not result["model_synthesis"]
