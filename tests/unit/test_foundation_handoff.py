"""Cross-cutting handoff must preserve evidence without manufacturing readiness."""

from copy import deepcopy
from pathlib import Path

import pytest

from oeis_learn.experiments.artifacts import canonical_bytes, load_json, compute_file_hash
from oeis_learn.evaluation.experiment_runner import implementation_identity
from oeis_learn.evaluation.readiness import release_readiness


def dispositions():
    record = load_json(
        Path("specs/007-experiment-foundation/validation/phase-11-dispositions.json").read_bytes()
    )
    record["implementation_sha256"] = implementation_identity()
    return record


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown",
        "missing_task",
        "wrong_disposition",
        "empty_rationale",
        "source_hash",
        "task_hash",
        "forged_tasks",
        "release_claim",
    ],
)
def test_disposition_count_cannot_replace_original_inventory(tmp_path, mutation):
    record = dispositions()
    if mutation == "duplicate":
        record["items"][1] = deepcopy(record["items"][0])
    elif mutation == "unknown":
        record["items"][0]["id"] = "R99"
    elif mutation == "missing_task":
        record["items"][0]["task_states"].pop("T097")
    elif mutation == "wrong_disposition":
        record["items"][0]["design_disposition"] = "deferred_loda"
    elif mutation == "empty_rationale":
        record["items"][0]["rationale"] = " "
    elif mutation == "source_hash":
        record["source_sha256"] = "sha256:" + "0" * 64
    elif mutation == "task_hash":
        record["task_inventory_sha256"] = "sha256:" + "0" * 64
    elif mutation == "release_claim":
        record["full_007_complete"] = True
    else:
        record["items"][0]["task_states"]["T097"] = "X"
    path = tmp_path / "dispositions.json"
    path.write_bytes(canonical_bytes(record))
    report = release_readiness(dispositions=path)
    assert report["gates"]["G10"]["checks"]["dispositions"]["status"] == "FAIL"
    assert not report["full_007_complete"]


def test_complete_dispositions_are_traceability_not_runtime_evidence(tmp_path):
    path = tmp_path / "dispositions.json"
    path.write_bytes(canonical_bytes(dispositions()))
    report = release_readiness(dispositions=path)
    assert report["gates"]["G10"]["checks"]["dispositions"]["status"] == "PASS"
    assert report["missing_gates"] == [f"G{i}" for i in range(11)]


def input_manifest(root, *, readiness=None):
    source = root / "identity.json"
    source.write_bytes(canonical_bytes({"profile": "diagnostic"}))
    return dict(
        version="handoff-inputs/v1",
        files=[dict(path=source.name, sha256=compute_file_hash(source))],
        readiness=readiness or {},
    )


def test_handoff_replay_is_immutable_and_missing_gates_stay_pending(tmp_path):
    from oeis_learn.evaluation.handoff import archive_handoff, replay_handoff

    manifest = input_manifest(tmp_path)
    output = tmp_path / "archive"
    report = archive_handoff(manifest, input_root=tmp_path, output=output)
    assert not report["readiness"]["full_007_complete"]
    assert len(report["readiness"]["gates"]) == 11
    assert replay_handoff(output) == report
    assert archive_handoff(manifest, input_root=tmp_path, output=output) == report
    (output / "evidence/identity.json").write_text("{}")
    with pytest.raises(ValueError, match="content"):
        replay_handoff(output)
    with pytest.raises(ValueError):
        archive_handoff(manifest, input_root=tmp_path, output=output)


@pytest.mark.parametrize(
    "mutation", ["escape", "symlink", "duplicate", "wrong_hash", "unknown_field", "unlisted"]
)
def test_handoff_rejects_unsafe_or_incomplete_inventory_before_publication(tmp_path, mutation):
    from oeis_learn.evaluation.handoff import archive_handoff

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = input_manifest(inputs)
    if mutation == "escape":
        manifest["files"][0]["path"] = "../outside.json"
    elif mutation == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (inputs / "identity.json").unlink()
        (inputs / "identity.json").symlink_to(outside)
    elif mutation == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    elif mutation == "wrong_hash":
        manifest["files"][0]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "unknown_field":
        manifest["qualified"] = True
    else:
        manifest["readiness"] = {"g7": "unlisted.json"}
    output = tmp_path / "archive"
    with pytest.raises(ValueError):
        archive_handoff(manifest, input_root=inputs, output=output)
    assert not output.exists()


def test_handoff_replays_supplied_gate_status_instead_of_trusting_label(tmp_path):
    from oeis_learn.evaluation.handoff import archive_handoff, replay_handoff

    gate = tmp_path / "g7.json"
    gate.write_bytes(canonical_bytes({"status": "PASS"}))
    manifest = input_manifest(tmp_path, readiness={"g7": gate.name})
    manifest["files"].append(dict(path=gate.name, sha256=compute_file_hash(gate)))
    report = archive_handoff(manifest, input_root=tmp_path, output=tmp_path / "archive")
    assert report["readiness"]["gates"]["G7"]["status"] == "FAIL"
    assert not report["readiness"]["full_007_complete"]
    assert replay_handoff(tmp_path / "archive") == report


def test_public_handoff_json_reports_archive_and_readiness_separately(tmp_path, capsys):
    from oeis_learn.cli.main import cli

    manifest = tmp_path / "inputs.json"
    manifest.write_bytes(canonical_bytes(input_manifest(tmp_path)))
    assert (
        cli(
            [
                "foundation",
                "handoff",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "archive"),
                "--json",
            ]
        )
        == 0
    )
    report = load_json(capsys.readouterr().out)
    assert report["status"] == "complete" and not report["readiness"]["full_release_qualified"]
    assert report["archive_is_authorization"] is False


def test_handoff_enforces_quota_before_copy(tmp_path, monkeypatch):
    import oeis_learn.evaluation.handoff as handoff

    manifest = input_manifest(tmp_path)
    monkeypatch.setattr(handoff, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="quota"):
        handoff.archive_handoff(manifest, input_root=tmp_path, output=tmp_path / "archive")
    assert not (tmp_path / "archive").exists()


def test_stale_implementation_cannot_replay_archived_readiness(tmp_path, monkeypatch):
    from oeis_learn.evaluation.handoff import archive_handoff, replay_handoff
    import oeis_learn.evaluation.experiment_runner as runner

    archive_handoff(input_manifest(tmp_path), input_root=tmp_path, output=tmp_path / "archive")
    monkeypatch.setattr(runner, "implementation_identity", lambda: "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="implementation"):
        replay_handoff(tmp_path / "archive")
