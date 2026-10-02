"""Crash/retry tests around the durable final decision and per-target seal."""

import pytest
from oeis_learn.experiments.artifacts import load_json, compute_canonical_digest as digest
from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file
from oeis_learn.evaluation.foundation_cohort import load_cohort, write_json
from oeis_learn.evaluation.finalization import create_finalization, final_output
from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation, ModelProcess


def setup_final(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "run")
    cohort = make_cohort(tmp_path / "data", dev_count=0, final_count=1)
    protocol = protocol_file(tmp_path, attempts=1)
    stopping = {
        "decision": "stop",
        "reason": "diagnostic software gate only",
        "checkpoint_sha256": load_json(checkpoint.read_bytes())["blob_sha256"],
        "protocol_sha256": digest(load_json(protocol.read_bytes())),
        "cohort_id": load_cohort(cohort)["cohort_id"],
    }
    stop = tmp_path / "stopping.json"
    write_json(tmp_path, stop.name, stopping)
    lockpath = tmp_path / "lock.json"
    lock = create_finalization(checkpoint.parent, checkpoint, protocol, cohort, stop, lockpath)
    return checkpoint, cohort, protocol, lockpath, final_output(lock)


@pytest.mark.parametrize("event", ["after_attempt", "after_seal", "after_exposure", "after_score"])
def test_final_crash_resume_and_no_postseal_regeneration(tmp_path, monkeypatch, event):
    checkpoint, cohort, protocol, lock, out = setup_final(tmp_path)

    def crash(phase, target):
        if phase == event:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        evaluate_foundation(
            checkpoint, cohort, protocol, out, split="final", finalization=lock, _hook=crash
        )
    if event != "after_attempt":

        def prohibited(*args, **kwargs):
            raise AssertionError("regeneration after seal")

        monkeypatch.setattr(ModelProcess, "generate", prohibited)
    report = evaluate_foundation(
        checkpoint, cohort, protocol, out, split="final", finalization=lock
    )
    assert report["N"] == report["success_any"] == report["success_top1"] == 1
    assert len(report["seal_ids"]) == 1 and report["finalization_id"]
    assert (
        evaluate_foundation(checkpoint, cohort, protocol, out, split="final", finalization=lock)
        == report
    )


def test_lost_seal_and_fresh_output_are_rejected_after_hidden_exposure(tmp_path):
    checkpoint, cohort, protocol, lock, out = setup_final(tmp_path)

    def crash(phase, target):
        if phase == "after_exposure":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        evaluate_foundation(
            checkpoint, cohort, protocol, out, split="final", finalization=lock, _hook=crash
        )
    next(out.glob("targets/*/seal.json")).unlink()
    with pytest.raises(ValueError, match="without committed seal"):
        evaluate_foundation(checkpoint, cohort, protocol, out, split="final", finalization=lock)
    with pytest.raises(ValueError, match="output is fixed"):
        evaluate_foundation(
            checkpoint, cohort, protocol, tmp_path / "fresh", split="final", finalization=lock
        )


def test_final_inputs_cannot_change_in_place(tmp_path):
    import json

    checkpoint, cohort, protocol, lock, out = setup_final(tmp_path)
    raw = load_json(protocol.read_bytes())
    raw["attempts"] = 2
    protocol.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cohort/protocol changed"):
        evaluate_foundation(checkpoint, cohort, protocol, out, split="final", finalization=lock)
    with pytest.raises(ValueError):
        create_finalization(
            checkpoint.parent, checkpoint, protocol, cohort, tmp_path / "stopping.json", lock
        )
