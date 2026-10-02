"""Actual HIP gate. CPU/fixture success never replaces these checks."""

from pathlib import Path
import pytest
import torch
from oeis_learn.rl.foundation_sft import prepare_run
from oeis_learn.tracking.foundation_controller import run_training
from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint


def test_unavailable_gpu_never_falls_back(tmp_path):
    if torch.cuda.is_available() and torch.version.hip:
        pytest.skip("negative CPU-host probe")
    with pytest.raises(ValueError, match="HIP GPU unavailable"):
        prepare_run("missing", "missing", tmp_path / "run", device="cuda", diagnostic=True)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="actual HIP device required"
)
def test_actual_hip_three_updates(training_inputs, tmp_path):
    # Full frozen small backbone, not the reduced CPU unit-test shape.
    root = tmp_path / "run"
    prepare_run(
        Path("configs/foundation/wat_smoke.yaml"),
        training_inputs[1],
        root,
        device="cuda",
        diagnostic=True,
    )
    report = run_training(root)
    assert report["completed_update"] == 3
    checkpoint = load_foundation_checkpoint(
        report["checkpoint"], device="cuda", include_payload=True
    )
    assert next(checkpoint.encoder.parameters()).device.type == "cuda"
    assert next(checkpoint.decoder.parameters()).device.type == "cuda"
    assert checkpoint.payload["torch_gpu_rng"]
    from oeis_learn.tracking.foundation_metrics import EventLog

    updates = [x for x in EventLog(root / "events").events.values() if x["kind"] == "update"]
    assert len(updates) == 3 and all(x["parameters_changed"] for x in updates)
    assert all(x["parameter_device"].startswith("cuda") for x in updates)
    assert all(x["resources"]["gpu_peak_allocated"]["value"] > 0 for x in updates)


pytest_plugins = ["tests.helpers.foundation_training"]
