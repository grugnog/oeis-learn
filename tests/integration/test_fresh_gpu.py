"""Actual HIP fresh-worker checkpoints. Use a workspace --basetemp, not broker /tmp."""

import shutil

import pytest
import torch

from oeis_learn.fresh import worker
from oeis_learn.fresh.common import write
from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore
from tests.unit.test_fresh_workflow import examples, settings

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.version.hip,
        reason="actual HIP accelerator required",
    ),
]


def test_actual_hip_fresh_update_and_complete_state_resume(tmp_path):
    def run(root, cap, previous=None):
        inputs, output = root / "input", root / "output"
        inputs.mkdir(parents=True)
        output.mkdir()
        data = examples()
        cfg = dict(
            settings(data, cap),
            device="cuda",
            free_disk_floor_bytes=100 << 30,
            seconds=120,
            checkpoint_reserve_seconds=20,
        )
        if previous:
            shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
            cfg["charged_budget_ns"] = 1_000_000_000_000
        write(inputs / "settings.json", cfg)
        write(inputs / "examples.json", data)
        result = worker.train(inputs, output)
        store = QualifiedCheckpointStore(output / "checkpoints", free_floor_bytes=0)
        _, _, state = store.load_learning(
            output / "checkpoints" / result["checkpoint"], **worker.identities(cfg)
        )
        return output, result, state

    initial, _, before = run(tmp_path / "initial", 0)
    first, result, state = run(tmp_path / "first", 1, initial)
    assert result["device"] == "cuda" and result["completed_updates"] == 1
    assert not torch.equal(state["decoder"]["lm_head.weight"], before["decoder"]["lm_head.weight"])
    _resumed, result, state = run(tmp_path / "resumed", 3, first)
    assert result["completed_updates"] == 3 and result["recovery"]["completed"] == 1
    assert state["rng"]["cuda"] and state["sampler"]["completed"] == 3
    assert all(torch.isfinite(p).all() for p in state["decoder"].values() if p.is_floating_point())
