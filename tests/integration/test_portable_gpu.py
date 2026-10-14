"""Actual HIP updates and checkpoint continuation; synthetic labels are fixture evidence."""

import json
import shutil

import pytest
import torch

from oeis_learn.fresh import worker
from oeis_learn.fresh.common import write
from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore
from tests.unit.test_fresh_workflow import examples, settings


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="actual ROCm GPU required"
)
def test_full_backbone_updates_and_exact_gpu_resume(tmp_path):
    root = tmp_path

    def train(name, cap, previous=None):
        inputs, output = root / name / "input", root / name / "output"
        data = examples()
        cfg = dict(
            settings(data, cap),
            device="cuda",
            seconds=180,
            checkpoint_reserve_seconds=10,
            free_disk_floor_bytes=1 << 30,
        )
        write(inputs / "examples.json", data)
        write(inputs / "settings.json", cfg)
        if previous:
            shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
            cfg["charged_budget_ns"] = 1_000_000_000_000
            write(inputs / "settings.json", cfg)
        output.mkdir(parents=True)
        result = worker.train(inputs, output)
        store = QualifiedCheckpointStore(output / "checkpoints", free_floor_bytes=1 << 30)
        _, _, state = store.load_learning(
            output / "checkpoints" / result["checkpoint"], **worker.identities(cfg)
        )
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        assert rows and all(
            r["parameters_changed"] and r["parameter_device"].startswith("cuda") for r in rows
        )
        assert all(r["resources"]["gpu_peak_allocated"]["value"] > 0 for r in rows)
        return output, result, state

    _, whole, expected = train("whole", 3)
    first, _, _ = train("first", 1)
    _, resumed, actual = train("resumed", 3, first)
    assert whole["completed_updates"] == resumed["completed_updates"] == 3
    assert resumed["recovery"]["completed"] == 1
    for component in ("encoder", "decoder"):
        for key in expected[component]:
            torch.testing.assert_close(
                expected[component][key], actual[component][key], atol=0, rtol=0
            )
    torch.testing.assert_close(expected["sampler"].pop("rng"), actual["sampler"].pop("rng"))
    assert expected["sampler"] == actual["sampler"]
