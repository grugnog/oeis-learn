"""CPU regressions for GPU initialization; no substitute for actual HIP training."""

from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from torch.cuda._utils import _get_device_index
from oeis_learn.rl.foundation_sft import FoundationTrainer, prepare_run, runtime_record
from oeis_learn.experiments.artifacts import load_json


@pytest.fixture
def initialization_inputs(tmp_path, monkeypatch):
    # Pool admission has its own integration coverage. Isolate constructor
    # initialization here so these CPU regressions require no evaluator/GPU.
    pool = tmp_path / "pool"
    trainer = pool / "trainer"
    trainer.mkdir(parents=True)
    examples = tuple({"program_id": f"program-{i}"} for i in range(4))
    (pool / "manifest.json").write_text('{"pool_id":"fixture-pool"}')
    (trainer / "manifest.json").write_text("{}")
    (trainer / "examples.json").write_text("[]")
    admitted = SimpleNamespace(examples=examples, pool_id="fixture-pool", view_id="fixture-view")
    monkeypatch.setattr("oeis_learn.rl.foundation_sft.load_pool", lambda *a, **k: admitted)
    monkeypatch.setattr("oeis_learn.rl.foundation_sft.load_trainer_view", lambda *a, **k: examples)
    raw = yaml.safe_load(Path("configs/foundation/wat_smoke.yaml").read_text())
    raw["model"].update(d_model=16, heads=2, encoder_layers=1, decoder_layers=1, feed_forward=32)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    return config, pool


@pytest.mark.parametrize("index", [0, 2])
@pytest.mark.parametrize("total_gib", [16, 120])
@pytest.mark.parametrize("resume", [False, True])
def test_gpu_allocator_receives_explicit_selected_index_and_frozen_limit(
    initialization_inputs, tmp_path, monkeypatch, index, total_gib, resume
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "fixture-hip")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: index)
    named = []

    def name(device):
        named.append(device)
        return f"fixture-gpu-{device}"

    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    root = tmp_path / "run"
    run = prepare_run(*initialization_inputs, root, device="cuda", diagnostic=True)
    assert run["device"] == "cuda" and run["free_disk_floor_bytes"] == 100 << 30
    assert load_json((root / "checkpoints/runtime.json").read_bytes())["device_name"] == (
        f"fixture-gpu-{index}"
    )
    capacities, limits = [], []

    def properties(device):
        # Properties accept the unindexed alias; allocator configuration does
        # not. Exercise PyTorch's actual index validator without GPU allocation.
        assert _get_device_index(device, optional=True) == index
        capacities.append(device)
        return SimpleNamespace(total_memory=total_gib << 30)

    class ConfiguredAllocator(RuntimeError):
        pass

    def memory_limit(fraction, device):
        resolved = _get_device_index(device)
        limits.append((fraction, resolved))
        raise ConfiguredAllocator("stop before actual GPU model allocation")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", memory_limit)
    with pytest.raises(ConfiguredAllocator):
        FoundationTrainer(root, resume=resume)
    assert capacities == [index]
    assert limits == [(min(0.8, 64 / total_gib), index)]
    assert named and set(named) == {index}


def test_cpu_trainer_does_not_configure_or_query_gpu_allocator(
    initialization_inputs, tmp_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU training must not select/query/configure the GPU allocator")

    for method in (
        "current_device",
        "get_device_name",
        "get_device_properties",
        "set_per_process_memory_fraction",
    ):
        monkeypatch.setattr(torch.cuda, method, unexpected)
    root = tmp_path / "run"
    prepare_run(*initialization_inputs, root, device="cpu", diagnostic=True)
    learner = FoundationTrainer(root)
    assert learner.device == torch.device("cpu")
    assert runtime_record("cpu")["device"] == "cpu"
