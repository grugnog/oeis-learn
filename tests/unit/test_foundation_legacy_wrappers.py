from types import SimpleNamespace
import pytest
from oeis_learn.curriculum.transition_gate import TransitionGateEvaluator


class Runner:
    def run_single(self, source, **kwargs):
        return SimpleNamespace(status="SUCCESS", output=[0] * 20 if source == "valid" else [1] * 20)


def test_transition_full_success_still_has_zero_reward_variance():
    evaluator = TransitionGateEvaluator(runner=Runner())
    result = evaluator.evaluate_rollouts([["valid"] * 4], [[0] * 20])
    assert result["advantage_collapse_rate"] == 1 and not result["gate_passed"]
    mixed = evaluator.evaluate_rollouts([["valid"] * 3 + ["wrong"]], [[0] * 20])
    assert mixed["gate_passed"] and mixed["qualified"] is False
    with pytest.raises(ValueError):
        evaluator.evaluate_rollouts([["valid"]], [])


def test_warmstart_gate_samples_model_and_restores_modes(monkeypatch):
    import torch
    import oeis_learn.cli.train_warmstart as wrapper

    calls = []

    class Encoder(torch.nn.Module):
        def forward_from_sequences(self, values, device=None):
            calls.append(values)
            return torch.zeros(len(values), 20, 8)

    class Sampler:
        def __init__(self, decoder, **kwargs):
            pass

        def sample_trajectories(self, memory, seeds, **kwargs):
            calls.append(seeds)
            return [SimpleNamespace(source="valid") for _ in seeds]

    monkeypatch.setattr(wrapper, "WatProgramSampler", Sampler)
    encoder, decoder = Encoder(), torch.nn.Linear(8, 8)
    encoder.train()
    decoder.eval()
    report = wrapper.evaluate_bridge_model(encoder, decoder, "cpu", [[0] * 20], runner=Runner())
    assert calls[0] == [[0] * 20] * 4 and calls[1] == [7001, 7002, 7003, 7004]
    assert report["candidate_origin"] == "model_sample"
    assert encoder.training and not decoder.training


def test_inventory_only_existing_files_and_historical_status():
    from oeis_learn.evaluation.legacy_wrappers import wrapper_inventory

    rows = wrapper_inventory()
    assert all(row["status"] == "legacy_unverified" for row in rows)
    assert not any("cli/train_run008.py" in row["path"] for row in rows)
    assert any(row["path"] == "scripts/run_008_learning_run.py" for row in rows)
