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


def test_gate_stage_does_not_train_or_replace_missing_parent_states(tmp_path, monkeypatch):
    import torch
    import oeis_learn.cli.train_warmstart as wrapper

    encoder, decoder = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    parent = tmp_path / "parent.pt"
    torch.save(dict(encoder_state_dict=encoder.state_dict(),
                    decoder_state_dict=decoder.state_dict()), parent)
    monkeypatch.setattr(wrapper, "TriStreamEncoder", lambda **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(wrapper, "WatTransformerDecoder", lambda **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(wrapper, "perform_vocabulary_surgery", lambda **kwargs: kwargs["decoder"])
    def training_forbidden(**kwargs):
        raise AssertionError("gate-only stage must not construct a trainer")
    monkeypatch.setattr(wrapper, "SftTrainer", training_forbidden)
    def gate(enc, dec, device, sequences):
        assert torch.equal(enc.weight, encoder.weight)
        assert torch.equal(dec.weight, decoder.weight)
        assert len({tuple(terms) for terms in sequences}) == len(sequences) > 1
        return dict(gate_passed=False, qualified=False)
    monkeypatch.setattr(wrapper, "evaluate_bridge_model", gate)
    result = wrapper.run_warmstart_pipeline(str(parent), str(tmp_path / "out"),
                                           stage="gate", warmup_steps=0, bridge_steps=0)
    assert result["stage"] == "gate" and not result["qualified"]
    torch.save(dict(encoder_state_dict=encoder.state_dict()), parent)
    with pytest.raises(ValueError, match="both trained model states"):
        wrapper.run_warmstart_pipeline(str(parent), str(tmp_path / "invalid"), stage="gate")
    with pytest.raises(ValueError, match="positive exact steps"):
        wrapper.run_warmstart_pipeline(str(parent), str(tmp_path / "zero"), warmup_steps=0)
