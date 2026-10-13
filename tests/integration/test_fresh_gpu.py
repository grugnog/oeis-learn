"""Actual HIP fresh-worker checkpoints. Use a workspace --basetemp, not broker /tmp."""

import math
import shutil

import pytest
import torch

from oeis_learn.decoder.program_codec import EOS_ID, FOUNDATION_VOCAB_SIZE, PAD_ID
from oeis_learn.fresh import worker
from oeis_learn.fresh.common import write
from oeis_learn.rl.policy import trajectory_mean
from oeis_learn.rl.token_loss import token_cross_entropy
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
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        store = QualifiedCheckpointStore(output / "checkpoints", free_floor_bytes=0)
        _, _, state = store.load_learning(
            output / "checkpoints" / result["checkpoint"], **worker.identities(cfg)
        )
        return output, result, state

    initial, baseline, before = run(tmp_path / "initial", 0)
    assert baseline["completed_updates"] == 0
    assert math.isfinite(baseline["validation_loss"])
    first, result, state = run(tmp_path / "first", 1, initial)
    assert result["device"] == "cuda" and result["completed_updates"] == 1
    assert math.isfinite(result["validation_loss"])
    assert not torch.equal(state["decoder"]["lm_head.weight"], before["decoder"]["lm_head.weight"])
    _resumed, result, state = run(tmp_path / "resumed", 3, first)
    assert result["completed_updates"] == 3 and result["recovery"]["completed"] == 1
    assert math.isfinite(result["validation_loss"])
    assert state["rng"]["cuda"] and state["sampler"]["completed"] == 3
    assert all(torch.isfinite(p).all() for p in state["decoder"].values() if p.is_floating_point())


@pytest.mark.parametrize("reduction", ["validation", "per_program"])
def test_actual_hip_deterministic_token_loss_forward_backward(reduction):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        values = torch.randn(
            2, 3, FOUNDATION_VOCAB_SIZE, generator=torch.Generator().manual_seed(118)
        )
        targets = torch.tensor([[4, 5, EOS_ID], [6, EOS_ID, PAD_ID]])
        if reduction == "validation":
            targets[1, -1] = EOS_ID

        def objective(logits, tokens):
            losses = token_cross_entropy(logits, tokens)
            return (
                losses.mean()
                if reduction == "validation"
                else trajectory_mean(losses, tokens.ne(PAD_ID))
            )

        reference = values.clone().requires_grad_()
        expected = objective(reference, targets)
        expected.backward()
        observed = []
        for _ in range(2):
            logits = values.to("cuda").requires_grad_()
            loss = objective(logits, targets.to("cuda"))
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            observed.append((loss.detach().cpu(), logits.grad.cpu()))
        for loss, gradient in observed:
            torch.testing.assert_close(loss, expected.detach(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradient, reference.grad, rtol=1e-5, atol=1e-6)
        for first, second in zip(observed[0], observed[1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
