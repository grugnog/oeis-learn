"""Actual HIP fresh-worker checkpoints; use a persistent --basetemp."""

import math
import shutil

import pytest
import torch

from oeis_learn.decoder.program_codec import EOS_ID, FOUNDATION_VOCAB_SIZE, PAD_ID
from oeis_learn.fresh import worker
from oeis_learn.fresh.common import digest, write
from oeis_learn.rl.policy import trajectory_mean
from oeis_learn.rl.token_loss import token_cross_entropy
from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore
from tests.unit.test_fresh_workflow import advance_test_ledger, examples, settings

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
            free_disk_floor_bytes=1 << 30,
            seconds=120,
            checkpoint_reserve_seconds=20,
        )
        if previous:
            shutil.copytree(previous / "checkpoints", inputs / "checkpoints")
            advance_test_ledger(cfg, inputs / "checkpoints")
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
    # Exercise the real CPU-deserialized checkpoint -> HIP cached-generation
    # path, which simple update/resume tests did not previously reach.
    generation_input = tmp_path / "generation-input"
    generation_input.mkdir()
    shutil.copytree(initial / "checkpoints", generation_input / "checkpoints")
    cfg = dict(
        settings(examples(), 0),
        device="cuda",
        checkpoint=baseline["checkpoint"],
        attempts=1,
        generation_tokens=8,
        per_group_seconds=10,
        protocol_sha256=digest("hip-generation-test"),
    )
    write(generation_input / "settings.json", cfg)
    write(
        generation_input / "visible.json",
        [{"group_id": digest("gpu-visible"), "values": ["0"] * 20}],
    )
    generation_output = tmp_path / "generation-output"
    generation_output.mkdir()
    seal = worker.generate(generation_input, generation_output)
    assert seal["checkpoint_sha256"] == baseline["checkpoint_sha256"]
    assert len(seal["records"]) == 1 and len(seal["records"][0]["attempts"]) == 1
    first, result, state = run(tmp_path / "first", 1, initial)
    assert result["device"] == "cuda" and result["completed_updates"] == 1
    assert math.isfinite(result["validation_loss"])
    assert not torch.equal(state["decoder"]["lm_head.weight"], before["decoder"]["lm_head.weight"])
    _resumed, result, state = run(tmp_path / "resumed", 3, first)
    assert result["completed_updates"] == 3 and result["recovery"]["completed"] == 1
    assert math.isfinite(result["validation_loss"])
    assert state["rng"]["cuda"] and state["sampler"]["completed"] == 3
    assert all(torch.isfinite(p).all() for p in state["decoder"].values() if p.is_floating_point())


@pytest.mark.parametrize("saved_length,destination_length", [(8, 8), (16, 8), (4, 8)])
def test_actual_hip_cpu_checkpoint_restores_positions_and_cached_logits(
    saved_length, destination_length
):
    from tests.unit.test_foundation_cache import decoder

    source = decoder(max_seq_len=saved_length, attention_backend="eager").to("cuda")
    source.pos_encoder.pe.add_(0.125)
    state = {name: value.cpu().clone() for name, value in source.state_dict().items()}
    target = decoder(max_seq_len=destination_length, attention_backend="eager").to("cuda")
    target.load_state_dict(state)
    device = next(target.parameters()).device
    assert all(value.device == device for value in target.buffers())
    torch.testing.assert_close(
        target.pos_encoder.pe[:, :saved_length].cpu(), state["pos_encoder.pe"], rtol=0, atol=0
    )
    tokens = torch.tensor([[1, 2, 3]], device=device)
    memory = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(19)).to(device)
    with torch.no_grad():
        expected = source(tokens, memory)
        actual = target(tokens, memory)
        cached, cache = target.prefill(tokens, memory, capacity=destination_length)
        continued = target.decode_step(torch.tensor([4], device=device), cache)
        full = target(torch.tensor([[1, 2, 3, 4]], device=device), memory)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cached, actual, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(continued, full[:, -1:], rtol=1e-4, atol=1e-5)


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
