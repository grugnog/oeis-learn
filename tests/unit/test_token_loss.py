"""Token loss preserves objectives/gradients without nondeterministic reduction."""

import pytest
import torch
import torch.nn.functional as F

from oeis_learn.decoder.program_codec import EOS_ID, FOUNDATION_VOCAB_SIZE, PAD_ID
from oeis_learn.rl.foundation_sft import per_program_loss
from oeis_learn.rl.token_loss import token_cross_entropy
from oeis_learn.tracking.role_worker import validation_loss


@pytest.fixture
def strict_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("ignore_padding", [False, True])
def test_token_losses_and_gradients_match_spatial_reference(strict_determinism, ignore_padding):
    generator = torch.Generator().manual_seed(117)
    # Noncontiguous inputs cover the actual logits/target slices from trainers.
    original = torch.randn(2, FOUNDATION_VOCAB_SIZE, 3, generator=generator, dtype=torch.float64)
    logits = original.transpose(1, 2).detach().requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    targets = torch.tensor([[EOS_ID, 4], [PAD_ID, 5], [PAD_ID, EOS_ID]]).transpose(0, 1)
    ignore = PAD_ID if ignore_padding else -100
    actual = token_cross_entropy(logits, targets, ignore_index=ignore)
    expected = F.cross_entropy(
        reference.transpose(1, 2), targets, ignore_index=ignore, reduction="none"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    weights = torch.tensor([[1.0, 0, 0], [1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-14, atol=1e-14)
    if ignore_padding:
        assert not actual[0, 1:].any()
    assert not logits.grad[0, 1:].any()
    assert logits.grad[0, 0, EOS_ID] != 0


def test_per_program_loss_preserves_equal_weight_padding_and_eos(strict_determinism):
    logits = torch.zeros(2, 3, FOUNDATION_VOCAB_SIZE, requires_grad=True)
    targets = torch.tensor([[EOS_ID, PAD_ID, PAD_ID], [4, 5, EOS_ID]])
    with torch.no_grad():
        logits[0, 0, EOS_ID] = 2
    expected = (
        F.cross_entropy(logits[0, :1], targets[0, :1]) + F.cross_entropy(logits[1], targets[1])
    ) / 2
    actual = per_program_loss(logits, targets)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(
        actual,
        F.cross_entropy(
            torch.cat([logits[0, :1], logits[1]]), torch.cat([targets[0, :1], targets[1]])
        ),
    )
    actual.backward()
    assert logits.grad[0, 0, EOS_ID] != 0
    assert not logits.grad[0, 1:].any()


def test_validation_avoids_hip_spatial_reduction_and_keeps_modes(monkeypatch, strict_determinism):
    original = F.cross_entropy

    def hip_reduction_guard(logits, targets, **kwargs):
        # Reproduce the observed HIP failure on CPU so the regression runs in CI.
        if logits.ndim > 2 and kwargs.get("reduction", "mean") != "none":
            raise RuntimeError("nll_loss2d_forward_out_cuda_template is not deterministic")
        return original(logits, targets, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", hip_reduction_guard)
    generator = torch.Generator().manual_seed(15)
    logits = [
        torch.randn(1, length, FOUNDATION_VOCAB_SIZE, generator=generator) for length in (2, 4)
    ]
    rows = [
        {"tokens": [4] * (length - 1) + [EOS_ID], "entry": {"conditioning_values": ["0"] * 20}}
        for length in (2, 4)
    ]
    iterator = iter(logits)
    encoder, decoder = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    encoder.train()
    decoder.eval()

    def forward(e, d, sequences, tokens, **kwargs):
        assert not e.training and not d.training and not torch.is_grad_enabled()
        return next(iterator)

    monkeypatch.setattr("oeis_learn.rl.sft_trainer.teacher_forced_logits", forward)
    expected = (
        sum(
            float(original(value[0], torch.tensor(row["tokens"])))
            for value, row in zip(logits, rows)
        )
        / 2
    )
    actual = validation_loss(encoder, decoder, rows, "cpu")
    assert actual == pytest.approx(expected)
    assert encoder.training and not decoder.training


@pytest.mark.parametrize(
    "logits,targets",
    [
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(2, 3, 4), torch.zeros(2, 2, dtype=torch.long)),
    ],
)
def test_token_shape_mismatch_is_rejected(logits, targets):
    with pytest.raises(ValueError, match="matching"):
        token_cross_entropy(logits, targets)
