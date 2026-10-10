import pytest
import torch
from torch import nn

from oeis_learn.rl.policy import (
    FrozenReference,
    group_advantages,
    masked_distribution,
    policy_loss,
    scoring_mode,
)


def test_two_step_enumeration_and_finite_difference():
    logits = torch.tensor(
        [[[0.2, -0.3, 9.0], [0.4, 0.1, -0.2]]], dtype=torch.float64, requires_grad=True
    )
    support = torch.tensor([[[True, True, False], [True, True, True]]])
    actions = torch.tensor([[1, 2]])
    logs, probabilities, entropy = masked_distribution(logits, support, 0.8)
    expected = torch.softmax(logits[0, 0, :2] / 0.8, -1)
    torch.testing.assert_close(probabilities[0, 0, :2], expected)
    # The four/six joint paths exhaust the exact two-step policy.
    assert (
        probabilities[0, 0, :, None] * probabilities[0, 1, None, :]
    ).sum().item() == pytest.approx(1)
    behavior = logs.detach().gather(-1, actions[..., None]).squeeze(-1)
    advantages = torch.tensor([0.7], dtype=torch.float64)
    loss = policy_loss(
        logits,
        actions,
        behavior,
        advantages,
        supports=support,
        temperatures=0.8,
        reference_logits=logits.detach() + 0.15,
        entropy_coefficient=0.03,
    )
    loss.backward()
    analytic = logits.grad.clone()
    for i in range(2):
        for j in range(3):
            high, low = logits.detach().clone(), logits.detach().clone()
            high[0, i, j] += 1e-6
            low[0, i, j] -= 1e-6

            def score(value):
                return policy_loss(
                    value,
                    actions,
                    behavior,
                    advantages,
                    supports=support,
                    temperatures=0.8,
                    reference_logits=logits.detach() + 0.15,
                    entropy_coefficient=0.03,
                )

            numerical = (score(high) - score(low)) / 2e-6
            torch.testing.assert_close(analytic[0, i, j], numerical, atol=1e-7, rtol=1e-5)
    assert analytic[0, 0, 2] == 0
    assert torch.isfinite(entropy).all() and torch.isfinite(analytic).all()


def test_actual_behavior_ratio_forced_pad_and_sampled_eos():
    original = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    changed = torch.tensor([[[0.1, -0.1], [0.3, -0.3], [0.2, -0.2]]], requires_grad=True)
    actions = torch.tensor([[0, 1, 0]])  # middle action is sampled EOS
    old = original.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
    mask = torch.tensor([[False, True, False]])
    loss = policy_loss(
        changed, actions, old, torch.tensor([1.0]), action_mask=mask, kl_coefficient=0
    )
    ratio = (changed.log_softmax(-1)[0, 1, 1] - old[0, 1]).exp()
    expected = -torch.minimum(ratio, ratio.clamp(0.8, 1.2))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert changed.grad[0, 0].abs().sum() == changed.grad[0, 2].abs().sum() == 0
    assert changed.grad[0, 1].abs().sum() > 0
    with pytest.raises(ValueError, match="fresh"):
        policy_loss(
            changed, actions, old, torch.tensor([1.0]), action_mask=mask, objective="reinforce"
        )
    for rewards in ([1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]):
        assert group_advantages(torch.tensor(rewards)).abs().sum() == 0


def test_nested_modes_restore_on_exception_and_full_reference_frozen():
    encoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    decoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    encoder.train()
    encoder[0].eval()
    decoder.eval()
    decoder[1].train()
    modes = [module.training for model in (encoder, decoder) for module in model.modules()]
    with pytest.raises(RuntimeError):
        with scoring_mode(encoder, decoder):
            assert all(not m.training for model in (encoder, decoder) for m in model.modules())
            raise RuntimeError("scoring failed")
    assert modes == [module.training for model in (encoder, decoder) for module in model.modules()]
    reference = FrozenReference(encoder, decoder)
    x = torch.ones(1, 2)
    before = reference.decoder(reference.encoder(x)).clone()
    with torch.no_grad():
        encoder[0].weight.add_(1)
        decoder[0].weight.add_(1)
    torch.testing.assert_close(reference.decoder(reference.encoder(x)), before)
    assert all(
        not p.requires_grad
        for model in (reference.encoder, reference.decoder)
        for p in model.parameters()
    )
