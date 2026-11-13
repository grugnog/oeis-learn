"""Independent acceptance contracts for the shared logical optimizer update."""

import copy

import pytest
import torch
from torch import nn

from oeis_learn.rl import sft_trainer
from tests.unit.test_progressive_optimizer import fixture, independent_objective


class Counter:
    def __init__(self):
        self.count = 0

    def step(self):
        self.count += 1


def test_accumulated_uneven_batch_matches_gold_with_one_clip_optimizer_lr_and_allocation(
    monkeypatch,
):
    e, d, x, t = fixture()
    gold_e, gold_d = copy.deepcopy(e), copy.deepcopy(d)
    opt = torch.optim.SGD([*e.parameters(), *d.parameters()], lr=0.03)
    gold = torch.optim.SGD([*gold_e.parameters(), *gold_d.parameters()], lr=0.03)
    loss = independent_objective(gold_e, gold_d, x, t)
    loss.backward()
    nn.utils.clip_grad_norm_([*gold_e.parameters(), *gold_d.parameters()], 1.0)
    gold.step()
    clip = nn.utils.clip_grad_norm_
    calls = []

    def measured(*args, **kwargs):
        calls.append(1)
        return clip(*args, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", measured)
    step = opt.step
    steps = []

    def counted(*args, **kwargs):
        steps.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(opt, "step", counted)
    scheduler, clock = Counter(), Counter()
    result = sft_trainer.logical_update(
        e,
        d,
        opt,
        x,
        t,
        device="cpu",
        pad_id=0,
        microbatch_size=2,
        scheduler=scheduler,
        allocation_clock=clock,
    )
    assert abs(result["loss"] - float(loss.detach())) < 1e-6
    assert len(calls) == len(steps) == scheduler.count == clock.count == 1
    for a, b in zip(
        [*e.parameters(), *d.parameters()], [*gold_e.parameters(), *gold_d.parameters()]
    ):
        assert torch.isfinite(a).all() and torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_interrupted_microbatch_never_steps_optimizer_or_clocks(monkeypatch):
    e, d, x, t = fixture()
    original = e.forward_from_sequences
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise InterruptedError("mid logical update")
        return original(*args, **kwargs)

    monkeypatch.setattr(e, "forward_from_sequences", interrupted)
    opt = torch.optim.SGD([*e.parameters(), *d.parameters()], lr=0.03)
    before = [p.detach().clone() for p in [*e.parameters(), *d.parameters()]]
    scheduler, clock = Counter(), Counter()
    with pytest.raises(InterruptedError):
        sft_trainer.logical_update(
            e,
            d,
            opt,
            x,
            t,
            device="cpu",
            pad_id=0,
            microbatch_size=2,
            scheduler=scheduler,
            allocation_clock=clock,
        )
    assert scheduler.count == clock.count == 0
    assert all(torch.equal(a, b) for a, b in zip(before, [*e.parameters(), *d.parameters()]))


@pytest.mark.parametrize("size", [1, 2, 3, 8])
def test_partition_metrics_and_gradients_match_independent_batch(size):
    e, d, x, t = fixture()
    ge, gd = copy.deepcopy(e), copy.deepcopy(d)
    loss = independent_objective(ge, gd, x, t)
    loss.backward()
    expected_norm = nn.utils.clip_grad_norm_([*ge.parameters(), *gd.parameters()], 1.0)
    gold = torch.optim.SGD([*ge.parameters(), *gd.parameters()], lr=0.03)
    gold.step()
    opt = torch.optim.SGD([*e.parameters(), *d.parameters()], lr=0.03)
    result = sft_trainer.logical_update(
        e,
        d,
        opt,
        x,
        t,
        device="cpu",
        pad_id=0,
        microbatch_size=size,
        observe=False,
        learning_metrics=True,
    )
    assert result["loss"] == pytest.approx(float(loss.detach()), abs=1e-6)
    assert result["gradient_norm"] == pytest.approx(float(expected_norm), abs=1e-6)
    assert result["token_metrics"]["tokens"] == 7
    assert result["token_metrics"]["programs"] == 3
    for a, b in zip([*e.parameters(), *d.parameters()], [*ge.parameters(), *gd.parameters()]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_failed_backward_clears_partial_gradients_and_leaves_clocks(monkeypatch):
    e, d, x, t = fixture()
    opt = torch.optim.AdamW([*e.parameters(), *d.parameters()], lr=0.03)
    scheduler, clock = Counter(), Counter()
    original = d.forward
    calls = []

    def nonfinite(*args, **kwargs):
        calls.append(1)
        logits = original(*args, **kwargs)
        # Finite forward, infinite backward on the second microbatch.
        if len(calls) == 2:
            logits.register_hook(lambda grad: grad * float("inf"))
        return logits

    monkeypatch.setattr(d, "forward", nonfinite)
    before = [p.detach().clone() for p in [*e.parameters(), *d.parameters()]]
    with pytest.raises(RuntimeError, match="non-finite"):
        sft_trainer.logical_update(
            e,
            d,
            opt,
            x,
            t,
            device="cpu",
            pad_id=0,
            microbatch_size=2,
            scheduler=scheduler,
            allocation_clock=clock,
        )
    assert scheduler.count == clock.count == 0
    assert not opt.state
    assert all(p.grad is None for p in [*e.parameters(), *d.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(before, [*e.parameters(), *d.parameters()]))
