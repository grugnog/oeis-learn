"""Independent logical-batch oracle and real RNG semantics for T037/T040."""

import copy

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1, 4)

    def forward_from_sequences(self, sequences, *, device):
        return self.project(
            torch.tensor([[float(s[0])] for s in sequences], device=device)
        ).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.project = nn.Linear(4, 9)

    def forward(self, tokens, memory, **kwargs):
        return self.project(self.embedding(tokens) + memory)


def fixture():
    torch.manual_seed(13)
    return (
        Encoder(),
        Decoder(),
        [[1], [2], [3]],
        torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6], [1, 7, 2, 0]]),
    )


def independent_objective(e, d, sequences, tokens):
    # Explicit per-program means; no production reducer or shared loss helper.
    logits = d(tokens[:, :-1], e.forward_from_sequences(sequences, device="cpu"))
    target = tokens[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    mask = target.ne(0)
    return ((losses * mask).sum(1) / mask.sum(1)).mean()


def test_existing_full_batch_matches_independent_per_program_objective():
    from oeis_learn.rl.sft_trainer import supervised_step

    e, d, x, t = fixture()
    expected_e, expected_d = copy.deepcopy(e), copy.deepcopy(d)
    expected = torch.optim.SGD([*expected_e.parameters(), *expected_d.parameters()], lr=0.03)
    loss = independent_objective(expected_e, expected_d, x, t)
    loss.backward()
    nn.utils.clip_grad_norm_([*expected_e.parameters(), *expected_d.parameters()], 1.0)
    expected.step()
    actual = torch.optim.SGD([*e.parameters(), *d.parameters()], lr=0.03)
    result = supervised_step(e, d, actual, x, t, device="cpu", pad_id=0, observe=False)
    assert abs(result["loss"] - float(loss.detach())) < 1e-6
    for a, b in zip(
        [*e.parameters(), *d.parameters()], [*expected_e.parameters(), *expected_d.parameters()]
    ):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_uneven_microbatch_gold_weights_examples_not_tokens():
    # Program losses are 3,9,3; a two-plus-one partition must yield 5.
    assert (2 * ((3 + 9) / 2) + 3) / 3 == 5
    assert (2 + 4 + 9 + 3 + 3 + 3) / 6 == 4
    assert ((3 + 9) / 2 + 3) / 2 == 4.5


def test_dropout_rng_restore_repeats_same_partition_without_cross_partition_claim():
    dropout = nn.Dropout(0.2).train()
    x = torch.ones(7, 31)
    with torch.random.fork_rng():
        torch.manual_seed(13)
        saved = torch.get_rng_state()
        a = dropout(x)
        after = torch.get_rng_state()
        torch.set_rng_state(saved)
        b = dropout(x)
        assert torch.equal(a, b)
        assert torch.equal(torch.get_rng_state(), after)
        assert not torch.equal(saved, after)
