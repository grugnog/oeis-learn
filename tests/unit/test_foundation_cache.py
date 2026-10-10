import pytest
import torch

from oeis_learn.decoder.wat_decoder import WatTransformerDecoder


def decoder(**kwargs):
    torch.manual_seed(12)
    return WatTransformerDecoder(
        vocab_size=19,
        d_model=16,
        n_heads=2,
        n_decoder_layers=2,
        d_ff=32,
        dropout=0,
        pad_idx=0,
        **kwargs,
    ).eval()


@pytest.mark.parametrize("length", [1, 2, 513, 1024])
def test_cached_logits_match_full_prefix(length):
    model = decoder()
    tokens = torch.randint(1, 19, (1, length))
    memory = torch.randn(1, 3, 16)
    with torch.no_grad():
        expected = model(tokens, memory)
        cached, cache = model.prefill(tokens, memory, capacity=1024)
    torch.testing.assert_close(cached, expected, atol=1e-5, rtol=1e-4)
    assert cache.lengths == [length]


@pytest.mark.parametrize("position", ["sinusoidal", "rope", "prefix_ancestor"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_positions_padding_reorder_and_finished(position, backend):
    model = decoder(position_style=position, attention_backend=backend)
    tokens = torch.tensor([[1, 2, 3], [4, 0, 0]])
    memory = torch.randn(2, 4, 16)
    with torch.no_grad():
        logits, cache = model.prefill(tokens, memory)
        for row, length in enumerate([3, 1]):
            expected = model(tokens[row : row + 1, :length], memory[row : row + 1])
            torch.testing.assert_close(logits[row, :length], expected[0], atol=1e-5, rtol=1e-4)
        cache.reorder([1, 0, 1])
        cache.finished[1] = True
        got = model.decode_step(torch.tensor([5, 6, 7]), cache)
        assert cache.lengths == [2, 3, 2]
        assert not got[1].any()
        expected = model(torch.tensor([[4, 5]]), memory[1:2])[:, -1:]
        torch.testing.assert_close(got[0:1], expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("change", ["weights", "prompt", "dtype", "position", "mode"])
def test_invalidations(change):
    model = decoder()
    memory = torch.randn(1, 2, 16)
    with torch.no_grad():
        _, cache = model.prefill(torch.tensor([[1]]), memory, capacity=2)
        if change == "weights":
            next(model.parameters()).add_(1)
        elif change == "prompt":
            memory.add_(1)
        elif change == "dtype":
            model.double()
        elif change == "position":
            model.position_style = "rope"
        else:
            model.train()
        with pytest.raises(ValueError, match="stale"):
            model.decode_step(torch.tensor([2]), cache)
        assert not cache.valid


def test_capacity_and_uncached_training():
    model = decoder()
    memory = torch.randn(1, 2, 16)
    with torch.no_grad():
        _, cache = model.prefill(torch.tensor([[1]]), memory, capacity=1)
        with pytest.raises(ValueError, match="capacity"):
            model.decode_step(torch.tensor([2]), cache)
    model.train()
    model(torch.tensor([[1, 2]]), memory).sum().backward()
    assert model.lm_head.weight.grad is not None


def test_preallocated_support_and_exact_prompt_buckets():
    from oeis_learn.decoder.grammar_masker import GrammarMasker
    from oeis_learn.decoder.sampler import prompt_length_buckets

    masker = GrammarMasker(9, static_legal_ids=[0, 2, 5])
    first = masker.fill_support([[0, 2], [5]])
    pointer = first.data_ptr()
    expected = first.clone()
    second = masker.fill_support([[5], [0, 2]])
    assert pointer == second.data_ptr()
    assert second[0, 5] and not second[0, 0]
    assert expected[0, 0] and expected[0, 2]
    with pytest.raises(ValueError, match="static"):
        masker.fill_support([[7]])
    assert prompt_length_buckets([[1] * 20, [2] * 5, [3] * 20]) == (
        ((5, 1024), (1,)),
        ((20, 1024), (0, 2)),
    )
