import pytest
import torch
from oeis_learn.fresh.conditioning_controls import derangement, condition_memory
from oeis_learn.fresh.synthesis_policy import (
    Portfolio,
    select_visible,
    transfer_success,
    rescue_identity,
)
from oeis_learn.fresh.template_baseline import template_bank


def test_derangement_absent_masks_and_fixed_templates():
    mapping = derangement(("a", "b", "c"), 17)
    assert mapping == derangement(("a", "b", "c"), 17) and all(k != v for k, v in mapping.items())
    memory = torch.randn(3, 4, 8)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    absent, padding = condition_memory(memory, mask, "absent")
    assert absent.shape == (3, 1, 8) and not padding.any()
    shuffled, _ = condition_memory(memory, mask, "shuffled", order=(1, 2, 0))
    torch.testing.assert_close(shuffled[0], memory[1])
    with pytest.raises(ValueError):
        condition_memory(memory, mask, "shuffled", order=(0, 1, 2))
    assert len(template_bank()["sources"]) == 8
    from oeis_learn.decoder import bigint_codec
    from oeis_learn.synthesis.reference_parser import parse

    for source in template_bank()["sources"]:
        parse(source, allow_parameters=True)
        tokens = bigint_codec.encode_body(source)
        assert bigint_codec.encode_body(bigint_codec.decode_body(tokens)) == tokens


def test_aggregate_portfolio_budget_and_incomplete_tokens():
    p = Portfolio(start=10, seed=17)
    for i in range(8):
        attempt = p.begin(now=10 + i)
        assert attempt["method"] == ("greedy" if i == 0 else "stochastic")
        p.finish_generation(
            attempt["id"], tokens=list(range(1024)), complete=False, source="", now=11 + i
        )
    with pytest.raises(ValueError):
        p.begin(now=20)
    assert p.fit_allowance(now=69) == (0, 1)
    assert p.remaining(now=71) == 0
    assert sum(len(r["tokens"]) for r in p.records.values()) == 8 * 1024
    with pytest.raises(ValueError):
        rescue_identity("a", "b")


def test_visible_selection_sealed_before_full_horizon_scoring():
    from oeis_learn.synthesis.admission import source_identity

    rows = [
        dict(
            source="long",
            verified=True,
            source_sha256=source_identity("long"),
            visible_count=20,
            cost=1,
        ),
        dict(
            source="x", verified=True, source_sha256=source_identity("x"), visible_count=20, cost=2
        ),
    ]
    selected = select_visible(rows)
    assert selected["source"] == "x"
    expected = [str(i) for i in range(100)]

    def evidence(values, **changes):
        return dict(
            source_sha256=selected["source_sha256"],
            indices=list(map(str, range(100))),
            values=values,
            outcome="ok",
            **changes,
        )

    assert transfer_success(
        selected, list(map(str, range(100))), expected, evidence(expected), evidence(expected)
    )
    bad = expected[:20] + ["0"] * 80
    assert not transfer_success(
        selected, list(map(str, range(100))), expected, evidence(bad), evidence(bad)
    )
    assert not transfer_success(
        selected,
        list(map(str, range(100))),
        expected,
        evidence(expected[:20]),
        evidence(expected[:20]),
    )
    with pytest.raises(ValueError):
        select_visible([{**rows[0], "hidden_score": 100}])
