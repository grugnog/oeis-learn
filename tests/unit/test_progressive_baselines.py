import pytest
from oeis_learn.generalization.progressive.baselines import (
    polynomial,
    linear_recurrence,
    candidates,
    Portfolio,
    load_checkpoint,
    load_random,
)
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.decoder.bigint_codec import encode_body


def prompt(values):
    return dict(id="x", indices=[str(i) for i in range(20)], values=[str(v) for v in values])


@pytest.mark.parametrize(
    "values,expected",
    [
        (list(range(20)), list(range(100))),
        ([i * i - 2 * i + 3 for i in range(20)], [i * i - 2 * i + 3 for i in range(100)]),
    ],
)
def test_conventional_polynomial_is_visible_only_and_independently_exact(values, expected):
    source = polynomial(prompt(values))
    encode_body(source)
    result = evaluate(source, [str(i) for i in range(100)])
    assert result.values == tuple(map(str, expected))


def test_recurrence_has_fixed_order_and_exact_iterative_meaning():
    y = [0, 1]
    while len(y) < 100:
        y.append(y[-1] + y[-2])
    source = linear_recurrence(prompt(y[:20]))
    encode_body(source)
    assert evaluate(source, [str(i) for i in range(100)]).values == tuple(map(str, y))
    with pytest.raises(ValueError):
        linear_recurrence(prompt(y[:20]), max_order=3)
    with pytest.raises(ValueError):
        polynomial(prompt(y[:20]), max_degree=4)
    assert polynomial(prompt(y[:20])) is None


def test_template_branches_share_one_allowance():
    p = Portfolio()
    result = candidates("templates", prompt(range(20)), p)
    assert len(result) == 8 and p.attempts == 8
    assert candidates("polynomial", prompt(range(20)), p) == []


def test_real_canonical_checkpoint_round_trip_and_identity(tmp_path):
    import torch
    from oeis_learn.fresh.bigint_config import resolve

    recipe = resolve(dict(version="bigint-bootstrap/v1"))
    encoder, decoder, counts = load_random(recipe, seed=13)
    weights = {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}
    torch.save(weights, tmp_path / "weights.pt")
    from oeis_learn.generalization.progressive.contracts import reference

    ref = reference(tmp_path, "weights.pt")
    restored_e, restored_d, loaded_counts = load_checkpoint(tmp_path, ref, recipe)
    assert loaded_counts["total_parameters"] == counts["total_parameters"]
    for model, key in [(restored_e, "encoder"), (restored_d, "decoder")]:
        assert all(
            torch.equal(value, weights[key][name]) for name, value in model.state_dict().items()
        )
        assert not model.training
    with (tmp_path / "weights.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, ref, recipe)


def test_per_method_costs_include_candidate_preparation():
    p = Portfolio()
    candidates("polynomial", prompt(range(20)), p)
    report = p.report()
    assert report["attempts"] == 1
    assert report["methods"]["polynomial"]["candidates"] == 1
    assert report["methods"]["polynomial"]["elapsed_us"] >= 0
    assert report["details"][0]["outcome"] == "candidate_generated"
