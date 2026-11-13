import pytest

from oeis_learn.decoder.bigint_codec import encode_body
from oeis_learn.generalization.progressive.baselines import (
    Portfolio,
    candidates,
    linear_recurrence,
    load_checkpoint,
    load_random,
    polynomial,
)
from oeis_learn.synthesis.reference import evaluate


def prompt(values):
    return {"id": "x", "indices": [str(i) for i in range(20)], "values": [str(v) for v in values]}


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

    recipe = resolve({"version": "bigint-bootstrap/v1"})
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


def test_frozen_conventional_policy_and_shared_attempt_costs():
    from dataclasses import FrozenInstanceError

    from oeis_learn.generalization.progressive.baselines import ConventionalPolicy

    policy = ConventionalPolicy(max_degree=1, max_order=1, coefficient_bits=8)
    with pytest.raises(FrozenInstanceError):
        policy.max_degree = 3
    p = Portfolio(conventional=policy)
    assert not candidates("polynomial", prompt([i * i for i in range(20)]), p)[0]["applicable"]
    assert candidates("polynomial", prompt(range(20)), p)[0]["applicable"]
    report = p.report()
    assert report["methods"]["polynomial"]["attempts"] == 2
    assert report["methods"]["polynomial"]["candidates"] == 1
    assert report["conventional_policy"]["max_degree"] == 1
    p.conventional = ConventionalPolicy()
    with pytest.raises(ValueError, match="frozen"):
        candidates("polynomial", prompt(range(20)), p)


def test_profile_mutation_cannot_renew_portfolio():
    from oeis_learn.generalization.progressive.cohorts import DEFAULT_PROFILE

    supplied = dict(DEFAULT_PROFILE)
    p = Portfolio(profile=supplied)
    supplied["attempts"] = 100
    assert p.profile["attempts"] == 8
    p.profile["attempts"] = 100
    with pytest.raises(ValueError):
        p.reserve("templates")


def test_candidate_stream_drains_before_next_attempt_and_starts_fit_clock_after_generation(
    monkeypatch,
):
    from oeis_learn.generalization.progressive import baselines

    clock = [0.0]
    monkeypatch.setattr(baselines.time, "monotonic", lambda: clock[0])
    original = baselines.polynomial

    def slow(*args, **kwargs):
        clock[0] += 3.0
        return original(*args, **kwargs)

    monkeypatch.setattr(baselines, "polynomial", slow)
    p = Portfolio(started=0.0)
    candidate = next(baselines.iter_candidates("polynomial", prompt(range(20)), p))
    assert candidate["fit_deadline"] == 7.0
    stream = baselines.iter_candidates("templates", prompt(range(20)), p)
    next(stream)
    assert p.attempts == 2
    clock[0] = 60.0
    assert list(stream) == []
    assert p.attempts == 2


def test_late_candidate_is_charged_but_not_returned(monkeypatch):
    from oeis_learn.generalization.progressive import baselines

    clock = [0.0]
    monkeypatch.setattr(baselines.time, "monotonic", lambda: clock[0])

    def late(*args, **kwargs):
        clock[0] = 60.0
        return "(program (return $index))"

    monkeypatch.setattr(baselines, "polynomial", late)
    p = Portfolio(started=0.0)
    assert candidates("polynomial", prompt(range(20)), p) == []
    assert p.report()["methods"]["polynomial"]["attempts"] == 1
    assert p.report()["details"][0]["outcome"] == "deadline"


def test_candidate_generation_error_retains_spent_cost(monkeypatch):
    from oeis_learn.generalization.progressive import baselines

    def broken(*args, **kwargs):
        raise RuntimeError("producer failure")

    monkeypatch.setattr(baselines, "polynomial", broken)
    p = Portfolio()
    with pytest.raises(RuntimeError, match="producer failure"):
        candidates("polynomial", prompt(range(20)), p)
    assert p.report()["methods"]["polynomial"]["attempts"] == 1
    assert p.report()["details"][0]["outcome"] == "error"


def test_conventional_bounds_and_token_limit_are_explicit():
    from oeis_learn.generalization.progressive.baselines import ConventionalPolicy

    p = Portfolio(conventional=ConventionalPolicy(coefficient_bits=8))
    row = candidates("polynomial", prompt([256] * 20), p)[0]
    assert row["source"] is None and row["reason"] == "outside_frozen_applicability"
    row = candidates("polynomial", prompt([10**1100] * 20), Portfolio())[0]
    assert row["source"] is None and row["reason"] == "codec_limit"
    for kwargs in ({"max_degree": True}, {"max_order": 3}, {"coefficient_bits": 4097}):
        with pytest.raises(ValueError):
            ConventionalPolicy(**kwargs)


@pytest.mark.parametrize("values", [[(-2) ** i for i in range(100)], [0] * 100, [7] * 100])
def test_recurrence_signed_zero_and_constant_sequences(values):
    source = linear_recurrence(prompt(values[:20]))
    assert evaluate(source, list(map(str, range(100)))).values == tuple(map(str, values))


def test_nonintegral_and_offset_applicability():
    assert polynomial(prompt([i * (i - 1) // 2 for i in range(20)])) is None
    shifted = {
        "id": "shift",
        "indices": list(map(str, range(-10, 10))),
        "values": [str(i * i) for i in range(-10, 10)],
    }
    assert evaluate(polynomial(shifted), ["-11", "10"]).values == ("121", "100")
    assert linear_recurrence(shifted) is None


def test_model_baselines_load_real_weights_and_enforce_earlier_compatibility(tmp_path):
    import torch

    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.contracts import reference

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    e, d, _ = load_baseline("random", recipe, seed=13)
    torch.save({"encoder": e.state_dict(), "decoder": d.state_dict()}, tmp_path / "baseline.pt")
    ref = reference(tmp_path, "baseline.pt")
    for method in ("retained_D", "earlier_checkpoint", "shuffled", "absent"):
        loaded_e, loaded_d, _ = load_baseline(
            method, recipe, root=tmp_path, ref=ref, reference_recipe=recipe
        )
        assert all(torch.equal(v, loaded_e.state_dict()[k]) for k, v in e.state_dict().items())
        assert all(torch.equal(v, loaded_d.state_dict()[k]) for k, v in d.state_dict().items())
    with pytest.raises(ValueError, match="reference recipe"):
        load_baseline("earlier_checkpoint", recipe, root=tmp_path, ref=ref)
    # Same tensor shapes do not establish identical input semantics.
    incompatible = resolve(
        {"version": "bigint-bootstrap/v1", "inputs": {"ablation_mode": "retrained"}}
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_baseline(
            "earlier_checkpoint", incompatible, root=tmp_path, ref=ref, reference_recipe=recipe
        )
    with pytest.raises(ValueError):
        load_baseline("random", recipe, seed=13, ref=ref)


def test_random_baseline_is_seeded_and_not_a_checkpoint_alias():
    import torch

    from oeis_learn.fresh.bigint_config import resolve

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    a = load_random(recipe, seed=11)[1].state_dict()
    b = load_random(recipe, seed=11)[1].state_dict()
    c = load_random(recipe, seed=12)[1].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_fitting_costs_do_not_double_count_candidates(monkeypatch):
    from oeis_learn.generalization.progressive import baselines

    clock = [0.0]
    monkeypatch.setattr(baselines.time, "monotonic", lambda: clock[0])
    p = Portfolio(started=0.0)
    row = candidates("polynomial", prompt(range(20)), p)[0]
    clock[0] = 4.0
    p.observe(
        "polynomial",
        candidate=baselines.digest(row["source"]),
        started=0.0,
        ended=4.0,
        outcome="complete",
        phase="fit",
    )
    report = p.report()
    assert report["methods"]["polynomial"] == {
        "attempts": 1,
        "candidates": 1,
        "elapsed_us": 4000000,
    }
    assert report["total_elapsed_us"] == 4000000


def test_conditioning_is_reproducible_closed_and_does_not_mutate_inputs():
    from copy import deepcopy

    from oeis_learn.generalization.progressive.baselines import conditioning

    prompts = [prompt(range(i, i + 20)) | {"id": str(i)} for i in range(4)]
    before = deepcopy(prompts)
    shuffled = conditioning(prompts, "shuffled", seed=13)
    assert shuffled == conditioning(prompts, "shuffled", seed=13)
    assert all(p["values"] != s["values"] for p, s in zip(prompts, shuffled))
    assert sorted(tuple(p["values"]) for p in prompts) == sorted(
        tuple(p["values"]) for p in shuffled
    )
    assert all(p["values"] is None for p in conditioning(prompts, "absent", seed=13))
    shuffled[0]["values"][0] = "999"
    assert prompts == before
    for invalid in (True, -1, 2**32):
        with pytest.raises(ValueError):
            conditioning(prompts, "shuffled", seed=invalid)
    with pytest.raises(ValueError):
        conditioning([prompts[0], prompts[0]], "shuffled", seed=13)
