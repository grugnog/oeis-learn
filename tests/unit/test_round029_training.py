import math
import pytest
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.velocity_trials import recipe_bindings, exposure


def test_grid_and_useful_crossing_after_discard():
    rows = velocity_policy()["recipes"]
    assert [(r["batch"], r["decay"]) for r in rows] == [(32,0)]*4 + [(16,0),(32,.01),(16,.01)]
    bindings = recipe_bindings(digest("new-registration"), initialization=13, stream_seed=72,
                               ordinary=digest("old stream"), broad=digest("broad"))
    assert {b["initialization"] for b in bindings.values()} == {13}
    for recipe in ("00","02","04","05","06"):
        assert bindings[recipe]["stream_ref"] == digest("old stream")
    assert bindings["01"]["stream_ref"] == bindings["03"]["stream_ref"]
    stream = {"order": ["a","b","c"]}
    data = {"a": {"tokens": [1,2]}, "b": {"tokens": [1]}, "c": {"tokens": [2]}}
    for size in (16,32):
        cut = exposure(stream, data, updates=2, batch=size, discarded=100)
        assert cut["cursor"] == 2*size
        assert cut["useful_tokens"] == 2*size + math.ceil(2*size/3)
        assert cut["charged_tokens"] == cut["useful_tokens"] + 100


def test_masked_mean_scores_and_batch_normalization():
    import torch
    from oeis_learn.rl.sft_trainer import prefix_matching_loss, calibrate_matching

    true = torch.tensor([[[0.,1.,2.],[0.,2.,1.]], [[0.,1.,2.],[0.,2.,1.]]],
                        requires_grad=True)
    donor = torch.zeros((1,2,3), requires_grad=True)
    target = torch.tensor([[2,0],[1,2]])
    aux, wins = prefix_matching_loss(true, donor, target, [0], pad_id=0)
    d = -math.log(3) - (2 - math.log(1 + math.e + math.exp(2)))
    assert aux.item() == pytest.approx(math.log1p(math.exp(d))/2)
    assert wins == 1
    aux.backward()
    assert true.grad[1].abs().sum() == 0 and torch.isfinite(donor.grad).all()
    assert calibrate_matching([2.,4.,6.], [1.,2.,3.]) == .5
    with pytest.raises(ValueError):
        calibrate_matching([2.], [])


def test_composition_preserves_familiar_slots_and_closes_only_shortage():
    from oeis_learn.generalization.progressive.generator import broader_stream
    def member(i, family, familiar=False, length=4):
        return dict(draw=i, composition=family, tokens=length, band="middle", familiar=familiar)
    slots = [member("a","A",True), member("b","B"), member("b","B"), member("b","B")]
    supply = [member(str(k), c) for k,c in enumerate("BCDE")]
    result = broader_stream(slots, supply, seed=12)
    assert result["available"] and result["order"][0] == "a"
    assert result["ordinary"]["tokens"] == result["broad"]["tokens"] == 16
    assert result["broad"]["distinct"] == 4
    assert not broader_stream(slots, supply[:1], seed=12)["available"]
