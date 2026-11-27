import math
import pytest
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.velocity_trials import recipe_bindings, exposure


def test_grid_and_useful_crossing_after_discard():
    rows = velocity_policy()["recipes"]
    assert [(r["batch"], r["decay"]) for r in rows] == [(32, 0)] * 4 + [
        (16, 0),
        (32, 0.01),
        (16, 0.01),
    ]
    bindings = recipe_bindings(
        digest("new-registration"),
        initialization=13,
        stream_seed=72,
        ordinary=digest("old stream"),
        broad=digest("broad"),
    )
    assert {b["initialization"] for b in bindings.values()} == {13}
    for recipe in ("00", "02", "04", "05", "06"):
        assert bindings[recipe]["stream_ref"] == digest("old stream")
    assert bindings["01"]["stream_ref"] == bindings["03"]["stream_ref"]
    stream = {"order": ["a", "b", "c"]}
    data = {"a": {"tokens": [1, 2]}, "b": {"tokens": [1]}, "c": {"tokens": [2]}}
    for size in (16, 32):
        cut = exposure(stream, data, updates=2, batch=size, discarded=100)
        assert cut["cursor"] == 2 * size
        assert cut["useful_tokens"] == 2 * size + math.ceil(2 * size / 3)
        assert cut["charged_tokens"] == cut["useful_tokens"] + 100


def test_masked_mean_scores_and_batch_normalization():
    import torch
    from oeis_learn.rl.sft_trainer import prefix_matching_loss, calibrate_matching

    true = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]], [[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]], requires_grad=True
    )
    donor = torch.zeros((1, 2, 3), requires_grad=True)
    target = torch.tensor([[2, 0], [1, 2]])
    aux, wins = prefix_matching_loss(true, donor, target, [0], pad_id=0)
    d = -math.log(3) - (2 - math.log(1 + math.e + math.exp(2)))
    assert aux.item() == pytest.approx(math.log1p(math.exp(d)) / 2)
    assert wins == 1
    zero, wins_zero = prefix_matching_loss(true, donor[:0], target, [], pad_id=0)
    assert zero.item() == 0 and wins_zero == 0
    aux.backward()
    assert true.grad[1].abs().sum() == 0 and torch.isfinite(donor.grad).all()
    assert calibrate_matching([2.0, 4.0, 6.0], [1.0, 2.0, 3.0]) == 0.5
    with pytest.raises(ValueError):
        calibrate_matching([2.0], [])


def test_composition_preserves_familiar_slots_and_closes_only_shortage():
    from oeis_learn.generalization.progressive.generator import broader_stream

    def member(i, family, familiar=False, length=4):
        return dict(draw=i, composition=family, tokens=length, band="middle", familiar=familiar)

    slots = [member("a", "A", True), member("b", "B"), member("b", "B"), member("b", "B")]
    supply = [member(str(k), c) for k, c in enumerate("BCDE")]
    result = broader_stream(slots, supply, seed=12)
    assert result["available"] and result["order"][0] == "a"
    assert result["ordinary"]["tokens"] == result["broad"]["tokens"] == 16
    assert result["broad"]["distinct"] == 4
    assert not broader_stream(slots, supply[:1], seed=12)["available"]



def test_frozen_study_registers_and_reopens_looks_without_sampler_feedback(tmp_path):
    from oeis_learn.generalization.progressive.campaign import velocity_registration
    from oeis_learn.generalization.progressive.graph import velocity_study_looks
    from oeis_learn.pipeline_validation.progressive_study import open_velocity_study, close_velocity_study

    development = [dict(task="a", group="f", stratum="oeis"), dict(task="b", group="f", stratum="oeis")]
    retention = [dict(task="r", group="known", stratum="retention")]
    args = dict(source="a" * 40, design=digest("frozen hypotheses, workers and prices"),
                development=development, retention=retention, initialization=53, stream_seed=67,
                ordinary=digest("ordinary stream"), broad=digest("broad stream"))
    reg = velocity_registration(**args)
    journal = open_velocity_study(tmp_path, reg)
    assert len(journal.trials) == 7
    assert [journal.velocity_trial(f"{i:02d}")["binding"]["recipe"]["id"] for i in range(7)] == [
        "00", "01", "02", "03", "04", "05", "06"]
    graph = velocity_study_looks(reg)
    cheap = graph["00"]["common"]
    assert [r["look"] for r in cheap] == [0, 1_000_000, 4_000_000]
    assert graph["00"]["full"][0]["members"] == development + retention
    assert all(r["members"] == development[:1] for r in cheap)
    with pytest.raises(ValueError, match="previous"):
        journal.observe_velocity(reg, cheap[1], evidence=digest("one million"))
    first = journal.observe_velocity(reg, cheap[0], evidence=digest("zero"))
    assert journal.observe_velocity(reg, cheap[0], evidence=digest("zero")) == first
    with pytest.raises(ValueError, match="differs"):
        journal.observe_velocity(reg, cheap[0], evidence=digest("other"))
    with pytest.raises(ValueError):
        journal.observe_velocity(reg, dict(cheap[1], members=[]), evidence=digest("one million"))
    journal.observe_velocity(reg, cheap[1], evidence=digest("one million"))
    journal.observe_velocity(reg, cheap[2], reason="physical budget exhausted")
    assert all(row["status"] == "PROPOSED" for row in journal.trials.values())
    assert open_velocity_study(tmp_path, reg).observe_velocity(reg, cheap[0], evidence=digest("zero")) == first
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id
    accepted = seal(dict(registration=reg["identity"], scope="exploration"))
    authority = physical_root_id(tmp_path)
    terms = dict(version="round029-grant/v1", study_id=SESSION, scope="cpu", state_root_id=authority,
                 registration_or_parent_ref=accepted["identity"], total_us=100, holdback_us=10,
                 allowed_operations=["cpu", "structure_search"], resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=100))
    approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
    grant = seal(dict(terms, authorizing_record_ref=approval["identity"]))
    funded = dict(decision=digest("observed deficit"), reason="price a fixed-order comparison",
                  grant=grant, approval=approval, accepted=accepted, root_id=authority)
    assert journal.record_velocity_branch("structure_search", **funded)["grant"] == grant["identity"]
    assert journal.record_velocity_branch("paired_extension", decision=first, reason="unfunded")["state"] == "unused"
    with pytest.raises(ValueError):
        journal.record_velocity_branch("confirmation", **funded)
    boundary = close_velocity_study(tmp_path, reg, state="resumable", evidence=first)
    child = velocity_registration(**dict(args, initialization=71, stream_seed=99, ordinary=digest("new stream"),
                                         development=[dict(task="new", group="fresh", stratum="oeis")],
                                         retention=[dict(task="new-r", group="fresh-r", stratum="retention")]),
                                  scope="confirmation", parent=boundary, finalist="03")
    assert set(velocity_study_looks(child)) == {"00", "03", "D"}
    assert len(open_velocity_study(tmp_path, child).trials) == 2
    with pytest.raises((FileNotFoundError, ValueError)):
        open_velocity_study(tmp_path / "foreign", child)
    with pytest.raises(ValueError):
        velocity_registration(**args, scope="confirmation", parent=dict(boundary, state="running"), finalist="03")
