"""New fixed-bin adapter does not reinterpret historical PAC policies."""

import pytest


def layout():
    return {
        "bins": ["familiar", "frontier", "composition"],
        "probes": {
            b: {"role": "training", "id": "probe/" + b}
            for b in ["familiar", "frontier", "composition"]
        },
    }


def test_fixed_floors_and_eight_update_feedback_clock():
    from oeis_learn.generalization.progressive.curriculum import Controller

    c = Controller(layout(), seed=13)
    p = c.probabilities()
    assert p["familiar"] >= 0.2 and p["frontier"] >= 0.1
    selected = c.select()
    for i in range(7):
        c.commit(tokens=5, success=True)
        assert c.select() == selected
    c.commit(tokens=5, success=True)
    with pytest.raises(ValueError):
        c.select()
    c.feedback(role="training", probe=layout()["probes"][selected["bin"]]["id"], gain=0.5)
    assert c.snapshot()["updates"] == 8
    assert c.snapshot()["charged_tokens"] == 40
    c.select()


def test_resume_replays_selection_rng_and_preserves_rejection_watermark():
    from oeis_learn.generalization.progressive.curriculum import Controller

    c = Controller(layout(), seed=13)
    c.select()
    c.commit(tokens=7, success=False)
    c.set_stream_cut(shard=2, cursor=3, rejection_watermark=11)
    state = c.snapshot()
    r = Controller.restore(state, layout())
    assert r.select() == c.select()
    assert r.snapshot() == c.snapshot()
    assert r.snapshot()["discarded_tokens"] == 7
    changed = layout()
    changed["bins"].append("new")
    with pytest.raises(ValueError):
        Controller.restore(state, changed)
    with pytest.raises(ValueError):
        c.feedback(role="development", probe="probe/familiar", gain=1)
    with pytest.raises(ValueError):
        Controller({"bins": ["familiar", "frontier"], "probes": {}}, seed=13)
