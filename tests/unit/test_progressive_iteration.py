import pytest
from oeis_learn.generalization.progressive.iteration import margin
from oeis_learn.generalization.progressive.contracts import digest, seal


def test_complete_update_probe_publication_margin_precedes_work(tmp_path):
    from tests.unit.test_progressive_recovery_transactions import setup
    from tests.integration.test_progressive_recovery import states
    from oeis_learn.generalization.progressive.recovery import Recovery

    t, j, cuts = setup(tmp_path)
    p = t["payload"]
    p["iteration_budget"] = dict(
        update_us=200,
        probe_us=300,
        checkpoint_us=500,
        producer=dict(path="producer.json", bytes=2, sha256=digest("producer")),
    )
    t = seal(t)
    with Recovery(j, t, digest("learning")) as r:
        r.initialize(states(), cuts)
        r.begin(boot="b", monotonic_us=0)
        before = j.path.read_bytes()
        with pytest.raises(ValueError, match="finalization"):
            r.reserve(
                ["slow-update"],
                1,
                boot="b",
                monotonic_us=r.clock["end"] - p["finalization_us"] - 1000,
            )
        assert j.path.read_bytes() == before
    with pytest.raises(ValueError):
        margin({}, research=True)
