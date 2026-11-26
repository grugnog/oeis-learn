from oeis_learn.generalization.progressive.training_rows import first_divergence, verified_training_pairs
from oeis_learn.pipeline_validation.progressive_preparation import verify_draw


def test_first_divergence_must_share_real_context_and_exclude_other_positives():
    good, bad = [2, 3, 4], [2, 6, 8]
    assert first_divergence(good, bad, [good]) == dict(position=1, positive=3, negative=6)
    assert first_divergence(good, bad, [good, [2, 6, 9]]) is None
    assert first_divergence(good, good, [good]) is None


def training_rows():
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import digest
    # Independent native fixture supplies values by index, separate from Python program evaluation.
    rows = []
    sources = ["(program (return (int 0)))",
               "(program (return (select (lt $index (int 20)) (int 0) (int 1))))"]
    for i, source in enumerate(sources):
        value = dict(source=source, ordinal=i, draw_identity=digest(i), source_policy=digest("mechanical"),
                     structure=digest(source), detector_limit="fixture")
        client = SimpleNamespace(execute=lambda s, xs, **kw: dict(outcome="ok",
                                 values=[str(int(i == 1 and int(x) >= 20)) for x in xs]))
        row, reason = verify_draw(value, client=client)
        assert reason is None
        rows.append(row)
    return rows


def test_audit_funding_uses_only_executable_pairs(monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round028_support as audit
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.program_search_routing import project, route
    from oeis_learn.generalization.progressive.training_rows import matched_stream
    from tests.contract.test_round028_routing import sample

    rows = training_rows()
    for probes, expected in [({"held": {"id": rows[0]["draw_identity"]}}, 0), ({}, 2)]:
        package = seal(dict(shards=[rows], layout={"probes": probes}))
        monkeypatch.setattr(audit, "training_package", lambda *args: package)
        support = audit.training_support(None, {})
        stream = matched_stream(package["shards"], package["layout"], digest("registered"), 0)
        assert stream["pairs"] == support["pairs"]
        assert support["eligible_rows"] == expected
        decision = route(project(sample(["P"]), training_support=bool(expected)))
        assert (decision["action"] == "MATCHED_GENERATOR") == bool(expected)


def test_only_independently_grounded_training_pairs_supply_a_label():
    rows = training_rows()
    result = verified_training_pairs([rows])
    assert (result["eligible_rows"], result["eligible_families"]) == (2, 2)
    assert verified_training_pairs([[rows[0]]])["eligible_rows"] == 0

def test_frozen_generator_pairs_exclude_probes_and_bind_every_target_and_seed():
    import pytest
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.training_rows import matched_batch, matched_stream, pair_seeds
    from oeis_learn.generalization.progressive.prepared_support import matched_package
    from oeis_learn.pipeline_validation.progressive_learner import validate_input, compatibility
    from tests.helpers.progressive_fixture import task
    from tests.unit.test_progressive_training_policy import profile

    rows = training_rows()
    probes = [seal(dict(r, draw_identity=digest(["probe", i]))) for i, r in enumerate(rows)]
    shards = [rows + probes]
    layout = {"bins": ["familiar", "frontier"], "probes": {
        b: {"role": "training", "id": r["draw_identity"]} for b, r in zip(("familiar", "frontier"), probes)}}
    reg = digest("acceptance")
    stream = matched_stream(shards, layout, reg, 0)
    assert stream == matched_stream([list(reversed(shards[0]))], layout, reg, 0)
    assert stream["eligible_rows"] == 2 and set(stream["order"]) == {r["draw_identity"] for r in rows}
    assert all(p["negative_row"] in stream["order"] for p in stream["pairs"])
    assert matched_batch(stream, 1, 32) == (stream["order"] * 17)[1:33]
    assert pair_seeds(reg, 1)["training_families"] != stream["seeds"]["training_families"]
    for arm in ("ordinary", "first_divergence"):
        pkg = matched_package(shards, layout, profile(), stream, arm)
        t = task(arm=arm, seed=stream["seeds"]["initialization"], token_target=4_000_000,
                 initializer=dict(mode="random", artifact=None, compatibility=compatibility(pkg)))
        assert validate_input(pkg, t) == pkg
        imported = dict(t, payload=dict(t["payload"], initializer=dict(t["payload"]["initializer"],
                              mode="weights_only", artifact=None)))
        with pytest.raises(ValueError, match="strict matched generator"):
            validate_input(pkg, imported)
        with pytest.raises(ValueError, match="independently generated"):
            validate_input(pkg, dict(t, purpose="research"))
        bad = seal(dict(pkg, matched_generator=seal(dict(stream, order=stream["order"][::-1]))))
        changed = dict(t, payload=dict(t["payload"], initializer=dict(t["payload"]["initializer"], compatibility=compatibility(bad))))
        with pytest.raises(ValueError, match="training rows, order"):
            validate_input(bad, changed)
