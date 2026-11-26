from oeis_learn.generalization.progressive.training_rows import first_divergence, verified_training_pairs
from oeis_learn.pipeline_validation.progressive_preparation import verify_draw


def test_first_divergence_must_share_real_context_and_exclude_other_positives():
    good, bad = [2, 3, 4], [2, 6, 8]
    assert first_divergence(good, bad, [good]) == dict(position=1, positive=3, negative=6)
    assert first_divergence(good, bad, [good, [2, 6, 9]]) is None
    assert first_divergence(good, good, [good]) is None


def test_only_independently_grounded_training_pairs_supply_a_label():
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
    result = verified_training_pairs([rows])
    assert (result["eligible_rows"], result["eligible_families"]) == (2, 2)
    assert verified_training_pairs([[rows[0]]])["eligible_rows"] == 0
