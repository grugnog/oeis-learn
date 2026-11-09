from oeis_learn.generalization.progressive.diversity import summarize
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.generator import structure_id


def row(source, values):
    return dict(
        draw_identity=digest(source),
        source=source,
        structure=structure_id(source),
        values=values,
        operations=["add"],
    )


def test_coefficients_do_not_manufacture_structures_and_consumption_is_separate():
    a = row("(program (return (add $index (int 1))))", ["1", "2"])
    b = row("(program (return (add $index (int 2))))", ["2", "3"])
    c = row("(program (return (mul $index $index)))", ["0", "1"])
    r = summarize([a, b, c], [a["draw_identity"], a["draw_identity"]])
    assert r["available"]["numeric_variants"] == 3
    assert r["available"]["canonical_structures"] == 2
    assert r["available"]["finite_behavior_clusters"] == 3
    assert r["consumed"]["examples"] == 2 and r["unique_consumed"] == 1
    assert r["consumed"]["canonical_structures"] == 1
    assert r["available"]["compositions"] == 2


def test_preparation_cost_uses_original_producers_not_guessed_elapsed(tmp_path):
    import pytest
    from oeis_learn.generalization.progressive.contracts import atomic_write, seal, reference
    from oeis_learn.pipeline_validation.progressive_training import preparation_cost

    shard = dict(identity=digest("shard"))
    atomic_write(
        tmp_path / "cost.json",
        seal(
            dict(
                version="progressive-preparation-cost/v1",
                shards=[shard["identity"]],
                elapsed_us=123,
                scope="observed producer invocation",
            )
        ),
    )
    ref = reference(tmp_path, "cost.json")
    assert preparation_cost(tmp_path, [shard], [ref]) == 123
    assert preparation_cost(tmp_path, [shard], []) is None
    with pytest.raises(ValueError):
        preparation_cost(tmp_path, [dict(identity=digest("other"))], [ref])
    with pytest.raises(ValueError):
        preparation_cost(tmp_path, [shard], [ref, ref])


def test_completed_resume_receipt_does_not_invent_lost_preparation_time(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write,seal,reference
    from oeis_learn.pipeline_validation.progressive_training import preparation_cost
    shard=dict(identity=digest('interrupted-shard'),payload={'preparation_elapsed_us':None})
    atomic_write(tmp_path/'completed.json',seal(dict(version='progressive-preparation-cost/v1',shards=[shard['identity']],elapsed_us=123,scope='observed completed invocation only')))
    assert preparation_cost(tmp_path,[shard],[reference(tmp_path,'completed.json')]) is None
