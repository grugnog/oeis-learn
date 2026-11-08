"""Real native/reference shards; CPU learner continuation acceptance added in T036."""

from oeis_learn.generalization.progressive.generator import default_policy
from oeis_learn.generalization.progressive.stream import load_shard
from oeis_learn.pipeline_validation.progressive_preparation import prepare_stream


def test_native_reference_shard_publishes_exact_independent_targets(tmp_path, fitting_client):
    p = default_policy(seed=13) | {"shard_rows": 2, "max_draws": 2}
    shards = prepare_stream(tmp_path, p, client=fitting_client)
    assert len(shards) == 1
    rows = load_shard(tmp_path, shards[0])
    assert len(rows) == 2
    assert all(r["native_verified"] and len(r["values"]) == 100 for r in rows)
    from oeis_learn.synthesis.reference import evaluate

    for row in rows:
        assert evaluate(row["source"], row["indices"]).values == tuple(row["values"])
