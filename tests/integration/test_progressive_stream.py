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


def _diagnostic_shards(root):
    from oeis_learn.generalization.progressive.stream import StreamStore

    p = default_policy(seed=13) | {"shard_rows": 16, "max_draws": 32}
    store = StreamStore(root, p)
    import pytest

    with pytest.raises(InterruptedError):
        store.step(fault="after_reserve")
    shards = []
    for index in range(2):
        for _ in range(16):
            store.step()
        if index == 0:
            with pytest.raises(InterruptedError):
                store.seal_shard(fault="after_shard_write")
            store = StreamStore(root, p)
            shards.append(store.shards()[0])
        else:
            shards.append(store.seal_shard())
    return shards


def test_real_cpu_updates_cross_shards_and_restore_spent_pending_work(tmp_path):
    import pytest

    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_training import acceptance_pilot

    shards = _diagnostic_shards(tmp_path / "data")
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    root = tmp_path / "data"
    with pytest.raises(InterruptedError):
        acceptance_pilot(
            root,
            "run",
            shards,
            recipe,
            arm="refreshed",
            seed=2,
            updates=2,
            diagnostic_shards=True,
            fault="after_reserve",
        )
    resumed = acceptance_pilot(
        root,
        "run",
        shards,
        recipe,
        arm="refreshed",
        seed=2,
        updates=2,
        diagnostic_shards=True,
        resume=True,
    )
    uninterrupted = acceptance_pilot(
        root, "full", shards, recipe, arm="refreshed", seed=2, updates=2, diagnostic_shards=True
    )
    assert resumed["final_weights"] == uninterrupted["final_weights"]
    assert resumed["controller"]["cursor"] == uninterrupted["controller"]["cursor"] == 2
    assert resumed["charged_tokens"] > uninterrupted["charged_tokens"]
    assert resumed["discarded_tokens"] > 0
    assert resumed["useful_tokens"] == uninterrupted["charged_tokens"]
    assert resumed["history"] == uninterrupted["history"]
    assert resumed['diversity']==uninterrupted['diversity']
    assert resumed['preparation_elapsed_us'] is None
    assert resumed["qualified"] is False and resumed["kind"] == "DIAGNOSTIC_ONLY"
    assert resumed["initial_weights"] != resumed["final_weights"]


def test_matched_static_refresh_have_same_initial_model_and_support(tmp_path):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_training import acceptance_pilot

    shards = _diagnostic_shards(tmp_path / "data")
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    values = [
        acceptance_pilot(
            tmp_path / "data",
            arm,
            shards,
            recipe,
            arm=arm,
            seed=2,
            updates=2,
            diagnostic_shards=True,
        )
        for arm in ("static", "refreshed")
    ]
    assert values[0]["initial_weights"] == values[1]["initial_weights"]
    assert values[0]["initial_shard"] == values[1]["initial_shard"]
    assert values[0]["updates"] == values[1]["updates"] == 2
    # The shared frozen seed selects frontier, so only refreshed support changes
    # at the declared one-update diagnostic boundary.
    assert values[0]["history"]["2"]["shard"] == 0
    assert values[1]["history"]["2"]["shard"] == 1
    from oeis_learn.pipeline_validation.progressive_evidence import diversity_report

    report = diversity_report(*values)
    assert report['arms']['static']['diversity']['consumed']['canonical_structures']<=report['arms']['refreshed']['diversity']['available']['canonical_structures']
    assert report['arms']['static']['diversity']!=report['arms']['refreshed']['diversity']
    assert report["next_action"] == "RETAIN_BASELINE"
    assert report["scientific_status"] == "UNMEASURED"
    assert report["training_loss_is_not_capability"] is True


def test_native_shards_feed_actual_canonical_cpu_update(tmp_path, fitting_client):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_training import acceptance_pilot

    p = default_policy(seed=13) | {"shard_rows": 16, "max_draws": 16}
    shards,cost = prepare_stream(tmp_path, p, client=fitting_client,return_receipt=True)
    result = acceptance_pilot(
        tmp_path,
        "run",
        shards,
        resolve({"version": "bigint-bootstrap/v1"}),
        arm="static",
        seed=2,
        updates=1,
        preparation_receipts=[cost],
    )
    assert result['preparation_elapsed_us']>0
    assert result['diversity']['consumed']['examples']==1
    assert result['accepted_rows']==16
    assert result["updates"] == 1
    assert result["initial_weights"] != result["final_weights"]
    assert result["charged_tokens"] > 0 and result["discarded_tokens"] == 0


def test_full_update_overshoot_and_binding_execution_ceiling(tmp_path):
    import pytest

    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_training import acceptance_pilot

    shards = _diagnostic_shards(tmp_path / "data")
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    kwargs = {"arm": "static", "seed": 2, "updates": 1, "diagnostic_shards": True}
    full = acceptance_pilot(tmp_path / "data", "full", shards, recipe, target_tokens=1, **kwargs)
    assert full["updates"] == 1
    assert full["full_update_overshoot"] == full["charged_tokens"] - 1 > 0
    stopped = acceptance_pilot(tmp_path / "data", "stopped", shards, recipe, token_cap=1, **kwargs)
    assert stopped["updates"] == 0 and stopped["charged_tokens"] == 0
    assert stopped["useful_tokens"] == 0 and stopped["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="terminal token ceiling"):
        acceptance_pilot(
            tmp_path / "data", "stopped", shards, recipe, token_cap=1, resume=True, **kwargs
        )


def test_real_training_probe_feedback_and_refresh_at_eight_update_boundary(tmp_path):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.progressive_training import acceptance_pilot

    shards = _diagnostic_shards(tmp_path / "data")
    result = acceptance_pilot(
        tmp_path / "data",
        "run",
        shards,
        resolve({"version": "bigint-bootstrap/v1"}),
        arm="refreshed",
        seed=2,
        updates=9,
        refresh_updates=8,
        diagnostic_shards=True,
        seconds=120,
    )
    assert result["updates"] == 9
    assert len(result["controller"]["rewards"]) == 1
    assert result["controller"]["kernel"]["valid_updates"] == 1
    assert result["history"]["8"]["shard"] == 0
    assert result["history"]["9"]["shard"] == 1
