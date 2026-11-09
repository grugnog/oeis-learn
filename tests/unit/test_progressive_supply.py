import pytest
from oeis_learn.generalization.progressive.supply import plan


def package(cadence=8):
    return dict(
        arm="refreshed",
        shards=[[dict(tokens=[1] * 10)] for _ in range(64)],
        update_policy=dict(effective_batch=16),
        refresh_updates=cadence,
    )


def test_default_25m_support_shortfall_is_a_preparation_deficit():
    with pytest.raises(ValueError, match="preparation deficit"):
        plan(package(), 25000000)


def test_prospective_cadence_makes_same_finite_inventory_feasible():
    assert plan(package(4096), 25000000)["required_shards"] == 39
    p = plan(package(), 81920)
    assert p["maximum_updates"] == 512 and p["required_shards"] == 64
    with pytest.raises(ValueError):
        plan(package(), 81921)


def test_final_frontier_shard_boundary_restores_without_reuse():
    from oeis_learn.generalization.progressive.supply import shard_index
    assert shard_index('refreshed','frontier',511,8,64)==63
    assert shard_index('refreshed','familiar',512,8,64)==0
    import pytest
    with pytest.raises(TimeoutError,match='no static fallback'):
        shard_index('refreshed','frontier',512,8,64)


@pytest.mark.parametrize("arm", ["replay", "continued_synthetic", "library", "no_library"])
def test_paired_mixture_arms_preserve_refreshed_synthetic_cursor(arm):
    from oeis_learn.generalization.progressive.supply import shard_index

    assert shard_index(arm, "frontier", 8, 8, 2) == 1
    assert shard_index(arm, "familiar", 16, 8, 2) == 0
    with pytest.raises(TimeoutError, match="no static fallback"):
        shard_index(arm, "frontier", 16, 8, 2)


def test_unknown_arm_cannot_select_support():
    from oeis_learn.generalization.progressive.supply import shard_index

    with pytest.raises(ValueError, match="support cursor"):
        shard_index("unknown", "familiar", 0, 8, 2)
