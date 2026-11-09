"""Selected real model sizes must fit the same finite installed acceptance path."""

import pytest


@pytest.mark.parametrize("shape", ["reference", "larger"])
def test_selected_profile_commits_updates_and_restores_successor(tmp_path, monkeypatch, shape):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation import progressive_device_acceptance as device
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    monkeypatch.setattr(
        device,
        "selected_profile",
        lambda p: resolve_profile(
            p["base_recipe"], shape=shape, seed=p["model"]["seed"], dropout=p["model"]["dropout"]
        ),
    )
    campaign, tasks, channel, charge = fixture(tmp_path, interrupt=False)
    first, one = invoke_cpu(tmp_path, campaign, tasks[0], channel, ordinal=0)
    assert first["mechanical_status"] == "COMPLETE", first
    assert first["metrics"]["useful_tokens"] == 3 * charge
    second, two = invoke_cpu(tmp_path, campaign, tasks[1], channel, ordinal=1)
    assert second["mechanical_status"] == "COMPLETE", second
    assert second["metrics"]["useful_tokens"] == 5 * charge
    initial = load(one / "role/checkpoint.json")["payload"]
    final = load(two / "role/checkpoint.json")["payload"]
    assert initial["model_update"] == 3 and final["model_update"] == 5
    assert initial["compatibility"] == final["compatibility"]


def test_profile_storage_deficit_fails_before_learner_publication(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_acceptance_limits as limits
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture

    def unavailable(*args, **kwargs):
        raise TimeoutError("free storage byte/inode reserve exhausted")

    monkeypatch.setattr(limits.StorageGuard, "check", unavailable)
    with pytest.raises(TimeoutError, match="storage"):
        fixture(tmp_path, interrupt=False)
    assert not (tmp_path / "campaign.json").exists()
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("shape", ["reference", "larger"])
def test_oeis_child_tasks_reserve_selected_profile_and_simultaneous_stores(
    tmp_path, monkeypatch, shape
):
    from oeis_learn.generalization.progressive.campaign import validate_campaign
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation import progressive_device_acceptance as device
    from tests.helpers.progressive_oeis_fixture import fixture

    monkeypatch.setattr(
        device,
        "selected_profile",
        lambda p: resolve_profile(
            p["base_recipe"], shape=shape, seed=p["model"]["seed"], dropout=p["model"]["dropout"]
        ),
    )
    campaign, _, _, _ = fixture(tmp_path)
    validate_campaign(campaign)
    storage = campaign["payload"]["storage"]
    assert storage["checkpoint_bytes"] > 64 << 20
    assert storage["task_store_bytes"] == sum(
        t["caps"]["disk_bytes"] for t in campaign["payload"]["tasks"]
    )
    assert all(
        t["caps"]["disk_bytes"] >= 2 * storage["checkpoint_bytes"]
        for t in campaign["payload"]["tasks"]
    )
    assert campaign["payload"]["caps"]["disk_bytes"] >= storage["required_bytes"]
