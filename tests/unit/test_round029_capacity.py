from oeis_learn.pipeline_validation.progressive_pilot_capacity import (
    mount_demand,
    owned_allocations,
)
from oeis_learn.generalization.progressive.program_search_costing import price_wave


def test_coexistent_writes_and_separate_owners(tmp_path):
    p = tmp_path / "parent.pt"
    p.write_bytes(b"w" * 4096)
    import os

    os.link(p, tmp_path / "other-name.pt")
    allocation = owned_allocations([p, tmp_path / "other-name.pt"])
    assert allocation == {"bytes": p.stat().st_blocks * 512, "inodes": 1}
    result = mount_demand(
        available={"bytes": 128, "inodes": 9},
        reserve={"bytes": 16, "inodes": 3},
        pending=[{"bytes": 40, "inodes": 4}, {"bytes": 65, "inodes": 3}],
    )
    assert result["deficit_bytes"] == 0 and result["deficit_inodes"] == 1


def test_expected_tail_and_protected_followup():
    prices = {
        "cold": {"expected_us": 10, "tail_us": 13},
        "case": {"expected_us": 60, "tail_us": 61},
    }
    plan = {"common_probe": {"cold": 1, "case": 2}, "wave": {"case": 4}, "followup": {"case": 1}}
    r = price_wave(plan, prices, remaining_us=250, credited={"wave": {"case": 1}})
    assert r["common_probe"]["tail_us"] == 135
    assert r["wave"]["expected_us"] == 180 and r["wave"]["affordable_with_reserve"]
    assert price_wave(plan, {}, remaining_us=250)["wave"]["expected_us"] is None


def test_guard_projects_same_scope_and_serial_recovery_peaks():
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import learner_mount

    need = learner_mount({"free_bytes": 20_000_000_000, "free_inodes": 100}, floor=1024)
    assert need["required_bytes"] == 2 * 3 * 1_257_820_683
    assert need["deficit_bytes"] == 0 and need["deficit_inodes"] > 0
    assert all(g["basis"] == "provisional" and g["rehearsal"] for g in need["guards"])


def test_paid_initialization_open_drain_and_atomic_smaller_prefix():
    from oeis_learn.generalization.progressive.program_search_costing import affordable_prefix

    prices = {"cold": (12_000_000, 18_000_000), "search": (50_000_000, 60_133_643), "score": (1_000_000, 2_000_000)}
    prices = {k: dict(expected_us=a, tail_us=b) for k,(a,b) in prices.items()}
    batches = [
        {"id": "a", "operations": {"init": "cold", "a-search": "search", "a-score": "score"}},
        {"id": "b", "operations": {"init": "cold", "b-search": "search", "b-score": "score"}},
        {"id": "c", "operations": {"init": "cold", "c-search": "search", "c-score": "score"}},
    ]
    found = affordable_prefix(batches, prices, remaining_us=75_000_000, holdback_us=10_000_000,
                              paid={"init"}, opened={"a-search"})
    assert found["batches"] == ["a", "b"] and found["expected_us"] == 52_000_000
    assert found["tail_us"] == 64_133_643 and found["unsettled"] == ["a-search"]
    assert found["next"] == "c" and found["deficit_us"] == 61_267_286
    prices.pop("score")
    missing = affordable_prefix(batches, prices, remaining_us=75_000_000, holdback_us=10_000_000,
                                paid={"init"}, opened={"a-search"})
    assert missing["batches"] == [] and missing["missing"] == ["score"] and missing["deficit_us"] is None
