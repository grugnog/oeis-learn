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


def test_distinct_mount_peaks_count_shared_inodes_once_and_keep_owner_limits_separate(tmp_path):
    import os
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_pilot_capacity import velocity_mounts

    state = tmp_path / "state"
    public = tmp_path / "public"
    state.mkdir()
    public.mkdir()
    existing = state / "cut.pt"
    existing.write_bytes(b"x" * 4096)
    alias = state / "checkpoint-alias.pt"
    os.link(existing, alias)
    size = existing.stat().st_blocks * 512

    def owner(name, path, retained, size, nodes):
        return dict(
            owner=name,
            scope=name + " simultaneous files",
            path=path,
            retained=retained,
            observed=dict(bytes=size, inodes=nodes),
            projected=dict(bytes=size, inodes=nodes),
            revision="a" * 40,
            harm="publication could exhaust the reserve",
            exhaustion="shrink the batch",
            rehearsal="T060 peak coexistence",
        )

    writers = [
        owner(
            "checkpoint", "state", [reference(tmp_path, p) for p in (existing, alias)], size * 2, 2
        ),
        owner("journal", "state", [], 900, 3),
        owner("temporary_archive", "public", [], 1000, 3),
        owner("published_archive", "public", [], 1000, 3),
    ]

    def facts(path):
        return (1, 100_000, 50) if path == state else (2, 20_000, 13)

    views = velocity_mounts(
        tmp_path,
        writers,
        [dict(path="state", bytes=9, inodes=2), dict(path="public", bytes=3, inodes=2)],
        measure=facts,
    )
    assert views[0]["retained_bytes"] == size and views[0]["retained_inodes"] == 1
    assert views[0]["required_bytes"] == 3 * size + 1800 and views[0]["required_inodes"] == 9
    assert views[1]["required_bytes"] == 4000 and views[1]["deficit_inodes"] == 1
    assert views[1]["deficit_bytes"] == 0 and views[0]["owners"][1]["owner"] == "journal"


def test_original_prices_count_retries_but_keep_profiles_and_missing_clocks_separate():
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_costing import original_probe_costs

    cuts = [
        {"method": k, "look": 4_000_000, "training_to_look_us": n}
        for k, n in (("00", 30), ("first_divergence", 31), ("D", None))
    ]
    looks = [dict(look=4_000_000, attempts=8, denominator=2)]
    paid = [
        seal(dict(block=k, role="search", charged_us=n, lease_us=n - 1))
        for k, n in (
            ("ordinary-4000000", 10),
            ("ordinary-4000000-resume-1", 7),
            ("incumbent-4000000", 20),
            ("ordinary-1000000", 100),
        )
    ]
    result = original_probe_costs(cuts, looks, paid)
    assert result[0]["roles"]["search"] == dict(
        charged_us=17, lease_us=15, average_per_case_us=9, reason=None
    )
    assert (
        result[0]["roles"]["score"]["charged_us"] is None
        and result[2]["roles"]["search"]["charged_us"] == 20
    )
