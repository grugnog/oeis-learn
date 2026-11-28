from copy import deepcopy

import pytest

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.program_search_costing import supported_velocity_wave


def inputs():
    origin = digest("synthetic fresh input")
    methods = ("00", "04", "05", "06")
    supply = dict(slots=3, ordinary=3, matching={"ordinary": dict(coverage=dict(full=True), eligible_slots=0)})
    fresh = dict(identity=origin, disposition="fresh_ordinary_only", supply=supply,
                 device=dict(methods=dict.fromkeys(methods)))
    rows = {m: dict(storage_us=dict(initial_save=1, first_save=2, second_save=4, resume_load=3, final_load=5))
            for m in methods}
    cycles = dict(origin=origin, receipt=digest("synthetic checkpoint"), methods=rows)
    operations = {"learner.one.search.parameter": 18, "learner.full.score.oeis": 6, "D.full.search.oeis": 2,
                  "ordinary_to_4000000_tokens": 1, "ordinary_matching_calibration": 1,
                  "trajectory_startup_checkpoint_io_and_drain": 1, "conventional_cpu_controls": 1}
    operations.update({f"optimizer_to_4M_recipe_0{i}": 1 for i in range(1, 7)})
    rates = {k: dict(expected_us=2, tail_us=3) for k in operations
             if k not in {"trajectory_startup_checkpoint_io_and_drain", "conventional_cpu_controls"}}
    return dict(wave=dict(operations=operations), unit_prices=rates, followup_reserve_us=7), fresh, cycles


def test_prices_only_supported_opportunities_and_keeps_remaining_work():
    prices, fresh, cycles = inputs()
    saved = deepcopy(prices)
    value = supported_velocity_wave(velocity_policy(), prices, fresh, cycles)
    assert value["methods"] == ["00", "04", "05", "06"] and set(value["omitted"]) == {"01", "02", "03"}
    ops = value["operations"]
    assert ops["learner.one.search.parameter"] == 9 and ops["learner.full.score.oeis"] == 3
    assert ops["D.full.search.oeis"] == 2 and "ordinary_matching_calibration" not in ops
    assert value["storage"]["00"] == dict(expected_us=10, tail_us=14)
    assert value["known_expected_us"] == (9 + 3 + 2 + 4) * 2 + 4 * 10
    assert set(value["missing"]) == {"conventional_cpu_controls", "trajectory_startup_extra_checkpoints_and_drain"}
    assert value["tail_us"] is None and value["followup_reserve_us"] == 7
    assert prices == saved


def test_partial_matching_or_a_different_paid_frame_cannot_identify_an_unsupported_arm():
    prices, fresh, cycles = inputs()
    for change in ("matching", "ceiling", "origin", "treatment"):
        f, c = deepcopy(fresh), deepcopy(cycles)
        if change == "matching":
            f["supply"]["matching"]["ordinary"]["coverage"]["full"] = False
        elif change == "ceiling":
            f["supply"]["ordinary"] -= 1
        elif change == "origin":
            c["origin"] = digest("other stream")
        else:
            del c["methods"]["04"]
        assert supported_velocity_wave(velocity_policy(), prices, f, c) is None


def test_launch_overheads_use_paid_cuts_and_price_each_cold_role_once():
    from oeis_learn.generalization.progressive.program_search_costing import trajectory_overheads

    hour = 3_600_000_000
    methods = ["00", "04", "05", "06"]
    timings = {m: dict(expected_us=hour + 11, tail_us=2 * hour + 11) for m in methods}
    rows = {m: dict(restore_us=25, first_cut_us=50, resumed_cut_us=35,
                    storage_us=dict(initial_save=1, first_save=2, second_save=4,
                                    resume_load=3, final_load=5),
                    checkpoint_bytes=1000, checkpoint_inodes=20) for m in methods}
    # The device charge encloses all 4 x (50 + 35 + 25) us measured cut calls.
    cycles = dict(methods=rows, device_charge_us=450)
    result = trajectory_overheads(timings, dict.fromkeys(methods, 11), cycles)
    m = result["methods"]["00"]
    assert m["outside_cut_us_per_role"] == 10
    assert m["fixed_us"] == 3 * (25 - 5) - 11 + 3 * 10
    assert m["hourly_saves"] == dict(expected=1, tail=2)
    # The baseline already has initial, 1M, 4M writes and one restore.
    assert m["extra_io_us"] == dict(expected_us=3 + 3, tail_us=5 + 2 * 4)
    assert result["known_expected_us"] == 4 * (m["fixed_us"] + 6)
    assert result["known_tail_us"] == 4 * (m["fixed_us"] + 13)
    assert "update_journal_and_stream_replay" in result["missing"]
    with pytest.raises(ValueError):
        trajectory_overheads(timings, dict.fromkeys(methods, 11), dict(cycles, device_charge_us=439))


def test_launch_capacity_prices_all_retained_recipes_and_smaller_prefix(tmp_path):
    from oeis_learn.pipeline_validation.progressive_velocity_launch import checkpoint_space

    rows = {m: dict(checkpoint_bytes=1000, checkpoint_inodes=20) for m in ("00", "04", "05", "06")}
    trajectory = dict(methods={m: {"hourly_saves": {"tail": 1}} for m in rows})
    result = checkpoint_space(tmp_path, dict(methods=rows, candidate="a"*40, receipt=digest("cuts")),
                              trajectory, floor=55)
    wave, prefix = result["wave"][0], result["first_ordinary_cut"][0]
    assert wave["required_bytes"] == 2 * (4 * 2000 + 1000)
    assert prefix["required_bytes"] == 2 * (1000 + 1000)
    assert wave["required_inodes"] == 2 * (4 * 40 + 20)
    assert prefix["reserved_bytes"] == wave["reserved_bytes"] == 55
    assert not list(tmp_path.iterdir())


def test_publication_proxy_prices_the_same_write_read_and_one_ledger_replay():
    from oeis_learn.pipeline_validation.progressive_velocity_launch import publication_proxy

    encodings = dict(compressible=dict(cold_us=11, warm_us=3), varied=dict(cold_us=23, warm_us=5))
    release = dict(corpora=encodings, journal=dict(cold_us=7, warm_us=1), receipt=digest("release"))
    price = publication_proxy(release)
    assert price["expected_us"] == 28 and price["tail_us"] == 35
    assert price["receipt"] == release["receipt"]
    assert publication_proxy(None) is None


def test_wave_storage_prices_resident_blocks_by_profile_and_keeps_gaps_visible(tmp_path):
    from oeis_learn.pipeline_validation.progressive_velocity_launch import wave_space

    wave = dict(methods=["00", "04", "05", "06"], operations={})
    for family, blocks, cases in (("ordinary.one", 3, 9), ("learner.one", 9, 9),
                                  ("ordinary.full", 1, 10), ("learner.full", 3, 10),
                                  ("D.full", 1, 10)):
        for phase in ("score", "search"):
            wave["operations"][family + "." + phase + ".startup"] = blocks
            wave["operations"][family + "." + phase + ".oeis"] = blocks * cases
    cycles = dict(candidate="a"*40, receipt=digest("cut"),
                  methods={m: dict(checkpoint_bytes=20, checkpoint_inodes=5) for m in wave["methods"]})
    trajectory = dict(methods={m: dict(hourly_saves=dict(tail=0)) for m in wave["methods"]})
    sample = dict(
        cases=8,
        checkpoint=cycles["receipt"],
        runtime=dict(bytes=5, inodes=2),
        roles={
            k: dict(bytes=10, inodes=2)
            for k in (
                "ordinary-common",
                "ordinary-full",
                "D-full",
                "template",
                "polynomial",
                "linear_recurrence",
            )
        },
    )
    release = dict(
        candidate="a" * 40,
        receipt=digest("release"),
        peak_allocated_bytes=300,
        peak_inodes=60,
        floor_bytes=45,
        corpora={"varied": dict(expanded_bytes=1000, compressed_bytes=900, files=100)},
        journal=dict(events=290000, bytes=20000),
    )
    registered = [dict(device=tmp_path.stat().st_dev, reserved_bytes=70, reserved_inodes=200)]
    result = wave_space(
        tmp_path,
        wave,
        cycles,
        trajectory,
        sample,
        release,
        floor=55,
        updates=dict.fromkeys(wave["methods"], 75),
        registered=registered,
    )
    # 34 neural resident chunks + 6 conventional chunks + 9 paired sample blocks.
    assert result["chunks"] == {
        "ordinary-common": 28,
        "ordinary-full": 12,
        "D-full": 3,
        "template": 2,
        "polynomial": 2,
        "linear_recurrence": 2,
    }
    assert not result["missing"]
    assert result["wave"][0]["required_bytes"] > result["first_ordinary_cut"][0]["required_bytes"]
    # The very first initialization starts in an empty learner directory. Its pending
    # blobs become that checkpoint, whereas the next cut must retain the original.
    initial = result["ordinary_initialization"][0]
    owners = {o["owner"] for o in initial["owners"]}
    assert owners == {"learner_00", "worker_staging", "first_account"}
    assert initial["required_bytes"] + 2 * 20 == result["first_ordinary_cut"][0]["required_bytes"]
    assert initial["reserved_inodes"] == 200
    assert result["wave"][0]["reserved_bytes"] == 70
    assert result["first_ordinary_cut"][0]["reserved_inodes"] == 200
    del sample["roles"]["D-full"]
    result = wave_space(
        tmp_path,
        wave,
        cycles,
        trajectory,
        sample,
        release,
        floor=55,
        updates=dict.fromkeys(wave["methods"], 75),
        registered=registered,
    )
    assert result["missing"] == ["D-full"] and result["fits_with_reserve"] is None
    assert not list(tmp_path.iterdir())


def test_private_replay_and_public_scale_use_measured_resident_work_and_separate_units():
    from oeis_learn.pipeline_validation.progressive_velocity_launch import verification_forecast

    wave = dict(
        missing=[], chunks={"common": 3, "full": 2}, projected_private=dict(bytes=210, inodes=30)
    )
    sample = dict(
        private_read_us={"common": dict(search=3, score=7), "full": dict(search=8, score=4)}
    )
    release = dict(
        receipt=digest("historical"),
        corpora=dict(
            varied=dict(expanded_bytes=100, files=5, cold_us=13, warm_us=7),
            compressible=dict(cold_us=6, warm_us=4),
        ),
        journal=dict(events=10, cold_us=5),
    )
    result = verification_forecast(wave, sample, release, 31, {"00": 3, "04": 2})
    assert result["historical_corpus_passes"] == 6
    assert result["historical_journal_passes"] == 3
    assert result["expected_us"] == 31 + 15 + 3 * 10 + 2 * 12 + 6 * 15
    assert result["tail_us"] == 31 + 15 + 5 * (8 + 7) + 6 * 20
    assert verification_forecast(dict(wave, missing=["D-full"]), sample, release, 31, {}) is None
