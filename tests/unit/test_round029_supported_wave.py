from copy import deepcopy

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
