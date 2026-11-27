import json

import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.generalization.progressive.velocity_metrics import (
    sampling_interval,
    contrast,
    rates,
    factorial,
    affordable_look,
)
from oeis_learn.generalization.progressive.program_search_costing import occupied_time


def obs(case, status, label):
    receipt = seal(dict(version="round029-case-outcome/v1", case=str(case),
                        outcome_ref=digest(label), choice_ref=digest(("choice", label)),
                        scorer_ref=digest("reference"), truth_ref=digest(("truth", str(case))),
                        status=status))
    return dict(status=status, receipt=receipt, complete=status != "unresolved", closed=True)


def cell(i, a, b):
    return dict(case=str(i), family=str(i), stratum="oeis",
                observations={"a": obs(i, a, f"a{i}"), "b": obs(i, b, f"b{i}")})


def frozen(families):
    members = [dict(task=str(i), group=str(f), stratum="oeis") for i, f in enumerate(families)]
    return {k: velocity_look(registration=digest("study"), method=m, seed=1, look=4_000_000,
                             profile="full", development=members, retention=[])
            for k, m in (("a", "00"), ("b", "01"))}


def test_turnover_bounds_and_cost_rates():
    rows = [cell(i, "solve" if i < 4 else "miss", "miss" if i < 4 else "solve") for i in range(8)]
    result = contrast(rows, {"b": 1, "a": -1}, stratum="oeis", looks=frozen(range(8)))
    assert result["observed_pp"] == 0 and result["gains"] == result["losses"] == 4
    assert result["first_solve_families"] == result["last_solve_families"] == 4
    assert rates(6, 3_000_000, 5_400_000_000) == {"pp_per_million": 2, "pp_per_hour": 4}
    assert rates(1, 100, 0) == {"pp_per_million": 10000, "pp_per_hour": None}
    rows = [
        cell(
            i, "solve" if i < 3 else "miss", "solve" if i < 2 else "unresolved" if i < 4 else "miss"
        )
        for i in range(10)
    ]
    r = contrast(rows, {"a": 1, "b": -1}, stratum="oeis", looks=frozen(range(10)))
    assert (r["observed_pp"], r["lower_pp"], r["upper_pp"]) == (10, -10, 10)
    rows[0]["observations"] = dict.fromkeys(("a", "b"), obs(0, "unresolved", "shared"))
    assert contrast(rows[:1], {"a": 1, "b": -1}, stratum="oeis", looks=frozen([0]))["upper_pp"] == 0


def test_family_weights_interactions_and_overlap():
    rows = [cell(i, "miss", "solve" if i < 3 else "miss") for i in range(4)]
    for row in rows[:3]:
        row["family"] = "large"
    assert contrast(rows, {"b": 1, "a": -1}, stratum="oeis", looks=frozen(["large"] * 3 + [3]))["observed_pp"] == 50
    f = factorial(dict(zip(("00", "01", "02", "03", "04", "05", "06"), (1, 2, 3, 6, 5, 7, 12))))
    assert f["composition_matching"]["interaction"] == 2 and f["batch_decay"]["interaction"] == 1
    looks = [dict(useful_tokens=k, verified=True) for k in (0, 101, 402)]
    assert affordable_look(looks, "useful_tokens", 300, overlap=(0, 402)) == looks[1]
    assert affordable_look(looks, "useful_tokens", 403, overlap=(0, 402)) is None


def test_shared_wall_union_and_unknown_boot_overlap():
    intervals = [
        dict(receipt="a", host="x", boot="1", start_us=0, end_us=6, beneficiaries=["00"]),
        dict(receipt="b", host="x", boot="1", start_us=3, end_us=9, beneficiaries=["01"]),
    ]
    r = occupied_time([*intervals, intervals[0]])
    assert r["union_us"] == 9 and r["attributed_us"] == {"00": 4.5, "01": 4.5}
    assert occupied_time([intervals[0], dict(intervals[1], boot="2")])["union_us"] is None


def test_both_reducers_require_the_registered_denominator():
    rows = [cell(0, "miss", "solve"), cell(1, "miss", "miss")]
    for row in rows:
        row["family"] = "F"
    scope = dict(stratum="oeis", looks=frozen(["F", "F"]))
    coeff = {"a": -1, "b": 1}
    assert contrast(rows, coeff, **scope)["observed_pp"] == 50
    assert sampling_interval(rows, coeff, seed=1, draws=5, **scope)["lower"] == 50
    for bad in [rows[:1], rows + [cell(2, "solve", "solve")],
                [rows[0], dict(rows[1], family="other")], [rows[0], rows[0]]]:
        with pytest.raises(ValueError, match="roster"):
            contrast(bad, coeff, **scope)
        with pytest.raises(ValueError, match="roster"):
            sampling_interval(bad, coeff, seed=1, **scope)
    altered = dict(scope["looks"]["a"], members=[])
    with pytest.raises(ValueError):
        contrast(rows, coeff, stratum="oeis", looks=dict(scope["looks"], a=altered))


def test_unresolved_identity_binds_case_choice_truth_and_scorer():
    rows = [cell(0, "miss", "unresolved"), cell(1, "unresolved", "miss")]
    opts = dict(stratum="oeis", looks=frozen([0, 1]))
    coeff = {"a": -1, "b": 1}
    result = contrast(rows, coeff, **opts)
    assert (result["lower_pp"], result["upper_pp"]) == (-50, 50)
    left = rows[0]["observations"]["b"]
    rows[1]["observations"]["a"] = left
    with pytest.raises(ValueError, match="receipt"):
        contrast(rows, coeff, **opts)
    rows = [cell(0, "unresolved", "unresolved")]
    for field in ("choice_ref", "truth_ref", "scorer_ref"):
        rows[0]["observations"]["a"] = left
        rows[0]["observations"]["b"] = dict(left, receipt=seal(dict(left["receipt"], **{field: digest("other")})))
        with pytest.raises(ValueError, match="receipt"):
            contrast(rows, coeff, stratum="oeis", looks=frozen([0]))
    rows[0]["observations"]["b"] = left
    result = contrast(rows, coeff, stratum="oeis", looks=frozen([0]))
    assert result["lower_pp"] == result["upper_pp"] == 0


def test_joint_family_cells_and_opaque_projection():
    from oeis_learn.generalization.progressive.velocity_report import public_family_cells

    coeff = {"b": 1, "a": -1}
    panels = frozen(["private-family-A", "private-family-B"])
    rows = [cell(0, "miss", "solve"), cell(1, "solve", "miss")]
    for row, f in zip(rows, ("private-family-A", "private-family-B")):
        row["family"] = f
    rows[0]["observations"]["b"]["closed"] = False
    original = contrast(rows, coeff, stratum="oeis", looks=panels)
    cells = original["families"]
    assert [f["frozen_size"] for f in cells] == [1, 1]
    first = cells[0]["patterns"][0]
    assert first["count"] == 1
    assert [(o["status"], o["complete"], o["closed"], o["equality"]) for o in first["outcomes"]] == [
        ("miss", True, True, 0), ("solve", True, False, 1)]
    for row in rows:
        a, b = row["observations"].values()
        row["observations"] = {"a": b, "b": a}
    swapped = contrast(rows, coeff, stratum="oeis", looks=panels)
    assert original["observed_pp"] == swapped["observed_pp"] == 0
    assert swapped["families"][0]["patterns"][0]["outcomes"][0]["status"] == "solve"
    public = public_family_cells(original, key=b"one-study-secret")
    assert [(c["method"], c["seed"], c["look"]) for c in public["coordinates"]] == [
        ("00", 1, 4_000_000), ("01", 1, 4_000_000)]
    assert sorted(json.dumps(f["patterns"]) for f in public["families"]) == sorted(
        json.dumps(f["patterns"]) for f in cells)
    assert "private-family" not in json.dumps(public) and "truth_ref" not in json.dumps(public)
    assert public == public_family_cells(original, key=b"one-study-secret")
    assert public != public_family_cells(original, key=b"other-study-secret")
    shared = obs(0, "unresolved", "common-zero")
    rows[0]["observations"] = {"a": shared, "b": shared}
    result = contrast(rows, coeff, stratum="oeis", looks=panels)
    assert [o["equality"] for o in result["families"][0]["patterns"][0]["outcomes"]] == [0, 0]


def test_crossing_clocks_use_latest_verified_look_and_measured_overlap():
    def look(tokens, time, level, verified=True):
        return dict(useful_tokens=tokens, charged_us=time, level=level, verified=verified)
    zero = look(0, 0, 0)
    first = [zero, look(1_000_000, 0, 2), look(4_000_000, 4, 9)]
    second = [zero, look(1_000_000, 2, 5), look(4_000_000, 6, 7)]
    for order in (first, first[::-1]):
        selected = affordable_look([*order, look(8_000_000, 0, 12, False)], "charged_us", 0, overlap=(0, 4))
        assert selected == first[1]
    for axis, budget, overlap, levels in [("useful_tokens", 1_000_000, (0, 4_000_000), (2, 5)),
                                           ("charged_us", 4, (0, 4), (9, 5))]:
        assert tuple(affordable_look(arm, axis, budget, overlap=overlap)["level"]
                     for arm in (first, second)) == levels
        assert affordable_look(first, axis, overlap[1] + 1, overlap=overlap) is None
    assert rates(2, 1_000_000, 0)["pp_per_hour"] is None


def test_missing_structural_stratum_preserves_cells_and_profiles_cannot_be_substituted():
    members = [dict(task="0", group="F", stratum="composition")]
    def look(method, profile):
        return velocity_look(registration=digest("study"), method=method, seed=1, look=4_000_000,
                             profile=profile, development=members, retention=[])
    views = {"a": look("00", "full"), "b": look("01", "full")}
    rows = [dict(cell(0, "miss", "solve"), stratum="composition", family="F")]
    coeff = {"a": -1, "b": 1}
    report = contrast(rows, coeff, stratum="structural", looks=views)
    assert report["observed_pp"] is None and report["reason"] and report["denominator"] == 1
    assert report["families"][0]["frozen_size"] == 1
    sampled = sampling_interval(rows, coeff, stratum="structural", looks=views, seed=1, draws=5)
    assert sampled["supported_draws"] == 0 and sampled["lower"] is None
    with pytest.raises(ValueError, match="profile"):
        contrast(rows, coeff, stratum="composition", looks=dict(views, a=look("00", "common")))


def test_original_and_incremental_work_keep_failed_debits_reuse_and_unknown_clocks():
    from oeis_learn.generalization.progressive.program_search_costing import workstation_accounting

    def work(receipt, role, start, end, **changes):
        return dict(
            dict(
                receipt=receipt,
                origin="incremental",
                state="complete",
                role=role,
                host="host",
                boot="boot",
                start_us=start,
                end_us=end,
                beneficiaries=["00", "02"],
                charged_us=end - start,
                lease_us=0,
                covered_us=0,
                busy_us=0,
            ),
            **changes,
        )

    old = work("old", "training", 1, 21, origin="historical")
    reused = dict(old, origin="incremental", state="reused", charged_us=0)
    failed = work("retry", "prepare", 3, 7, state="failed", beneficiaries=["00"])
    fresh = work("device", "measure", 5, 11, lease_us=5, covered_us=3, busy_us=2, worker_cpu_us=8)
    records = [old, reused, failed, fresh, fresh]
    launch = dict(host="host", boot="boot", at_us=2)
    decision = dict(launch, at_us=15)
    account = workstation_accounting(records, launch=launch, decision=decision)
    assert account["historical"]["ledger_us"] == 20
    marginal = account["incremental"]
    assert marginal["ledger_us"] == 10 and marginal["occupied"]["union_us"] == 8
    assert marginal["occupied"]["attributed_us"] == {"00": 5, "02": 3}
    assert marginal["dispositions"] == {"reused": 1, "failed": 1, "complete": 1}
    assert marginal["roles"] == {"prepare": 4, "measure": 6}
    assert marginal["resources"]["worker_cpu_us"] == {"observed": 8, "missing_operations": 1}
    assert marginal["resources"]["busy_us"] == {"observed": 2, "missing_operations": 0}
    assert account["launch_to_decision_us"] == 13
    open_work = dict(failed, receipt="open", state="open", end_us=None, charged_us=15)
    unknown = workstation_accounting(
        [*records, open_work], launch=launch, decision=dict(decision, boot="other")
    )
    marginal = unknown["incremental"]
    assert marginal["ledger_us"] == 25 and marginal["occupied"]["lower_us"] == 8
    assert marginal["occupied"]["upper_us"] is None and marginal["occupied"]["union_us"] is None
    assert unknown["launch_to_decision_us"] is None
    with pytest.raises(ValueError, match="both cost scopes"):
        workstation_accounting([old, dict(old, origin="incremental")])
    with pytest.raises(ValueError, match="no new charge"):
        workstation_accounting([dict(reused, charged_us=1)])
