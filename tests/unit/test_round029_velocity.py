import pytest
from oeis_learn.generalization.progressive.velocity_metrics import contrast, rates, factorial, affordable_look
from oeis_learn.generalization.progressive.program_search_costing import occupied_time


def obs(status, receipt):
    return dict(status=status, receipt=receipt, complete=status!="unresolved", closed=True)


def cell(i, a, b):
    return dict(case=str(i), family=str(i), stratum="oeis",
                observations={"a":obs(a, f"a{i}"), "b":obs(b, f"b{i}")})


def test_turnover_bounds_and_cost_rates():
    rows = [cell(i, "solve" if i<4 else "miss", "miss" if i<4 else "solve") for i in range(8)]
    result = contrast(rows, {"b":1, "a":-1}, stratum="oeis")
    assert result["observed_pp"] == 0 and result["gains"] == result["losses"] == 4
    assert result["first_solve_families"] == result["last_solve_families"] == 4
    assert rates(6, 3_000_000, 5_400_000_000) == {"pp_per_million":2, "pp_per_hour":4}
    assert rates(1, 100, 0) == {"pp_per_million":10000, "pp_per_hour":None}
    rows = [cell(i, "solve" if i<3 else "miss", "solve" if i<2 else "unresolved" if i<4 else "miss")
            for i in range(10)]
    r = contrast(rows, {"a":1,"b":-1}, stratum="oeis")
    assert (r["observed_pp"], r["lower_pp"], r["upper_pp"]) == (10,-10,10)
    rows[0]["observations"] = dict.fromkeys(("a","b"), obs("unresolved","shared"))
    assert contrast(rows[:1], {"a":1,"b":-1}, stratum="oeis")["upper_pp"] == 0


def test_family_weights_interactions_and_overlap():
    rows = [cell(i, "miss", "solve" if i<3 else "miss") for i in range(4)]
    for row in rows[:3]:
        row["family"]="large"
    assert contrast(rows, {"b":1, "a":-1}, stratum="oeis")["observed_pp"] == 50
    f = factorial(dict(zip(("00","01","02","03","04","05","06"), (1,2,3,6,5,7,12))))
    assert f["composition_matching"]["interaction"] == 2 and f["batch_decay"]["interaction"] == 1
    looks = [dict(useful_tokens=k, verified=True) for k in (0, 101, 402)]
    assert affordable_look(looks, "useful_tokens", 300, overlap=(0,402)) == looks[1]
    assert affordable_look(looks, "useful_tokens", 403, overlap=(0,402)) is None


def test_shared_wall_union_and_unknown_boot_overlap():
    intervals = [dict(receipt="a",host="x",boot="1",start_us=0,end_us=6,beneficiaries=["00"]),
                 dict(receipt="b",host="x",boot="1",start_us=3,end_us=9,beneficiaries=["01"])]
    r = occupied_time([*intervals,intervals[0]])
    assert r["union_us"] == 9 and r["attributed_us"] == {"00":4.5,"01":4.5}
    assert occupied_time([intervals[0],dict(intervals[1],boot="2")])["union_us"] is None
