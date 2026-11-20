import math
import pytest
from oeis_learn.generalization.progressive.combinatorial import prepare


def test_complete_collections_and_independent_closed_form():
    result = prepare(["inversions", "fixed_points"], moments=[1, 2])
    assert result["cardinalities"] == [str(math.factorial(n)) for n in range(9)]
    rows = {(r["statistic"], r["moment"]): r["values"] for r in result["rows"]}
    assert rows["inversions", 1] == [str(math.factorial(n) * n * (n - 1) // 4) for n in range(9)]
    assert rows["fixed_points", 1] == ["0"] + [str(math.factorial(n)) for n in range(1, 9)]
    assert rows["inversions", 2][:4] == ["0", "0", "1", "19"]
    assert result["objects"] == 46234
    assert result["track"] == "external_assisted"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(statistics=["download.py"]),
        dict(statistics=["inversions"] * 33),
        dict(statistics=["inversions"], moments=[3]),
        dict(statistics=["inversions"], max_objects=1),
        dict(statistics=["inversions"], max_work=1),
        dict(statistics=["inversions"], max_bytes=1),
    ],
)
def test_frozen_enumeration_budgets_fail_before_admission(kwargs):
    with pytest.raises((ValueError, TimeoutError)):
        prepare(**kwargs)


def test_reviewed_small_example_all_allowlisted_statistics():
    from oeis_learn.generalization.progressive.combinatorial import statistic, ALLOWLIST

    expected = dict(
        inversions=3,
        descents=2,
        ascents=1,
        major_index=4,
        fixed_points=0,
        excedances=2,
        cycles=1,
        records=2,
    )
    assert {s: statistic(s, (2, 0, 3, 1)) for s in ALLOWLIST} == expected
    assert all(statistic(s, ()) == 0 for s in ALLOWLIST)
