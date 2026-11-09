import copy
import pytest
from oeis_learn.generalization.progressive.sources import validate_source, freeze_source, groups


def source(**changes):
    return freeze_source(
        dict(
            family="FindStat/Sage",
            revision="reviewed-snapshot-1",
            snapshot="sha256:" + "1" * 64,
            access="available",
            use_conditions="approved finite diagnostic",
            indexing="collection-size",
            offset="0",
            representation="exact_integer_totals",
            objects="permutations",
            relations=["statistic:inversions"],
            transformation="first_moment",
            verification="independent complete enumeration",
            finite_support=9,
            values=list(map(str, range(9))),
            code=None,
            track="external_assisted",
        )
        | changes
    )


def test_source_is_immutable_explicit_and_grouped():
    s = source()
    before = copy.deepcopy(s)
    assert validate_source(s) == s
    assert s == before
    a = source()
    b = source()
    b["identity"] = "other"
    assert groups([a, a]) == [["0", "1"]]


@pytest.mark.parametrize(
    "key,value",
    [
        ("values", ["0.5"] * 9),
        ("values", [1] * 9),
        ("offset", None),
        ("use_conditions", ""),
        ("access", "unresolved"),
        ("representation", "approximate"),
        ("indexing", "arbitrary-object-order"),
        ("transformation", "zero-pad-polynomial"),
        ("code", "https://download/statistic.py"),
        ("finite_support", 100),
        ("track", "synthetic_strict"),
    ],
)
def test_unsuitable_sources_cannot_be_admitted(key, value):
    with pytest.raises(ValueError):
        source(**{key: value})
