import pytest
from oeis_learn.generalization.progressive.short_profile import visible, validate_terms


def test_exact_four_visible_nine_total_no_padding():
    row = dict(indices=list(map(str, range(9))), values=list(map(str, range(9))))
    assert visible(row) == dict(indices=["0", "1", "2", "3"], values=["0", "1", "2", "3"])
    assert len(row["values"]) == 9
    for size in (4, 8, 20, 100):
        with pytest.raises(ValueError):
            validate_terms(dict(indices=list(map(str, range(size))), values=["0"] * size))


def test_short_input_bucket_has_explicit_position_mask_and_old_guard_survives():
    from oeis_learn.generalization.progressive.short_profile import collate
    from oeis_learn.generalization.progressive.cohorts import validate_prompt

    row = dict(indices=["0", "1", "2", "3"], values=["0", "1", "3", "6"])
    batch = collate([row, row])
    assert batch["position_mask"] == [[True] * 4, [True] * 4]
    assert batch["values"] == [row["values"], row["values"]]
    with pytest.raises(ValueError):
        validate_prompt(dict(id="short", **row))
    with pytest.raises(ValueError):
        collate([dict(indices=["0"] * 20, values=["0"] * 20)])
