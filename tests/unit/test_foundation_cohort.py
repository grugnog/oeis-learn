"""Independent first-100 and grouping regressions for the evaluator boundary."""

import pytest


def record(name, values, offset=0):
    return {
        "record_id": "fixture:" + name,
        "first_index": offset,
        "indices": list(range(offset, offset + len(values))),
        "values": [str(v) for v in values],
        "metadata": {},
    }


def test_source_continuity_census_and_signed_bounds():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    rows = [
        record("good", range(100), -10),
        record("short", range(20)),
        record("large", [2**255] * 100),
    ]
    gap = record("gap", range(100))
    gap["indices"][50] += 1
    rows.append(gap)
    result = build_cohort(rows, seed=1, dev_count=1, final_count=0)
    assert result["census"]["total_records"] == 4
    assert result["census"]["complete100"] == 2
    assert result["census"]["in_range_complete100"] == 1
    assert result["records"]["fixture:good"]["indices"][0] == -10
    assert len(result["exclusions"]) == 3


def test_transitive_groups_and_fixed_denominator():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    a = record("a", range(100))
    duplicate = record("b", range(100), 7)
    shift = record("c", range(10, 110))
    ambiguous = record("d", list(range(20)) + list(range(200, 280)))
    other = record("e", [7] * 100)
    result = build_cohort(
        [other, shift, duplicate, a, ambiguous], seed=2, dev_count=1, final_count=1
    )
    assert len(result["groups"]) == 2
    group = next(g for g in result["groups"] if len(g["members"]) == 4)
    assert group["representative"] == "fixture:a" and len(group["witnesses"]) == 3
    reverse = build_cohort(
        [ambiguous, a, duplicate, shift, other], seed=2, dev_count=1, final_count=1
    )
    assert result == reverse
    with pytest.raises(ValueError, match="shortfall"):
        build_cohort([a, duplicate], seed=2, dev_count=1, final_count=1)


def test_hash_window_is_not_sufficient_shift_evidence():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    a = record("a", range(100))
    b = record("b", list(range(1, 81)) + [1000] * 20)
    result = build_cohort([a, b], seed=2, dev_count=1, final_count=1)
    assert len(result["groups"]) == 2


def test_duplicate_indices_offsets_and_exact_range():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    rows = [record("min", [-(2**255)] * 100, -100), record("max", [2**255 - 1] * 100, 17)]
    duplicate = record("dup", [-(2**255)] * 100, -100)
    duplicate["indices"].append(-100)
    duplicate["values"].append(str(-(2**255)))
    conflict = record("conflict", range(100))
    conflict["indices"].append(0)
    conflict["values"].append("99")
    rows += [duplicate, conflict]
    result = build_cohort(rows, seed=7, dev_count=1, final_count=1)
    assert len(result["groups"]) == 2 and len(result["records"]) == 3
    assert result["exclusions"] == [
        {"record_id": "fixture:conflict", "reason": "conflicting_index"}
    ]
    for bad in [True, 1.5, "1"]:
        with pytest.raises(ValueError):
            build_cohort(rows, seed=bad, dev_count=1, final_count=1)


def test_large_duplicate_bucket_retains_linear_witnesses():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    rows = [record(f"{i:04}", [7] * 100) for i in range(500)]
    result = build_cohort(rows, seed=1, dev_count=1, final_count=0)
    assert len(result["groups"]) == 1
    assert len(result["groups"][0]["witnesses"]) == 499
    assert result["census"]["group_size_distribution"] == {"500": 1}


def test_freeze_separates_views_and_rejects_corruption(tmp_path):
    from tests.helpers.foundation_fixture import make_cohort
    from oeis_learn.evaluation.foundation_cohort import load_cohort, read_ref
    from oeis_learn.experiments.models import validate_artifact

    root = make_cohort(tmp_path)
    manifest = load_cohort(root)
    group = manifest["groups"][0]
    prompt = read_ref(root, group["prompt"])
    validate_artifact(prompt)
    assert set(prompt) == {
        "kind",
        "schema_version",
        "request_nonce",
        "language_profile",
        "observed_terms",
    }
    assert len(prompt["observed_terms"]) == 20
    assert len(read_ref(root, group["truth"])["values"]) == 100
    (root / group["truth"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="truth"):
        load_cohort(root)
