from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import pytest
from tests.unit.test_foundation_cohort import record


def analyzer():
    spec = spec_from_file_location("precision", Path("scripts/analyze_oeis_corpus_precision.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_denominators_and_signed_minimum():
    report = analyzer().analyze_sequence_cohort(
        [
            {"terms": [-(1 << 255)] * 120},
            {"terms": [1 << 255] * 100},
            {"terms": [0] * 20},
            {"terms": []},
        ]
    )
    h = report["horizons"]["horizon_100"]
    assert h["sequences_evaluated"] == 2
    assert h["available_prefixes"] == 3
    assert h["coverage_counts"]["signed256"] == 1
    assert h["coverage_pct"]["signed256"] == 50
    assert report["horizons"]["horizon_120"]["sequences_evaluated"] == 1


def test_growth_is_full_series_description():
    module = analyzer()
    assert module.classify_growth_profile([1 << 300, 1, 0, 1]) == "nonmonotone_magnitude"
    assert module.classify_growth_profile([1, 2, 4, 8]) == "nondecreasing_magnitude"
    assert (
        "asymptotic"
        not in module.generate_markdown_report({"cohorts": {}, "total_sequences": 0}).lower()
    )


def test_release_affine_shift_grouping_and_full_overlap():
    from oeis_learn.evaluation.foundation_cohort import build_cohort

    a = [n * n for n in range(100)]
    b = [-3 * n * n + 7 for n in range(5, 105)]
    fake = list(b)
    fake[10] += 1
    fake[90] += 1
    report = build_cohort(
        [record("a", a), record("b", b), record("fake", fake)],
        seed=3,
        dev_count=1,
        final_count=1,
        affine=True,
    )
    assert sorted(len(g["members"]) for g in report["groups"]) == [1, 2]
    witness = next(g for g in report["groups"] if len(g["members"]) == 2)["witnesses"][0]
    assert witness["kind"] == "affine_shift"
    assert witness["matched_overlap"] == 95


def test_database_path_required_and_metadata_retained(tmp_path):
    import duckdb

    module = analyzer()
    with pytest.raises(TypeError):
        module.run_precision_analysis()
    db = tmp_path / "source.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute(
            'CREATE TABLE sequences(oeis_id VARCHAR, terms_json VARCHAR, "offset" INTEGER, tags VARCHAR)'
        )
        con.execute(
            "INSERT INTO sequences VALUES ('A1', ?, -2, 'fini,tabl'), ('A2', 'invalid', 0, '')",
            ["[" + ",".join(["0"] * 120) + "]"],
        )
    report = module.run_precision_analysis(
        str(db), str(tmp_path / "report.json"), str(tmp_path / "report.md")
    )
    assert report["invalid_records"] == 1
    assert report["source_metadata"]["finite"] == 1
    assert report["source_metadata"]["table"] == 1
    assert report["source_metadata"]["offsets"] == {"-2": 1}
