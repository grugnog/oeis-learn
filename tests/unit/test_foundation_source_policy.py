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


def test_indexed_gaps_do_not_compress_horizon():
    from oeis_learn.evaluation.source_policy import analyze_sequence_cohort

    row = record("gapped", range(120), offset=-7)
    row["indices"].pop(40)
    row["values"].pop(40)
    report = analyze_sequence_cohort([row])
    assert report["horizons"]["horizon_20"]["sequences_evaluated"] == 1
    assert report["horizons"]["horizon_50"]["sequences_evaluated"] == 0
    assert report["source_metadata"]["offsets"] == {"-7": 1}


def test_release_profile_roundtrip_and_all_member_reserved_prefixes(tmp_path):
    import json
    import yaml
    from oeis_learn.evaluation.foundation_cohort import freeze_cohort, load_cohort, read_ref
    from oeis_learn.data.program_admission import load_membership
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    rows = [
        record("a", [n * n for n in range(120)], offset=-3),
        record("b", [-3 * n * n + 7 for n in range(120)], offset=7),
    ]
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.json").write_text(json.dumps(rows))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                schema_version="foundation/v1",
                profile="prefix20_total100_affine_v2",
                seed=3,
                dev_count=1,
                final_count=0,
            )
        )
    )
    root = tmp_path / "cohort"
    frozen = freeze_cohort(source, config, root)
    assert load_cohort(root) == frozen
    assert len(frozen["groups"]) == 1 and frozen["census"]["complete120"] == 2
    membership, _ = load_membership(root)
    assert membership == {digest(row["values"][:20]) for row in rows}
    prompt = read_ref(root, frozen["groups"][0]["prompt"])
    assert "metadata" not in prompt and "first_index" not in prompt


def test_indexed_metadata_preserves_keyword_arrays():
    report = analyzer().analyze_sequence_cohort(
        [
            dict(
                first_index=-2,
                indices=list(range(-2, 118)),
                values=["0"] * 120,
                metadata=dict(keywords=["fini", "tabl", "nonn"]),
            )
        ]
    )
    assert report["source_metadata"]["finite"] == 1
    assert report["source_metadata"]["table"] == 1
    assert report["source_metadata"]["keywords"] == dict(fini=1, tabl=1, nonn=1)
