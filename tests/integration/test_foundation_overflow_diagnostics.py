import pytest
from oeis_learn.evaluation.overflow_diagnostics import diagnose_overflow


def sample(source, origin="generic_sample"):
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    return dict(
        source=source, origin=origin, provenance_sha256=digest({"source": source, "origin": origin})
    )


def test_wider_intermediate_rescue_and_in_range_benchmark(tmp_path):
    half = (1 << 255) - 1
    overflowing = f"(pure (return (sub (add (const {half}) (const 1)) (const 1))))"
    report = diagnose_overflow(
        {
            "random_generic": [sample(overflowing), sample("(pure (return (index)))")],
            "model_samples": [],
            "verified_solutions": [],
        },
        tmp_path,
    )
    assert report["qualified"] is False
    assert report["strata"]["model_samples"]["availability"] == "unavailable"
    rows = report["strata"]["random_generic"]["records"]
    rescued = next(r for r in rows if r["wide"]["violations"])
    assert rescued["wide"]["outcome"] is None and rescued["wide"]["final_outputs_in_range"]
    assert rescued["benchmark"] is None
    safe = next(r for r in rows if not r["wide"]["violations"])
    assert safe["benchmark"]["repeats"] == 30
    assert safe["benchmark"]["outputs_equal"]
    assert report["rescued_candidates"] == 1 and report["added_development_groups"] is None


def test_4096_bit_cap_unknown_and_duplicate_or_wrong_provenance_rejected(tmp_path):
    exploding = (
        "(pure (set 0 (const 2)) (repeat 20 (set 0 (mul (get 0) (get 0)))) (return (get 0)))"
    )
    report = diagnose_overflow(
        {"random_generic": [sample(exploding)], "model_samples": [], "verified_solutions": []},
        tmp_path,
    )
    result = report["strata"]["random_generic"]["records"][0]["wide"]
    assert result["outcome"] == "unknown" and result["reason"] == "wider_reference_4096_bit_cap"
    with pytest.raises(ValueError, match="origin"):
        diagnose_overflow(
            {
                "random_generic": [sample("(pure (return (index)))", "imported")],
                "model_samples": [],
                "verified_solutions": [],
            },
            tmp_path / "bad",
        )
