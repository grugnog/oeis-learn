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


def test_body_wide_overflow_and_native_wraps_are_separate():
    from oeis_learn.evaluation.overflow_diagnostics import _trace_body

    maximum = (1 << 255) - 1
    wide = _trace_body(f"i256.const {maximum} i256.const 1 i256.add i256.const 1 i256.sub")
    assert wide["outcome"] is None and wide["outputs"] == [str(maximum)] * 100
    assert wide["violations"] == 100 and wide["native_limb_wraps"] == 0
    word = _trace_body(
        "i64.const 9223372036854775807 i64.const 1 i64.add i64.const 0 i64.const 0 i64.const 0"
    )
    assert word["violations"] == 0 and word["native_limb_wraps"] == 100


def test_raw_generic_draws_replay_before_strict_overflow_admission():
    from pathlib import Path
    import yaml
    from copy import deepcopy
    from oeis_learn.data.generic_programs import GenericSampler
    from oeis_learn.evaluation.overflow_diagnostics import sampled_strata, verify_origins

    sampler = GenericSampler(
        yaml.safe_load(Path("configs/foundation/generic_sampler.yaml").read_text())
    )
    strata, packets = sampled_strata(sampler, count=4)
    assert strata["random_generic"] and not strata["model_samples"]
    verified = verify_origins(strata, provenance_packets=packets)
    assert verified["snapshot_sha256"] is None
    forged = deepcopy(strata)
    forged["random_generic"][0]["source"] = "i256.const 7"
    with pytest.raises(ValueError, match="provenance"):
        verify_origins(forged, provenance_packets=packets)


def test_sealed_development_scoring_counts_unique_programs_and_groups(tmp_path):
    import json, yaml
    from oeis_learn.evaluation.foundation_cohort import freeze_cohort, write_json
    from oeis_learn.evaluation.overflow_diagnostics import score_development, _trace_body
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
    from tests.unit.test_foundation_cohort import record

    maximum = (1 << 255) - 1
    source = f"i256.const {maximum} i256.const 1 i256.add i256.const 1 i256.sub"
    inputs = tmp_path / "source"
    inputs.mkdir()
    (inputs / "records.json").write_text(json.dumps([record("a", [maximum] * 100)]))
    config = tmp_path / "cohort.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                schema_version="foundation/v1",
                profile="prefix20_total100_v1",
                seed=3,
                dev_count=1,
                final_count=0,
            )
        )
    )
    cohort = tmp_path / "cohort"
    freeze_cohort(inputs, config, cohort)
    output = tmp_path / "diagnostic"
    strata = {
        "random_generic": [sample(source)],
        "model_samples": [sample(source, "generic_model")],
        "verified_solutions": [],
    }
    write_json(output, "strata.json", strata)
    write_json(output, "seal.json", dict(strata_sha256=digest(strata), provenance=None))
    row = dict(source=source, source_sha256=digest(source), wide=_trace_body(source))
    report = dict(
        source_sha256=digest(strata),
        strata={name: dict(records=[row] if values else []) for name, values in strata.items()},
    )
    result = score_development(report, cohort, output)
    assert result["prefix_matching_rescues"] == 1 and result["development_matching_rescues"] == 1
    assert result["added_development_groups"] == 1 and not result["feedback_to_candidates"]
