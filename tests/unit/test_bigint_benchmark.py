"""Measurement fixtures must obey production widths and independent expectations."""

import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_bigint.py"
    spec = importlib.util.spec_from_file_location("bigint_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_width_matrix_uses_legal_inputs_and_independent_exact_expectations():
    from oeis_learn.synthesis.contracts import canonical_integer
    from oeis_learn.synthesis.reference import evaluate

    cases = benchmark().cases()
    assert {64, 256, 512, 1024, 4096} <= {c["supplied_bits"] for c in cases}
    assert max(c["result_bits"] for c in cases if c["expected_reason"] is None) == 16384
    assert any(c["expected_reason"] == "integer_limit" for c in cases)
    for case in cases:
        assert all(abs(canonical_integer(n)).bit_length() <= 4096 for n in case["indices"])
        result = evaluate(case["source"], case["indices"])
        if case["expected_reason"]:
            assert result.outcome != "ok" and result.reason == case["expected_reason"]
        else:
            assert result.outcome == "ok" and list(result.values) == case["expected"]
        assert case["old_source"] is None or case["supplied_bits"] <= 256


def test_measurement_summary_rejects_missing_failed_or_fabricated_samples():
    summarize = benchmark().summarize
    with pytest.raises(ValueError):
        summarize([])
    for sample in (float("nan"), -1, True):
        with pytest.raises(ValueError):
            summarize([sample])
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["median_seconds"] == 3 and result["p95_seconds"] == 5


def test_cost_admission_rechecks_forecast_and_rejects_cpu_or_incomplete_dataset(tmp_path):
    from copy import deepcopy
    from oeis_learn.fresh.bigint_config import defaults, resolve

    module = benchmark()
    recipe = defaults()
    report = dict(
        status="PASS",
        image="sha256:" + "a" * 64,
        neural=dict(hip="fixture", device="cuda:0", fixture_sha256="sha256:" + "b" * 64),
        costs={
            key + "_p95": 0.001
            for key in (
                "training",
                "teacher_forcing",
                "generation",
                "fit",
                "reference",
                "hidden_scoring",
                "checkpoint",
            )
        },
        cases=[dict(case="fixture")],
    )
    shapes = dict(
        counts=dict(training=512, parameter_development=128, structural_development=128),
        tiny_count=32,
        max_input_bits=4096,
        max_tokens=1024,
        native_admitted=True,
        stages={s: 1 for s in "ABCD"},
        prepared_sha256="sha256:" + "c" * 64,
        split_sha256="sha256:" + "d" * 64,
    )
    for kind in ("cpu", "partial", "expensive", "failed", "not_admitted"):
        bad, data = deepcopy(report), deepcopy(shapes)
        if kind == "cpu":
            bad["neural"]["device"] = "cpu"
        if kind == "partial":
            data["counts"]["training"] = 511
        if kind == "expensive":
            bad["costs"]["generation_p95"] = 60
        if kind == "failed":
            bad["status"] = "FAIL"
        if kind == "not_admitted":
            data["native_admitted"] = False
        with pytest.raises(ValueError):
            module.admit_cost_recipe(bad, recipe, data, tmp_path)
        assert not (tmp_path / "measured-recipe.yaml").exists()
    result = module.admit_cost_recipe(report, recipe, shapes, tmp_path)
    import yaml

    value = resolve(yaml.safe_load((tmp_path / result["path"]).read_text()))
    assert value["learning"]["cost_preflight"]["scoring_data"] is False
    assert value["learning"]["cost_preflight"]["image"] == report["image"]
