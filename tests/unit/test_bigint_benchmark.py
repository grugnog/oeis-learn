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

