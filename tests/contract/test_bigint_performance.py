"""Actual packaged measurements must agree with independent production fixtures."""

from pathlib import Path
import os
import pytest

from tests.unit.test_bigint_benchmark import benchmark


def test_packaged_measurement_path_preserves_values_failures_and_cold_warm_boundaries():
    binary = Path(
        os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
    )
    if not binary.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("mandatory benchmark runtime missing")
        pytest.skip("requires pinned native qualification image")
    module = benchmark()
    for case in module.cases():
        row = module.runtime_measurements(case, batch=1, repetitions=2)
        assert len(row["samples"]) == 2
        assert row["runtime_compile_seconds"] >= 0
        assert row["warm"]["execution_seconds"]["samples"] == 1
        assert row["cold"]["total_seconds"] >= row["cold"]["prepare_seconds"]
        assert row["expected_reason"] == case["expected_reason"]
