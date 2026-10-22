"""Actual packaged measurements must agree with independent production fixtures."""

from pathlib import Path
import os
import pytest

from tests.unit.test_bigint_benchmark import benchmark


@pytest.mark.parametrize(
    "profile,compare,detail",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_packaged_measurement_path_preserves_values_failures_and_cold_warm_boundaries(
    profile, compare, detail
):
    binary = Path(
        os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
    )
    if not binary.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("mandatory benchmark runtime missing")
        pytest.skip("requires pinned native qualification image")
    module = benchmark()
    for case in module.cases():
        row = module.runtime_measurements(
            case,
            batch=1,
            repetitions=2,
            profile=profile,
            compare_host_lookups=compare,
            profile_input_load=detail,
        )
        if detail:
            probe = row["input_load_profiling"]["first_byte_probe"]
            assert row["input_load_profiling"]["policy"] == "bigint-input-load-profile/v2"
            assert probe["terms_observed"] == probe["memory_probe_terms"] == [1, 1]
            assert probe["first_byte_calls"] == [1, 1]
            assert set(row["input_load_profiling"]["phases"]) == set(module.INPUT_LOAD_PHASES)
        if compare:
            assert row["host_lookups"]["all_results_match"], row
        if profile:
            assert row["profiling"]["paired_results_match"], row
            assert len(row["profiling"]["phases"]) == 10
            for sample in row["samples"]:
                stages = sample["timings"]["stages"]
                assert stages["terms_started"] == stages["terms_finished"] == 1
                assert stages["phases"]["clock_stop"]["calls"] == 1
                assert stages["phases"]["store_drop"]["calls"] == 1
        assert len(row["samples"]) == 2
        assert row["runtime_compile_seconds"] >= 0
        assert row["warm"]["execution_seconds"]["samples"] == 1
        assert row["cold"]["total_seconds"] >= row["cold"]["prepare_seconds"]
        assert row["expected_reason"] == case["expected_reason"]


@pytest.mark.parametrize("order", ["zero-first", "reserved-first"])
def test_packaged_growth_comparison_preserves_exact_fuel_and_failures(order, tmp_path):
    binary = Path(
        os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
    )
    if not binary.is_file():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("mandatory benchmark runtime missing")
        pytest.skip("requires pinned native qualification image")
    mod = benchmark()
    for case in mod.cases():
        row = mod.memory_growth_measurements(
            case, 1, 2, order=order, failure_output=tmp_path / "failed-native.log"
        )
        assert row["memory_growth"]["all_results_match"], row
        assert (
            row["engine_configuration"]["manifest"]["dynamic_memory_reserved_for_growth"] == 65536
        )
        assert row["engine_configuration"]["manifest"]["purpose"] == "production"
        assert (
            row["memory_growth"]["zero"]["engine_configuration"]["manifest"]["purpose"]
            == "qualification-zero-growth-control"
        )
        for arm in [row, row["memory_growth"]["zero"]]:
            assert arm["input_load_profiling"]["first_byte_probe"]["memory_in_first"][
                "requests"
            ] == [1, 1]
            assert arm["input_load_profiling"]["first_byte_probe"]["memory_in_first"][
                "requested_increase_bytes"
            ] == [65536, 65536]
