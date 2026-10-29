"""Lookup speed comparisons must preserve all result fields and observation controls."""

from copy import deepcopy
import json
import sys

import pytest

from tests.unit.test_bigint_benchmark import benchmark
from tests.unit.test_bigint_runtime_profile import report
from tests.unit.test_bigint_phase5 import helper


def lookup_report():
    data = report()
    data["host_lookup_comparison"] = dict(
        version="per-term-host-lookups/v1", cached="lazy-per-term", baseline="per-call"
    )
    for index, sample in enumerate(data["samples"]):
        sample["lookup_baseline"] = dict(
            result=deepcopy(sample["control"]["result"]),
            timings=dict(prepare_seconds=0.01, execution_seconds=0.04, total_seconds=0.05),
            parity=True,
            order="baseline-first" if index % 2 == 0 else "baseline-last",
        )
    return data


def test_lookup_savings_use_ordinary_calls_separately_from_profiler_overhead():
    module = benchmark()
    data = lookup_report()
    result = module.lookup_summary(data)
    assert result["all_results_match"]
    assert result["cached_to_lookup_execution_ratio"] == pytest.approx(0.5)
    assert result["warm_paired_savings_seconds"] == pytest.approx(0.02)
    assert module.profile_summary(data, 1)["warm_median_execution_ratio"] == pytest.approx(1.5)


@pytest.mark.parametrize("field", ["values", "fuel", "coverage", "reason", "resource_identity"])
def test_lookup_comparison_independently_rejects_forged_parity(field):
    data = lookup_report()
    data["samples"][1]["lookup_baseline"]["result"][field] = "wrong"
    result = benchmark().lookup_summary(data)
    assert not result["all_results_match"]
    assert field in result["mismatches"][0]["fields"]


@pytest.mark.parametrize(
    "kind", ["missing", "policy", "order", "timing", "negative", "instrumented", "empty"]
)
def test_lookup_comparison_rejects_incomplete_or_mislabelled_evidence(kind):
    data = lookup_report()
    row = data["samples"][0]["lookup_baseline"]
    if kind == "missing":
        del data["samples"][0]["lookup_baseline"]
    elif kind == "policy":
        data["host_lookup_comparison"]["baseline"] = "cached"
    elif kind == "order":
        row["order"] = "baseline-last"
    elif kind == "timing":
        row["timings"]["total_seconds"] = 1
    elif kind == "negative":
        row["timings"]["execution_seconds"] = -1
    elif kind == "instrumented":
        row["timings"]["stages"] = {}
    else:
        data["samples"] = []
    with pytest.raises((ValueError, KeyError)):
        benchmark().lookup_summary(data)


def test_lookup_comparison_requires_profile_before_native_launch(tmp_path):
    with pytest.raises(ValueError, match="profil"):
        benchmark().run(tmp_path, compare_host_lookups=True)
    assert json.loads((tmp_path / "benchmark.json").read_text())["status"] == "FAIL"


def test_host_lookup_comparison_requires_cpu_profile_scope_before_engine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qualify", "--compare-host-lookups", "--output", "unused"])
    with pytest.raises(SystemExit) as exc:
        helper().main()
    assert exc.value.code == 2


@pytest.mark.parametrize("arm", ["control", "lookup_baseline"])
def test_missing_nullable_result_field_cannot_match_an_explicit_null(arm):
    data = lookup_report()
    del data["samples"][0][arm]["result"]["reason"]
    module = benchmark()
    result = module.profile_summary(data, 1) if arm == "control" else module.lookup_summary(data)
    key = "paired_results_match" if arm == "control" else "all_results_match"
    assert result[key] is False
    assert "reason" in result["mismatches"][0]["fields"]


def test_lookup_disagreement_retains_all_three_results_in_failed_archive_stage(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from oeis_learn.fresh import bigint_config, common

    module = benchmark()
    data = lookup_report()
    data["samples"][1]["lookup_baseline"]["result"]["fuel"] += 1
    monkeypatch.setattr(bigint_config, "packaged_artifacts", lambda: {})
    monkeypatch.setattr(common, "file_hash", lambda p: "sha256:" + "a" * 64)
    monkeypatch.setattr(
        module, "cases", lambda: [dict(name="fixture", source="fixture", indices=["0"])]
    )

    def native(argv, **kwargs):
        request = json.loads(kwargs["input"])
        assert request["compare_host_lookups"] is True and request["profile"] is True
        return SimpleNamespace(returncode=0, stdout=json.dumps(data).encode(), stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", native)
    with pytest.raises(RuntimeError, match="disagreement"):
        module.run(tmp_path, profile_runtime=True, compare_host_lookups=True)
    for name in ("partial.json", "benchmark.json"):
        evidence = json.loads((tmp_path / name).read_text())
        assert evidence["status"] == "FAIL"
        row = evidence["cases"][0]
        assert row["profiling"]["paired_results_match"] is True
        assert row["host_lookups"]["all_results_match"] is False
        for sample in row["samples"]:
            for result in module.measurement_results(sample):
                assert "values_sha256" in result and "values" not in result
