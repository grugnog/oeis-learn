"""Nested input timings belong inside their parent and cannot invent completed work."""

import json
import sys

import pytest
from tests.unit.test_bigint_benchmark import benchmark
from tests.unit.test_bigint_runtime_profile import report
from tests.unit.test_bigint_phase5 import helper

PHASES = (
    "index_parse",
    "begin_load",
    "magnitude_encode",
    "byte_transfer",
    "finish_load",
    "status_check",
    "seal",
)


def input_report():
    data = report()
    data["profile_input_load"] = True
    for sample in data["samples"]:
        sample["timings"]["stages"]["input_load"] = dict(
            policy="bigint-input-load-profile/v1",
            phases={
                name: dict(calls=3 if name not in ("index_parse", "seal") else 1, seconds=0.0001)
                for name in PHASES
            },
            literal_values_started=2,
            index_values_started=1,
            values_finished=3,
            byte_calls_attempted=3,
            byte_calls_returned=3,
            unattributed_seconds=0.0003,
        )
    return data


def test_nested_phases_are_a_partition_of_input_loading_not_additional_execution():
    data = input_report()
    result = benchmark().input_load_summary(data, 1)
    assert result["policy"] == "bigint-input-load-profile/v1"
    assert set(result["phases"]) == set(PHASES)
    assert result["parent_input_load_warm"]["median_seconds"] == pytest.approx(0.001)
    assert result["unattributed"]["median_seconds"] == pytest.approx(0.0003)
    assert result["phases"]["byte_transfer"]["warm_fraction_of_input_load"] == pytest.approx(0.1)
    assert benchmark().profile_summary(data, 1)["warm_median_execution_ratio"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "policy",
        "extra",
        "negative",
        "nan",
        "overlap",
        "calls",
        "started",
        "finished",
        "bytes",
        "returned",
        "order",
        "flag",
    ],
)
def test_invented_or_overlapping_input_measurements_are_rejected(kind):
    data = input_report()
    detail = data["samples"][0]["timings"]["stages"]["input_load"]
    if kind == "missing":
        del data["samples"][0]["timings"]["stages"]["input_load"]
    elif kind == "policy":
        detail["policy"] = "other"
    elif kind == "extra":
        detail["phases"]["unknown"] = dict(calls=1, seconds=0)
    elif kind == "negative":
        detail["unattributed_seconds"] = -1
    elif kind == "nan":
        detail["phases"]["seal"]["seconds"] = float("nan")
    elif kind == "overlap":
        detail["phases"]["seal"]["seconds"] = 1
    elif kind == "calls":
        detail["phases"]["begin_load"]["calls"] = True
    elif kind == "started":
        detail["index_values_started"] = 2
    elif kind == "finished":
        detail["values_finished"] = 4
    elif kind == "bytes":
        detail["byte_calls_attempted"] = 10000000
    elif kind == "returned":
        detail["byte_calls_returned"] = 4
    elif kind == "order":
        detail["phases"]["finish_load"]["calls"] = 2
    else:
        data["profile_input_load"] = False
    with pytest.raises((ValueError, KeyError)):
        benchmark().input_load_summary(data, 1)


def test_early_failure_can_have_partial_values_and_byte_calls():
    data = input_report()
    for sample in data["samples"]:
        detail = sample["timings"]["stages"]["input_load"]
        detail.update(
            literal_values_started=1,
            index_values_started=0,
            values_finished=0,
            byte_calls_attempted=1,
            byte_calls_returned=0,
        )
        for name, phase in detail["phases"].items():
            phase["calls"] = int(name in ("begin_load", "magnitude_encode", "byte_transfer"))
            phase["seconds"] = 0.0001 if phase["calls"] else 0
        detail["unattributed_seconds"] = 0.0007
    assert benchmark().input_load_summary(data, 1)["counters"]["byte_calls_returned"] == [0, 0, 0]


def test_image_side_input_profile_requires_runtime_profile(tmp_path):
    with pytest.raises(ValueError, match="profil"):
        benchmark().run(tmp_path, profile_input_load=True)
    assert json.loads((tmp_path / "benchmark.json").read_text())["status"] == "FAIL"


def test_host_input_profile_rejects_unscoped_request_before_engine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qualify", "--profile-input-load", "--output", "unused"])
    with pytest.raises(SystemExit) as error:
        helper().main()
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["nan", "invalid-json", "non-object"])
def test_malformed_native_detail_keeps_bounded_raw_evidence_and_failure_status(
    tmp_path, monkeypatch, kind
):
    from types import SimpleNamespace
    from oeis_learn.fresh import bigint_config, common

    module = benchmark()
    data = input_report()
    data["samples"][0]["timings"]["stages"]["input_load"]["unattributed_seconds"] = float("nan")
    raw = (
        b"[]"
        if kind == "non-object"
        else b"{bad JSON"
        if kind == "invalid-json"
        else json.dumps(data).encode()
    )
    monkeypatch.setattr(bigint_config, "packaged_artifacts", lambda: {})
    monkeypatch.setattr(common, "file_hash", lambda p: "sha256:" + "a" * 64)
    monkeypatch.setattr(
        module, "cases", lambda: [dict(name="fixture", source="fixture", indices=["0"])]
    )

    def native(argv, **kwargs):
        request = json.loads(kwargs["input"])
        assert request["profile_input_load"] is True and request["profile"] is True
        return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", native)
    with pytest.raises(ValueError):
        module.run(tmp_path, profile_runtime=True, profile_input_load=True)
    result = json.loads((tmp_path / "benchmark.json").read_text())
    assert result["status"] == "FAIL" and result["active_measurement"]["case"] == "fixture"
    assert (tmp_path / "failed-native.log").read_bytes() == raw
    assert not (tmp_path / "failed-native.log.tmp").exists()
