"""Profiling evidence cannot invent parity, costs, qualification or work scope."""

from copy import deepcopy
import json
from types import SimpleNamespace
import sys

import pytest
from tests.unit.test_bigint_benchmark import benchmark
from tests.unit.test_bigint_phase5 import helper


def report():
    module = benchmark()
    result = dict(
        outcome="ok",
        reason=None,
        values=["4"],
        coverage=1,
        fuel=10,
        resource_identity="sha256:" + "1" * 64,
    )
    return dict(
        version="bigint-measurements/v2",
        samples=[
            dict(
                result=deepcopy(result),
                parity=True,
                order="profiled-first" if i % 2 == 0 else "control-first",
                control=dict(
                    result=deepcopy(result),
                    timings=dict(prepare_seconds=0.01, execution_seconds=0.02, total_seconds=0.03),
                ),
                timings=dict(
                    prepare_seconds=0.01,
                    execution_seconds=0.03,
                    total_seconds=0.04,
                    stages=dict(
                        policy="bigint-runtime-profile/v1",
                        terms_started=1,
                        terms_finished=1,
                        unattributed_seconds=0.02,
                        phases={
                            name: dict(calls=1, seconds=0.001) for name in module.PROFILE_PHASES
                        },
                    ),
                ),
            )
            for i in range(3)
        ],
    )


def test_profile_keeps_unattributed_time_and_observer_cost_separate():
    summary = benchmark().profile_summary(report(), 1)
    assert summary["paired_results_match"]
    assert summary["warm_median_execution_ratio"] == pytest.approx(1.5)
    assert summary["warm_paired_delta_seconds"] == pytest.approx(0.01)
    assert summary["unattributed"]["median_seconds"] == pytest.approx(0.02)
    assert len(summary["phases"]) == 10


@pytest.mark.parametrize(
    "field,value", [("values", ["16"]), ("fuel", 11), ("coverage", 0), ("reason", "deadline")]
)
def test_profile_does_not_trust_native_parity_claim(field, value):
    data = report()
    data["samples"][1]["control"]["result"][field] = value
    summary = benchmark().profile_summary(data, 1)
    assert not summary["paired_results_match"]
    assert field in summary["mismatches"][0]["fields"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "nan",
        "negative",
        "bool_calls",
        "extra_time",
        "order",
        "finished",
        "control_stages",
    ],
)
def test_profile_rejects_malformed_or_invented_timing_evidence(mutation):
    data = report()
    sample = data["samples"][0]
    stages = sample["timings"]["stages"]
    if mutation == "missing":
        stages["phases"].pop("clock_stop")
    elif mutation == "nan":
        stages["phases"]["clock_stop"]["seconds"] = float("nan")
    elif mutation == "negative":
        stages["unattributed_seconds"] = -1
    elif mutation == "bool_calls":
        stages["phases"]["store_setup"]["calls"] = True
    elif mutation == "extra_time":
        stages["phases"]["compute"]["seconds"] = 5
    elif mutation == "order":
        sample["order"] = "control-first"
    elif mutation == "control_stages":
        sample["control"]["timings"]["stages"] = stages
    else:
        stages["terms_finished"] = 2
    with pytest.raises(ValueError):
        benchmark().profile_summary(data, 1)


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpu"],
        ["--prepare"],
        ["--inputs", "fixture"],
        ["--performance-only"],
        ["--fitting-only"],
        ["--checkpoint-run", "fixture"],
    ],
)
def test_runtime_profile_rejects_unrelated_work_before_engine(extra, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--runtime-profile-only", "--output", "unused", *extra]
    )
    with pytest.raises(SystemExit) as error:
        helper().main()
    assert error.value.code == 2


@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.parametrize("kind", ["pass", "failed", "old-source"])
def test_profile_host_bounds_and_archives_even_failure(tmp_path, monkeypatch, kind, compare):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.synthesis.contracts import identity

    module = helper()
    output = tmp_path / "profile"
    snapshot = {"fixture": "immutable"}
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: snapshot)

    class Host:
        def __init__(self, args):
            self.engine = SimpleNamespace(name="podman")
            self.policy = dict(runtime={})

        def ensure(self):
            return "sha256:" + "a" * 64

    monkeypatch.setitem(
        sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=lambda s: s)
    )
    stages = []

    def stage(host, image, path, argv, **kwargs):
        assert kwargs == dict(seconds=600)
        assert "--runtime-profile" in argv and "--neural" not in argv
        assert ("--compare-host-lookups" in argv) == compare
        stages.append(path.name)
        module.write(
            path / "benchmark.json",
            dict(
                status="FAIL" if kind == "failed" else "PASS",
                compare_host_lookups=compare,
                source_sha256="old" if kind == "old-source" else identity(snapshot),
            ),
        )

    monkeypatch.setattr(module, "container_stage", stage)
    monkeypatch.setattr(module, "launch", lambda *a, **kw: pytest.fail("unrelated work"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--runtime-profile-only",
            *(["--compare-host-lookups"] if compare else []),
            "--output",
            str(output),
        ],
    )
    assert module.main() == (0 if kind == "pass" else 1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scope"] == "runtime-profile-only" and summary["qualified"] is False
    assert stages == ["runtime-profile"] and output.with_suffix(".zip").is_file()


@pytest.mark.parametrize(
    "extra", [dict(neural=True), dict(device="cuda"), dict(config="fixture"), dict(shapes={})]
)
def test_image_side_profile_rejects_unrelated_work(tmp_path, extra):
    with pytest.raises(ValueError, match="CPU-only"):
        benchmark().run(tmp_path, profile_runtime=True, **extra)
    assert json.loads((tmp_path / "benchmark.json").read_text())["status"] == "FAIL"


def test_semantic_mismatch_retains_partial_and_final_evidence(tmp_path, monkeypatch):
    from oeis_learn.fresh import bigint_config, common

    module = benchmark()
    monkeypatch.setattr(bigint_config, "packaged_artifacts", lambda: {})
    monkeypatch.setattr(common, "file_hash", lambda path: "sha256:" + "a" * 64)
    monkeypatch.setattr(module, "cases", lambda: [dict(name="fixture")])
    row = dict(
        case="fixture",
        batch=1,
        profiling=dict(paired_results_match=False, mismatches=[dict(sample=1, fields=["fuel"])]),
    )
    monkeypatch.setattr(module, "runtime_measurements", lambda *a, **kw: row)
    monkeypatch.setattr(module, "fitting_measurements", lambda *a: pytest.fail("unrelated fitting"))
    with pytest.raises(RuntimeError, match="disagreement"):
        module.run(tmp_path, profile_runtime=True)
    for name in ("partial.json", "benchmark.json"):
        evidence = json.loads((tmp_path / name).read_text())
        assert evidence["status"] == "FAIL" and evidence["cases"] == [row]
    assert "disagreement" in evidence["error"]
