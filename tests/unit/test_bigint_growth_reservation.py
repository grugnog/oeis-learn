"""The growth control cannot silently become production or claim timing-only parity."""

import json
import sys

import pytest

from tests.unit.test_bigint_benchmark import benchmark
from tests.unit.test_bigint_first_byte import byte_report
from tests.unit.test_bigint_phase5 import helper


def configuration(reserve, *, legacy=False):
    if not legacy:
        from oeis_learn.synthesis.engine_policy import engine_configuration

        return engine_configuration("sha256:" + "a" * 64, zero_control=reserve == 0)

    from oeis_learn.synthesis.contracts import identity

    manifest = dict(
        version="bigint-engine-configuration/v1",
        purpose="qualification-growth-control" if reserve else "production",
        runtime_wasm_sha256="sha256:" + "a" * 64,
        wasmtime="20.0.2",
        consume_fuel=True,
        epoch_interruption=True,
        wasm_threads=False,
        parallel_compilation=False,
        static_memory_maximum_size=0,
        static_memory_guard_size=0,
        dynamic_memory_guard_size=0,
        dynamic_memory_reserved_for_growth=reserve,
    )
    return dict(manifest=manifest, sha256=identity(manifest))


def arm(reserve):
    data = byte_report()
    data.update(
        engine_configuration=configuration(reserve),
        peak_process_rss_bytes=64 << 20,
        runtime_compile_seconds=0.1,
    )
    return data


def test_growth_comparison_uses_ordinary_costs_and_separate_identities():
    mod = benchmark()
    zero = arm(0)
    reserved = arm(65536)
    for sample in reserved["samples"]:
        sample["control"]["timings"].update(execution_seconds=0.01, total_seconds=0.02)
    summary = mod.growth_comparison_summary(zero, reserved, "reserved-first", 1)
    assert summary["all_results_match"]
    assert summary["warm_execution_ratio_reserved_over_zero"] == pytest.approx(0.5)
    assert summary["configurations"]["zero"] != summary["configurations"]["reserved"]
    assert summary["peak_process_rss_bytes"] == dict(zero=64 << 20, reserved=64 << 20)
    assert summary["ordinary_warm"]["reserved"]["total_seconds"]["median_seconds"] == 0.02


@pytest.mark.parametrize(
    "kind",
    [
        "purpose",
        "reserve",
        "guest",
        "hash",
        "guard",
        "missing",
        "rss",
        "compile",
        "order",
        "pair_count",
        "nested",
    ],
)
def test_unbound_or_malformed_growth_evidence_is_rejected(kind):
    from oeis_learn.synthesis.contracts import identity

    zero = arm(0)
    reserved = arm(65536)
    order = "zero-first"
    config = reserved["engine_configuration"]
    m = config["manifest"]
    if kind == "purpose":
        m["purpose"] = "qualification-growth-control"
    elif kind == "reserve":
        m["dynamic_memory_reserved_for_growth"] = 1 << 30
    elif kind == "guest":
        m["runtime_wasm_sha256"] = "sha256:" + "b" * 64
    elif kind == "hash":
        config["sha256"] = "sha256:" + "0" * 64
    elif kind == "guard":
        m["dynamic_memory_guard_size"] = 65536
    elif kind == "missing":
        del reserved["engine_configuration"]
    elif kind == "rss":
        reserved["peak_process_rss_bytes"] = True
    elif kind == "compile":
        reserved["runtime_compile_seconds"] = float("nan")
    elif kind == "order":
        order = "same-process"
    elif kind == "pair_count":
        reserved["samples"].pop()
    else:
        del reserved["samples"][0]["timings"]["stages"]["input_load"]["first_byte_probe"]
    if kind not in ["hash", "missing"]:
        config["sha256"] = identity(m)
    with pytest.raises((ValueError, KeyError)):
        benchmark().growth_comparison_summary(zero, reserved, order, 1)


@pytest.mark.parametrize(
    "field,value", [("fuel", 11), ("coverage", 0), ("reason", "memory_limit"), ("values", ["9"])]
)
def test_cross_setting_disagreement_cannot_hide_behind_native_parity(field, value):
    zero = arm(0)
    reserved = arm(65536)
    for label in ["result", "control"]:
        target = reserved["samples"][1][label]
        (target["result"] if label == "control" else target)[field] = value
    summary = benchmark().growth_comparison_summary(zero, reserved, "zero-first", 1)
    assert not summary["all_results_match"]
    assert field in summary["mismatches"][0]["fields"]


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--runtime-profile-only"],
        ["--runtime-profile-only", "--profile-input-load", "--gpu"],
        ["--runtime-profile-only", "--profile-input-load", "--compare-host-lookups"],
    ],
)
def test_host_growth_scope_is_rejected_before_engine(extra, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["qualify", "--compare-memory-growth", *extra, "--output", "unused"]
    )
    with pytest.raises(SystemExit) as exc:
        helper().main()
    assert exc.value.code == 2


@pytest.mark.parametrize("order", ["zero-first", "reserved-first"])
def test_separate_process_arm_order_and_control_selection(order, monkeypatch):
    mod = benchmark()
    seen = []

    def measure(case, batch, repetitions, **kwargs):
        reserve = kwargs.get("zero_growth_control", False)
        seen.append(reserve)
        return arm(0 if reserve else 65536)

    monkeypatch.setattr(mod, "runtime_measurements", measure)
    row = mod.memory_growth_measurements({}, 1, 3, order=order, failure_output="fixture.log")
    assert seen == ([True, False] if order == "zero-first" else [False, True])
    assert row["memory_growth"]["all_results_match"]
    assert row["memory_growth"]["zero"]["engine_configuration"] == configuration(0)
    assert row["engine_configuration"] == configuration(65536)


@pytest.mark.parametrize("kind", ["pass", "missing", "failed"])
def test_host_routes_only_growth_profile_and_archives_every_result(kind, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.synthesis.contracts import identity

    mod = helper()
    snapshot = {"fixture": "source"}
    output = tmp_path / "run"
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: snapshot)

    class Host:
        def __init__(self, args):
            self.engine = SimpleNamespace(name="podman")
            self.policy = dict(runtime={})

        def ensure(self):
            return "sha256:" + "a" * 64

    monkeypatch.setitem(sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=str))

    def stage(host, image, path, argv, **kw):
        assert kw == dict(seconds=600)
        assert argv == [
            "python",
            "scripts/benchmark_bigint.py",
            "--runtime-profile",
            "--profile-input-load",
            "--compare-memory-growth",
            "--output",
            "/work/output",
        ]
        mod.write(
            path / "benchmark.json",
            dict(
                status="FAIL" if kind == "failed" else "PASS",
                profile_input_load=True,
                compare_memory_growth=kind != "missing",
                source_sha256=identity(snapshot),
            ),
        )

    monkeypatch.setattr(mod, "container_stage", stage)
    monkeypatch.setattr(mod, "launch", lambda *a, **k: pytest.fail("unrelated work"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--runtime-profile-only",
            "--profile-input-load",
            "--compare-memory-growth",
            "--output",
            str(output),
        ],
    )
    assert mod.main() == (0 if kind == "pass" else 1)
    assert output.with_suffix(".zip").is_file()
    assert json.loads((output / "summary.json").read_text())["compare_memory_growth"] is True


def test_completed_arm_survives_later_native_failure(tmp_path, monkeypatch):
    mod = benchmark()
    calls = []

    def measure(*args, **kwargs):
        calls.append(kwargs["zero_growth_control"])
        if len(calls) == 2:
            raise ValueError("bad native JSON")
        return arm(0)

    monkeypatch.setattr(mod, "runtime_measurements", measure)
    progress = tmp_path / "growth-active.json"
    with pytest.raises(ValueError, match="bad native JSON"):
        mod.memory_growth_measurements(
            dict(name="fixture"),
            1,
            3,
            order="zero-first",
            failure_output=tmp_path / "failed-native.log",
            progress_output=progress,
        )
    saved = json.loads(progress.read_text())
    assert saved["status"] == "FAIL" and saved["active_arm"] == "reserved"
    assert saved["arms"]["zero"]["engine_configuration"] == configuration(0)


def test_latest_raw_native_record_survives_configuration_or_expected_value_failure(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    mod = benchmark()
    raw = json.dumps(arm(65536)).encode()
    capture = tmp_path / "growth-latest-native.log"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=raw, stderr=b""),
    )
    # The native answer is4 whereas this fixture's independently declared value is9.
    case = dict(
        name="mismatch",
        source="(program (return (int 4)))",
        indices=["0"],
        expected=["9"],
        expected_reason=None,
    )
    with pytest.raises(RuntimeError, match="independent"):
        mod.runtime_measurements(
            case,
            1,
            3,
            profile=True,
            profile_input_load=True,
            zero_growth_control=False,
            native_output=capture,
        )
    assert capture.read_bytes() == raw


def test_archived_v1_growth_reader_keeps_original_production_meaning():
    zero = arm(0)
    reserved = arm(65536)
    zero["engine_configuration"] = configuration(0, legacy=True)
    reserved["engine_configuration"] = configuration(65536, legacy=True)
    result = benchmark().growth_comparison_summary(zero, reserved, "zero-first", 1)
    assert result["version"] == "bigint-growth-reservation-comparison/v1"
    assert "production_setting" not in result
    assert result["all_results_match"]


def test_current_growth_collection_rejects_archived_configuration_labels(monkeypatch):
    mod = benchmark()

    def archived(*args, **kwargs):
        reserve = 0 if kwargs["zero_growth_control"] else 65536
        row = arm(reserve)
        row["engine_configuration"] = configuration(reserve, legacy=True)
        return row

    monkeypatch.setattr(mod, "runtime_measurements", archived)
    with pytest.raises(ValueError, match="v2 engine evidence"):
        mod.memory_growth_measurements({}, 1, 3, order="zero-first", failure_output="fixture.log")
