"""Profile evidence is diagnostic, bounded and independent of research authority."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oeis_learn.tracking.pipeline_profile import (
    Recorder, cpu_capacity, exclusive_seconds, gpu_sensor_path, interval_union,
)


def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/profile_bigint_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_wall_accounting_does_not_double_count():
    rows = [dict(start=0, end=10, name="generation", depth=0),
            dict(start=2, end=5, name="decode", depth=1),
            dict(start=3, end=4, name="step", depth=2)]
    assert exclusive_seconds(rows) == {"generation": 7, "decode": 2, "step": 1}
    assert interval_union([(0, 5), (2, 7), (9, 10)]) == 8


def test_capacity_respects_fractional_quota():
    assert cpu_capacity(16, "150000 100000", "0-7") == 1.5
    assert cpu_capacity(16, "max 100000", "0-3,8-11") == 8


def test_gpu_mapping_never_chooses_first_card(tmp_path):
    first = tmp_path / "renderD128/device"
    second = tmp_path / "renderD129/device"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    assert gpu_sensor_path("/dev/dri/renderD129", tmp_path) == second.resolve()
    assert gpu_sensor_path("/dev/dri/renderD130", tmp_path) is None


def test_recorder_retains_exception_and_stops_collector(tmp_path):
    r = Recorder(tmp_path, instrumented=True)
    with pytest.raises(RuntimeError), r:
        with r.phase("failed"):
            raise RuntimeError("expected")
    rows = [json.loads(s) for s in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[-1]["error"] == "RuntimeError"
    assert not r.thread.is_alive()
    assert json.loads((tmp_path / "telemetry.json").read_text())["samples"] >= 1


def test_host_help_and_scope_rejection_without_site_packages():
    module = runner()
    p = subprocess.run([sys.executable, "-S", module.__file__, "--help"], capture_output=True)
    assert p.returncode == 0
    p = subprocess.run([sys.executable, "-S", module.__file__, "--train"], capture_output=True)
    assert p.returncode != 0


def test_archive_allows_reports_only_and_reserves_core(tmp_path):
    module = runner()
    (tmp_path / "summary.json").write_text('{}')
    (tmp_path / "manifest.json").write_text('{}')
    (tmp_path / "private").mkdir()
    (tmp_path / "private/truth.json").write_text('"secret"')
    (tmp_path / "weights.pt").write_bytes(b"weights")
    (tmp_path / "trace.json").symlink_to(tmp_path / "private/truth.json")
    import zipfile
    with zipfile.ZipFile(module.archive(tmp_path)) as z:
        assert set(z.namelist()) == {"summary.json", "manifest.json", "archive-manifest.json"}


def test_frozen_plan_counts_budget_and_no_authority():
    plan = runner().workload()
    assert plan["updates_per_arm"] == 500
    assert plan["arms"] == ["ordinary", "telemetry"]
    assert plan["attempts"] == dict(correct=8, shuffled=1, absent=1, template=8)
    assert plan["qualification"] is False and plan["resume"] is False
    assert sum(plan["stage_caps"].values()) <= 5100
