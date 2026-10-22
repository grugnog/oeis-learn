"""Profile evidence is diagnostic, bounded and independent of research authority."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oeis_learn.tracking.pipeline_profile import (
    Recorder,
    cpu_capacity,
    exclusive_seconds,
    gpu_sensor_path,
    interval_union,
)


def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/profile_bigint_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_wall_accounting_does_not_double_count():
    rows = [
        dict(start=0, end=10, name="generation", depth=0),
        dict(start=2, end=5, name="decode", depth=1),
        dict(start=3, end=4, name="step", depth=2),
    ]
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
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
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


def test_state_restoration_repeats_optimizer_and_rng_without_mutating_parent():
    import torch
    from oeis_learn.fresh.pipeline_profile import snapshot, restore, parameter_digest

    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW([*a.parameters(), *b.parameters()], lr=0.01)
    state = snapshot(a, b, optimizer)
    parent = parameter_digest(a, b)
    ends = []
    for _ in range(2):
        restore(state, a, b, optimizer)
        optimizer.zero_grad()
        b(a(torch.randn(4, 2))).sum().backward()
        optimizer.step()
        ends.append(parameter_digest(a, b))
    assert ends[0] == ends[1] != parent
    restore(state, a, b, optimizer)
    assert parameter_digest(a, b) == parent and not optimizer.state


def test_scoped_instrumentation_restores_hooks_on_failure(tmp_path):
    from oeis_learn.fresh.pipeline_profile import instrument
    from oeis_learn.synthesis.client import SynthesisClient

    before = SynthesisClient.request
    with pytest.raises(RuntimeError), Recorder(tmp_path, instrumented=True) as r:
        with instrument(r):
            assert SynthesisClient.request is not before
            raise RuntimeError("stop")
    assert SynthesisClient.request is before


def test_deadline_work_difference_not_reported_as_observer_speedup():
    from oeis_learn.fresh.pipeline_report import pair_attempts

    a = dict(
        prompt="p",
        mode="correct",
        position=0,
        seed=3,
        tokens_sha256="a",
        batch_size=8,
        generation={"complete": True},
        fitting={"complete": True},
        raw={},
    )
    b = dict(a, generation={"complete": False})
    report = pair_attempts(dict(ordinary=[a], telemetry=[b]))
    assert report["changed"] == 1 and report["matching"] == 0
    assert pair_attempts(dict(ordinary=[a], telemetry=[]))["unpaired"] == 1


def test_schedule_projection_keeps_unknown_costs_unknown():
    from oeis_learn.fresh.pipeline_report import schedule_projection

    result = schedule_projection(
        [], {}, {"learning": {"max_updates": 3000, "validation_every": 100}}, {}
    )
    assert result["total_seconds"] is None and result["amdahl_bound"] is None
    assert result["measured_weighted_seconds"]["training"] is None


def parent_fixture(tmp_path):
    from oeis_learn.fresh.common import write, digest, file_hash

    run, inputs, output = (tmp_path / x for x in ("parent", "data", "child"))
    output.mkdir()
    row = lambda i: dict(
        id=str(i),
        family=str(i % 4),
        indices=[str(n) for n in range(20)],
        values=["0"] * 20,
        horizon_indices=[str(n) for n in range(100)],
        tokens=[1],
    )
    examples = dict(
        tiny_ids=[str(i) for i in range(32)],
        views=dict(
            concrete=dict(
                training=[row(i) for i in range(512)],
                parameter_development=[row(i) for i in range(512, 640)],
                structural_development=[row(i) for i in range(640, 768)],
            )
        ),
    )
    write(inputs / "examples.json", examples)
    write(
        inputs / "private/synthetic-truth.json",
        {
            str(i): dict(indices=[str(n) for n in range(100)], values=["0"] * 100)
            for i in range(512)
        },
    )
    prepared = dict(
        native_admitted=True,
        files={
            str(p.relative_to(inputs)): dict(sha256=file_hash(p)) for p in inputs.rglob("*.json")
        },
    )
    prepared["identity"] = digest(prepared)
    write(inputs / "prepared.json", prepared)
    write(run / "checkpoints/state/weights.pt", b"weights")
    sha = file_hash(run / "checkpoints/state/weights.pt")
    write(
        run / "checkpoints/state/checkpoint.json",
        dict(
            version="bigint-checkpoint/v1",
            binding=dict(contract="parent-contract", weights=sha),
            files=dict(weights=dict(path="state/weights.pt", sha256=sha)),
        ),
    )
    pointer = dict(
        path="state/checkpoint.json", sha256=file_hash(run / "checkpoints/state/checkpoint.json")
    )
    write(run / "checkpoint-identity.json", dict(selected=pointer))
    write(run / "status.json", dict(status="completed"))
    write(
        run / "run.json",
        dict(
            version="bigint-run/v1",
            final_access="sealed",
            inputs_identity=prepared["identity"],
            contract_sha256="parent-contract",
            recipe=dict(
                model=dict(seed=20261016),
                inputs=dict(profile="hybrid-chunks/v1"),
                learning=dict(target_view="literal"),
            ),
        ),
    )
    return run, inputs, output


def test_preparation_never_copies_truth_or_resume_state_into_gpu_input(tmp_path):
    from oeis_learn.fresh.common import read, file_hash

    run, inputs, output = parent_fixture(tmp_path)
    before = {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}
    runner().prepare(
        run, inputs, output, image="image", source="source", render_node="/dev/dri/renderD128"
    )
    work = read(output / "input/work.json")
    assert set(p.name for p in (output / "input").iterdir()) == {"work.json", "weights.pt"}
    assert set(work) == {"manifest", "recipe", "prompts", "training", "validation"}
    assert len(work["prompts"]) == 32
    assert "oracle_source" not in str(work)
    assert {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("damage", ["weights", "examples", "private/synthetic-truth", "checkpoint"])
def test_preparation_rejects_changed_bound_inputs(tmp_path, damage):
    run, inputs, output = parent_fixture(tmp_path)
    target = {
        "weights": run / "checkpoints/state/weights.pt",
        "examples": inputs / "examples.json",
        "private/synthetic-truth": inputs / "private/synthetic-truth.json",
        "checkpoint": run / "checkpoints/state/checkpoint.json",
    }[damage]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|mismatch"):
        runner().prepare(
            run, inputs, output, image="image", source="source", render_node="/dev/dri/renderD128"
        )


def test_reduced_portfolio_counts_are_explicit_and_default_is_unchanged(monkeypatch):
    import torch
    import time
    from oeis_learn.fresh import bigint_diagnostics as d

    class Encoder:
        def eval(self):
            pass

        def parameters(self):
            yield torch.zeros(1)

        def forward_from_sequences(self, rows, device):
            return torch.ones(len(rows), 20, 4)

    class Decoder:
        def eval(self):
            pass

    counts = []

    def portfolio(*a, **kw):
        counts.append(kw["attempt_count"])
        arm = dict(complete=True)
        return dict(
            complete=True,
            raw=None,
            rescue=None,
            equal_budget=None,
            arm_evidence=dict(raw=arm, rescue=arm, equal_budget=arm),
        )

    monkeypatch.setattr(d, "_portfolio", portfolio)
    monkeypatch.setattr(d, "_drive_portfolios", lambda decoder, generators: generators)
    rows = [
        dict(id=str(i), indices=[str(n) for n in range(20)], values=["0"] * 20) for i in range(8)
    ]
    for reduced in (False, True):
        counts.clear()
        d.generate(
            Encoder(),
            Decoder(),
            rows,
            dict(model=dict(seed=4)),
            None,
            deadline=time.monotonic() + 100,
            diagnostic_attempts=runner().workload()["attempts"] if reduced else None,
        )
        assert counts == ([8] * 8 + [1] * 16 + [8] * 8 if reduced else [8] * 32)


def test_gpu_role_rejects_source_before_loading_or_executing_weights(tmp_path, monkeypatch):
    from oeis_learn.fresh import pipeline_profile as p
    from oeis_learn.fresh.common import write, digest
    from oeis_learn.experiments import source_snapshot

    manifest = dict(
        plan=runner().workload(),
        recipe_sha256=digest({}),
        payload_sha256=digest(dict(prompts=[], training=[], validation=[])),
        source_sha256="wrong",
    )
    manifest["identity"] = digest(manifest)
    write(
        tmp_path / "input/work.json",
        dict(manifest=manifest, recipe={}, prompts=[], training=[], validation=[]),
    )
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: {})
    assert p.inside(tmp_path / "input", tmp_path / "out", "gpu") == 1
    assert (
        "image/source differs" in json.loads((tmp_path / "out/summary.json").read_text())["error"]
    )
