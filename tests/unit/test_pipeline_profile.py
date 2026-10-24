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


def check_diagnostic_checkpoint_device_roundtrip(device, tmp_path):
    import torch
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.pipeline_profile import snapshot
    from oeis_learn.pipeline_validation.neural import checkpoint_roundtrip

    encoder=torch.nn.Linear(2,2).to(device)
    decoder=torch.nn.Linear(2,2).to(device)
    optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()])
    decoder(encoder(torch.ones(1,2,device=device))).sum().backward()
    optimizer.step()
    state=snapshot(encoder,decoder,optimizer)
    recipe=resolve({'version':'bigint-bootstrap/v1'})
    report=checkpoint_roundtrip(tmp_path/'diagnostic',recipe,state,'fixture',None)
    assert 'not research resumable' in report['authority']
    assert state['encoder']['weight'].device.type==device
    assert encoder.weight.device.type==device


def test_diagnostic_checkpoint_cpu_roundtrip(tmp_path):
    check_diagnostic_checkpoint_device_roundtrip('cpu',tmp_path)


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


def test_complete_host_path_without_site_packages(tmp_path):
    """Exercise preparation, real role audits/report/archive with a fake engine."""
    from oeis_learn.fresh.common import read, write, file_hash

    run, inputs, output = parent_fixture(tmp_path)
    output.rmdir()
    parent = read(run / "run.json")
    parent["recipe"]["learning"].update(max_updates=3000, validation_every=100)
    write(run / "run.json", parent)
    before = {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}
    code = r"""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

script, root = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(script.parents[1] / "src"), str(script.parent)]
spec = importlib.util.spec_from_file_location("pipeline_runner", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import foundation_local
from oeis_learn.tracking import pipeline_profile
from oeis_learn.fresh.common import read, write

commands, removed = [], []
class Engine:
    def command(self, **kw):
        commands.append(kw)
        return kw
    def call(self, argv, **kw):
        assert argv == ["inspect", commands[-1]["name"]]
        inspection = [dict(
            HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True,
                            Privileged=False, Devices=[]),
            Mounts=[dict(Type="bind", Destination=dst, RW=rw)
                    for _, dst, rw in commands[-1]["mounts"]],
        )]
        return SimpleNamespace(stdout=json.dumps(inspection))
    def remove(self, name):
        removed.append(name)
class Host:
    def __init__(self, args):
        self.policy = dict(render_node="/dev/dri/renderD128", cpus=16, memory_bytes=1 << 30)
        self.engine = Engine()
    def ensure(self):
        return "sha256:" + "a" * 64
    def select_gpu(self):
        pass
    def gpu_lock(self):
        return nullcontext()
def launch(command, output, **kw):
    mode = command["argv"][3]
    devices = json.loads(command["environment"]["OEIS_EXPECTED_PIPELINE_DEVICES"])
    assert bool(devices) == (mode != "scorer")
    write(output / "devices.json", dict(version="pipeline-device-view/v1", devices=devices))
    write(output / "summary.json", dict(status="COMPLETE", updates={}, work={}, validation={}))
    if mode == "gpu":
        write(output / "selections.json", {})
        write(output / "attempts.json", dict(ordinary=[], telemetry=[]))
sys.argv = [str(script), "--run", str(root / "parent"), "--inputs", str(root / "data"),
            "--output", str(root / "child")]
with patch.object(foundation_local, "Host", Host), \
     patch.object(foundation_local, "relative", str), \
     patch.object(pipeline_profile, "character_device", return_value=dict(major=1, minor=3)), \
     patch.object(module, "launch", launch):
    assert module.main() == 0, read(root / "child/summary.json")
assert [x["argv"][3] for x in commands] == ["preflight", "gpu", "scorer"]
assert len(removed) == 3
assert read(root / "child/summary.json")["status"] == "COMPLETE"
assert (root / "child/timeline.html").is_file()
assert (root / "child.zip").is_file()
assert not {"yaml", "torch", "numpy"}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, runner().__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()} == before


def test_tracking_public_exports_remain_compatible():
    from oeis_learn import tracking
    from oeis_learn.tracking.run_manager import RunContext, RunManager

    assert tracking.RunManager is RunManager
    assert tracking.RunContext is RunContext
    with pytest.raises(AttributeError, match="no_such_export"):
        tracking.no_such_export


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


def test_scorer_keeps_independent_arm_clients_and_original_order(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.fresh import bigint_worker, bigint_diagnostics
    from oeis_learn.fresh.pipeline_profile import scorer_work
    from oeis_learn.fresh.common import write, digest, file_hash

    instances, calls = [], []

    class Client:
        def __init__(self, settings, folder):
            self.folder = folder
            self.worker = SimpleNamespace(stats={})
            self.caches = SimpleNamespace(metrics=lambda: {})
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def score(selections, truth, client, **kw):
        calls.append((str(client.folder), list(truth)))
        assert set(selections) == set(truth)
        return dict(outcomes={}, splits={})

    monkeypatch.setattr(bigint_worker, "client", Client)
    monkeypatch.setattr(bigint_diagnostics, "score", score)
    truth = {str(i): {} for i in range(32)}
    write(tmp_path / "input/truth.json", truth)
    write(tmp_path / "input/selections.json", {arm: truth for arm in ("ordinary", "telemetry")})
    manifest = dict(
        identity="analysis",
        plan=runner().workload(),
        truth_sha256=digest(truth),
        prompt_ids=list(truth),
    )
    work = dict(
        manifest=manifest,
        recipe={},
        selections_sha256=file_hash(tmp_path / "input/selections.json"),
    )
    summary = {}
    scorer_work(tmp_path / "input", tmp_path / "out", work, summary)
    assert len(instances) == 2 and len(calls) == 8
    for arm in ("ordinary", "telemetry"):
        assert [ids for folder, ids in calls if folder.endswith(arm)] == [
            list(truth)[i : i + 8] for i in range(0, 32, 8)
        ]


@pytest.mark.parametrize("bad_audit", [False, True])
def test_role_isolation_audit_and_cleanup_run_even_after_failure(tmp_path, monkeypatch, bad_audit):
    from types import SimpleNamespace

    module = runner()
    calls = []
    inspection = [
        dict(
            HostConfig=dict(
                NetworkMode="host" if bad_audit else "none",
                ReadonlyRootfs=True,
                Privileged=False,
                Devices=[],
            ),
            Mounts=[
                dict(Type="bind", Destination="/work/input", RW=False),
                dict(Type="bind", Destination="/work/output", RW=True),
            ],
        )
    ]

    class Engine:
        def command(self, **kw):
            calls.append(kw)
            return ["fake"]

        def call(self, *a, **kw):
            return SimpleNamespace(stdout=json.dumps(inspection))

        def remove(self, name):
            calls.append("removed")

    (tmp_path / "input").mkdir()
    monkeypatch.setattr(
        module,
        "launch",
        lambda *a, **kw: (tmp_path / "out/devices.json").write_text(
            json.dumps(dict(version="pipeline-device-view/v1", devices={}))
        ),
    )
    host = SimpleNamespace(engine=Engine(), policy=dict(memory_bytes=1 << 30, cpus=16))
    if bad_audit:
        with pytest.raises(RuntimeError, match="isolation"):
            module.container_stage(
                host,
                "image",
                tmp_path / "input",
                tmp_path / "out",
                mode="scorer",
                seconds=5,
                gpu=False,
            )
    else:
        module.container_stage(
            host, "image", tmp_path / "input", tmp_path / "out", mode="scorer", seconds=5, gpu=False
        )
    assert calls[-1] == "removed"
    assert calls[0]["gpu"] is False and calls[0]["mounts"][0][2] is False
    assert json.loads((tmp_path / "out/cleanup.json").read_text())["within_bound"]


def test_external_watchdog_kills_blocked_role(tmp_path):
    with pytest.raises(TimeoutError, match="external deadline"):
        runner().launch(
            [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path, seconds=0.03
        )


def test_report_exports_timeline_without_inventing_missing_gpu_counters(tmp_path):
    from oeis_learn.fresh.common import write
    from oeis_learn.fresh.pipeline_report import report

    write(
        tmp_path / "input/work.json",
        dict(recipe=dict(learning=dict(max_updates=3000, validation_every=100))),
    )
    write(tmp_path / "gpu/summary.json", dict(updates={}, work={}, validation={}))
    write(tmp_path / "gpu/attempts.json", dict(ordinary=[], telemetry=[]))
    with Recorder(tmp_path / "gpu/arms/telemetry/group-0", instrumented=True) as r:
        with r.phase("portfolio", group=0):
            pass
    result = report(tmp_path)
    assert result["gpu_busy"] == {}
    assert (tmp_path / "timeline.html").is_file()
    assert result["schedule_weighted"]["total_seconds"] is None


@pytest.mark.parametrize("role_failure", [None, "preflight", "gpu", "scorer"])
def test_host_exports_success_or_partial_failure_without_parent_writes(
    tmp_path, monkeypatch, role_failure
):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from oeis_learn.fresh.common import read, write, file_hash
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.fresh import pipeline_report

    run, inputs, output = parent_fixture(tmp_path)
    output.rmdir()
    parent_hashes = {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}
    module = runner()
    module.ROOT = tmp_path

    class Host:
        def __init__(self, args):
            self.policy = dict(render_node="/dev/dri/renderD128")

        def ensure(self):
            return "sha256:" + "a" * 64

        def select_gpu(self):
            pass

        def gpu_lock(self):
            return nullcontext()

    monkeypatch.setitem(sys.modules, "foundation_local", SimpleNamespace(Host=Host, relative=str))
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda root: {})
    monkeypatch.setattr(pipeline_report, "report", lambda output: dict(mock=True))
    calls = []

    def stage(host, image, source, destination, *, mode, **kw):
        destination.mkdir()
        calls.append(mode)
        if mode in {"gpu", "preflight"}:
            assert not (source / "truth.json").exists()
            if role_failure == mode:
                raise RuntimeError("injected role failure")
            if mode == "gpu":
                write(destination / "selections.json", {})
        else:
            assert not (source / "weights.pt").exists()
            work = read(source / "work.json")
            assert work["selections_sha256"] == file_hash(source / "selections.json")
            if role_failure == mode:
                raise RuntimeError("injected scorer failure")
        write(destination / "summary.json", dict(status="COMPLETE"))

    monkeypatch.setattr(module, "container_stage", stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [module.__file__, "--run", str(run), "--inputs", str(inputs), "--output", str(output)],
    )
    assert module.main() == (1 if role_failure else 0)
    expected_calls = ["preflight", "gpu", "scorer"]
    if role_failure:
        expected_calls = expected_calls[: expected_calls.index(role_failure) + 1]
    assert calls == expected_calls
    assert {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()} == parent_hashes
    import zipfile

    with zipfile.ZipFile(output.with_suffix(".zip")) as z:
        assert "summary.json" in z.namelist()
        assert not any(
            "/private/" in "/" + n or "/input/" in "/" + n or n.endswith(".pt")
            for n in z.namelist()
        )


def test_podman_empty_devices_requires_matching_actual_device_attestation():
    from oeis_learn.tracking.pipeline_profile import audit_device_view

    expected = {
        "/dev/kfd": {"major": 235, "minor": 0},
        "/dev/dri/renderD128": {"major": 226, "minor": 128},
    }
    view = dict(version="pipeline-device-view/v1", devices=expected)
    assert audit_device_view([], view, expected) == "role-character-devices"
    with pytest.raises(ValueError, match="device"):
        audit_device_view([], dict(view, devices={}), expected)
    with pytest.raises(ValueError, match="device"):
        audit_device_view(
            [],
            dict(
                view,
                devices=dict(expected, **{"/dev/dri/renderD129": {"major": 226, "minor": 129}}),
            ),
            expected,
        )
    with pytest.raises(ValueError, match="device"):
        audit_device_view([dict(PathOnHost="/dev/kfd")], view, expected)
    with pytest.raises(ValueError, match="device"):
        audit_device_view([], view, {})


def test_trace_cap_failure_preserves_explicit_partial_evidence(tmp_path, monkeypatch):
    import torch
    from oeis_learn.fresh.pipeline_profile import trace_probe

    class Profiler:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def export_chrome_trace(self, path):
            with open(path + ".tmp", "wb") as f:
                f.truncate(40 << 20)

    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Profiler())
    result = trace_probe(tmp_path, "capped", lambda step: None)
    assert result["status"] == "UNAVAILABLE"
    assert "byte cap" in result["reason"]
    assert result["partial"]["bytes"] == 40 << 20
    assert (tmp_path / "traces/capped.partial.json").stat().st_size == 40 << 20


def test_scheduled_trace_exports_real_cpu_operators_but_never_claims_hip(tmp_path):
    import torch
    from oeis_learn.fresh.pipeline_profile import trace_probe

    def operation(step):
        for i in range(5):
            torch.ones(16) @ torch.ones(16)
            step()

    result = trace_probe(
        tmp_path, "scheduled", operation, wait_steps=1, warmup_steps=1, active_steps=1
    )
    assert result["status"] == "INCOMPLETE" and not result["gpu_kernels"]
    assert result["schedule"]["active_steps"] == 1
    assert result["bytes"] < 40 << 20


def test_trace_sampler_step_hooks_preserve_budget_and_restore_method(tmp_path, monkeypatch):
    from oeis_learn.decoder.sampler import WatProgramSampler
    from oeis_learn.fresh.pipeline_profile import instrument

    calls = []

    def sample(self, memory, seeds, *, budget_hook):
        for i in range(3):
            budget_hook(i)
        return seeds

    monkeypatch.setattr(WatProgramSampler, "sample_trajectories", sample)
    obj = object.__new__(WatProgramSampler)
    with Recorder(tmp_path, instrumented=False) as r:
        with instrument(r, trace=True, profiler_step=lambda: calls.append("step")):
            assert obj.sample_trajectories(None, [37], budget_hook=lambda i: calls.append(i)) == [
                37
            ]
    assert calls == ["step", 0, "step", 1, "step", 2]
    assert WatProgramSampler.sample_trajectories is sample


def test_ordinary_sampler_never_receives_profiler_step_hook(tmp_path, monkeypatch):
    from oeis_learn.decoder.sampler import WatProgramSampler
    from oeis_learn.fresh.pipeline_profile import instrument

    before = WatProgramSampler.sample_trajectories
    with Recorder(tmp_path, instrumented=False) as r:
        with instrument(r, profiler_step=lambda: pytest.fail("ordinary trace hook invoked")):
            assert WatProgramSampler.sample_trajectories is before


def test_device_view_rejects_noncharacter_and_wrong_major_minor(tmp_path):
    from oeis_learn.tracking.pipeline_profile import character_device, audit_device_view

    file = tmp_path / "regular"
    file.touch()
    with pytest.raises(ValueError, match="character device"):
        character_device(file)
    with pytest.raises(ValueError, match="device"):
        audit_device_view(
            [],
            dict(version="pipeline-device-view/v1", devices={"/dev/kfd": dict(major=1, minor=3)}),
            {"/dev/kfd": dict(major=235, minor=0)},
        )


def test_preflight_rejects_cpu_before_model_or_update_work(tmp_path, monkeypatch):
    import torch
    from oeis_learn.fresh.pipeline_profile import preflight_work

    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="actual HIP"):
        preflight_work(tmp_path, tmp_path, {}, {})


def test_host_startup_gap_is_separate_from_container_work(tmp_path):
    from oeis_learn.fresh.common import write
    from oeis_learn.fresh.pipeline_report import host_timings

    write(tmp_path / "gpu/host-timing.json", dict(seconds_through_cleanup=985.93))
    write(
        tmp_path / "gpu/container-inspect.json",
        [
            dict(
                State=dict(
                    StartedAt="2026-10-22T22:41:39+00:00", FinishedAt="2026-10-22T22:54:03+00:00"
                )
            )
        ],
    )
    r = host_timings(tmp_path)["gpu"]
    assert r["container_lifetime_seconds"] == 744
    assert r["host_minus_container_seconds"] == pytest.approx(241.93)


def check_baseline_replay_role(device,tmp_path,*,batching=False):
    """Start with real frozen threads, without changing pytest's global pools."""
    import subprocess
    import sys
    from pathlib import Path
    import torch
    before=(torch.get_num_threads(),torch.get_num_interop_threads())
    code=('import sys; from pathlib import Path; import torch; '
        'torch.set_num_interop_threads(1); '
        'from tests.unit.test_pipeline_profile import _check_baseline_replay_role; '
        '_check_baseline_replay_role(sys.argv[1],Path(sys.argv[2]),batching=sys.argv[3]=="batching")')
    subprocess.run([sys.executable,'-c',code,device,str(tmp_path),'batching' if batching else 'bookkeeping'],
        cwd=Path(__file__).resolve().parents[2],check=True,timeout=180)
    assert (torch.get_num_threads(),torch.get_num_interop_threads())==before


def _check_baseline_replay_role(device,tmp_path,*,batching=False):
    """Actual tensor/reference files and role functions; tiny frozen fixture weights."""
    import time
    import torch
    from unittest.mock import patch
    from tests.integration.test_bigint_decoding import model
    from oeis_learn.pipeline_validation.replay import prepare_inputs,reference_role,replay_role,cached_replay,verify_outputs,tensor_record
    from oeis_learn.pipeline_validation.identity import file_hash,seal
    candidate_cached=cached_replay
    if batching:
        from oeis_learn.pipeline_validation.replay_batching import prepare_inputs,reference_role,batch_role as replay_role,batch_cached
        def cached_replay(decoder,memory,plan,deadline):
            decoder.cache_bookkeeping='hoisted/v1'
            return batch_cached(decoder,memory,plan,deadline,batch_size=8)
        def candidate_cached(decoder,memory,plan,deadline):
            decoder.cache_bookkeeping='hoisted/v1'
            return batch_cached(decoder,memory,plan,deadline,batch_size=16)

    assert torch.get_num_interop_threads()==1
    def already_configured(count):
        # These role functions normally run once per isolated process. The
        # fixture calls several in one child; PyTorch permits setting this pool
        # only once. Check the real setting instead of inventing observations.
        assert count==torch.get_num_interop_threads()==1

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(1,16)
        def forward_from_sequences(self,values,device):
            return self.linear(torch.tensor(values,dtype=torch.float32,device=device)[:,:,None])

    def models(recipe,selected):
        return Encoder().to(selected),model().to(selected),{}

    torch.manual_seed(20261024)
    weights=tmp_path/'weights.pt'
    torch.save({'encoder':Encoder().state_dict(),'decoder':model().state_dict()},weights)
    before=file_hash(weights)
    prepared=tmp_path/'prepared';prepared.mkdir()
    prepare_inputs({'fixture':'frozen'},[{'id':str(i),'values':[str(n+i if batching else n) for n in range(20)]} for i in range(32 if batching else 8)],weights,prepared,source='fixture-source')
    inputs=prepared/'input';reference=tmp_path/'reference';reference.mkdir()
    cfg=dict(source='fixture-source',contract='fixture-contract',options=dict(seconds=570,memory_ceiling=1<<30,render_node=None))
    if batching:cfg['options'].update(batch_size=8,memory_ceiling=32<<30)
    with patch('oeis_learn.fresh.bigint_worker.models',models),patch.object(torch,'set_num_interop_threads',side_effect=already_configured):
        result=reference_role(inputs,reference,cfg)
        ref=json.loads((reference/'reference.json').read_text())
        assert result['reference']==ref['identity']
        payload=torch.load(reference/'private/reference.pt',weights_only=True)
        assert ref['memory']==tensor_record(payload['memory'])
        decoder=model()
        decoder.load_state_dict(torch.load(weights,weights_only=True)['decoder'])
        plan=json.loads((inputs/'work.json').read_text())['manifest']['plan']
        out=cached_replay(decoder,payload['memory'],plan,time.monotonic()+570)
        count=160 if batching else 12
        assert len(verify_outputs(out,payload['expected']))==count
        decoder.cache_bookkeeping='hoisted/v1'
        hoisted=candidate_cached(decoder,payload['memory'],plan,time.monotonic()+570)
        assert len(verify_outputs(hoisted,payload['expected']))==count
        missing=dict(out);missing.pop(next(iter(missing)))
        with pytest.raises(ValueError):verify_outputs(missing,payload['expected'])
        bad={k:v.clone() for k,v in out.items()};bad[next(iter(bad))].fill_(float('nan'))
        with pytest.raises(RuntimeError,match='nonfinite'):verify_outputs(bad,payload['expected'])
        with pytest.raises(TimeoutError):cached_replay(decoder,payload['memory'],plan,time.monotonic()-1)
        with pytest.raises(TimeoutError):verify_outputs(out,payload['expected'],deadline=time.monotonic()-1)
        if device=='cuda':
            import shutil
            shutil.copyfile(reference/'reference.json',inputs/'reference.json')
            shutil.copyfile(reference/'private/reference.pt',inputs/'reference.pt')
            output=tmp_path/'gpu';output.mkdir()
            result=replay_role(inputs,output,cfg)
            assert result['status']=='COMPLETE' and result['boundaries']==count
            assert result['torch_threads']=={'intraop':2,'interop':1}
            assert result['reference']==ref['identity'] and result['device']['hip']
            # Same public child/slot parser and actual role used by the operator;
            # tiny fixture weights are correctness evidence, not performance.
            from oeis_learn.pipeline_validation.replay import TRIALS,replay_report
            from oeis_learn.pipeline_validation.replay_comparison import freeze_comparison,comparison_role,SLOTS
            if batching:
                from oeis_learn.pipeline_validation.replay_batching import freeze_batching as freeze_comparison,comparison_role,BATCH_SLOTS as SLOTS
            baseline={key:dict(state='PASS',identity=key,observations={'result':seal(dict(result,
                resources={'qualified':True,'ceiling_bytes':cfg['options']['memory_ceiling'],'role_peak_upper_bytes':1<<30}))}) for key in TRIALS}
            child=freeze_comparison(replay_report(baseline,ref['identity'],ref['workload']),baseline,ref)
            (inputs/'comparison-contract.json').write_text(json.dumps(child))
            comparison_config=dict(cfg,options=dict(cfg['options'],seconds=child['limits']['internal_seconds'],comparison=child['identity']))
            for slot in SLOTS[1:3]:
                comparison_config['options']['slot']=slot['id']
                if batching:comparison_config['options']['batch_size']=slot['batch_size']
                (inputs/'comparison-slot.json').write_text(json.dumps(slot))
                output=tmp_path/slot['arm'];output.mkdir()
                compared=comparison_role(inputs,output,comparison_config)
                assert compared['bookkeeping']==slot['bookkeeping']
                assert compared['comparison']==child['identity'] and compared['boundaries']==count
            bad_slot=dict(SLOTS[2],bookkeeping='baseline')
            (inputs/'comparison-slot.json').write_text(json.dumps(bad_slot))
            output=tmp_path/'bad-slot';output.mkdir()
            with pytest.raises(ValueError,match='slot/source/validation/deadline'):
                comparison_role(inputs,output,comparison_config)
            changed=seal(dict(ref,contract='changed'))
            (inputs/'reference.json').write_text(json.dumps(changed))
            output=tmp_path/'changed';output.mkdir()
            with pytest.raises(ValueError,match='changed replay reference'):replay_role(inputs,output,cfg)
    assert file_hash(weights)==before


@pytest.mark.parametrize('batching',[False,True])
def test_baseline_replay_cpu_reference_and_completed_work(tmp_path,batching):
    check_baseline_replay_role('cpu',tmp_path,batching=batching)


def check_hardware_capacity_role(device,tmp_path):
    """Fresh real thread pools and real128-row tensors; no fixture performance claim."""
    import torch
    before=(torch.get_num_threads(),torch.get_num_interop_threads())
    code=('import sys; from pathlib import Path; import torch; torch.set_num_interop_threads(1); '
        'from tests.unit.test_pipeline_profile import _check_hardware_capacity_role; '
        '_check_hardware_capacity_role(sys.argv[1],Path(sys.argv[2]))')
    subprocess.run([sys.executable,'-c',code,device,str(tmp_path)],cwd=Path(__file__).resolve().parents[2],check=True,timeout=180)
    assert (torch.get_num_threads(),torch.get_num_interop_threads())==before


def _check_hardware_capacity_role(device,tmp_path):
    import shutil
    import time
    import torch
    from unittest.mock import patch
    from tests.integration.test_bigint_decoding import model
    from oeis_learn.pipeline_validation.hardware_ladder import prepare_inputs,reference_role,numeric_role
    from oeis_learn.pipeline_validation.replay_batching import batch_cached
    from oeis_learn.pipeline_validation.replay import verify_outputs,tensor_record
    from oeis_learn.pipeline_validation.identity import file_hash,seal
    assert torch.get_num_interop_threads()==1
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(1,16)
        def forward_from_sequences(self,values,device):
            return self.linear(torch.tensor(values,dtype=torch.float32,device=device)[:,:,None])
    def models(recipe,selected):return Encoder().to(selected),model().to(selected),{}
    def configured(count):assert count==torch.get_num_interop_threads()==1
    torch.manual_seed(20261024)
    weights=tmp_path/'weights.pt';torch.save({'encoder':Encoder().state_dict(),'decoder':model().state_dict()},weights)
    before=file_hash(weights);prepared=tmp_path/'prepared';prepared.mkdir()
    prepare_inputs({'fixture':'frozen'},[{'id':str(i),'values':[str(n+i) for n in range(20)]} for i in range(32)],weights,prepared,source='fixture-source')
    inputs=prepared/'input';reference=tmp_path/'reference';reference.mkdir()
    cfg=dict(source='fixture-source',contract='fixture-contract',options=dict(seconds=570,memory_ceiling=32<<30,
        render_node=None,batch_size=16,measurement='aggregate1Hz/v1'))
    with patch('oeis_learn.fresh.bigint_worker.models',models),patch.object(torch,'set_num_interop_threads',side_effect=configured):
        reference_role(inputs,reference,cfg)
        ref=json.loads((reference/'reference.json').read_text());payload=torch.load(reference/'private/reference.pt',weights_only=True)
        memory=payload['memory'];assert len(memory)==128 and memory.is_contiguous()
        assert memory.untyped_storage().nbytes()==memory.numel()*memory.element_size()
        assert torch.equal(memory[:32],memory[32:64])
        assert ref['memory']==tensor_record(memory) and len(payload['expected'])==640
        plan=json.loads((inputs/'work.json').read_text())['manifest']['plan']
        if device=='cpu':
            decoder=model();decoder.load_state_dict(torch.load(weights,weights_only=True)['decoder']);decoder.cache_bookkeeping='hoisted/v1'
            for batch in (16,32,64,128):
                outputs=batch_cached(decoder,memory,plan,time.monotonic()+570,batch_size=batch)
                assert len(verify_outputs(outputs,payload['expected']))==640
            with pytest.raises(TimeoutError):batch_cached(decoder,memory,plan,time.monotonic()-1,batch_size=16)
        else:
            shutil.copyfile(reference/'reference.json',inputs/'reference.json');shutil.copyfile(reference/'private/reference.pt',inputs/'reference.pt')
            for batch in (16,32,64,128):
                output=tmp_path/str(batch);output.mkdir()
                options=dict(cfg['options'],batch_size=batch,measurement='ordinary/v1' if batch==128 else 'aggregate1Hz/v1')
                result=numeric_role(inputs,output,dict(cfg,options=options))
                assert result['status']=='COMPLETE' and result['boundaries']==640 and result['device']['hip']
                assert result['torch_threads']=={'intraop':2,'interop':1} and result['measurement_mode']==options['measurement']
            (inputs/'reference.json').write_text(json.dumps(seal(dict(ref,contract='changed'))))
            output=tmp_path/'changed';output.mkdir()
            with pytest.raises(ValueError,match='changed replay reference'):numeric_role(inputs,output,cfg)
    assert file_hash(weights)==before


def test_hardware_capacity128_cpu_reference_and_batches(tmp_path):
    check_hardware_capacity_role('cpu',tmp_path)
