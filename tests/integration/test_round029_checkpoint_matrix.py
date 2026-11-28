"""All host clocks, sizes and receipts are synthetic. The learner runs its own small CPU model."""

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from tests.integration.test_round029_resume import profile


def test_four_recipe_checkpoint_matrix_saves_restores_and_prices_each_worker(tmp_path):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.pipeline_validation.progressive_velocity_worker import disposable_batch
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint as checkpoint

    model = profile()
    batch = disposable_batch(model, 16)
    enc, dec, _ = construct(model, device="cpu")
    ordinary = seal(
        dict(
            profile=model,
            initial_weights=model_snapshot(enc, dec),
            rows=batch["rows"],
            order=batch["order"],
        )
    )
    view = seal(dict(streams={"ordinary": ordinary}, fresh_start=digest("qualified start")))
    design = checkpoint.design(view, microbatch=8, origin=digest("paid fresh qualification"))
    measured = checkpoint.measure(view, design, tmp_path, device="cpu")
    rows = checkpoint.checked_methods(design, measured)
    assert set(rows) == {"00", "04", "05", "06"}
    assert sum(r["useful_tokens"] for r in rows.values()) == design["work"]["tokens"]
    for row in rows.values():
        assert row["updates"] == 2 and row["restored"] == row["written"]
        assert all(
            row[k] > 0
            for k in (
                "restore_us",
                "first_cut_us",
                "resumed_cut_us",
                "checkpoint_bytes",
                "checkpoint_inodes",
            )
        )
    assert {p.name for p in tmp_path.iterdir()} == {f"recipe-{m}.json" for m in rows}
    for method, row in rows.items():
        assert load(tmp_path / f"recipe-{method}.json") == checkpoint.treatment(design, method, row)
        assert all(v > 0 for v in row["storage_us"].values())
    assert (
        checkpoint.package(ordinary, "04", 32, design["origin"])["policy"]["microbatch_size"] == 16
    )
    changed = dict(rows["04"], restored=rows["00"]["restored"])
    with pytest.raises(ValueError):
        checkpoint.checked_methods(
            design, seal(dict(measured, methods=dict(rows, **{"04": changed})))
        )


@pytest.mark.parametrize("interrupt", [False, True])
def test_checkpoint_matrix_grant_routes_checks_capacity_and_replays_one_device_lease(
    tmp_path, monkeypatch, interrupt
):
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_round028, progressive_round029
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint_host as host
    from oeis_learn.pipeline_validation import progressive_velocity_timing_host as timing
    import zipfile
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary

    source = Path(__file__).resolve().parents[2]
    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    revision = "b" * 40
    pointer = seal(dict(grant=digest("paid grant"), receipt=digest("fresh receipt")))
    atomic_write(folder / host.broad.FRESH_MARKER, pointer)
    projection = tmp_path / "private" / "projection.json"
    atomic_write(projection, seal(dict(fresh_start=digest("start"))))
    image, physical = digest("immutable image"), digest("physical")
    methods = {
        m: dict(package=digest(m), first=5, second=10, batch=16 if m in {"04", "06"} else 32)
        for m in host.matrix.METHODS
    }
    plan = seal(dict(origin=pointer["receipt"], methods=methods, work=dict(tokens=40, attempts=8)))
    monkeypatch.setattr(host, "inputs", lambda *_: (pointer, projection, physical, plan))
    monkeypatch.setattr(host, "collect_snapshot", lambda *_: {"fixture": True})
    monkeypatch.setattr(timing, "collect_snapshot", host.collect_snapshot)
    monkeypatch.setattr(host, "physical_scope", lambda *a, **kw: physical)
    monkeypatch.setattr(host, "cached_runtime", lambda *_: {"image": image})
    monkeypatch.setattr(host, "package_evidence", lambda *_: None)
    monkeypatch.setattr(progressive_round029, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        progressive_round028,
        "git",
        lambda _, *a: (
            None
            if "symbolic-ref" in a
            else progressive_round029.velocity_policy()["donor_revision"]
        ),
    )
    machine = SimpleNamespace(
        policy={"free_disk_floor_bytes": 0}, engine=SimpleNamespace(image=lambda value: value)
    )
    calls = []
    blocked = [True]

    def check_capacity(root, area, revision, phase, floor):
        if blocked[0]:
            raise host.broad.CapacityShortfall(
                phase, [dict(deficit_bytes=0, deficit_inodes=9)], None
            )
        return [
            dict(
                available_bytes=30,
                required_bytes=18,
                reserved_bytes=2,
                deficit_bytes=0,
                available_inodes=10,
                required_inodes=2,
                reserved_inodes=1,
                deficit_inodes=0,
            )
        ]

    def pack(root, source, area, out, revision, parent, source_hash, worker, stage, seconds):
        calls.append("package")
        wheel, bootstrap = out / "candidate.whl", out / "candidate.py"
        wheel.write_text("fixture wheel")
        bootstrap.write_text("fixture bootstrap")
        return seal(
            dict(parent=parent, wheel=reference(area, wheel), bootstrap=reference(area, bootstrap))
        )

    def stage(machine, source, ref, out, argv, *, gpu, record_lease, readonly_inputs, seconds):
        assert gpu and record_lease and argv[4] == "checkpoint-matrix" and ref == image
        calls.append("device")
        request = load(next(p for p, dest in readonly_inputs if dest.endswith("/request.json")))
        rows = {
            m: dict(
                package=r["package"],
                batch=r["batch"],
                updates=2,
                useful_tokens=10,
                charged_tokens=10,
                written=digest(m),
                restored=digest(m),
                first_cut_us=2,
                resumed_cut_us=3,
                restore_us=1,
                checkpoint_inodes=3,
                checkpoint_bytes=100,
                storage_us=dict(
                    initial_save=1, first_save=1, second_save=1, resume_load=1, final_load=1
                ),
            )
            for m, r in methods.items()
        }
        for method, row in rows.items():
            atomic_write(out / f"recipe-{method}.json", host.matrix.treatment(plan, method, row))
            if interrupt and calls.count("device") == 1:
                atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
                raise ValueError("interrupted after the first treatment")
        (out / "telemetry").mkdir()
        (out / "telemetry/samples.jsonl").write_text("")
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    request=request["identity"],
                    device="hip",
                    matrix=seal(dict(design=plan["identity"], methods=rows)),
                    elapsed_us=30,
                    peak_device_bytes=9,
                    worker_cpu_us=12,
                    telemetry=usage_summary(out / "telemetry", 30),
                )
            ),
        )
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=[
                        dict(Type="bind", Destination=dest, Source=str(p), RW=False)
                        for p, dest in readonly_inputs
                    ]
                    + [dict(Type="bind", Destination="/work/output", Source=str(out), RW=True)],
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        atomic_write(out / "gpu-lease.json", dict(elapsed_us=35))

    monkeypatch.setattr(host, "capacity", check_capacity)
    monkeypatch.setattr(host, "package_worker", pack)
    monkeypatch.setattr(host, "stage", stage)
    authority = host.authorize(tmp_path, source, revision, 5)

    def invoke(name, grant=None):
        args = SimpleNamespace(
            workspace=tmp_path,
            archive=None,
            qualification_archive=None,
            action=name,
            authority=grant,
        )
        return progressive_round029.execute(args, source, host_factory=lambda *a, **kw: machine)

    refused, code = invoke("qualify", authority)
    assert (
        code == 2 and refused["deficit_inodes"] == 9 and refused["elapsed_us"] == 0 and not calls
    ), refused
    assert refused["diagnostics"]["capacity"]["mounts"][0]["deficit_inodes"] == 9
    blocked[0] = False
    api = runpy.run_path(
        str(source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py")
    )

    def action(name, grant=None):
        if name == "preflight":
            return dict(
                inventory=dict(
                    gpu_dispatches=0,
                    prices=dict(
                        checkpoint_cycles=host.inventory(tmp_path, source),
                        wave=dict(missing=["trajectory_startup_checkpoint_io_and_drain"]),
                    ),
                )
            )
        answer, code = invoke(name, grant)
        if code:
            raise SystemExit(json.dumps(answer))
        return answer

    monkeypatch.setitem(
        api["cli"].__globals__,
        "main",
        lambda rev, mins, mode: api["donor_checkpoints"](action, tmp_path, source, rev, mins),
    )
    if interrupt:
        partial = tmp_path / "interrupted.zip"
        assert api["cli"]([revision, "5", "checkpoints", "--output", str(partial)]) == 1
        with zipfile.ZipFile(partial) as archive:
            assert json.loads(archive.read("summary.json"))["status"] == "FAILED"
            assert set(archive.namelist()) == {
                "failure.json",
                "summary.json",
                "unverified-attempt-0-recipe-00.json",
            }
    paid, code = invoke("qualify", authority)
    expected_calls = ["package", "device", "device"] if interrupt else ["package", "device"]
    charged = 80 if interrupt else 40
    assert code == 0 and paid["charged_tokens"] == charged and paid["research_dispatches"] == 0, (
        paid
    )
    assert invoke("qualify", authority) == (paid, 0) and calls == expected_calls
    checked, code = invoke("verify")
    assert code == 0 and checked["last_committed_ref"] == paid["receipt"]["identity"], checked
    assert checked["verification_scope"] == "private_checkpoint_matrix"
    output = tmp_path / "matrix.zip"
    assert api["cli"]([revision, "5", "checkpoints", "--output", str(output)]) == 0
    with zipfile.ZipFile(output) as archive:
        assert all("recipe-" + m + ".json" in archive.namelist() for m in methods)
        assert json.loads(archive.read("summary.json"))["charged_tokens"] == charged
        assert json.loads(archive.read("capacity.json"))["device"][0]["required_bytes"] == 18
        forecast = json.loads(archive.read("forecast.json"))
        assert forecast["checkpoint_receipt"] == paid["receipt"]["identity"]
        assert forecast["wave"]["missing"] == ["trajectory_startup_checkpoint_io_and_drain"]
    assert calls == expected_calls
    marker = folder / host.MARKER
    marker.write_text(json.dumps(seal(dict(load(marker), receipt=digest("unrelated receipt")))))
    with pytest.raises(ValueError, match="paid boundary differs"):
        host.inventory(tmp_path, source)


def test_preflight_reports_only_cycles_from_the_qualified_stream(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round029 as host
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint_host as checkpoints

    original_forecast = host.velocity_forecast
    fresh = dict(identity=digest("synthetic fresh stream"), disposition="fresh_ordinary_only")

    def with_origin(info, policy):
        return dict(original_forecast(info, policy), fresh_start=fresh)

    monkeypatch.setattr(host, "velocity_forecast", with_origin)
    cycle = dict(receipt=digest("synthetic cycles"), origin=fresh["identity"])
    monkeypatch.setattr(checkpoints, "inventory", lambda *_: cycle)
    prices = host.preflight(tmp_path)["inventory"]["prices"]
    assert prices["checkpoint_cycles"] == cycle
    assert prices["wave"]["tail_us"] is None
    monkeypatch.setattr(
        checkpoints, "inventory", lambda *_: dict(cycle, origin=digest("unrelated stream"))
    )
    prices = host.preflight(tmp_path)["inventory"]["prices"]
    assert "checkpoint_cycles" not in prices
    assert "different fresh stream" in prices["checkpoint_reason"]


def test_control_history_cli_takes_only_zero_minutes_and_requires_a_new_zip(tmp_path):
    validation = Path(__file__).resolve().parents[2] / "specs/016-measure-learning-velocity/validation"
    op = runpy.run_path(str(validation / "qualify_continuation.py"))
    rev = "a" * 40
    with pytest.raises(SystemExit, match="zero grant minutes"):
        op["main"](rev, "1", "control-history")
    with pytest.raises(SystemExit, match="positive maximum"):
        op["main"](rev, "0", "checkpoints")
    with pytest.raises(SystemExit):
        op["cli"]([rev, "0", "control-history", "--output", str(tmp_path / "result.json")])


def test_one_grant_preserves_other_methods_and_replays_six_cpu_roles(tmp_path):
    import shutil
    import zipfile
    from copy import deepcopy
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.contracts import reference
    from tests.integration.test_round029_isolated_control_prices import (
        api,
        panel,
        EmptyClient,
        OPERATOR,
    )
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.generalization.progressive.velocity_records import SESSION

    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("synthetic/package.py", "pass")
    image = digest("fixture qualified image")
    runtime = dict(
        path=wheel,
        wheel=reference(tmp_path, wheel)["sha256"],
        source=digest("fixture source"),
        image=image,
    )
    members, visible, truth = panel()
    inputs = (
        members,
        visible,
        truth,
        resolve({"version": "bigint-bootstrap/v1"}),
        digest("fixture exposed roster"),
    )
    wave = dict(
        checkpoint_receipt=digest("fixture checkpoint"),
        operations={
            **{"D.full.search." + s: 32 for s in (*STRATA, "retention")},
            "D.full.search.startup": 1,
            "conventional_cpu_controls": 1,
        },
        missing=["conventional_cpu_controls", "paired_conditioning_sample"],
        known_expected_us=8,
        known_tail_us=10,
    )
    host = SimpleNamespace(policy={"free_disk_floor_bytes": 0, "runtime": {"engine": "fixture"}})
    calls = []
    fail = True

    def supervise(_host, source, im, out, argv, *, gpu, seconds, readonly_inputs):
        assert _host == host and not gpu and 0 < seconds <= 660 and im == image
        paths = {dest: path for path, dest in readonly_inputs}
        req = load(paths["/work/inputs/request.json"])
        data = load(paths["/work/inputs/data.json"])
        key = req["method"], req["phase"]
        calls.append(key)
        assert all(not path.is_dir() for path in paths.values())
        assert all(len(r["indices"]) == (20 if key[1] == "search" else 100) for r in data["rows"])
        assert ("/work/inputs/candidates.json" in paths) == (key[1] == "score")
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if fail and key == ("polynomial", "score"):
            raise ValueError("fixture stopped before scoring")
        work = tmp_path / ("role-" + str(len(calls)))
        (work / "output").mkdir(parents=True)
        (work / "inputs").mkdir()
        if key[1] == "score":
            shutil.copyfile(paths["/work/inputs/candidates.json"], work / "inputs/candidates.json")
        artifact = api["control_role"](work, req, data, EmptyClient())
        rel = artifact["artifact"]["path"]
        (out / rel).parent.mkdir(parents=True)
        shutil.copyfile(work / "output" / rel, out / rel)
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    request=req["identity"],
                    worker_us=1,
                    native=digest("fixture-native"),
                    **artifact,
                )
            ),
        )
        binds = [
            dict(Type="bind", Source=str(p), Destination=k, RW=False) for k, p in paths.items()
        ]
        binds.append(dict(Type="bind", Source=str(out), Destination="/work/output", RW=True))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=im,
                    Mounts=binds,
                    HostConfig=dict(
                        ReadonlyRootfs=True,
                        Privileged=False,
                        NetworkMode="none",
                        Devices=[],
                        DeviceRequests=[],
                    ),
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=False)
        )

    args = (
        tmp_path,
        OPERATOR.parents[3],
        folder,
        "a" * 40,
        12,
        runtime,
        inputs,
        wave,
        host,
        supervise,
    )
    partial = api["isolated_control_matrix"](*args)
    assert partial["status"] == "PARTIAL" and partial["errors"] == {
        "polynomial": dict(phase="score", error="ValueError")
    }, partial["errors"]
    assert (
        len(calls) == 6 and partial["files"]["forecast.json"]["measured_control_projection"] is None
    )
    assert {
        "control-template.json",
        "control-linear_recurrence.json",
        "polynomial-search.json",
    } <= partial["files"].keys()
    fail = False
    full = api["isolated_control_matrix"](*args)
    assert full["status"] == "COMPLETE" and full["grant"] == partial["grant"] and len(calls) == 7
    assert (
        full["charged_work"]["tokens"] == 0
        and full["elapsed_us"] + full["remaining_us"] == 12 * 60_000_000
    )
    assert full["files"]["forecast.json"]["measured_control_projection"]["missing"] == [
        "paired_conditioning_sample"
    ]
    again = api["isolated_control_matrix"](*args)
    assert (
        len(calls) == 7
        and again["charged_work"] == full["charged_work"]
        and again["receipt"] == full["receipt"]
    )
    assert "private/" not in json.dumps(full)
    target = tmp_path / "isolated-controls.zip"
    report = deepcopy(again)
    api["write_readiness_zip"](target, report)
    import hashlib

    with zipfile.ZipFile(target) as z:
        summary = json.loads(z.read("summary.json"))
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in summary["sha256"].items())
