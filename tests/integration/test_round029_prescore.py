"""A registered Tiny CPU worker stands in for the physical device in the host lifecycle."""

import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import zipfile

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.velocity_records import SESSION
from oeis_learn.pipeline_validation import progressive_round029 as controller
from oeis_learn.pipeline_validation import progressive_velocity_checkpoint_host as host
from oeis_learn.pipeline_validation import progressive_velocity_initialization as initial
from oeis_learn.pipeline_validation import progressive_velocity_prescore as freeze


def test_funded_zero_scope_freezes_all_trials_and_recovers_before_device(tmp_path, monkeypatch):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from oeis_learn.pipeline_validation import progressive_velocity_preparation as funded
    from oeis_learn.pipeline_validation import progressive_round028
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.pipeline_validation.progressive_velocity_donation import retained_qualification
    from oeis_learn.pipeline_validation.progressive_velocity_worker import disposable_batch

    source = Path(__file__).resolve().parents[2]
    rev = "b" * 40
    folder = tmp_path / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    pointer = seal(dict(grant=digest("fresh grant"), receipt=digest("fresh")))
    marker = seal(dict(grant=digest("checkpoint grant"), receipt=digest("checkpoint")))
    family = seal(dict(kind="families", grant=digest("families grant"), receipt=digest("families")))
    for path, value in (
        (host.broad.FRESH_MARKER, pointer),
        (host.MARKER, marker),
        ("preparation/supply/families/funded.json", family),
    ):
        atomic_write(folder / path, value)
    authority = digest("seed authority")
    seeds = pair_seeds(authority, 0)
    profile = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}),
        shape="smaller_width",
        seed=seeds["initialization"],
    )
    enc, dec, _ = construct(profile, device="cpu")
    weights = model_snapshot(enc, dec)
    origin = seal(
        dict(
            version="round029-initial-state/v1",
            seed_authority=authority,
            pair=0,
            initialization=seeds["initialization"],
            stream_seed=seeds["training_families"],
            ordinary=digest("stream"),
            broad=None,
            initial_weights=weights,
            common_policy=digest("policy"),
            codec=digest("codec"),
        )
    )
    batch = disposable_batch(profile, 16)
    view = seal(
        dict(
            streams=dict(
                ordinary=seal(
                    dict(
                        profile=profile,
                        initial_weights=weights,
                        rows=batch["rows"],
                        order=batch["order"],
                    )
                )
            ),
            fresh_start=origin["identity"],
        )
    )
    projection = tmp_path / "input/projection.json"
    atomic_write(projection, view)
    paid = host.matrix.design(view, microbatch=8, origin=pointer["receipt"])
    physical, image = digest("machine"), digest("image")
    monkeypatch.setattr(host, "inputs", lambda *_: (pointer, projection, physical, paid))
    monkeypatch.setattr(host, "physical_scope", lambda *a, **kw: physical)
    monkeypatch.setattr(host.broad, "archive_size", lambda *_: 1024)
    monkeypatch.setattr(host, "cached_runtime", lambda *_: dict(image=image))
    monkeypatch.setattr(host, "package_evidence", lambda *_: None)
    monkeypatch.setattr(controller, "candidate_revision", lambda *_: rev)
    monkeypatch.setattr(
        progressive_round028,
        "git",
        lambda _, *a: (
            None if "symbolic-ref" in a else controller.velocity_policy()["donor_revision"]
        ),
    )
    protected = "fixture-private-final-group"
    members = seal(
        dict(
            panels=dict(
                development=[dict(task="private-development", family="g", stratum="oeis")],
                development_retention=[
                    dict(task="private-retention", family="h", stratum="retention")
                ],
                assessment=[dict(task=protected, family=protected, stratum="oeis")],
            )
        )
    )
    families = dict(measurement=family["receipt"])
    monkeypatch.setattr(funded, "funded_inventory", lambda *a, **k: (families, members))
    owners = [
        dict(owner=n, incremental=dict(bytes=100_000, inodes=25))
        for n in ("first_account", "worker_staging", "learner_00")
    ]
    rows = [dict(owners=owners, device=tmp_path.stat().st_dev, reserved_bytes=0, reserved_inodes=0)]
    prices = dict(
        supported_wave=dict(
            methods=list(host.matrix.METHODS),
            omitted={m: ["unavailable"] for m in ("01", "02", "03")},
        ),
        fresh_start=dict(start=origin, identity=pointer["receipt"]),
        launch_preparation=dict(
            wave_space=dict(ordinary_initialization=rows), initialization_price=dict(tail_us=100)
        ),
        launch_estimate=dict(
            missing=[],
            expected_us=9000,
            tail_us=10000,
            followup_reserve_us=900,
            provisional_publication=dict(checkpoint_read_us=140),
            sources=dict(fresh=pointer["receipt"], checkpoint=marker["receipt"]),
        ),
    )
    options = dict(
        research_authority=None,
        scopes=dict(ordinary_initialization=dict(tail_us=100, tail_with_reserve_us=1000)),
        followup_reserve_us=900,
    )
    prices["launch_options"] = options
    info = dict(cuts=[], families=families, prices=prices, assessment="sealed", gpu_dispatches=0)
    monkeypatch.setattr(controller, "preflight", lambda *a, **kw: dict(inventory=info))
    machine = SimpleNamespace(
        policy=dict(free_disk_floor_bytes=0), engine=SimpleNamespace(image=lambda v: v)
    )
    calls = []

    def pack(root, source, area, out, revision, parent, source_hash, host, stage, seconds):
        calls.append("package")
        wheel, bootstrap = out / "candidate.whl", out / "bootstrap.py"
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr("fixture.py", "pass")
        bootstrap.write_text("pass")
        return seal(
            dict(parent=parent, wheel=reference(area, wheel), bootstrap=reference(area, bootstrap))
        )

    monkeypatch.setattr(host, "package_worker", pack)
    grant_path = freeze.authorize(tmp_path, rev, 15)
    grant = load(grant_path)
    captured = retained_qualification(tmp_path, grant)

    def stage(machine, source, ref, out, argv, *, gpu, record_lease, readonly_inputs, seconds):
        assert gpu and record_lease and ref == image and argv[4] == "ordinary-initialization"
        assert captured.is_file()
        assert all(
            dest.endswith((".whl", ".py", "request.json", "projection.json"))
            for _, dest in readonly_inputs
        )
        calls.append("initialization")
        req = load(next(p for p, dest in readonly_inputs if dest.endswith("request.json")))
        start = time.monotonic_ns()
        cut = initial.measure(view, req["design"], out, device="cpu")
        (out / "telemetry").mkdir()
        (out / "telemetry/samples.jsonl").write_text("")
        us = (time.monotonic_ns() - start) // 1000
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    request=req["identity"],
                    device="hip",
                    matrix=cut,
                    elapsed_us=us,
                    telemetry=usage_summary(out / "telemetry", us),
                    peak_device_bytes=1,
                    worker_cpu_us=1,
                )
            ),
        )
        atomic_write(out / "gpu-lease.json", dict(elapsed_us=us + 10))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=[
                        dict(Type="bind", Source=str(p), Destination=d, RW=False)
                        for p, d in readonly_inputs
                    ]
                    + [dict(Type="bind", Source=str(out), Destination="/work/output", RW=True)],
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=False)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))

    monkeypatch.setattr(host, "stage", stage)

    def invoke(action, grant=None):
        return controller.execute(
            SimpleNamespace(
                workspace=tmp_path,
                action=action,
                authority=grant,
                archive=None,
                qualification_archive=None,
                free_disk_gib=0,
            ),
            source,
            host_factory=lambda *a, **kw: machine,
        )

    original_space = freeze.dispatch_space

    def blocked(*a, **kw):
        raise host.broad.CapacityShortfall(
            "ordinary_initialization", [dict(deficit_inodes=1, deficit_bytes=0)], None
        )

    monkeypatch.setattr(freeze, "dispatch_space", blocked)
    first, code = invoke("qualify", grant_path)
    assert code == 2 and first["deficit_inodes"] == 1 and calls == ["package"], json.dumps(first)
    assert captured.is_file() and not (folder / "accepted.json").exists()
    before = captured.read_bytes()
    assert protected.encode() not in before and b"private-development" not in before
    monkeypatch.setattr(freeze, "dispatch_space", original_space)

    def action(name, grant=None):
        result, code = invoke(name, grant)
        assert code == 0, json.dumps(result)
        return result

    paid_result = action("qualify", grant_path)
    assert paid_result["charged_tokens"] == 0 and calls == ["package", "initialization"]
    assert (
        action("qualify", grant_path) == paid_result
        and action("qualify", grant_path) == paid_result
    )
    api = runpy.run_path(
        str(source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py")
    )
    bundle = api["donor_initialization"](action, tmp_path, rev, 15)
    assert bundle["status"] == "COMPLETE", bundle
    dest = tmp_path / "result.zip"
    api["write_readiness_zip"](dest, bundle)
    assert api["verify_initialization_zip"](dest, captured)["restores"] == 2
    other = tmp_path / "unrelated.zip"
    other.write_bytes(b"another pre-score package")
    with pytest.raises(ValueError, match="independently retained"):
        api["verify_initialization_zip"](dest, other)
    assert before == captured.read_bytes() and calls == ["package", "initialization"]
    assert len(list((folder / "registrations").glob("*/trial-*.json"))) == 7
    assert not list((folder / "registrations").glob("*/look-*.json"))
    captured.unlink()
    failed, code = invoke("verify")
    assert code == 1 and failed["mechanical_status"] == "FAILED" and not captured.exists()
