"""A registered Tiny CPU worker stands in for the physical device in the host lifecycle."""

import json
from pathlib import Path
import runpy
import subprocess
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
from oeis_learn.pipeline_validation.progressive_velocity_launch import launch_options


def launch_fixture():
    mount = dict(
        device=7,
        available_bytes=1000,
        required_bytes=700,
        reserved_bytes=100,
        available_inodes=80,
        required_inodes=50,
        reserved_inodes=3,
        deficit_bytes=0,
        deficit_inodes=0,
    )
    return launch_options(
        dict(expected_us=9000, tail_us=10000, followup_reserve_us=900),
        dict(
            ordinary_cut_price=dict(expected_us=500, tail_us=600),
            initialization_price=dict(expected_us=90, tail_us=100),
            wave_space={
                n: [mount] for n in ("wave", "first_ordinary_cut", "ordinary_initialization")
            },
        ),
    )


def test_partial_initialization_checks_capacity_and_original_balance(tmp_path):
    source = Path(__file__).resolve().parents[2]
    api = runpy.run_path(
        str(source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py")
    )
    output = tmp_path / "partial.zip"
    launch = launch_fixture()
    failure = dict(
        mechanical_status="FAILED", stage="package", elapsed_us=10, remaining_authority=59_999_990
    )
    report = dict(
        scope=freeze.VERSION,
        status="PARTIAL",
        research_run_requested=False,
        scientific_verdict=None,
        grant=digest("original"),
        candidate="b" * 40,
        grant_minutes=1,
        errors=dict(qualification="SystemExit"),
        files={"launch.json": launch, "failure.json": failure},
    )

    def verify():
        api["write_readiness_zip"](output, report)
        return api["verify_initialization_zip"](output)

    checked = verify()
    assert checked["capacity"]["wave"][0] == dict(device=7, margin_bytes=200, margin_inodes=27)
    assert checked["grant"] == report["grant"] and checked["next_scope"] == "wave"
    for record, field, invalid in (
        (failure, "remaining_authority", 60_000_000),
        (failure, "elapsed_us", True),
        (launch, "research_authority", report["grant"]),
        (launch, "next_scope", "first_ordinary_cut"),
        (launch["scopes"]["wave"], "fits", False),
        (launch["scopes"]["wave"], "tail_with_reserve_us", 10_000),
        (launch["scopes"]["wave"]["mounts"][0], "available_bytes", 600),
    ):
        previous = record[field]
        record[field] = invalid
        with pytest.raises(ValueError):
            verify()
        record[field] = previous


def test_pinned_handoff_returns_partial_zip_and_keeps_terminal_output_private(
    tmp_path, monkeypatch
):
    import os

    root = tmp_path / "donor"
    home = tmp_path / "operator"
    root.mkdir()
    home.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("HOME", str(home))
    subprocess.run(["git", "init", "-q"], check=True)
    relative = Path("specs/016-measure-learning-velocity/validation/qualify_continuation.py")
    script = root / relative
    script.parent.mkdir(parents=True)
    api = runpy.run_path(str(Path(__file__).resolve().parents[2] / relative))
    # A pinned, stdlib-only donor process exercises the exact CLI transport without a device.
    source = Path(api["__file__"]).read_text()
    script.write_text(
        source.replace(
            'if __name__ == "__main__":',
            """
def fixture(revision, minutes, mode, supplied, recovery):
    assert (minutes, mode, supplied, recovery) == ("30", "initialization", None, [INITIAL_WORKER, INITIAL_GRANT])
    print("private-fixture-error", file=sys.stderr)
    return dict(scope=INITIALIZATION, candidate=INITIAL_WORKER, grant=INITIAL_GRANT, grant_minutes=30,
                research_run_requested=False, scientific_verdict=None, errors={"qualification": "SystemExit"},
                status="PARTIAL", files={"failure.json": dict(stage="package", mechanical_status="FAILED",
                                                              elapsed_us=1, remaining_authority=1799999999)})
main = fixture
if __name__ == "__main__":""",
        )
    )
    subprocess.run(["git", "add", str(relative)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=fixture@example.org",
            "-c",
            "user.name=Fixture",
            "commit",
            "-qm",
            "pinned reader",
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "-q"], check=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    api["resume_initialization"].__globals__["DONOR"] = current
    prior = os.umask(0o077)
    try:
        assert api["resume_initialization"](current) == 1
    finally:
        os.umask(prior)
    (directory,) = home.iterdir()
    assert directory.stat().st_mode & 0o077 == 0
    assert "private-fixture-error" in (directory / "operator.log").read_text()
    assert (
        api["verify_initialization_zip"](directory / "initialization.zip")["failure"]["elapsed_us"]
        == 1
    )
    with zipfile.ZipFile(directory / "initialization.zip") as archive:
        assert all(b"private-fixture-error" not in archive.read(p) for p in archive.namelist())


@pytest.mark.parametrize("installed_before_failure", (False, True))
def test_funded_zero_scope_freezes_all_trials_and_recovers_before_device(
    tmp_path, monkeypatch, installed_before_failure
):
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

    def verify_package(root, area, runtime, binding):
        # The verifier runs in a separate process, just as it does on the physical host.
        if installed_before_failure and not (area / "first-error.json").exists():
            raise ValueError("fixture reader stopped after installation")

    monkeypatch.setattr(host, "package_evidence", verify_package)
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
    prices["launch_options"] = launch_fixture()
    info = dict(cuts=[], families=families, prices=prices, assessment="sealed", gpu_dispatches=0)
    monkeypatch.setattr(controller, "preflight", lambda *a, **kw: dict(inventory=info))
    machine = SimpleNamespace(
        policy=dict(free_disk_floor_bytes=0), engine=SimpleNamespace(image=lambda v: v)
    )
    calls = []

    def pack(root, source, area, out, revision, parent, source_hash, host, stage, seconds):
        calls.append("package")
        if calls == ["package"] and not installed_before_failure:
            child = out / "package"
            child.mkdir()
            (child / "stderr.log").write_text(
                "private-canary\nOSError: [Errno 28] No space left on device"
            )
            atomic_write(child / "process.json", dict(returncode=1))
            atomic_write(child / "cleanup.json", dict(acknowledged=True, within_bound=True))
            raise RuntimeError("exit 1: " + str(child / "stderr.log"))
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

    reader_source = tmp_path / "reader"

    def invoke(action, grant=None, *, resume=None):
        return controller.execute(
            SimpleNamespace(
                workspace=tmp_path,
                action=action,
                authority=grant,
                archive=None,
                qualification_archive=None,
                resume_initialization=resume,
                free_disk_gib=0,
            ),
            reader_source if resume else source,
            host_factory=lambda *a, **kw: machine,
        )

    original_space = freeze.dispatch_space

    def blocked(*a, **kw):
        raise host.broad.CapacityShortfall(
            "ordinary_initialization", [dict(deficit_inodes=1, deficit_bytes=0)], None
        )

    monkeypatch.setattr(freeze, "dispatch_space", blocked)
    first, code = invoke("qualify", grant_path)
    assert code == 1 and first["stage"] == "package" and calls == ["package"], first
    api = runpy.run_path(
        str(source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py")
    )
    from oeis_learn.pipeline_validation import (
        progressive_velocity_initialization_recovery as recovery,
    )

    authority = recovery.original_grant(tmp_path, rev, grant["identity"], 15)
    with pytest.raises(ValueError):
        recovery.original_grant(tmp_path, rev, grant["identity"], 30)
    area = folder / "qualification" / grant["identity"][7:]
    from oeis_learn.generalization.progressive.lifecycle import RunStore

    with RunStore.owner(folder), pytest.raises(BlockingIOError):
        recovery.history(area, grant)
    history = recovery.history(area, grant)
    assert history["elapsed_us"] == first["elapsed_us"]
    assert history["operations"]["freeze_trials"]["receipt"]
    failed = history["operations"]["package"]["attempts"]
    assert len(failed) == 1
    if not installed_before_failure:
        assert failed[0]["process_exit"] == 1 and failed[0]["diagnosis"] == "storage_full"
    assert "private-canary" not in json.dumps(history) and str(tmp_path) not in json.dumps(history)
    assert not history["operations"]["measure"]["attempts"] and not captured.exists()

    reader_rev = "c" * 40

    def worker_source(root, reader, worker):
        assert root == tmp_path and reader == reader_source and worker == rev
        return source

    monkeypatch.setattr(recovery, "worker_source", worker_source)

    def action(name, grant=None):
        result, code = invoke(name, grant, resume=rev if name in {"qualify", "verify"} else None)
        if code:
            raise SystemExit(json.dumps(result))
        return result

    monkeypatch.setattr(
        freeze, "authorize", lambda *_: pytest.fail("original grant required on recovery")
    )

    def resume():
        return api["donor_initialization"](
            action, tmp_path, rev, 15, recovery=dict(reader=reader_rev, grant=grant["identity"])
        )

    partial = resume()
    packages = ["package"] if installed_before_failure else ["package", "package"]
    assert partial["status"] == "PARTIAL" and calls == packages
    dest = tmp_path / "result.zip"
    api["write_readiness_zip"](dest, partial)
    checked = api["verify_initialization_zip"](dest, captured)
    assert checked["prescore"] == freeze.read_member(captured, freeze.MEMBER)["identity"]
    assert checked["failure"]["deficit_inodes"] == 1 and checked["recovery"]["before"] == history
    assert checked["recovery"]["after"]["operations"]["package"]["receipt"]
    assert (
        bool(checked["recovery"]["after"]["operations"]["verify_paid_package"]["receipt"])
        == installed_before_failure
    )
    assert captured.is_file() and not (folder / "accepted.json").exists()
    before = captured.read_bytes()
    assert protected.encode() not in before and b"private-development" not in before
    monkeypatch.setattr(freeze, "dispatch_space", original_space)

    bundle = resume()
    assert bundle["status"] == "COMPLETE", bundle
    pre, post = (bundle["files"]["recovery.json"][k] for k in ("before", "after"))
    assert pre == checked["recovery"]["after"] and post["operations"]["measure"]["receipt"]
    assert post["elapsed_us"] > pre["elapsed_us"]
    assert post["first_error"] == pre["first_error"] == history["first_error"]
    paid_result = action("qualify", authority)
    assert paid_result["charged_tokens"] == 0 and calls == packages + ["initialization"]
    assert (
        action("qualify", grant_path) == paid_result
        and action("qualify", grant_path) == paid_result
    )
    api["write_readiness_zip"](dest, bundle)
    assert api["verify_initialization_zip"](dest, captured)["restores"] == 2
    import copy

    altered_history = copy.deepcopy(bundle)
    old_package = altered_history["files"]["recovery.json"]["before"]["operations"]["package"]
    old_package["attempts"][0]["process_exit"] = 99
    altered_file = tmp_path / "altered-history.zip"
    api["write_readiness_zip"](altered_file, altered_history)
    with pytest.raises(ValueError, match="settled initialization debit"):
        api["verify_initialization_zip"](altered_file)
    assert (
        bundle["files"]["recovery.json"]["before"]["operations"]["package"]["attempts"][0]
        == failed[0]
    )
    other = tmp_path / "unrelated.zip"
    other.write_bytes(b"another pre-score package")
    with pytest.raises(ValueError, match="independently retained"):
        api["verify_initialization_zip"](dest, other)
    assert before == captured.read_bytes() and calls == packages + ["initialization"]
    assert len(list((folder / "registrations").glob("*/trial-*.json"))) == 7
    assert not list((folder / "registrations").glob("*/look-*.json"))
    from oeis_learn.generalization.progressive import velocity_donation_report as public

    canonical = folder / "qualification" / grant["identity"][7:] / "share" / public.REPORT
    offline = SimpleNamespace(
        action="verify",
        workspace=None,
        authority=None,
        archive=canonical,
        qualification_archive=captured,
    )
    independent, code = controller.execute(offline, source)
    assert (
        code == 0 and independent["verification_scope"] == "paired_archives_ordinary_initialization"
    )
    assert independent["last_committed_ref"] == paid_result["receipt"]["identity"]
    altered = seal(dict(public.read_member(canonical, freeze.REPORT_MEMBER), tokens=1))
    changed = tmp_path / "changed-report.zip"
    public.install(changed, freeze.REPORT_MEMBER, altered)
    with pytest.raises(ValueError, match="zero-cut"):
        freeze.verify_archives(changed, captured)
    captured.unlink()
    failed, code = invoke("verify")
    assert code == 1 and failed["mechanical_status"] == "FAILED" and not captured.exists()
