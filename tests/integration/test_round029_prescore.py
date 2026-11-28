"""A registered Tiny CPU worker stands in for the physical device in the host lifecycle."""

import hashlib
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
    try:
        assert api["prepare_trials"](current, "15") == 1
    finally:
        os.umask(prior)
    (combined,) = home.glob("round029-trial-prep-*/trial-preparation.zip")
    returned = api["verify_trial_preparation_zip"](combined)
    assert (
        returned["status"] == "PARTIAL" and returned["initialization"]["failure"]["elapsed_us"] == 1
    )
    with pytest.raises(SystemExit, match="not found"):
        api["recover_trial_preparation"](current, "0" * 64, combined)
    with pytest.raises(SystemExit, match="completed initialization"):
        api["recover_trial_preparation"](current, hashlib.sha256(combined.read_bytes()).hexdigest())
    assert not list(home.glob("round029-trial-readback-*"))
    assert not (root / "run-state/discovery-authorizations").exists()


@pytest.mark.parametrize("failure_mode", ("failed_build", "installed", "missing_image"))
def test_funded_zero_scope_freezes_all_trials_and_recovers_before_device(
    tmp_path, monkeypatch, failure_mode
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
    missing = failure_mode == "missing_image"
    monkeypatch.setattr(host.runtime_build, "context_size", lambda *_: dict(bytes=4096, inodes=2))
    installed_before_failure = failure_mode == "installed"
    monkeypatch.setattr(host, "cached_runtime", lambda *_: None if missing else dict(image=image))

    def verify_package(root, area, runtime, binding, **kwargs):
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
            sources=dict(
                fresh=pointer["receipt"],
                checkpoint=marker["receipt"],
                **{k: digest(k) for k in ("controls", "paired", "bank", "accounting")},
            ),
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
        if calls == ["package"] and failure_mode == "failed_build":
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
    assert (
        code == 1 and first["stage"] == "package" and calls == ([] if missing else ["package"])
    ), first
    if missing:
        # Reproduce both settled lookups from the returned workstation ZIP.
        first, code = invoke("qualify", grant_path)
        assert code == 1 and first["reason"] == "settled compatible runtime required"
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
    assert len(failed) == (2 if missing else 1)
    if failure_mode == "failed_build":
        assert failed[0]["process_exit"] == 1 and failed[0]["diagnosis"] == "storage_full"
    assert "private-canary" not in json.dumps(history) and str(tmp_path) not in json.dumps(history)
    assert not history["operations"]["measure"]["attempts"] and not captured.exists()

    reader_rev = "c" * 40

    def worker_source(root, reader, worker):
        assert root == tmp_path and reader == reader_source and worker == rev
        return source

    monkeypatch.setattr(recovery, "worker_source", worker_source)
    if missing:
        graph = tmp_path / "engine-store"
        graph.mkdir()
        machine.engine.name = "podman"
        machine.engine.details = {"store": {"graphRoot": str(graph)}}

        def build(source_root, *, target, tag, seconds, evidence, check_resources, iidfile):
            assert source_root == source and target == "gpu" and seconds > 0
            calls.append("image-build")
            check_resources()
            evidence.mkdir(exist_ok=True)
            iidfile.write_text(image)
            atomic_write(
                evidence / "process.json",
                dict(outcome="COMPLETE", returncode=0, cleanup_failure=None),
            )
            atomic_write(evidence / "cleanup.json", dict(acknowledged=True, within_bound=True))
            atomic_write(
                evidence / "command.json",
                dict(
                    argv=[
                        "/usr/bin/podman",
                        "build",
                        "--target",
                        "gpu",
                        "--tag",
                        tag,
                        "--iidfile",
                        str(iidfile),
                        str(source),
                    ]
                ),
            )
            return image

        machine.engine.build = build

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

    if missing:
        with monkeypatch.context() as limits:
            limits.setattr(host.runtime_build, "capacity", blocked)
            no_space = resume()
        assert no_space["files"]["failure.json"]["stage"] == "build_runtime"
        assert no_space["files"]["failure.json"]["deficit_inodes"] == 1
        assert no_space["files"]["recovery.json"]["after"] == history and calls == []

    partial = resume()
    packages = (
        ["image-build", "package"]
        if missing
        else ["package"] if installed_before_failure else ["package", "package"]
    )
    assert partial["status"] == "PARTIAL" and calls == packages
    dest = tmp_path / "result.zip"
    api["write_readiness_zip"](dest, partial)
    checked = api["verify_initialization_zip"](dest, captured)
    assert checked["prescore"] == freeze.read_member(captured, freeze.MEMBER)["identity"]
    assert checked["failure"]["deficit_inodes"] == 1 and checked["recovery"]["before"] == history
    assert checked["failure"]["cause"]["error"] == "CapacityShortfall"
    assert checked["failure"]["cause"]["identity"] != history["first_error"]["identity"]
    acquired = checked["recovery"]["after"]["operations"]
    if missing:
        assert acquired["package"]["attempts"] == failed and acquired["package"]["receipt"] is None
        assert acquired["build_runtime"]["receipt"] and acquired["package_runtime"]["receipt"]
        observed = acquired["build_runtime"]["image_build"]
        assert observed["image"] == image and observed["samples"] >= 2
        assert [r["device"] for r in observed["lowest"]] == [tmp_path.stat().st_dev]
        from oeis_learn.generalization.progressive.budgets import BudgetJournal
        from oeis_learn.generalization.progressive.velocity_records import operation_key

        balance = BudgetJournal.inspect_existing(
            area / "accounting", grant["identity"], grant["resources"], include_work=True
        )
        logical = operation_key(host.coordinates(grant, "build_runtime"), 0)["logical"]
        build_reserve = balance["reserved_work"][logical + "/dispatch/0"]["elapsed_us"]
        assert build_reserve == history["remaining_us"] - grant["holdback_us"] - 242_001_040
    else:
        assert acquired["package"]["receipt"]
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
    if missing:
        image_id = next(area.glob("operations/*/attempt-*/image-build/image-id"))
        image_id.write_text(digest("different image"))
        rejected, code = invoke("verify", resume=rev)
        assert code and rejected["mechanical_status"] == "FAILED"
        image_id.write_text(image)
        assert action("verify")["report_ref"] == paid_result["receipt"]["identity"]
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
    if missing:
        from oeis_learn.pipeline_validation import progressive_velocity_trial_preparation as trial

        monkeypatch.setattr(
            controller,
            "candidate_revision",
            lambda path, _: reader_rev if path == reader_source else rev,
        )
        approval = trial.authorize(tmp_path, reader_rev, 15)

        def trial_action(action, authority=None):
            return controller.execute(
                SimpleNamespace(
                    workspace=tmp_path,
                    action=action,
                    authority=authority,
                    archive=None,
                    qualification_archive=None,
                    free_disk_gib=0,
                ),
                reader_source,
            )

        with monkeypatch.context() as short:
            short.setitem(prices["launch_options"]["scopes"]["wave"], "fits", False)
            short.setitem(
                prices["launch_options"]["scopes"]["wave"]["mounts"][0], "available_bytes", 799
            )
            short.setitem(
                prices["launch_options"]["scopes"]["wave"]["mounts"][0], "deficit_bytes", 1
            )
            blocked_result, code = trial_action("qualify", approval)
            assert code and blocked_result["deficit_bytes"] == 1
            assert not (folder / "accepted.json").exists()
        prepared, code = trial_action("qualify", approval)
        assert code == 0 and prepared["scope"] == "trial_preparation", prepared
        assert prepared["charged_tokens"] == prepared["research_dispatches"] == 0
        accepted = load(folder / "accepted.json")
        assert accepted["research_authority"] is None and accepted["supported"] == [
            "00",
            "04",
            "05",
            "06",
        ]
        assert set(accepted["omitted"]) == {"01", "02", "03"} and accepted["followup_us"] == 900
        assert accepted["initialization"] == paid_result["receipt"]["identity"]
        assert trial.published(tmp_path, folder)["accepted"] == accepted["identity"]
        assert len(accepted["learners"]) == 4 and all(
            v["package"] for v in accepted["learners"].values()
        )
        assert (
            trial_action("qualify", approval)[0] == prepared == trial_action("qualify", approval)[0]
        )
        assert trial_action("verify")[0]["verification_scope"] == "private_trial_preparation"
        prep_grant = load(approval)
        prep_area = folder / "qualification" / prep_grant["identity"][7:]
        transport = retained_qualification(tmp_path, prep_grant)
        assert (
            trial.verify_archives(prep_area / "share" / public.REPORT, transport)["accepted"]
            == accepted["identity"]
        )
        issued = prep_area / "share" / public.REPORT
        trial_bundle = dict(
            scope=trial.VERSION,
            reader=reader_rev,
            prep_minutes=15,
            grant=prep_grant["identity"],
            status="COMPLETE",
            research_run_requested=False,
            scientific_verdict=None,
            errors={},
            files={
                "initialization.json": api["copied_zip"](dest),
                "qualification.json": api["copied_zip"](transport),
                "report.json": api["copied_zip"](issued),
                "verification.json": dict(
                    private_receipt=prepared["last_committed_ref"],
                    public_report=public.read_member(issued, trial.REPORT_MEMBER)["identity"],
                ),
            },
        )
        packed = tmp_path / "trial-preparation.zip"
        api["write_readiness_zip"](packed, trial_bundle)
        reviewed = api["verify_trial_preparation_zip"](packed)
        assert reviewed["accepted"] == accepted["identity"]
        assert (reviewed["training_targets"], reviewed["looks"]) == (12, 20)
        assert (
            reviewed["expected_us"],
            reviewed["tail_us"],
            reviewed["followup_reserve_us"],
            reviewed["proposed_total_us"],
        ) == (9000, 10000, 900, 10900)
        assert reviewed["capacity"] == [
            dict(launch_fixture()["scopes"]["wave"]["mounts"][0], margin_bytes=200, margin_inodes=27)
        ]
        substituted = tmp_path / "substituted-trials.zip"
        bad = seal(dict(accepted, training=accepted["training"][:-1]))
        public.install(substituted, trial.MEMBER, bad)
        trial_bundle["files"]["qualification.json"] = api["copied_zip"](substituted)
        api["write_readiness_zip"](packed, trial_bundle)
        with pytest.raises(ValueError, match="wave differs"):
            api["verify_trial_preparation_zip"](packed)
        # Exercise the full operator command with the actual returned archives and original grant.
        import os
        import shutil

        namespace = api["prepare_trials"].__globals__
        home = tmp_path / "operator"
        home.mkdir()

        def existing_initialization(revision, *, directory):
            assert revision == reader_rev
            shutil.copyfile(dest, directory / "initialization.zip")
            return 0

        def checked_action(argv, root, name, **kwargs):
            assert root == tmp_path and argv[2:4] == [reader_rev, name]
            value, code = trial_action(name, Path(argv[4]) if name == "qualify" else None)
            assert code == 0, value
            return value

        with monkeypatch.context() as operator:
            operator.chdir(tmp_path)
            operator.setenv("HOME", str(home))
            operator.setitem(namespace, "resume_initialization", existing_initialization)
            operator.setitem(namespace, "pinned_source", lambda *_: reader_source)
            operator.setitem(namespace, "checked_json", checked_action)
            mask = os.umask(0o077)
            try:
                assert api["prepare_trials"](reader_rev, "15") == 0
            finally:
                os.umask(mask)
        (delivered,) = home.glob("round029-trial-prep-*/trial-preparation.zip")
        assert api["verify_trial_preparation_zip"](delivered)["accepted"] == accepted["identity"]
        # Reproduce the original transport loss after a successful qualification.
        previous = dict(
            trial_bundle,
            status="PARTIAL",
            errors={"verification": "SystemExit"},
            files={
                "initialization.json": api["copied_zip"](dest),
                "failure.json": {"mechanical_status": "FAILED", "remaining_authority": None},
            },
        )
        api["write_readiness_zip"](delivered, previous)
        stopped = controller.result(
            "verify",
            "FAILED",
            "unknown object member",
            "inspect_inputs",
            verification_scope="unverified_inputs",
            first_error=dict(error="ValueError", stage="verify", reason="unknown object member"),
        )
        delivered.with_name("operator.log").write_text(
            "private-canary\nverification: " + json.dumps(stopped)
        )
        prior_hash = hashlib.sha256(delivered.read_bytes()).hexdigest()
        new_reader, new_source = "d" * 40, tmp_path / "repaired-reader"
        init_journal = (area / "accounting/budget.jsonl").read_bytes()
        prep_journal = (prep_area / "accounting/budget.jsonl").read_bytes()
        authorities = sorted((tmp_path / "run-state/discovery-authorizations").iterdir())
        for reader, minutes in ((new_reader, 15), (reader_rev, 30)):
            with pytest.raises(ValueError, match="original trial preparation"):
                trial.authorize(tmp_path, reader, minutes)
        # The session's latest status is unrelated. The receipt and journal still name this grant.
        from oeis_learn.pipeline_validation.preparation_recovery import revise

        revise(folder / "status.json", seal(paid_result))

        def pinned_worker(root, reader, worker):
            if (reader, worker) == (new_source, reader_rev):
                return reader_source
            return worker_source(root, reader, worker)

        with monkeypatch.context() as replay:
            replay.chdir(tmp_path)
            replay.setenv("HOME", str(home))
            replay.setitem(namespace, "require_donor_root", lambda *_: None)
            replay.setitem(namespace, "pinned_source", lambda *_: new_source)
            replay.setattr(recovery, "worker_source", pinned_worker)
            for key in ("checked_json", "resume_initialization"):
                replay.setitem(namespace, key, lambda *_a, **_k: pytest.fail("reissued paid work"))
            replay.setattr(trial, "authorize", lambda *_: pytest.fail("created a new grant"))
            mask = os.umask(0o077)
            try:
                from oeis_learn.pipeline_validation.progressive_velocity_verify import private_result

                with RunStore.owner(folder):
                    blocked, code = private_result(
                        tmp_path, folder, source=new_source, trial_grant=prep_grant["identity"]
                    )
                    assert code == 2 and blocked["mechanical_status"] == "BLOCKED"
                for prior_path in (None, delivered):
                    assert api["recover_trial_preparation"](new_reader, prior_hash, prior_path) == 0
                assert len(list(home.glob("round029-trial-readback-*/trial-preparation.zip"))) == 2
                for archive in home.glob("round029-trial-readback-*/trial-preparation.zip"):
                    assert (
                        api["verify_trial_preparation_zip"](archive)["accepted"]
                        == accepted["identity"]
                    )
                    summary, parts = api["diagnostic_zip_members"](archive, trial.VERSION)
                    assert (
                        summary["reader"] == reader_rev
                        and summary["readback"]["reader"] == new_reader
                    )
                    assert (
                        summary["readback"]["prior_failure"]["category"] == "schema_unknown_member"
                    )
                    assert parts["initialization.json"] == previous["files"]["initialization.json"]
                    assert "private-canary" not in json.dumps([summary, parts])
                inputs = prep_area / "trial-inputs.json"
                with inputs.open("rb") as stream:
                    saved_inputs = stream.read()
                try:
                    inputs.write_text("{}")
                    assert api["recover_trial_preparation"](new_reader, prior_hash) == 1
                    partials = [
                        api["verify_trial_preparation_zip"](p)
                        for p in home.glob("round029-trial-readback-*/trial-preparation.zip")
                    ]
                    (partial,) = [p for p in partials if p["status"] == "PARTIAL"]
                    assert partial["published"]["accepted"] == accepted["identity"]
                    assert partial["failure"]["category"] == "trial_inputs_changed"
                finally:
                    inputs.write_bytes(saved_inputs)
            finally:
                os.umask(mask)
        assert (area / "accounting/budget.jsonl").read_bytes() == init_journal
        assert (prep_area / "accounting/budget.jsonl").read_bytes() == prep_journal
        assert sorted((tmp_path / "run-state/discovery-authorizations").iterdir()) == authorities
        assert load(folder / "status.json") == seal(paid_result)
        assert not list((folder / "registrations").glob("*/look-*.json"))
        assert protected.encode() not in transport.read_bytes()
        action("qualify", grant_path)
    captured.unlink()
    failed, code = invoke("verify")
    assert code == 1 and failed["mechanical_status"] == "FAILED" and not captured.exists()
