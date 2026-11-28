"""Controller-only fixtures. Hardware measurements come exclusively from the installed worker."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.pipeline_validation import progressive_handoff, progressive_round028_rehearsal
from oeis_learn.pipeline_validation import progressive_velocity_qualification as controller
from oeis_learn.pipeline_validation.progressive_round029 import inspect_status
from oeis_learn.generalization.progressive.velocity_records import validate_result


def test_operator_reads_only_the_terminal_round029_result(tmp_path):
    import json
    import sys

    helper = (
        Path(__file__).resolve().parents[2]
        / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    )
    read = runpy.run_path(str(helper))["checked_json"]
    reply = inspect_status(tmp_path)
    log = "[1/2] STEP 1/3: FROM base\n" + json.dumps({"unrelated": True}) + "\n"

    def read_stdout(text):
        return read(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", text], tmp_path, "status"
        )

    assert read_stdout(log + json.dumps(reply)) == reply
    for malformed in (
        log,
        log + json.dumps(reply) + "\ntrailing build log",
        json.dumps(dict(reply, action="verify")),
    ):
        with pytest.raises(ValueError, match="terminal status result"):
            read_stdout(malformed)


@pytest.mark.parametrize("missing", [False, True])
def test_operator_writes_a_complete_attachment_on_success_or_failure(
    tmp_path, monkeypatch, capsys, missing
):
    import json
    import stat

    helper = (
        Path(__file__).resolve().parents[2]
        / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    )
    cli = runpy.run_path(str(helper))["cli"]
    path = tmp_path / "qualification.json"
    candidate = "1" * 40
    expected = {"candidate": candidate, "verified_replays": 2}
    visits = []

    def review(revision, minutes, mode):
        visits.append((revision, minutes, mode))
        if missing:
            raise ValueError("No saved child qualification")
        return expected

    monkeypatch.setitem(cli.__globals__, "main", review)
    args = [candidate, "20", "review-children", "--output", str(path)]
    assert cli(args) == int(missing)
    capture = capsys.readouterr()
    assert capture.out == "" and str(path) in capture.err
    saved = json.loads(path.read_bytes())
    if missing:
        assert saved["status"] == "UNVERIFIED" and saved["reason"] == "No saved child qualification"
    else:
        assert saved == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and list(tmp_path.iterdir()) == [path]
    with pytest.raises(SystemExit):
        cli(args)
    assert json.loads(path.read_bytes()) == saved
    assert visits == [(candidate, "20", "review-children")]


def test_readiness_zip_keeps_independent_checks_and_failed_grant(tmp_path, monkeypatch):
    import hashlib
    import json
    import stat
    import zipfile

    source = Path(__file__).resolve().parents[2]
    helper = source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    api = runpy.run_path(str(helper))
    cli = api["cli"]
    driver = api["donor_readiness"]
    rev = "a" * 40
    grant = digest("existing fresh grant")
    receipt_path = tmp_path / "fresh-start.json"
    problem = dict(stage="prepare", error="CapacityShortfall", reason="PRIVATE-CANARY",
                   deficit_bytes=100, deficit_inodes=2)
    receipt_path.write_text(json.dumps(dict(
        candidate=rev, scope="fresh_start_timings", grant=grant, status="BLOCKED", receipt=None,
        elapsed_us=17, remaining_us=60_000_000-17, charged_tokens=0, research_dispatches=0,
        next_action="inspect_capacity", first_error=problem, problem=problem,
    )))
    monkeypatch.setitem(driver.__globals__, "reopen_child_rehearsal", lambda *_: {"verified_replays": 2})
    from oeis_learn.pipeline_validation.progressive_round029 import preflight
    inspected = preflight(tmp_path)
    inspected["inventory"]["private_rows"] = ["PRIVATE-CANARY"]
    calls = []
    def action(name, *args):
        calls.append(name)
        return {"preflight": inspected, "status": inspect_status(tmp_path)}[name]
    monkeypatch.setitem(cli.__globals__, "main", lambda revision, minutes, mode, supplied:
                        driver(action, tmp_path, source, revision, minutes, supplied))
    output = tmp_path / "readiness.zip"
    argv = [rev, "1", "readiness", "--from-result", str(receipt_path), "--output", str(output)]
    assert cli(argv) == 1
    assert sorted(calls) == ["preflight", "status"]
    assert "verify" not in calls and stat.S_IMODE(output.stat().st_mode) == 0o600
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        summary = json.loads(archive.read("summary.json"))
        assert summary["status"] == "PARTIAL" and summary["research_run_requested"] is False
        assert summary["checks"]["fresh-start"] == "BLOCKED"
        assert summary["checks"]["verification"] == "NOT_RUN"
        for name, checksum in summary["sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        assert set(archive.namelist()) == {"summary.json", *summary["sha256"]}
        assert all(b"PRIVATE-CANARY" not in archive.read(n) for n in archive.namelist())
        receipt = json.loads(archive.read("fresh-start.json"))
        assert receipt["elapsed_us"] == 17 and receipt["grant"] == grant and receipt["deficit_inodes"] == 2
        assert json.loads(archive.read("accounting-fixture.json"))["occupied_us"] == 9
        assert json.loads(archive.read("velocity-fixture.json"))["rates"]["pp_per_hour"] == 4
        assert json.loads(archive.read("preflight.json"))["prices"]["wave"]["expected_us"] is None
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        cli(argv)
    assert output.read_bytes() == original and sorted(calls) == ["preflight", "status"]


def test_readiness_writes_bootstrap_failure_as_zip_and_requires_zip_output(tmp_path, monkeypatch):
    import json
    import zipfile

    helper = Path(__file__).resolve().parents[2] / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    cli = runpy.run_path(str(helper))["cli"]
    monkeypatch.chdir(tmp_path)
    args = ["a" * 40, "37", "readiness", "--from-result", "fresh-start.json"]
    for suffix in ([], ["--output", "wrong.json"]):
        with pytest.raises(SystemExit):
            cli(args + suffix)
    with pytest.raises(SystemExit):
        cli([*args[:3], "--output", "missing-input.zip"])
    assert not (tmp_path / "wrong.json").exists()
    assert cli(args + ["--output", "diagnostic.zip"]) == 1
    with zipfile.ZipFile("diagnostic.zip") as archive:
        assert archive.namelist() == ["summary.json"]
        summary = json.loads(archive.read("summary.json"))
        assert summary["status"] == "UNVERIFIED" and summary["sha256"] == {}


def test_readiness_reopens_the_published_child_grant_with_its_own_candidate(tmp_path, monkeypatch):
    import json
    import sys

    source = Path(__file__).resolve().parents[2]
    validation = source / "specs/016-measure-learning-velocity/validation"
    helper = validation / "qualify_continuation.py"
    expected = load(validation / "round029-child-qualification.json")
    review = runpy.run_path(str(helper))["reopen_child_rehearsal"]
    returned, commands = expected.copy(), []

    def replay(argv, *, cwd):
        commands.append((argv, cwd))
        return json.dumps(returned).encode()

    monkeypatch.setattr("subprocess.check_output", replay)
    assert review(tmp_path, source) == expected
    assert commands == [([sys.executable, "-I", "-S", "-B", str(helper),
                          expected["candidate"], "20", "review-children"], tmp_path)]
    returned["charged_tokens"] += 1
    with pytest.raises(ValueError, match="published receipt"):
        review(tmp_path, source)


@pytest.mark.parametrize(
    "fault,expected_calls",
    [
        ("none", 2),
        ("missing_receipt", 0),
        ("other_grant", 0),
        ("other_status", 0),
        ("extra_charge", 2),
    ],
)
def test_saved_learner_review_uses_the_original_grant(tmp_path, fault, expected_calls):
    validation = (
        Path(__file__).resolve().parents[2] / "specs/016-measure-learning-velocity/validation"
    )
    helper = runpy.run_path(str(validation / "replay_learner.py"))
    grant_id = helper["GRANT"]
    authority = tmp_path / "run-state/discovery-authorizations" / f"grant-{grant_id}.json"
    retained_grant = digest("other") if fault == "other_grant" else "sha256:" + grant_id
    atomic_write(authority, {"identity": retained_grant})
    receipt = {"identity": helper["RECEIPT_ID"]}
    saved = dict(
        receipt_path="retained/receipt.json",
        receipt=receipt,
        elapsed_us=100,
        charged_tokens=20,
        research_dispatches=0,
    )
    if fault != "missing_receipt":
        atomic_write(tmp_path / saved["receipt_path"], receipt)
    calls = []

    def invoke(action, *args):
        if action == "status":
            status_id = digest("other") if fault == "other_status" else helper["STATUS_ID"]
            return dict(
                owner_active=False,
                last_committed_ref=receipt["identity"],
                progress=dict(saved, identity=status_id),
            )
        assert action == "qualify" and args == (str(authority),)
        calls.append(action)
        if fault == "extra_charge" and len(calls) == 2:
            return dict(saved, elapsed_us=101)
        return saved.copy()

    if fault == "none":
        result = helper["replay"](tmp_path, invoke)
        assert result["verified_replays"] == 2 and result["receipt"] == receipt["identity"]
    else:
        with pytest.raises(SystemExit):
            helper["replay"](tmp_path, invoke)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("children", [False, True])
def test_original_grant_retries_after_capacity_block_and_credits_one_receipt(
    tmp_path, monkeypatch, children
):
    revision = "1" * 40
    root = tmp_path / "donor"
    root.mkdir()
    folder = root / "run-state/discovery/016-research-round-029"
    folder.mkdir(parents=True)
    grant = load(controller.authorize_learner_check(root, 10, revision, children=children))
    assert controller.authorize_learner_check(root, 10, revision, children=children).is_file()
    low = {"free_bytes": 1 << 40, "free_inodes": 1}
    monkeypatch.setattr(progressive_round028_rehearsal, "space", lambda root: dict(low))
    monkeypatch.setattr(
        progressive_round028_rehearsal, "physical_scope", lambda *a, **k: digest("fixture physical")
    )
    monkeypatch.setattr(controller, "_original_profile", lambda root: {"fixture": True})
    from oeis_learn.experiments import source_snapshot

    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda source: {"fixture": True})
    image = digest("fixture image")
    host = SimpleNamespace(
        policy={"free_disk_floor_bytes": 0}, engine=SimpleNamespace(image=lambda ref: image)
    )
    args = SimpleNamespace(image=image, gpu_base_image=None)
    issued = []

    def stage(host, source, image, out, argv, *, gpu, record_lease, seconds, readonly_inputs):
        issued.append(out)
        assert gpu and record_lease and seconds > 0
        path, dest = readonly_inputs[0]
        req = load(path)
        from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearsal_counts

        counts = rehearsal_counts(children=children)
        batches = {}
        for size in req["batch_sizes"]:
            batches[str(size)] = dict(
                batch=size,
                full_restore=True,
                useful_updates=2,
                discarded_updates=1,
                **counts[str(size)],
                restored_weights=digest("fixture weights"),
                uninterrupted_weights=digest("fixture weights"),
                next_rows=digest("fixture next"),
                uninterrupted_next_rows=digest("fixture next"),
                restored_optimizer=digest("fixture moments"),
                uninterrupted_optimizer=digest("fixture moments"),
                scheduler={"updates": 2, "tokens": counts[str(size)]["charged_tokens"]},
                interrupted_peak_device_bytes=1,
                checkpoint_bytes=1,
                checkpoint_inodes=1,
                failed_us=1,
                matching_eligible=int(size == 32),
                matching_passes=int(size == 32),
            )
            if children:
                assert req["version"] == "round029-device-request/v2"
                row = batches[str(size)]
                inc = row["child_update_tokens"]
                comparison = {k: digest(k) for k in ("weights", "optimizer", "rng", "next_rows")}
                row["child"] = dict(
                    parent=digest("parent"),
                    checkpoint=digest("child"),
                    reference=comparison,
                    restored=comparison,
                    horizon=8_000_000,
                    updates=3,
                    verified_replays=2,
                    incremental_charged_tokens=inc,
                    useful_tokens=row["useful_tokens"] + inc,
                    charged_tokens=row["charged_tokens"] + inc,
                    discarded_tokens=row["discarded_tokens"],
                )
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    version="round029-device-result/v1",
                    device="hip",
                    request=req["identity"],
                    batches=batches,
                    literal_probes={"fixture": {"status": "fixture"}},
                    peak_device_bytes=1,
                    elapsed_us=2,
                    telemetry={},
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
                        dict(Type="bind", Destination=dest, Source=str(path), RW=False),
                        dict(Type="bind", Destination="/work/output", Source=str(out), RW=True),
                    ],
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        atomic_write(out / "gpu-lease.json", {"elapsed_us": 10})

    monkeypatch.setattr(progressive_handoff, "stage", stage)

    def run():
        return controller.qualify(args, root, folder, root, revision, grant, lambda *a, **k: host)

    blocked, code = run()
    assert validate_result(blocked) == blocked
    assert blocked["deficit_inodes"] == blocked["first_error"]["deficit_inodes"]
    assert code == 2 and blocked["first_error"]["deficit_inodes"] > 0
    assert blocked["charged_tokens"] == 0 and not issued
    assert inspect_status(folder)["progress"]["next_action"] == "inspect_capacity"
    low["free_inodes"] = 10000
    accepted, code = run()
    assert validate_result(accepted) == accepted
    assert code == 0 and len(issued) == 1
    assert accepted["charged_tokens"] == grant["resources"]["tokens"] > 0
    assert accepted["first_error"] == blocked["first_error"]
    receipt = root / accepted["receipt_path"]
    assert receipt.exists() and accepted["receipt"]["lease_us"] == 10
    summaries = set((folder / "recovery-records").iterdir())
    receipt.unlink()  # clean supervisor termination preceded final host publication
    for _ in range(2):
        repeated, code = run()
        assert code == 0 and repeated == accepted and len(issued) == 1
    assert inspect_status(folder)["last_committed_ref"] == accepted["receipt"]["identity"]
    assert set((folder / "recovery-records").iterdir()) == summaries
    if children:
        from oeis_learn.generalization.progressive.lifecycle import RunStore
        from oeis_learn.pipeline_validation.progressive_velocity_verify import private_result

        with RunStore.owner(folder):
            pass

        def invoke(action, *params):
            if action == "status":
                return inspect_status(folder)
            if action == "verify":
                response, code = private_result(root, folder)
            else:
                assert action == "qualify" and load(params[0]) == grant
                response, code = run()
            assert code == 0, response
            return response

        helper = (
            Path(__file__).resolve().parents[2]
            / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
        )
        review = runpy.run_path(str(helper))
        report = review["child_rehearsal"](invoke, root, revision, "10")
        assert (
            report["charged_tokens"] == 4182
            and report["verified_replays"] == 2
            and len(issued) == 1
        )
        assert report["children"]["32"]["incremental_charged_tokens"] == 400
        # The session's latest status may already show a different qualification.
        (folder / "status.json").unlink()
        before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
        assert review["saved_children"](root, root, revision, 10) == report
        with pytest.raises(ValueError, match="No saved child"):
            review["saved_children"](root, root, "2" * 40, 10)
        with pytest.raises(ValueError, match="No saved child"):
            review["saved_children"](root, root, revision, 20)
        with RunStore.owner(folder), pytest.raises(ValueError, match="active"):
            review["saved_children"](root, root, revision, 10)
        assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*")}
        assert len(issued) == 1
    from oeis_learn.pipeline_validation.progressive_velocity_runtime import learner_runtime

    base = learner_runtime(root, grant["identity"][7:])
    assert base["image"] == image and base["receipt"] == accepted["receipt"]["identity"]
    assert base["candidate"] == revision
    mounts = load(issued[0] / "container-inspect.json")
    mounts[0]["Mounts"][0]["RW"] = True
    (issued[0] / "container-inspect.json").unlink()
    atomic_write(issued[0] / "container-inspect.json", mounts)
    with pytest.raises(ValueError, match="MOUNTS"):
        learner_runtime(root, grant["identity"][7:])
    if children:
        with pytest.raises(ValueError, match="MOUNTS"):
            review["saved_children"](root, root, revision, 10)
    with pytest.raises(ValueError, match="MOUNTS"):
        run()

    # An operator interrupt with no cleanup proof cannot create time for a second launch.
    def interrupted(*args, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(progressive_handoff, "stage", interrupted)
    grant = load(controller.authorize_learner_check(root, 11, revision))
    failed, code = run()
    assert validate_result(failed) == failed
    assert code == 130 and failed["first_error"]["exit"] == 130
    assert failed["remaining_authority"] == 0 and failed["remaining_tokens"] == 0
    stopped, code = run()
    assert code == 2 and stopped["first_error"] == failed["first_error"]


def test_matrix_requires_prior_paid_supply_before_creating_a_grant(tmp_path):
    from oeis_learn.pipeline_validation.progressive_velocity_timing_host import authorize_timing

    with pytest.raises(FileNotFoundError, match="donor supply qualification"):
        authorize_timing(tmp_path, 30, "1" * 40)
    assert not (tmp_path / "run-state").exists()


@pytest.mark.parametrize("route", ["supplied", "cold_retry", "cached_wheel", "prelaunch"])
def test_single_funded_matrix_replays_all_prices_without_another_dispatch(
    tmp_path, monkeypatch, velocity_supply_rows, route
):
    from types import SimpleNamespace
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.generalization.progressive.velocity_trials import update_policy
    from oeis_learn.pipeline_validation import progressive_velocity_timing_host as host
    from oeis_learn.pipeline_validation import progressive_round029, progressive_round028
    from oeis_learn.pipeline_validation.progressive_velocity_timing import (
        timing_projection,
        timing_design,
    )

    from oeis_learn.pipeline_validation import progressive_velocity_runtime as runtime
    from oeis_learn.experiments.source_snapshot import file_digest
    import json
    import subprocess
    import tarfile
    import zipfile

    cold_retry = route == "cold_retry"
    warm = route == "cached_wheel"
    revision = "4" * 40
    image = digest("image")
    snapshot = ["source bytes"]
    if warm:

        def git(*args):
            return (
                subprocess.check_output(
                    [
                        "git",
                        "-c",
                        "user.name=Fixture",
                        "-c",
                        "user.email=fixture@example.invalid",
                        *args,
                    ],
                    cwd=tmp_path,
                )
                .decode()
                .strip()
            )

        git("init", "-q")
        (tmp_path / "uv.lock").write_text("fixture locked runtime")
        git("add", "uv.lock")
        git("commit", "-qm", "fixture runtime")
        base_revision = git("rev-parse", "HEAD")
        bootstrap = tmp_path / runtime.BOOTSTRAP
        bootstrap.parent.mkdir(parents=True)
        bootstrap.write_bytes(b"# fixture bootstrap")
        snapshot = {runtime.BOOTSTRAP: file_digest(bootstrap.read_bytes())}
        git("add", runtime.BOOTSTRAP)
        git("commit", "-qm", "fixture application")
        revision = git("rev-parse", "HEAD")
        base_grant = digest("paid learner")
        base = seal(
            dict(
                grant=base_grant,
                receipt=digest("paid receipt"),
                candidate=base_revision,
                image=image,
                source=digest("base source"),
            )
        )
        original = (
            tmp_path / "run-state/discovery/016-research-round-029/qualification" / base_grant[7:]
        )
        original.mkdir(parents=True)
        atomic_write(
            original / "binding.json",
            seal(dict(version="round029-learner-rehearsal/v1", candidate=base_revision)),
        )
        atomic_write(original / "receipt.json", seal(dict(fixture=True)))

        def verified_base(root, previous):
            assert root == tmp_path and previous == base_grant[7:]
            return base

        monkeypatch.setattr(runtime, "learner_runtime", verified_base)
    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), seed=29, shape="smaller_width"
    )
    rows = velocity_supply_rows
    ids = [r["draw_identity"] for r in rows]
    stream = seal(
        dict(
            order=ids,
            seeds={"training_families": 17},
            row_identities=digest([r["identity"] for r in rows]),
        )
    )
    pairs = matching_pairs([rows], ids, seed=29)
    payment = seal(
        dict(kind="matching_pairs", grant=digest("prior grant"), receipt=digest("prior payment"))
    )
    projection = timing_projection(
        seal(dict(profile=model, microbatch=8, stream=stream, rows=rows)),
        pairs,
        digest("initial weights"),
        payment["receipt"],
    )
    design = timing_design(projection, len(ids), 8)
    monkeypatch.setattr(host, "pair_pointer", lambda _: payment)

    def unopened(*_):
        pytest.fail("funding opened the private training archive before a time reservation")

    monkeypatch.setattr(host, "donor_inputs", unopened)
    monkeypatch.setattr(host, "collect_snapshot", lambda _: snapshot)
    monkeypatch.setattr(host, "physical_scope", lambda *_a, **_k: digest("test card"))
    monkeypatch.setattr(progressive_round029, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        progressive_round028,
        "git",
        lambda _, op, *a: velocity_policy()["donor_revision"] if op == "rev-parse" else None,
    )
    grant = host.authorize_timing(tmp_path, 10, revision)
    for k, charge in design["cost"].items():
        assert load(grant)["resources"][k] >= 2 * charge
    monkeypatch.setattr(
        host, "donor_inputs", lambda *_: (projection, design, host.matching_scope(pairs, stream))
    )
    args = SimpleNamespace(
        action="qualify",
        authority=grant,
        workspace=str(tmp_path),
        image=image if route in {"supplied", "prelaunch"} else None,
        gpu_base_image=None,
        archive=None,
        qualification_archive=None,
    )
    launches = []

    packs = []

    def settle(out, img, readonly_inputs):
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        binds = [
            dict(Type="bind", Destination=dest, Source=str(path), RW=False)
            for path, dest in readonly_inputs
        ]
        binds.append(dict(Type="bind", Destination="/work/output", Source=str(out), RW=True))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=img,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=binds,
                )
            ],
        )

    def stage(_host, _source, img, out, argv, *, seconds, readonly_inputs, gpu, record_lease=False):
        assert img == image and seconds > 0
        mounts = {dest: path for path, dest in readonly_inputs}
        settle(out, img, readonly_inputs)
        if not gpu:
            assert warm and not record_lease and argv[4] == "package"
            assert set(mounts) == {
                "/work/inputs/candidate.tar.gz",
                "/work/inputs/candidate-wheel.py",
            }
            with tarfile.open(mounts["/work/inputs/candidate.tar.gz"]) as archive:
                assert all(not name.startswith("run-state/") for name in archive.getnames())
            wheel = out / "oeis_learn-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as body:
                body.writestr(runtime.INSTALLED, json.dumps(snapshot))
                body.writestr(runtime.BOOTSTRAP[4:], bootstrap.read_bytes())
            packs.append(wheel)
            return
        assert record_lease and "/work/inputs/candidate.tar.gz" not in mounts
        req = load(mounts["/work/inputs/request.json"])
        if warm:
            assert argv[4] == "run" and argv[-1] == digest(snapshot)
            assert mounts[argv[5]] == packs[0] and req["runtime"]["parent"] == base
        else:
            assert argv[-1].endswith("progressive_velocity_timing")
        launches.append(out)
        eligible = sum(k in projection["donors"] for k in projection["order"])
        calibration = seal(
            dict(
                version="round029-matching-observation/v1",
                request=projection["identity"],
                initial_weights=projection["initial_weights"],
                positive_rows=32,
                eligible_rows=eligible,
                wins=0,
                extra_passes=1,
                median_positive=1.0,
                median_auxiliary=1.0,
                weight=0.25,
                reason=None,
                elapsed_us=20,
                device="cuda",
            )
        )
        methods = {}
        for method, terms in design["methods"].items():
            policy = update_policy(
                model, method, microbatch=8, matching_weight=0.25 if terms["matching"] else None
            )
            methods[method] = dict(
                startup_us=5,
                policy=policy["identity"],
                samples={
                    k: dict(w, elapsed_us=17, auxiliary_forward_us=int(w["passes"] > 0))
                    for k, w in terms["samples"].items()
                },
            )
        matrix = seal(
            dict(
                version="round029-update-timings/v1",
                design=design["identity"],
                calibration=calibration,
                methods=methods,
            )
        )
        from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary

        samples = out / "telemetry/samples.jsonl"
        samples.parent.mkdir()
        samples.write_text(
            '{"time":0,"gpu":{"gpu_busy_percent":20}}\n'
            '{"time":0.0005,"gpu":{"gpu_busy_percent":60}}\n'
        )
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    request=req["identity"],
                    matrix=matrix,
                    device="hip",
                    elapsed_us=1000,
                    worker_cpu_us=1500,
                    telemetry=usage_summary(samples.parent, 1000),
                    peak_device_bytes=1024,
                )
            ),
        )
        atomic_write(out / "gpu-lease.json", {"elapsed_us": 1200})

    monkeypatch.setattr(host, "stage", stage)

    engine = SimpleNamespace(image=lambda img: img)
    if route == "prelaunch":
        inspected = []

        def missing_once(ref):
            inspected.append(ref)
            if len(inspected) == 2:
                raise RuntimeError("image unavailable before device dispatch")
            return ref

        engine = SimpleNamespace(image=missing_once)
    if cold_retry:
        from oeis_learn.container_runtime import Engine

        cli = tmp_path / "podman-fixture"
        cli.write_text("#!/bin/sh\necho 'storage fixture detail' >&2\nexit 125\n")
        cli.chmod(0o700)
        engine = Engine("podman", executable=str(cli), details={"host": {"serviceIsRemote": False}})

    def run():
        return progressive_round029.execute(
            args,
            tmp_path,
            host_factory=lambda *_a, **_k: SimpleNamespace(engine=engine),
        )

    if route == "prelaunch":
        failed, code = run()
        assert code and "before device dispatch" in failed["reason"] and not launches
        assert 0 < failed["elapsed_us"] < load(grant)["total_us"]
        assert failed["research_dispatches"] == 0
    if cold_retry:
        from oeis_learn.pipeline_validation.progressive_velocity_image import failure_details

        failed, code = run()
        assert code and failed["mechanical_status"] == "FAILED" and not launches
        assert "storage fixture detail" in failed["reason"]
        assert 0 < failed["elapsed_us"] < load(grant)["total_us"]
        assert failed["remaining_authority"] + failed["elapsed_us"] == load(grant)["total_us"]
        assert failed["charged_tokens"] == failed["research_dispatches"] == 0
        details = failure_details(tmp_path, load(grant)["identity"], failed["first_error"])
        item = details["steps"]["image_build"]
        assert (
            item["process"]["returncode"] == 125 and item["stderr_tail"] == "storage fixture detail"
        )
        assert details["steps"]["operation"]["storage-before"]["workspace"]["free_bytes"] > 0
        cli.write_text(f"#!/bin/sh\necho '{image}'\n")
    completed, code = run()
    assert code == 0, completed
    allocation = completed["receipt"]["storage"]
    assert allocation["after"]["workspace"]["free_bytes"] > 0
    assert allocation["transport"]["inodes"] == 3 if warm else allocation["transport"] is None
    assert run() == (completed, 0) and len(launches) == 1
    assert (
        completed["charged_tokens"] == design["cost"]["tokens"] * (1 + int(route == "prelaunch"))
        and completed["research_dispatches"] == 0
    )
    args.action = "verify"
    args.authority = None
    verified, code = run()
    assert (
        code == 0 and verified["last_committed_ref"] == completed["last_committed_ref"]
    ), verified["reason"]
    args.action = "preflight"
    info, code = run()
    assert code == 0 and info["inventory"]["gpu_dispatches"] == 0
    prices = info["inventory"]["prices"]
    assert prices["timed_updates"]["update_count"] == design["cost"]["attempts"]
    scope = prices["timed_updates"]["preparation_cost"]
    assert scope["launch_to_decision_us"] >= scope["incremental"]["occupied"]["union_us"] > 0
    assert scope["incremental"]["ledger_us"] == completed["elapsed_us"]
    assert scope["production_occupied"] == scope["incremental"]["occupied"]
    assert sum(row["production_us"] for row in scope["by_beneficiary"].values()) == pytest.approx(
        scope["production_occupied"]["union_us"]
    )
    assert scope["incremental"]["resources"]["worker_cpu_us"] == {
        "observed": 1500,
        "missing_operations": 1 + int(cold_retry or route == "prelaunch"),
    }
    assert scope["incremental"]["resources"]["busy_us"] == {
        "observed": 200,
        "missing_operations": int(route == "prelaunch"),
    }
    assert scope["incremental"]["resources"]["covered_us"]["observed"] == 500
    assert "optimizer_to_4M_recipe_03" in prices["wave"]["missing"]
    assert "optimizer_to_4M_recipe_02" in prices["wave"]["missing"]
    assert (
        prices["timed_updates"]["sampled_methods"]["02"]
        == completed["receipt"]["units"]["methods"]["02"]
    )
    assert prices["timed_updates"]["matching_scope"]["trial_seed"] == 17
    assert prices["wave"]["operations"]["ordinary_matching_calibration"] == 1
    assert prices["wave_with_reserve_us"] is None
    samples = launches[0] / "telemetry/samples.jsonl"
    original = samples.read_bytes()
    samples.write_text('{"time":0,"gpu":null}\n')
    changed, code = run()
    assert code == 0 and "timed_updates" not in changed["inventory"]["prices"]
    assert "sampled activity" in changed["inventory"]["prices"]["timing_reason"]
    samples.write_bytes(original)
    if warm:
        assert len(packs) == len(launches) == 1
        data = packs[0].read_bytes()
        packs[0].write_bytes(data + b"modified")
        changed, code = run()
        assert code == 0 and "timed_updates" not in changed["inventory"]["prices"]
        assert len(packs) == len(launches) == 1
        packs[0].write_bytes(data)
        (tmp_path / "uv.lock").write_text("different dependencies")
        git("commit", "-qam", "change a dependency")
        assert not runtime.compatible(tmp_path, base_revision, git("rev-parse", "HEAD"))
    path = launches[0] / "gpu-lease.json"
    path.write_text('{"elapsed_us":1}')
    new, code = run()
    assert code == 0 and "timed_updates" not in new["inventory"]["prices"] and len(launches) == 1


def test_broader_disjointness_failure_reopens_both_treatments_without_device_or_new_charge(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_round028, progressive_round029
    from oeis_learn.pipeline_validation import progressive_velocity_broad_host as broad
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_velocity_supply import _area

    folder = tmp_path / "run-state/discovery" / SESSION
    pointer = seal(dict(grant=digest("ordinary payment"), receipt=digest("paid matrix")))
    family = seal(
        dict(kind="families", grant=digest("prior supply"), receipt=digest("family census"))
    )
    atomic_write(folder / "preparation/update-prices.json", pointer)
    atomic_write(_area(tmp_path, "families") / "funded.json", family)
    # Grant routing can encounter the older ordinary payment before testing the broader parent.
    atomic_write(
        _area(tmp_path, "matching_pairs") / "funded.json",
        seal(dict(kind="matching_pairs", grant=family["grant"], receipt=digest("pairs"))),
    )
    revision = "a" * 40
    monkeypatch.setattr(progressive_round029, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        progressive_round028,
        "git",
        lambda root, *args: (
            None
            if "symbolic-ref" in args
            else progressive_round029.velocity_policy()["donor_revision"]
        ),
    )
    source = Path(__file__).resolve().parents[2]
    plan = dict(stream=digest("stream"), start=25, draws=2, slots=8, input_bytes=8192)
    from oeis_learn.generalization.progressive.generator import default_policy
    from oeis_learn.generalization.progressive.training_rows import pair_seeds

    old = dict(seeds=pair_seeds(digest("old"), 0), law=default_policy(seed=29))

    def work_plan(root, *, overlap=None):
        if overlap is None:
            return plan
        return dict(plan, start=0, frame=broad.fresh_start.frame(overlap, old, 2))

    monkeypatch.setattr(broad, "preparation_plan", work_plan)
    args = SimpleNamespace(
        workspace=tmp_path,
        action="qualify",
        authority=broad.authorize(tmp_path, 5, revision),
        archive=None,
        qualification_archive=None,
    )
    calls = []

    def custody(argv, out, **options):
        calls.append(argv)
        binding = load(Path(argv[-1]))
        fresh = "overlap" in binding["origins"]
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(
            out / "custody.json",
            seal(
                dict(
                    origins=binding["origins"],
                    ordinary_slots=0 if fresh else 8,
                    stream=plan["stream"],
                    sources=[],
                    physical=digest("device"),
                    calibration=dict(
                        observation=digest("weight"), measurement=digest("paid matrix")
                    ),
                    disposition="no_disjoint_ordinary_support"
                    if fresh
                    else "requires_fresh_disjoint_start",
                    counts=(
                        dict(
                            candidates=2,
                            eligible=0,
                            excluded=2,
                            reference_rejected=0,
                            ordinary_eligible=0,
                            prospective_eligible=0,
                            prospective_familiar=0,
                            prospective_native=0,
                        )
                        if fresh
                        else dict(candidates=8, excluded=3)
                    ),
                    drawn=2 if fresh else 0,
                    training=None,
                )
            ),
        )

    monkeypatch.setattr(supervision, "launch", custody)

    def no_device(*args, **kwargs):
        pytest.fail("a protected ordinary stream acquired a device")

    def run():
        return progressive_round029.execute(args, source, host_factory=no_device)

    import os
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_research_resources import register_velocity_capacity

    def reserve_inodes(n):
        path = tmp_path / f"mount-{n}.json"
        atomic_write(path, seal(dict(version="round029-mount-plan/v1", owners=[],
                                     floors=[dict(path=".", bytes=0, inodes=n)])))
        register_velocity_capacity(tmp_path, dict(owner="custodian", format="round029-mount-plan/v1",
                                                  artifact=reference(tmp_path, path)))

    reserve_inodes(os.statvfs(tmp_path).f_files + 1000)
    blocked, code = run()
    assert code == 2 and blocked["next_action"] == "inspect_capacity", blocked
    assert blocked["deficit_inodes"] >= 1000 and blocked["deficit_bytes"] == 0
    assert not calls and blocked["elapsed_us"] == 0
    reserve_inodes(0)
    first, code = run()
    assert code == 0, first
    assert first["receipt"]["disposition"] == "requires_fresh_disjoint_start"
    assert first["charged_tokens"] == 0 and first["research_dispatches"] == 0
    assert run() == (first, 0) and len(calls) == 1
    args.action, args.authority = "verify", None
    checked, code = run()
    assert code == 0 and checked["verification_scope"] == "private_broader_timings", checked
    assert checked["last_committed_ref"] == first["receipt"]["identity"]
    assert broad.inventory(tmp_path, source) == first["receipt"]
    c = next(folder.glob("qualification/*/operations/*/attempt-*/custodian/custody.json"))
    original_marker = load(folder / broad.MARKER)
    args.authority = broad.authorize(tmp_path, 5, revision, fresh=True)
    args.action = "qualify"
    fresh, code = run()
    assert code == 0 and fresh["receipt"]["disposition"] == "no_disjoint_ordinary_support", fresh
    assert fresh["charged_tokens"] == fresh["research_dispatches"] == 0
    assert fresh["receipt"]["charged_work"] == dict(draws=2, native_work=0, attempts=0, tokens=0)
    assert fresh["receipt"]["overlap"] == original_marker["receipt"]
    assert fresh["receipt"]["start"] is None and fresh["receipt"]["device"] is None
    assert run() == (fresh, 0) and len(calls) == 2
    args.action, args.authority = "verify", None
    verified, code = run()
    assert code == 0 and verified["verification_scope"] == "private_fresh_start_timings", verified
    assert load(folder / broad.MARKER) == original_marker
    assert broad.inventory(tmp_path, source, fresh=True) == fresh["receipt"]

    helper = source / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    def action(name, grant=None):
        if name == "preflight":
            prices = dict(fresh_start=fresh["receipt"], broader_updates=first["receipt"],
                          timed_updates=dict(measurement=fresh["receipt"]["ordinary_measurement"]))
            return dict(inventory=dict(prices=prices), reason="inspect the fresh frame")
        args.action, args.authority = name, grant
        value, code = run()
        assert code == 0, value
        return value
    summary = runpy.run_path(str(helper))["donor_timings"](action, tmp_path, revision, "5", fresh=True)
    assert summary["scope"] == "fresh_start_timings" and summary["fresh_start"] == fresh["receipt"]
    assert summary["elapsed_us"] + summary["remaining_us"] == 300_000_000
    assert len(calls) == 2

    api = runpy.run_path(str(helper))
    driver = api["donor_readiness"]
    child = load(source / "specs/016-measure-learning-velocity/validation/round029-child-qualification.json")
    monkeypatch.setitem(driver.__globals__, "reopen_child_rehearsal", lambda *_: child)
    pre = progressive_round029.preflight(tmp_path)
    pre["inventory"]["prices"].update(action("preflight")["inventory"]["prices"])
    def ready_action(name, grant=None):
        return pre if name == "preflight" else action(name, grant)
    saved = tmp_path / "fresh-start.json"
    atomic_write(saved, summary)
    ready = driver(ready_action, tmp_path, source, revision, "5", saved)
    assert ready["status"] == "COMPLETE" and len(calls) == 2
    assert ready["files"]["fresh-start.json"]["fresh_start"] == fresh["receipt"]
    assert ready["files"]["verification.json"]["last_committed_ref"] == fresh["receipt"]["identity"]
    assert ready["files"]["status.json"]["last_committed_ref"] == fresh["receipt"]["identity"]
    assert ready["files"]["recovery.json"] == child
    assert ready["checks"]["receipt-links"] == "PASS"

    # A different balance or historical price must leave the other checks intact.
    for fault in ("debit", "price"):
        altered = tmp_path / (fault + ".json")
        if fault == "debit":
            atomic_write(altered, dict(summary, elapsed_us=summary["elapsed_us"] + 1,
                                       remaining_us=summary["remaining_us"] - 1))
        else:
            atomic_write(altered, summary)
            pre["inventory"]["prices"]["broader_updates"] = dict(identity=digest("other overlap"))
        partial = driver(ready_action, tmp_path, source, revision, "5", altered)
        assert partial["status"] == "PARTIAL" and partial["checks"]["receipt-links"] == "FAILED"
        assert partial["checks"]["recovery"] == partial["checks"]["verification"] == "PASS"
        assert "fresh_start" not in partial["files"]["fresh-start.json"]
        assert partial["files"]["status.json"]["progress"]["elapsed_us"] == summary["elapsed_us"]
    assert len(calls) == 2

    import json

    c.write_text(json.dumps(seal(dict(load(c), counts=dict(candidates=8, excluded=0)))))
    with pytest.raises(ValueError, match="measured exclusion"):
        broad.inventory(tmp_path, source)


def test_broader_prices_survive_curve_reporting_but_still_pin_shared_math(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_velocity_timing_host as timing

    relative = "src/oeis_learn/generalization/progressive/"
    metrics, trials = relative + "velocity_metrics.py", relative + "trials.py"
    binding = {"version": "round029-broader-timing/v2"}
    original, current = tmp_path / "original", tmp_path / "current"
    for root, journal in ((original, "pass"), (current, "def velocity_report(self): return {}")):
        path = root / trials
        path.parent.mkdir(parents=True)
        path.write_text("def integer(n): return n\nclass TrialJournal:\n    " + journal + "\n")
    monkeypatch.setattr(timing, "original_price_source", lambda *_: (
        original, {metrics: "old", trials: "old"}, {metrics: "reader", trials: "new journal"}))
    timing.checked_price_source(tmp_path, current, binding)
    path = current / trials
    path.write_text(path.read_text().replace("return n", "return 0"))
    with pytest.raises(ValueError, match="workload changed"):
        timing.checked_price_source(tmp_path, current, binding)
