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


@pytest.mark.parametrize(
    "fault,expected_calls",
    [("none", 2), ("missing_receipt", 0), ("other_grant", 0), ("other_status", 0), ("extra_charge", 2)],
)
def test_saved_learner_review_uses_the_original_grant(tmp_path, fault, expected_calls):
    validation = Path(__file__).resolve().parents[2] / "specs/016-measure-learning-velocity/validation"
    helper = runpy.run_path(str(validation / "replay_learner.py"))
    grant_id = helper["GRANT"]
    authority = tmp_path / "run-state/discovery-authorizations" / f"grant-{grant_id}.json"
    retained_grant = digest("other") if fault == "other_grant" else "sha256:" + grant_id
    atomic_write(authority, {"identity": retained_grant})
    receipt = {"identity": helper["RECEIPT_ID"]}
    saved = dict(
        receipt_path="retained/receipt.json", receipt=receipt, elapsed_us=100,
        charged_tokens=20, research_dispatches=0,
    )
    if fault != "missing_receipt":
        atomic_write(tmp_path / saved["receipt_path"], receipt)
    calls = []

    def invoke(action, *args):
        if action == "status":
            status_id = digest("other") if fault == "other_status" else helper["STATUS_ID"]
            return dict(
                owner_active=False, last_committed_ref=receipt["identity"],
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
    mounts = load(issued[0] / "container-inspect.json")
    mounts[0]["Mounts"][0]["RW"] = True
    (issued[0] / "container-inspect.json").unlink()
    atomic_write(issued[0] / "container-inspect.json", mounts)
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
