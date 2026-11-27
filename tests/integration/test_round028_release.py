import pytest
from tests.integration.test_round028_receipts import original as original  # noqa: F401


def test_scale_reads_both_encodings_and_replays_every_charge_twice(tmp_path):
    from oeis_learn.pipeline_validation.progressive_release_checks import exercise_scale

    targets = dict(
        candidate_score_bytes=64000,
        campaign_bytes=4800,
        calibration_bytes=4900,
        tasks=12,
        journal_events=80,
    )
    result = exercise_scale(tmp_path, targets)
    assert result["targets"] == targets
    assert result["journal"]["committed"] == 40
    assert result["journal"]["events"] == 80
    assert result["journal"]["cold_us"] > 0 and result["journal"]["warm_us"] > 0
    assert (
        result["corpora"]["compressible"]["compressed_bytes"]
        < result["corpora"]["varied"]["compressed_bytes"]
    )
    assert all(
        v["expanded_bytes"] >= 64000 and v["files"] == 12 for v in result["corpora"].values()
    )
    assert result["campaign_bytes"] >= 4800 and result["calibration_bytes"] >= 4900
    assert result["peak_allocated_bytes"] > 0 and result["peak_inodes"] >= 12


def test_release_scale_uses_the_larger_measured_workload():
    from oeis_learn.pipeline_validation.progressive_release_checks import scale_targets

    targets = scale_targets(dict(candidate_score_bytes=300_000_000, campaign_bytes=10), 420)
    assert targets == dict(
        candidate_score_bytes=300_000_000,
        campaign_bytes=56_568_256,
        calibration_bytes=49_000_000,
        tasks=420,
        journal_events=290_000,
    )


def test_real_learner_killed_in_backward_replays_only_the_second_update(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, seal
    from oeis_learn.pipeline_validation.progressive_generator_probe import measured_updates
    from oeis_learn.pipeline_validation.progressive_release_faults import learner_fault
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import sample_stream
    from oeis_learn.generalization.progressive.models import resolve_profile
    from tests.unit.test_round028_training import training_rows
    from tests.unit.test_progressive_training_policy import profile

    rows = training_rows()
    base = seal(
        dict(
            shards=[rows],
            layout={"probes": {}},
            profile=profile(),
            update_policy={"microbatch_size": 8},
        )
    )
    sample = sample_stream(base, digest("release"), base["identity"])
    sample = seal(
        dict(
            sample,
            profile=resolve_profile(
                base["profile"]["base_recipe"], shape="smaller_width", seed=7, dropout=0
            ),
        )
    )
    path = tmp_path / "training.json"
    atomic_write(path, sample)
    reference = measured_updates(
        sample["profile"],
        sample["batches"],
        "ordinary",
        tmp_path / "baseline",
        device="cpu",
        microbatch=8,
    )
    receipt = learner_fault(tmp_path, path, reference["final"], device="cpu")
    assert receipt["useful_tokens"] == sum(reference["charged_tokens"])
    assert receipt["discarded_updates"] == 1 and receipt["failed_us"] > 0


def test_native_fit_interruption_preserves_unresolved_work_and_restarts(
    fitting_client, monkeypatch
):
    import os
    from oeis_learn.synthesis.admission import VisiblePrompt

    client = fitting_client
    prompt = VisiblePrompt(("-7", "19"), ("-19", "59"))
    source = "(program (return (add (mul (param.get p0) $index) (param.get p1))))"
    client.runtime_identity("bigint-body/v1")
    written = os.write
    interrupted = []

    def interrupt(fd, body):
        n = written(fd, body)
        process = client.worker.process
        if not interrupted and process and fd == process.stdin.fileno() and n == len(body):
            interrupted.append(process.pid)
            process.kill()
        return n

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", interrupt)
        result = client.fit(source, prompt)
    assert interrupted and result["outcome"] != "verified_solution"
    assert client.execute("(program (return $index))", ["-7", "19"])["values"] == ["-7", "19"]


@pytest.mark.parametrize("failure", [True, False])
def test_one_workstation_command_collects_release_or_diagnostics(
    tmp_path, monkeypatch, capsys, failure
):
    import json
    import runpy
    import subprocess
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_round028 import SESSION
    from oeis_learn.generalization.progressive.program_search_release import PHASES

    source = Path(__file__).resolve().parents[2]
    collector = runpy.run_path(
        str(source / "specs/015-learned-program-search/validation/collect_audit.py")
    )
    revision = "a" * 40
    (tmp_path / "run-state/discovery-tools").mkdir(parents=True)
    share = tmp_path / SESSION / "share/release-acceptance" / revision / "round028-release.zip"
    share.parent.mkdir(parents=True)
    share.write_bytes(b"fixture")
    receipt = dict(status="VERIFIED", candidate=revision, pilot="child-report")
    calls = []

    def dispatch(argv, **kwargs):
        action = argv[argv.index("--action") + 1]
        calls.append(action)
        if failure:
            value = dict(
                mechanical_status="BLOCKED",
                completed=["scale"],
                failures={"hardware": "ReceiptGap"},
                capacity={"deficit_bytes": 11, "private_source": "sentinel"},
            )
        elif action == "verify" and "--workspace" in argv:
            value = dict(
                mechanical_status="COMPLETE",
                pilot=dict(
                    private_replay="VERIFIED", public_recomputation={"report": receipt["pilot"]}
                ),
            )
        elif action == "verify":
            assert "--workspace" not in argv and Path(argv[-1]).read_bytes() == b"fixture"
            value = receipt
        else:
            value = dict(mechanical_status="COMPLETE", receipt=receipt, completed=PHASES)
        kwargs["stdout"].write(json.dumps(value).encode())
        return SimpleNamespace(returncode=2 if failure else 0)

    monkeypatch.setattr(subprocess, "run", dispatch)
    code = collector["collect"](tmp_path, revision, "collect-release")
    packet = Path(capsys.readouterr().out.strip())
    if failure:
        assert code == 2 and calls == ["release-acceptance"]
        with zipfile.ZipFile(packet) as archive:
            assert archive.namelist() == ["summary.json"]
            output = archive.read("summary.json")
            assert b"sentinel" not in output
            step = json.loads(output)["steps"]["acceptance"]
            assert step["failures"] == {"hardware": "ReceiptGap"} and step["capacity"] == {
                "deficit_bytes": 11
            }
    else:
        assert code == 0 and packet.read_bytes() == b"fixture"
        assert calls == ["release-acceptance"] * 3 + ["verify", "verify"]


def test_candidate_release_recovers_stages_and_rejects_nested_private_bytes(original, monkeypatch):
    """The installed controller and standalone reader share only their published packet."""
    import errno
    import io
    import json
    import zipfile
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive import program_search_pilot
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        reference,
        seal,
    )
    from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
        verify_transport,
    )
    from oeis_learn.pipeline_validation import progressive_round028 as cli
    from oeis_learn.pipeline_validation import progressive_round028_receipts as receipts
    from oeis_learn.pipeline_validation import progressive_round028_release as release
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_release_checks import release_cases
    from oeis_learn.pipeline_validation.progressive_round028_audit import ORIGINAL
    from tests.integration.test_round028_receipts import REVISION, SOURCE

    o = original
    training = receipts.recover(o.workspace, o.folder, SOURCE, REVISION, o.store)
    monkeypatch.setattr(receipts, "recover", lambda *a, **kw: training)
    with zipfile.ZipFile(o.folder / cli.ARCHIVE) as z:
        audit = archive_bytes({"audit.json": z.read("audit.json")})
    pilot_path = o.folder / release.PILOT_SHARE
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    from oeis_learn.generalization.progressive.program_search_pilot import ARMS, LOOKS

    prices = [
        [f"{arm}-{look}", role, 8]
        for arm in ARMS
        for look in LOOKS
        for role in ("train", "search", "score")
    ]
    prices += [[f"incumbent-{LOOKS[-1]}", role, 5] for role in ("search", "score")]
    prior = dict(
        looks=[{"zero_reused": True}], design={"prices": prices}, costs=[], research_gpu_us=10
    )
    pilot_path.write_bytes(
        archive_bytes(
            {
                "pilot.json": json.dumps(prior).encode(),
                "round028-parent.zip": archive_bytes({"round028.zip": audit}),
            }
        )
    )
    pilot = dict(report=digest("prior-pilot"), next_action="RETAIN_REFERENCE", assessment="SEALED")
    # The original pilot has its own archive tests. Supply only its decision in this fixture.
    monkeypatch.setattr(release, "verify_pilot", lambda *a: pilot)
    monkeypatch.setattr(program_search_pilot, "verify_pilot", lambda *a: pilot)
    campaign = seal(dict(payload=dict(tasks=["fixture"])))
    path = o.workspace / ORIGINAL / "prepare-research/frozen/campaign.json"
    atomic_write(path, campaign)
    parent = SimpleNamespace(
        campaign=dict(o.store.campaign, policy={"campaign": campaign["identity"]}),
        state=o.store.state,
    )
    monkeypatch.setattr(cli, "reopen", lambda *a, **kw: {"inputs": [reference(o.workspace, path)]})

    expected = digest("uninterrupted-model")

    def rehearse(*args, output_area, **kwargs):
        probe = o.folder / output_area
        req = digest("hip-request")
        stage = probe / "stages" / req[7:]
        retain(probe / "training.json", o.data)
        proof = seal(dict(details={"arms": {"ordinary": {"final": expected}}}))
        retain(stage / "producer.json", proof)
        boundary = seal(dict(producer=reference(stage, stage / "producer.json")))
        retain(stage / "receipt.json", boundary)
        return (
            dict(
                qualification_receipts={
                    "hip": {
                        "request": req,
                        "boundary": boundary["identity"],
                        "producer": proof["identity"],
                    }
                }
            ),
            0,
        )

    monkeypatch.setattr(release, "rehearse", rehearse)

    class Roles:
        phase = "cpu"

        def __init__(self, args, source, root, rev, area, reg, host):
            self.area = area
            self.native = digest("native-" + self.phase)

        def __call__(self, role, data, **kwargs):
            self.image = training[self.phase]["image"]
            self.last_producer = seal(dict(role=role, request=data["identity"]))
            self.last_wall_us, self.last_lease_us = 10, 3
            stage = self.area / "fixture" / data["block"]
            self.last_output = stage
            retain(stage / "end.json", seal(dict(elapsed_us=10)))
            self.last_boundary = stage / "boundary.json"
            retain(self.last_boundary, seal(dict(timing=reference(stage, stage / "end.json"))))
            if role == "release_interrupt":
                assert self.phase == "hip" and data["expected"] == expected
                return dict(
                    final=expected,
                    useful_updates=2,
                    discarded_updates=1,
                    useful_tokens=sum(training["hip"]["arms"]["ordinary"]["charged_tokens"]),
                    failed_us=2,
                    restored_us=1,
                )
            if role == "storage":
                return dict(errno=errno.ENOSPC, restored=True, bytes_exercised=8)
            if role == "release_scale":
                t = data["targets"]
                corpus = dict(
                    expanded_bytes=t["candidate_score_bytes"],
                    compressed_bytes=42,
                    files=t["tasks"],
                    cold_us=10,
                    warm_us=5,
                )
                return dict(
                    targets=t,
                    corpora=dict(varied=corpus, compressible=dict(corpus, compressed_bytes=10)),
                    campaign_bytes=t["campaign_bytes"],
                    calibration_bytes=t["calibration_bytes"],
                    peak_allocated_bytes=t["candidate_score_bytes"],
                    peak_inodes=t["tasks"] + 10,
                    journal=dict(
                        events=t["journal_events"],
                        committed=t["journal_events"] // 2,
                        bytes=500,
                        cold_us=20,
                        warm_us=3,
                    ),
                )
            retain(stage / "test.json", seal(dict(role=role)))
            ref = reference(stage, stage / "test.json")
            if role == "recovery":
                return dict(tests=3, report=ref)
            return dict(
                mode=data["mode"],
                files=len(release_cases()),
                tests=200,
                roster=digest(release_cases()),
                elapsed_us=10,
                junit=ref,
            )

    monkeypatch.setattr(release, "ReleaseRole", Roles)
    host = lambda *a, **kw: SimpleNamespace(policy={"free_disk_floor_bytes": 1})
    result, code = release.release_command(
        None, SOURCE, o.workspace, o.folder, REVISION, parent, host
    )
    assert code == 0 and result["completed"] == list(release.PHASES)
    repeated, code = release.release_command(
        None, SOURCE, o.workspace, o.folder, REVISION, parent, host
    )
    assert code == 0 and repeated == result
    path = o.folder / result["archive"]
    assert verify_transport(path) == result["receipt"]
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    with pytest.raises(ValueError):
        verify_transport(
            io.BytesIO(
                archive_bytes(
                    dict(parts, **{"rehearsal.zip": parts["rehearsal.zip"] + b"private-trace"})
                )
            )
        )
    record = json.loads(parts["release.json"])
    record["stages"]["interrupt"]["result"]["useful_updates"] = 1
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode

    with pytest.raises(ValueError):
        verify_transport(
            io.BytesIO(archive_bytes(dict(parts, **{"release.json": encode(seal(record))})))
        )


def test_release_scale_credits_only_the_bound_completed_assignment(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        reference,
        seal,
    )
    from oeis_learn.pipeline_validation import progressive_round028_release as release
    from oeis_learn.pipeline_validation.progressive_release_checks import HISTORICAL

    reg = seal(dict(source=digest("source")))
    packet = seal(dict(block="release-scale", targets=HISTORICAL))
    req = seal(
        dict(
            registration=reg["identity"],
            source=reg["source"],
            role="release_scale",
            data=packet["identity"],
        )
    )
    key = req["identity"]
    area = tmp_path
    stage = area / "roles" / key[7:]
    atomic_write(stage / "request.json", req)
    atomic_write(area / "assignments/release-scale-release_scale.json", seal(dict(request=key)))
    monkeypatch.setattr(release, "space", lambda _: dict(free_bytes=1000, free_inodes=1))

    def admission():
        return release.release_admission(area, HISTORICAL, 500, registration=reg)

    before = admission()
    assert (
        not before["credited_scale"] and before["deficit_bytes"] == before["required_bytes"] - 500
    )
    producer = seal(
        dict(
            version=release.ReleaseRole.producer_version,
            request=key,
            result={"targets": HISTORICAL},
        )
    )
    atomic_write(stage / "producer.json", producer)
    atomic_write(
        stage / "boundary.json",
        seal(dict(request=key, producer=reference(stage, stage / "producer.json"))),
    )
    assert admission() == dict(
        free_bytes=1000,
        free_inodes=1,
        required_bytes=0,
        required_inodes=0,
        credited_scale=True,
        deficit_bytes=0,
        deficit_inodes=0,
    )
    (stage / "producer.json").write_text("{}")
    with pytest.raises(ValueError):
        admission()
