import pytest


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
    receipt = dict(status="VERIFIED", candidate=revision)
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
        assert calls == ["release-acceptance"] * 3 + ["verify"]
