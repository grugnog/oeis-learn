"""Host protocol fixtures use a recorded container boundary. Numerical parity has a real CPU test."""

import shutil
import subprocess
import sys
import json
from pathlib import Path
from types import SimpleNamespace

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.pipeline_validation import progressive_round028, progressive_round029
from oeis_learn.pipeline_validation import progressive_velocity_donation as donation


def test_scoped_donation_uses_one_installed_dispatch_and_reopens_twice(tmp_path, monkeypatch):
    revision = "1" * 40
    image = digest("qualified image")
    payload = tmp_path / "original-training.json"
    atomic_write(payload, {"fixture": True})
    plan = seal(
        dict(
            parent=digest("milestone"),
            checkpoint=digest("4M full state"),
            package=digest("training"),
            old_policy=digest("ordinary"),
            new_policy=digest("00"),
            parent_updates=3150,
            prior_useful=4_000_210,
            prior_charge=4_000_210,
            update_tokens=1683,
            rows=digest("matched batch"),
            next_rows=digest("next 32"),
        )
    )
    inputs = dict(
        parent=dict(milestone=dict(seeds=dict(initialization=782196838), stream=digest("stream"))),
        plan=plan,
        training=dict(path="training.json", sha256=digest("fixture"), bytes=21),
    )
    monkeypatch.setattr(
        donation, "donated_ordinary", lambda _: (inputs, [(payload, "/work/inputs/training.json")])
    )
    monkeypatch.setattr(donation, "collect_snapshot", lambda _: ["source"])
    monkeypatch.setattr(donation, "physical_scope", lambda *_a, **_k: digest("card"))
    monkeypatch.setattr(progressive_round029, "candidate_revision", lambda *_: revision)
    monkeypatch.setattr(
        progressive_round028,
        "git",
        lambda _, op, *a: velocity_policy()["donor_revision"] if op == "rev-parse" else None,
    )
    authority = donation.authorize_donation_check(tmp_path, 30, revision)
    args = SimpleNamespace(
        action="qualify",
        authority=authority,
        workspace=str(tmp_path),
        archive=None,
        qualification_archive=None,
        image=image,
    )
    host = SimpleNamespace(engine=SimpleNamespace(image=lambda v: v))
    runs = []

    def stage(_host, _source, img, out, argv, *, seconds, readonly_inputs, gpu, record_lease):
        runs.append(out)
        assert 0 < seconds <= 1680 and gpu and record_lease and img == image
        assert argv[-1].endswith("progressive_velocity_continuation")
        request = load(
            dict((dst, src) for src, dst in readonly_inputs)["/work/inputs/request.json"]
        )
        row = dict(
            parent_weights=digest("parent state"),
            weights=digest("next state"),
            optimizer=digest("adamw"),
            rng=digest("all rngs"),
            updates=3151,
            tokens=4_001_893,
            useful_tokens=4_001_893,
            next_rows=plan["next_rows"],
        )
        comparison = seal(
            dict(
                plan=plan["identity"],
                parent_updates=3150,
                candidate=row,
                reference=row,
                charged_tokens=3366,
            )
        )
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    version="round029-donation-result/v1",
                    request=request["identity"],
                    device="hip",
                    comparison=comparison,
                    elapsed_us=1,
                    peak_device_bytes=999,
                )
            ),
        )
        binds = [
            dict(Type="bind", Source=str(src), Destination=dst, RW=False)
            for src, dst in readonly_inputs
        ]
        binds.append(dict(Type="bind", Source=str(out), Destination="/work/output", RW=True))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    Mounts=binds,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(returncode=0, outcome="COMPLETE", cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        atomic_write(out / "gpu-lease.json", dict(elapsed_us=2))

    monkeypatch.setattr(donation, "stage", stage)
    first, code = progressive_round029.execute(args, tmp_path, host_factory=lambda *_a, **_k: host)
    assert code == 0, first
    assert first["mechanical_status"] == "COMPLETE" and first["charged_tokens"] == 3366
    assert first["research_dispatches"] == 0 and len(runs) == 1
    for _ in range(2):
        assert progressive_round029.execute(args, tmp_path) == (first, 0)
    assert first["last_committed_ref"] == load(tmp_path / first["receipt_path"])["identity"]
    assert len(runs) == 1
    verify = SimpleNamespace(**dict(vars(args), action="verify", authority=None))
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    private, code = progressive_round029.execute(verify, tmp_path)
    assert code == 0 and private["verification_scope"] == "private_donated_next_update", private
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    area = (tmp_path / first["receipt_path"]).parent
    copied = tmp_path / "copied"
    copied.mkdir()
    report = shutil.copyfile(area / "share/round029-report.zip", copied / "r.zip")
    prescore = shutil.copyfile(area / "share/round029-qualification.zip", copied / "q.zip")
    cmd = [
        sys.executable,
        "-S",
        "-B",
        str(Path(__file__).resolve().parents[2] / "scripts/foundation_run.py"),
        "--engine",
        "podman",
        "discovery",
        "round029",
        "--action",
        "verify",
        "--archive",
        str(report),
        "--qualification-archive",
        str(prescore),
    ]
    exported = json.loads(subprocess.check_output(cmd, cwd=copied))
    assert exported["verification_scope"] == "paired_archives_donated_next_update"
    assert (
        exported["report_ref"] == private["report_ref"] and exported["scientific_verdict"] is None
    )
    with report.open("ab") as stream:
        stream.write(b"private canary")
    rejected = subprocess.run(cmd, capture_output=True)
    assert rejected.returncode == 1 and json.loads(rejected.stdout)["mechanical_status"] == "FAILED"
    from oeis_learn.pipeline_validation.progressive_velocity_verify import donor_continuation
    monkeypatch.setattr(progressive_round028, "candidate_revision", lambda *_: revision)
    cut = {"full_state": plan["checkpoint"], "lineage": digest("stream")}
    assert donor_continuation(tmp_path, area.parent.parent, tmp_path, cut)["receipt"] == first["last_committed_ref"]
    assert donor_continuation(tmp_path, area.parent.parent, tmp_path, dict(cut, lineage=digest("changed"))) is None
    original_process = (runs[0] / "process.json").read_bytes()
    (runs[0] / "process.json").write_text('{"outcome":"STOPPED"}')
    failure, code = progressive_round029.execute(args, tmp_path)
    assert code == 1 and failure["first_error"]["stage"] == "train"
    assert failure["charged_tokens"] == 3366 and len(runs) == 1
    (runs[0] / "process.json").write_bytes(original_process)
