import json
import subprocess
import sys
from pathlib import Path

import pytest

from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLES, ROLE_POLICY
from oeis_learn.generalization.progressive.velocity_records import visible_prompt, registered_worker

ROOT = Path(__file__).resolve().parents[2]


def test_isolated_inputs_and_original_role_policy():
    # Independently pinned for the historical 013/028 receipts.
    assert ROLE_POLICY == "sha256:8cec9ffba6aa2f3431034edd84c359be393b0ea09aa3b7b694b80ca91abf7aea"
    assert set(ROUND029_ROLES["search"].channels) == {"visible", "guide"}
    assert set(ROUND029_ROLES["score"].channels) == {"choices", "score_truth"}
    row = dict(indices=[str(i) for i in range(-7, 13)], values=[str(i * i) for i in range(-7, 13)])
    assert visible_prompt(row) == row
    for extra in ({"witness": "private"}, {"hidden": ["0"]}, {"values": row["values"] + ["105"]}):
        with pytest.raises(ValueError):
            visible_prompt(dict(row, **extra))
    frozen = dict(
        source="a" * 40, image="sha256:" + "1" * 64, device="cpu", role_policy=digest("roles")
    )
    assert registered_worker(frozen, frozen) == frozen
    with pytest.raises(ValueError):
        registered_worker(frozen, dict(frozen, source="b" * 40))


def test_read_only_entry_point_and_legacy_help(tmp_path):
    cmd = [
        sys.executable,
        "-S",
        "-B",
        str(ROOT / "scripts/foundation_run.py"),
        "--engine",
        "podman",
        "discovery",
    ]
    for study in ("round028", "round029"):
        out = subprocess.run([*cmd, study, "--help"], capture_output=True)
        assert out.returncode == 0 and b"preflight" in out.stdout
    old = subprocess.run([*cmd, "round028", "--workspace", str(tmp_path), "--action", "status"],
                         capture_output=True, check=True)
    assert json.loads(old.stdout)["mechanical_status"] == "UNREGISTERED"
    for action in ("status", "preflight"):
        out = subprocess.run(
            [*cmd, "round029", "--workspace", str(tmp_path), "--action", action],
            capture_output=True,
        )
        assert out.returncode == 0, out.stdout
        assert list(tmp_path.iterdir()) == []
    for action in ("qualify", "cpu", "run", "preflight", "status"):
        args = [*cmd, "round029", "--workspace", str(tmp_path), "--action", action]
        if action in {"preflight", "status"}:
            args += ["--authority", "unexpected"]
        denial = subprocess.run(args, capture_output=True)
        assert denial.returncode != 0 and not list(tmp_path.iterdir())
    for opts in (
        ["--archive", "x"],
        ["--workspace", str(tmp_path), "--archive", "x"],
        ["--workspace", str(tmp_path), "--qualification-archive", "x"],
    ):
        assert (
            subprocess.run(
                [*cmd, "round029", "--action", "verify", *opts], capture_output=True
            ).returncode
            != 0
        )


def test_two_controllers_share_physical_device_and_session(tmp_path):
    import runpy
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.lifecycle import RunStore

    Host = runpy.run_path(str(ROOT / "scripts/foundation_local.py"))["Host"]
    controllers = [
        SimpleNamespace(workspace=tmp_path, source_root=tmp_path / name, engine=None)
        for name in ("candidate", "controller")
    ]
    with Host.gpu_lock(controllers[0]), pytest.raises(RuntimeError, match="another foundation GPU"):
        with Host.gpu_lock(controllers[1]):
            pytest.fail("duplicate device dispatch")
    with RunStore.owner(tmp_path), pytest.raises(ValueError, match="another invocation"):
        with RunStore.owner(tmp_path):
            pytest.fail("duplicate session dispatch")


@pytest.mark.parametrize("operation,other_channel", [("search", "guide"), ("select", "admissions")])
def test_installed_live_projection_accepts_only_registered_visible_inputs(tmp_path, monkeypatch, operation, other_channel):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, seal
    from oeis_learn.generalization.progressive.projections import describe
    from oeis_learn.pipeline_validation.progressive_role import main
    from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLE_POLICY
    from oeis_learn.pipeline_validation.progressive_velocity_roles import project_live_inputs

    worker = dict(source="a" * 40, image=digest("installed image"), device="cpu", role_policy=ROUND029_ROLE_POLICY)
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", worker["image"])
    manifest = digest(source_snapshot())
    prompt = dict(indices=list(map(str, range(-7, 13))), values=list(map(str, range(20))))
    role = ROUND029_ROLES[operation].role

    def inputs(data, dest, current=worker):
        envelope = seal(dict(version="discovery-channel/v1", kind="round029_visible", role=role, data=data))
        path = tmp_path / (envelope["identity"][7:] + ".json")
        if not path.exists():
            atomic_write(path, envelope)
        item = describe(tmp_path, path)
        request = seal(dict(version="round029-live-input/v1", operation=operation, worker=worker,
                            source_manifest=manifest, channels={"visible": [item.identity], other_channel: []}))
        project_live_inputs(tmp_path, dest, request=request, current_worker=current,
                            artifacts={"visible": [item], other_channel: []}, authorized=[item.ref])
        return dest

    dest = inputs(prompt, tmp_path / "valid")
    output = tmp_path / "output"
    def accept(dest, output):
        return main(["round029-" + operation], inputs=dest / "inputs",
                    authority=dest / "authority.json", output=output)
    assert accept(dest, output) == 0
    # The actual supervisor mounts files only. An empty registered channel has no directory.
    import shutil
    from oeis_learn.pipeline_validation.progressive_velocity_roles import role_mounts
    work = tmp_path / "work"
    req = load(dest / "authority.json")["request"]
    for src, target in role_mounts(dest, req):
        arrived = work / Path(target).relative_to("/work")
        arrived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, arrived)
    assert main(["round029-" + operation], inputs=work / "inputs/channels", authority=work / "inputs/authority.json",
                output=work / "output") == 0
    accepted = load(output / "accepted-input.json")
    assert accepted["scope"] == "live_input_projection" and accepted["operation"] == operation
    for i, bad in enumerate([dict(prompt, witness="PRIVATE-WITNESS"), dict(prompt, hidden=["999"]),
                             dict(prompt, values=prompt["values"] + ["999"])]):
        target = tmp_path / ("invalid-" + str(i))
        with pytest.raises(ValueError):
            inputs(bad, target)
        assert not (target / "inputs").exists()
    with pytest.raises(ValueError, match="worker"):
        inputs(prompt, tmp_path / "wrong-source", dict(worker, source="b" * 40))
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", digest("different installed image"))
    assert accept(dest, tmp_path / "wrong-image") != 0
    assert not (tmp_path / "wrong-image/accepted-input.json").exists()
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", worker["image"])
    # Recheck physical mounts independently of the descriptor checked by the host.
    (dest / "inputs/visible").chmod(0o700)
    (dest / "inputs/visible/witness.bin").write_bytes(b"PRIVATE-WITNESS")
    assert accept(dest, tmp_path / "extra") != 0
    assert not (tmp_path / "extra/accepted-input.json").exists()



@pytest.mark.parametrize("operation", sorted(ROUND029_ROLES))
def test_each_registered_role_reopens_only_its_own_projection(tmp_path, monkeypatch, operation):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, seal
    from oeis_learn.generalization.progressive.projections import describe
    from oeis_learn.pipeline_validation import progressive_qualification
    from oeis_learn.pipeline_validation.progressive_role import main
    from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLE_POLICY
    from oeis_learn.pipeline_validation.progressive_velocity_roles import project_role_inputs

    spec = ROUND029_ROLES[operation]
    prompt = dict(indices=list(map(str, range(-3, 17))), values=list(map(str, range(20))))
    target = encode_body("(program (return $index))")
    channels = {
        "visible": ("visible", prompt),
        "training": ("training", dict(prompt=prompt, target=target, family_ref=digest("training family"))),
        "guide": None,
        "admissions": None,
        "model": ("model", dict(weights_ref=digest("eligible weights"), profile_ref=digest("profile"))),
        "skeleton": ("skeleton", dict(tokens=encode_body("(program (return (param.get p0)))"))),
        "choices": ("choice", dict(prompt_ref=digest(prompt), source=None, closed=True, complete=True)),
        "sealed_bank": ("bank", dict(prompt_ref=digest(prompt), choice_ref=digest("sealed"), sources=[], closed=True)),
        "score_truth": ("truth", dict(indices=list(map(str, range(-3, 97))), values=list(map(str, range(100))))),
    }
    worker = dict(source="a" * 40, image=digest("installed"), device="hip" if spec.gpu else "cpu",
                  role_policy=ROUND029_ROLE_POLICY)
    monkeypatch.setenv("OEIS_QUALIFICATION_IMAGE", worker["image"])
    hip = []
    monkeypatch.setattr(progressive_qualification, "require_hip", lambda: hip.append(operation))
    artifacts = {channel: [] for channel in spec.channels}
    for channel, items in artifacts.items():
        if channels[channel]:
            kind, data = channels[channel]
            path = tmp_path / (channel + ".json")
            atomic_write(path, seal(dict(version="discovery-channel/v1", kind="round029_" + kind,
                                         role=spec.role, data=data)))
            items.append(describe(tmp_path, path))
    request = seal(dict(version="round029-role-input/v1", operation=operation, worker=worker,
                        source_manifest=digest(source_snapshot()),
                        channels={c: [a.identity for a in items] for c, items in artifacts.items()}))
    dest = tmp_path / "role"
    project_role_inputs(tmp_path, dest, request=request, current_worker=worker, artifacts=artifacts,
                        authorized=[a.ref for items in artifacts.values() for a in items])
    output = tmp_path / "out"
    assert main(["round029-" + operation], inputs=dest / "inputs", authority=dest / "authority.json", output=output) == 0
    assert hip == ([operation] if spec.gpu else [])
    assert load(output / "accepted-input.json")["request_ref"] == request["identity"]
    foreign = next(c for c in ("score_truth", "training", "visible") if c not in spec.channels)
    (dest / "inputs").chmod(0o755)
    (dest / "inputs" / foreign).mkdir()
    assert main(["round029-" + operation], inputs=dest / "inputs", authority=dest / "authority.json",
                output=tmp_path / "bad") != 0
    assert not (tmp_path / "bad/accepted-input.json").exists()
    assert hip == ([operation] if spec.gpu else [])
