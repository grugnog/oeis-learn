import subprocess
import sys
from pathlib import Path

import pytest

from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLES, ROLE_POLICY
from oeis_learn.generalization.progressive.velocity_records import visible_prompt, registered_worker

ROOT = Path(__file__).resolve().parents[2]


def test_isolated_inputs_and_original_role_policy():
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    assert ROLE_POLICY == digest(
        {k: dict(role=v.role, channels=list(v.channels), gpu=v.gpu) for k, v in REGISTRY.items()}
    )
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
