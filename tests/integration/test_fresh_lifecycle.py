"""CPU orchestration fixture with real finite execution/scoring; not host/GPU acceptance."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from oeis_learn.experiments.profiles import qualified_model_profile
from oeis_learn.fresh import worker
from oeis_learn.fresh.bundle import pack
from oeis_learn.fresh.common import digest, file_hash, inventory, read, write
from oeis_learn.sandbox.pipeline import Runtime
from tests.unit.test_fresh_workflow import FakeClock, host


class DirectRunner:
    def __init__(self, root):
        self.runtime = Runtime()

    def run_single(self, source, indices, **kwargs):
        return self.runtime.evaluate(source, list(indices))


def fixture_inputs(root):
    examples = {
        "training": [
            {"program_id": digest(n), "values": [str(n)] * 20, "tokens": [1]} for n in range(128)
        ],
        "development": [{"program_id": digest("dev"), "values": ["0"] * 20, "tokens": [1]}],
    }
    write(root / "examples.json", examples)
    write(
        root / "truth.json",
        {
            split: [{"group_id": digest(split), "values": ["0"] * 100, "members": [split]}]
            for split in ("development", "final")
        },
    )
    write(
        root / "archive-preparation-report.json",
        {"training_prompts": 128, "development_structures": 1},
    )
    config = {
        "run": {
            "seed": 7,
            "training_seconds": 60,
            "model": qualified_model_profile(attention_backend="eager"),
            "evaluation": {"groups": 1, "attempts": 2, "tokens": 8, "per_group_seconds": 1},
        }
    }
    manifest = {"version": "fresh-inputs/v1", "config": config, "files": inventory(root)}
    manifest["identity"] = digest(manifest)
    write(root / "prepared.json", manifest)


def fixture_coordinator(clock, *, interrupt_stage=None):
    class CPUFixture:
        calls: ClassVar[list] = []

        def __init__(self, root, policy, image, ledger):
            self.root = root
            self.root.mkdir(parents=True, exist_ok=True)

        def run(self, operation, role, inputs, output, seconds):
            self.calls.append((operation, role))
            clock.n += 10**9
            output.mkdir(parents=True, exist_ok=True)
            names = inventory(inputs)
            if role in {"learner", "generation"}:
                assert "truth.json" not in names
                assert not any("indexed-source" in n or "archive/" in n for n in names)
            if operation == "train":
                settings = read(inputs / "settings.json")
                cap = settings["update_cap"]
                if interrupt_stage is not None and cap == interrupt_stage:
                    raise RuntimeError("injected interruption")
                name = f"qualified-{cap:08d}"
                write(output / "checkpoints" / (name + ".pt"), f"fixture-state-{cap}".encode())
                blob = file_hash(output / "checkpoints" / (name + ".pt"))
                write(
                    output / "checkpoints" / (name + "-base.json"),
                    {"blob_sha256": blob, "blob_path": name + ".pt"},
                )
                write(
                    output / "checkpoints" / (name + ".json"),
                    {"base": name + "-base.json", "qualified": {}},
                )
                result = {
                    "checkpoint": name + ".json",
                    "checkpoint_sha256": blob,
                    "completed_updates": cap,
                }
            elif operation == "generate":
                settings = read(inputs / "settings.json")
                assert "visible.json" in names and "examples.json" not in names
                visible = read(inputs / "visible.json")
                base = read(
                    inputs / "checkpoints" / (Path(settings["checkpoint"]).stem + "-base.json")
                )
                result = {
                    "version": "fresh-candidate-seal/v1",
                    "checkpoint_sha256": base["blob_sha256"],
                    "protocol_sha256": settings["protocol_sha256"],
                    "visible_sha256": digest(visible),
                    "records": [
                        {
                            "group_id": g["group_id"],
                            "selected_sha256": digest("i256.zero"),
                            "attempts": [
                                {
                                    "source": "i256.zero",
                                    "complete": True,
                                    "method": "greedy",
                                    "failure": None,
                                }
                            ],
                        }
                        for g in visible
                    ],
                }
                result["identity"] = digest(result)
            else:
                assert role == "evaluator" and "truth.json" in names
                with patch("oeis_learn.tracking.role_coordinator.QueueRunner", DirectRunner):
                    result = worker.score(inputs, output)
            write(output / "result.json", result)
            return result

        def close(self):
            clock.n += 10**9

    return CPUFixture


def run_environment(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixture_inputs(workspace / "inputs")
    policy = {
        "workspace": str(workspace),
        "home": str(tmp_path / "private"),
        "podman": "podman",
        "project": "test",
        "renderNode": "/dev/dri/renderD128",
    }
    return workspace, policy


def test_complete_private_smoke_and_final_evaluation_reuses_seal(tmp_path):
    workspace, policy = run_environment(tmp_path)
    clock = FakeClock()
    coordinator = fixture_coordinator(clock)
    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        assert (
            host.execute(
                policy,
                "sha256:" + "a" * 64,
                ["run", "inputs", "runs/smoke", "smoke"],
                coordinator_factory=coordinator,
            )
            == 0
        )
        report = read(workspace / "runs/smoke/status.json")
        assert report["status"] == "complete" and report["completed_updates"] == 3
        assert (
            report["final_score"]["denominator"] == 1
            and len(report["final_score"]["solved_groups"]) == 1
        )
        assert report["budget"]["charged_ns"] == clock.n
        generated = coordinator.calls.count(("generate", "generation"))
        host.execute(
            policy,
            "sha256:" + "a" * 64,
            ["evaluate", "runs/smoke", "final", "latest"],
            coordinator_factory=coordinator,
        )
        assert coordinator.calls.count(("generate", "generation")) == generated
        with pytest.raises(ValueError, match="finalized"):
            host.execute(
                policy,
                "sha256:" + "a" * 64,
                ["resume", "runs/smoke"],
                coordinator_factory=coordinator,
            )
    output = tmp_path / "analysis.zip"
    pack(workspace / "runs/smoke", output)
    assert output.exists()
    assert not any("-input" in str(p) for p in (workspace / "runs/smoke/roles").glob("*"))


def test_interruption_retains_budget_and_resume_skips_completed_stages(tmp_path):
    workspace, policy = run_environment(tmp_path)
    clock = FakeClock()
    failed = fixture_coordinator(clock, interrupt_stage=3)
    resumed = fixture_coordinator(clock)
    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        with pytest.raises(RuntimeError, match="injected"):
            host.execute(
                policy,
                "sha256:" + "a" * 64,
                ["run", "inputs", "runs/smoke", "smoke"],
                coordinator_factory=failed,
            )
        before = read(workspace / "runs/smoke/budget.json")["charged_ns"]
        clock.n += 5 * 10**9
        host.execute(
            policy, "sha256:" + "a" * 64, ["resume", "runs/smoke"], coordinator_factory=resumed
        )
        after = read(workspace / "runs/smoke/budget.json")["charged_ns"]
        assert after >= before + 5 * 10**9
        assert resumed.calls.count(("train", "learner")) == 1
        assert resumed.calls.count(("generate", "generation")) == 2
        assert read(workspace / "runs/smoke/status.json")["status"] == "complete"


def test_changed_image_cannot_resume_and_new_run_cannot_reissue_budget(tmp_path):
    _workspace, policy = run_environment(tmp_path)
    clock = FakeClock()
    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        with pytest.raises(RuntimeError):
            host.execute(
                policy,
                "sha256:" + "a" * 64,
                ["run", "inputs", "runs/smoke", "smoke"],
                coordinator_factory=fixture_coordinator(clock, interrupt_stage=1),
            )
        with pytest.raises(ValueError, match="requires resume"):
            host.execute(policy, "sha256:" + "a" * 64, ["run", "inputs", "runs/smoke", "smoke"])
        with pytest.raises(ValueError, match="image changed"):
            host.execute(policy, "sha256:" + "b" * 64, ["resume", "runs/smoke"])


def test_interrupted_initialization_can_resume_without_refunding_time(tmp_path):
    workspace, policy = run_environment(tmp_path)
    clock = FakeClock()
    with (
        patch.object(host, "clock", clock),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        with pytest.raises(RuntimeError, match="injected"):
            host.execute(
                policy,
                "sha256:" + "a" * 64,
                ["run", "inputs", "runs/smoke", "smoke"],
                coordinator_factory=fixture_coordinator(clock, interrupt_stage=0),
            )
        clock.n += 5 * 10**9
        host.execute(
            policy,
            "sha256:" + "a" * 64,
            ["resume", "runs/smoke"],
            coordinator_factory=fixture_coordinator(clock),
        )
        status = read(workspace / "runs/smoke/status.json")
        assert status["status"] == "complete" and status["budget"]["charged_ns"] >= 5 * 10**9
