"""Recovery of the workstation's staged CPU scorer after its first resident GPU search."""

import subprocess
from pathlib import Path

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.pipeline_validation import progressive_handoff, progressive_round028_pilot as pilot
from oeis_learn.pipeline_validation import progressive_round028_rehearsal as probe
from tests.integration.test_round028_pilot import parent
from tests.unit import test_foundation_local as foundation

workstation = foundation.host

SOURCE = Path(__file__).resolve().parents[2]
HELPER = Path("specs/015-learned-program-search/validation/candidate_source.py")


@pytest.mark.parametrize("reboot", [False, True])
def test_staged_scorer_resumes_after_gpu_discovery(workstation, tmp_path, monkeypatch, reboot):
    from oeis_learn.pipeline_validation.preparation_recovery import retain

    area = tmp_path / "pilot"
    area.mkdir()
    registry = seal({"source": digest("qualified source")})
    boot = tmp_path / "boot"
    boot.write_text("original-boot")
    drm = tmp_path / "drm"
    node = tmp_path / "renderD128"
    node.touch()
    (drm / node.name / "device").mkdir(parents=True)
    workstation.engine.render_node.return_value = str(node)
    paths = {Path("/sys/class/drm"): drm, Path("/proc/sys/kernel/random/boot_id"): boot}
    monkeypatch.setattr(probe, "Path", lambda p: paths.get(Path(p), Path(p)))
    calls = []

    def stage(host, source, image, output, argv, *, gpu, seconds, readonly_inputs, **options):
        inputs = {dest: path for path, dest in readonly_inputs}
        request = load(inputs["/work/inputs/request.json"])
        calls.append(request["role"])
        output.mkdir()
        result = {}
        if gpu:
            atomic_write(output / "gpu-lease.json", {"elapsed_us": 1})
            atomic_write(output / "choices.json", seal({"chosen": "exposed case"}))
            result = {"choices": reference(output, output / "choices.json")}
        if request["role"] == "score":
            assert load(inputs["/work/inputs/candidates.json"]) == seal({"chosen": "exposed case"})
        atomic_write(
            output / "producer.json",
            seal(
                dict(
                    version=pilot.PilotRole.producer_version,
                    request=request["identity"],
                    native=digest("native"),
                    result=result,
                )
            ),
        )
        mounts = [(path, dest, False) for path, dest in readonly_inputs] + [
            (output, "/work/output", True)
        ]
        atomic_write(
            output / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=[
                        dict(Type="bind", Source=str(p), Destination=d, RW=w) for p, d, w in mounts
                    ],
                )
            ],
        )

    monkeypatch.setattr(progressive_handoff, "stage", stage)

    def role(phase):
        runner = pilot.PilotRole(None, SOURCE, tmp_path, "a" * 40, area, registry, None)
        runner.host, runner.image, runner.phase = workstation, foundation.IMAGE, phase
        return runner

    cpu, gpu = role("cpu"), role("hip")
    workstation.policy["free_disk_floor_bytes"] = 0
    cpu("gradient", seal({"block": "qualify-training-signal"}), seconds=10)
    machine_file = area / "physical-cpu.json"
    initial_machine = machine_file.read_bytes()
    original_boot = probe.physical_scope(workstation, "cpu")
    search = seal({"block": "qualify-resident-search"})
    found = gpu("search", search, seconds=10)
    choices = gpu.last_output / found["choices"]["path"]
    data = seal({"block": "qualify-resident-search", "candidates": found["choices"]})
    mounts = [(choices, "/work/inputs/candidates.json")]

    # Exact on-disk state at the failed release's bind_physical call, before attempt-0000.
    legacy_boot = digest(["original-boot", workstation.policy["runtime"], {}])
    assert legacy_boot != original_boot
    req = seal(
        dict(
            version=cpu.request_version,
            registration=registry["identity"],
            source=registry["source"],
            engine="podman",
            image=foundation.IMAGE,
            candidate_native=None,
            role="score",
            data=data["identity"],
            physical=legacy_boot,
        )
    )
    claim = area / "assignments/qualify-resident-search-score.json"
    original = area / "roles" / req["identity"][7:]
    retain(claim, seal({"request": req["identity"]}))
    retain(original / "request.json", req)
    retain(original / "data.json", data)
    if reboot:
        boot.write_text("subsequent-boot")
    gpu("search", search, seconds=10)
    assert cpu("score", data, seconds=10, inputs=mounts) == {}
    assert cpu("score", data, seconds=10, inputs=mounts) == {}
    assert calls == ["gradient", "search", "score"]
    assert machine_file.read_bytes() == initial_machine and load(original / "request.json") == req
    assert load(claim)["request"] == req["identity"]
    assert list(original.glob("attempt-*")) == [original / "attempt-0000"]
    started = load(original / "attempt-0000/start.json")
    assert started["dispatch_scope"] == probe.physical_scope(workstation, "cpu")
    assert (started["dispatch_scope"] == original_boot) is not reboot

    # A real CPU resource change still prevents a new dispatch.
    workstation.policy["runtime"]["cpus"] += 1
    with pytest.raises(ValueError):
        cpu("score", seal(dict(data, block="different-case")), seconds=10, inputs=mounts)
    assert calls == ["gradient", "search", "score"]


def test_repaired_controller_authenticates_original_source_and_registration(tmp_path, monkeypatch):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.generalization.progressive.program_search_pilot import pilot_registration
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    import runpy

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    def commit(message):
        git("add", "src/oeis_learn/worker.py", str(HELPER))
        git(
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.org",
            "commit",
            "-qm",
            message,
        )
        return git("rev-parse", "HEAD")

    git("init", "--quiet")
    (tmp_path / HELPER).parent.mkdir(parents=True)
    (tmp_path / HELPER).write_bytes((SOURCE / HELPER).read_bytes())
    worker = Path("src/oeis_learn/worker.py")
    (tmp_path / worker).parent.mkdir(parents=True)
    (tmp_path / worker).write_text("old worker")
    older = commit("Original installed worker")
    snapshot = lambda p: {str(worker): (p / worker).read_text()}
    monkeypatch.setattr(source_snapshot, "collect_snapshot", snapshot)
    stored = pilot_registration(parent(), older, digest(snapshot(tmp_path)))
    area = tmp_path / "pilot"
    path = area / "registration.json"
    retain(path, stored)
    initial = path.read_bytes()
    (tmp_path / worker).write_text("different candidate")
    controller = commit("Controller repair")
    materialize = runpy.run_path(str(tmp_path / HELPER))["materialize"]
    caller = materialize(tmp_path, controller)
    for _ in range(2):
        source, reg = pilot.registered_pilot_source(tmp_path, caller, controller, area, parent())
        assert source == materialize(tmp_path, older)
        assert reg == stored and path.read_bytes() == initial
        assert (source / worker).read_text() == "old worker"
    changed_parent = dict(parent(), timing=digest("other timing"))
    with pytest.raises(ValueError):
        pilot.registered_pilot_source(tmp_path, caller, controller, area, changed_parent)
    assert path.read_bytes() == initial
    (source / worker).write_text("changed in the cached worker tree")
    with pytest.raises(ValueError):
        pilot.registered_pilot_source(tmp_path, caller, controller, area, parent())
