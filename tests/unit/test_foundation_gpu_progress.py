"""Controller safety and recovery checks run without the broker or GPU."""

import importlib.util
import os
import fcntl
import stat
from pathlib import Path
import subprocess
import sys
import pytest

spec = importlib.util.spec_from_file_location(
    "gpu_progress", Path(__file__).resolve().parents[2] / "scripts/gpu_progress.py"
)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


def test_all_earlier_partials_have_owning_steps():
    rows = g.task_inventory()
    assert {"T047", "T113", "T114"} <= {r["task"] for r in rows}
    names = {s.name for s in g.STEPS}
    assert all(set(r["checks"]) <= names for r in rows)
    assert g.STEPS[-1].name == "end-to-end" and g.STEPS[-1].seconds == 43200


def test_new_unmapped_partial_is_rejected(tmp_path):
    p = tmp_path / "specs/007-experiment-foundation/tasks.md"
    p.parent.mkdir(parents=True)
    p.write_text("- [/] T004 New partial\n")
    with pytest.raises(ValueError, match="Unmapped"):
        g.task_inventory(tmp_path)


@pytest.mark.parametrize("limit", [True, 599, 43201, 1.5])
def test_final_limit_closed(tmp_path, limit):
    p = tmp_path / "settings.json"
    g.write(p, dict(final_seconds=limit))
    with pytest.raises(ValueError):
        g.load_settings(p)


def test_unknown_settings_and_unresolved_image_fail(tmp_path):
    p = tmp_path / "settings.json"
    for value in (
        dict(force=True),
        dict(image_id="gpu:latest"),
        dict(docker_host="tcp://host:2375"),
        dict(docker_host="ssh://labgpu-fixture@10.77.0.1:2222"),
    ):
        g.write(p, value)
        with pytest.raises(ValueError):
            g.load_settings(p)


def test_manifest_references_are_hashed(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"parent")
    g.write(tmp_path / "case.json", dict(checkpoint="weights.pt"))
    a = g.input_hashes(tmp_path / "case.json")
    (tmp_path / "weights.pt").write_bytes(b"changed")
    assert a != g.input_hashes(tmp_path / "case.json")
    g.write(tmp_path / "a.json", dict(path="b.json"))
    g.write(tmp_path / "b.json", dict(path="a.json"))
    assert len(g.input_hashes(tmp_path / "a.json")) == 2


def test_cached_success_rejects_output_tampering(tmp_path):
    g.write(tmp_path / "result.json", dict(value=1))
    r = g.success_receipt("a", tmp_path, {}, 1, g.Step("test", (), 1))
    assert g.cached_success(r, "a", tmp_path)
    assert not g.cached_success(r, "b", tmp_path)
    g.write(tmp_path / "result.json", dict(value=2))
    assert not g.cached_success(r, "a", tmp_path)


def test_budget_charges_offline_and_rejects_limit_clock_changes(tmp_path):
    assert g.final_remaining(tmp_path, 600, now=1000) == 600
    assert g.final_remaining(tmp_path, 600, now=1100) == 500
    with pytest.raises(g.Blocked, match="change"):
        g.final_remaining(tmp_path, 601, now=1100)
    with pytest.raises(g.Blocked, match="regressed"):
        g.final_remaining(tmp_path, 600, now=1099)
    with pytest.raises(g.Blocked, match="exhausted"):
        g.final_remaining(tmp_path, 600, now=1600)


def test_active_remote_worker_blocks_retry(tmp_path):
    held = g.lock(tmp_path / "worker.lock")
    try:
        with pytest.raises(g.Blocked, match="lock"):
            g.require_idle(tmp_path)
    finally:
        held.close()
    g.require_idle(tmp_path)
    g.write(tmp_path / "reclamation.json", dict(status="failed"))
    with pytest.raises(g.Blocked, match="cleanup"):
        g.require_idle(tmp_path)


def test_gpu_command_uses_broker_without_environment_overrides(tmp_path):
    argv = g.worker_argv(g.Step("test", (), 2, True), tmp_path / "cfg", tmp_path, 2)
    assert argv[:4] == ["lab-gpu", "exec", "--", "python"]
    assert "_watchdog" in argv and "--env" not in argv


def driver(monkeypatch, tmp_path, steps):
    monkeypatch.setattr(g, "STEPS", steps)
    monkeypatch.setattr(g, "task_inventory", lambda: [])
    monkeypatch.setattr(g, "source_identity", lambda: "source")
    # Exercise existing issuance/receipt semantics with a hypothetical future
    # launcher. Separate tests assert actual broker stages are blocked today.
    monkeypatch.setattr(g, "HOST_STEPS", set())
    calls = []

    def fake(argv, log, seconds):
        name = argv[-4]
        out = tmp_path / name
        g.write(out / "step-result.json", dict(result=dict(value=name)))
        calls.append(name)
        return 0

    monkeypatch.setattr(g, "supervise", fake)
    return calls, fake


def test_rerun_skips_success_but_retests_changed_output(monkeypatch, tmp_path):
    steps = (g.Step("first", (), 1), g.Step("second", ("first",), 1))
    calls, _ = driver(monkeypatch, tmp_path, steps)
    assert g.run({"final_seconds": 600}, tmp_path) == 0
    assert g.run({"final_seconds": 600}, tmp_path) == 0 and calls == ["first", "second"]
    g.write(tmp_path / "first/step-result.json", dict(result=dict(value="tamper")))
    assert g.run({"final_seconds": 600}, tmp_path) == 0 and calls == ["first", "second", "first"]


def test_temporary_dependency_failure_preserves_success(monkeypatch, tmp_path):
    steps = (g.Step("first", (), 1), g.Step("second", ("first",), 1))
    calls, _ = driver(monkeypatch, tmp_path, steps)
    assert g.run({"final_seconds": 600}, tmp_path) == 0
    original = g.read(tmp_path / "receipts.json")["second"]
    monkeypatch.setattr(g, "PROBES", {"first"})
    monkeypatch.setattr(g, "supervise", lambda *a: 75)
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    assert g.read(tmp_path / "receipts.json")["second"] == original


def test_one_shot_measurement_never_reissued(monkeypatch, tmp_path):
    calls, _ = driver(monkeypatch, tmp_path, (g.Step("measurement", (), 10),))
    monkeypatch.setattr(g, "supervise", lambda *a: 124)
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    monkeypatch.setattr(g, "supervise", lambda *a: pytest.fail("must not reissue"))
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    assert "already issued" in g.read(tmp_path / "summary.json")["errors"]["measurement"]


def test_broker_busy_before_issuance_can_retry(monkeypatch, tmp_path):
    _, fake = driver(monkeypatch, tmp_path, (g.Step("measurement", (), 10, True),))
    monkeypatch.setattr(g, "supervise", lambda *a: 75)
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    assert not g.read(tmp_path / "receipts.json")["measurement"]["issued"]
    monkeypatch.setattr(g, "supervise", fake)
    assert g.run({"final_seconds": 600}, tmp_path) == 0


def test_final_changed_inputs_preserve_old_issuance(monkeypatch, tmp_path):
    driver(monkeypatch, tmp_path, (g.Step("end-to-end", (), 600),))
    monkeypatch.setattr(g, "supervise", lambda *a: 124)
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    original = g.read(tmp_path / "receipts.json")["end-to-end"]
    assert g.run({"final_seconds": 601}, tmp_path) == 1
    assert g.read(tmp_path / "receipts.json")["end-to-end"] == original


def test_watchdog_uses_broker_cleanup_without_docker_or_lease_environment(monkeypatch, tmp_path):
    settings = tmp_path / "cfg.json"
    g.write(settings, {})
    monkeypatch.delenv("OEIS_ROLE_LEASE", raising=False)
    calls = []
    monkeypatch.setattr(g, "supervise", lambda argv, log, seconds: calls.append(seconds) or 0)
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **kw: pytest.fail("no Docker/SSH cleanup"))
    assert g.remote_watchdog("gpu-smoke", settings, tmp_path, 600) == 0
    lease = g.read(tmp_path / "gpu-smoke/lease.json")
    assert lease["backend"] == "lab-gpu" and lease["cleanup_owner"] == "host broker"
    assert "docker_host" not in lease and "OEIS_ROLE_LEASE" not in g.os.environ
    assert calls == [560]
    assert not (tmp_path / "gpu-smoke/reclamation.json").exists()
    # No residual worker lock after normal execution.
    g.require_idle(tmp_path / "gpu-smoke")


def test_reclamation_requires_administrator_without_another_broker_request(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(g.SCRIPT), "--reclaim", "baseline"])
    monkeypatch.setattr(g, "supervise", lambda *a: pytest.fail("no remote cleanup authority"))
    with pytest.raises(SystemExit) as e:
        g.main()
    assert e.value.code == 2


def test_junit_requires_cases_and_explicit_skips(tmp_path):
    p = tmp_path / "suite.xml"
    p.write_text("<testsuites/>")
    with pytest.raises(ValueError, match="Empty"):
        g.check_junit(p)
    p.write_text('<testsuite><testcase name="deferred"><skipped/></testcase></testsuite>')
    with pytest.raises(ValueError, match="skipped"):
        g.check_junit(p)
    assert g.check_junit(p, ("deferred",)) == 1


def test_timeout_reclaims_process_group(tmp_path):
    log = tmp_path / "log"
    assert g.supervise([sys.executable, "-c", "import time; time.sleep(20)"], log, 0.05) == 124
    assert log.exists()


def test_role_lease_is_label_not_workload_environment(tmp_path, monkeypatch):
    from oeis_learn.tracking.foundation_container import container_command

    source = tmp_path / "input"
    target = tmp_path / "output"
    source.mkdir()
    target.mkdir()
    monkeypatch.setenv("OEIS_ROLE_LEASE", "b" * 32)
    argv = container_command(
        role="worker",
        image="sha256:" + "a" * 64,
        inputs=source,
        output=target,
        name="test",
        command=["python"],
        gpu=False,
    )
    assert "oeis.progression=" + "b" * 32 in argv and "OEIS_ROLE_LEASE=" + "b" * 32 not in argv
    monkeypatch.setenv("OEIS_ROLE_LEASE", "bad")
    with pytest.raises(ValueError, match="lease"):
        container_command(
            role="worker",
            image="sha256:" + "a" * 64,
            inputs=source,
            output=target,
            name="test",
            command=["python"],
            gpu=False,
        )


def worker_module():
    spec = importlib.util.spec_from_file_location(
        "gpu_progress_steps", g.ROOT / "scripts/gpu_progress_steps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_preflight_binds_root_dockerfile_and_lock(tmp_path):
    worker = worker_module()
    path = worker.preflight_config({"image_id": "sha256:" + "a" * 64}, tmp_path)
    from oeis_learn.cli.foundation_preflight import load_config

    cfg = load_config(str(path))
    assert (
        cfg.final_image_id == "sha256:" + "a" * 64 and cfg.updates == 3 and cfg.body_tokens == 1024
    )
    assert (path.parent / cfg.dependency_lock).read_bytes() == (g.ROOT / "uv.lock").read_bytes()
    assert (path.parent / cfg.dockerfile).read_bytes() == (g.ROOT / "Dockerfile").read_bytes()
    assert len(cfg.fixtures) == 3


def test_minimal_cohort_keeps_original_groups_and_partitions(tmp_path):
    from oeis_learn.evaluation.release_source import prepare_release
    from oeis_learn.evaluation.foundation_cohort import load_cohort
    from oeis_learn.experiments.artifacts import compute_file_hash

    source = tmp_path / "source"
    source.mkdir()
    records = []
    for n in range(6):
        terms = source / f"b{n}.txt"
        terms.write_text("\n".join(f"{i} {i**(n+2)}" for i in range(120)))
        meta = source / f"m{n}.json"
        g.write(
            meta,
            dict(
                offset=0,
                keywords=["nonn"],
                finite=False,
                table=False,
                locator=f"https://example.invalid/fixture/{n}",
            ),
        )
        records.append(
            dict(
                record_id=f"fixture:A{n:06}",
                terms=dict(path=terms.name, sha256=compute_file_hash(terms)),
                metadata=dict(path=meta.name, sha256=compute_file_hash(meta)),
            )
        )
    g.write(
        source / "manifest.json",
        dict(version="indexed-source/v1", release="explicit fixture", records=records),
    )
    config = tmp_path / "config.json"
    g.write(
        config,
        dict(
            schema_version="foundation/v1",
            profile="prefix20_total100_affine_v2",
            seed=3,
            dev_count=3,
            final_count=3,
        ),
    )
    release = tmp_path / "release"
    prepare_release(source / "manifest.json", config, release)
    out = tmp_path / "final"
    out.mkdir()
    cohort = worker_module().minimal_cohort(release, out)
    parent = load_cohort(release / "cohort")
    actual = load_cohort(cohort)
    assert len(actual["groups"]) == 4
    known = {(r["group_id"], r["partition"]) for r in parent["groups"]}
    assert {(r["group_id"], r["partition"]) for r in actual["groups"]} <= known
    assert worker_module().minimal_cohort(release, out) == cohort


def test_detached_controller_reclaimed_when_caller_interrupted(tmp_path, monkeypatch):
    import contextlib
    import oeis_learn.tracking.foundation_controller as c

    monkeypatch.setattr(c, "run_lock", lambda root: contextlib.nullcontext())
    monkeypatch.setattr(c, "remaining_on_disk", lambda root: 1_000_000_000)

    class Process:
        pid = 123456

        def start(self):
            pass

        def join(self, seconds):
            raise KeyboardInterrupt()

    class Context:
        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(c.mp, "get_context", lambda name: Context())
    killed = []
    reaped = []
    monkeypatch.setattr(c.os, "killpg", lambda pid, signal: killed.append(pid))
    monkeypatch.setattr(c, "_reap", lambda process: reaped.append(process.pid))
    with pytest.raises(KeyboardInterrupt):
        c.run_training(tmp_path)
    assert killed == reaped == [123456]


@pytest.mark.parametrize("name", sorted(g.HOST_STEPS))
def test_private_roles_are_rejected_before_lease_or_work_issuance(name, tmp_path, monkeypatch):
    monkeypatch.setattr(g, "supervise", lambda *a: pytest.fail("unsupported role issued"))
    with pytest.raises(g.Blocked, match="no Docker/Podman API"):
        g.remote_watchdog(name, tmp_path / "missing-settings", tmp_path, 600)
    worker = worker_module()
    with pytest.raises(g.Blocked, match="private mounts"):
        worker.execute(name, {}, tmp_path, tmp_path / name, 600)
    assert not (tmp_path / name).exists()


def test_build_products_do_not_invalidate_source_but_source_edits_do(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.py").write_text("original")
    original = g.source_identity(tmp_path)
    (src / "project.egg-info").mkdir()
    (src / "project.egg-info/PKG-INFO").write_text("generated")
    target = tmp_path / "crates/oeis_wasm_evaluator/target/debug"
    target.mkdir(parents=True)
    (target / "artifact").write_bytes(b"compiled")
    assert g.source_identity(tmp_path) == original
    (src / "model.py").write_text("changed")
    assert g.source_identity(tmp_path) != original


@pytest.mark.parametrize("raw", ["a" * 64 + "\n", "sha256:" + "a" * 64 + "\n"])
def test_broker_image_stdout_resolves_without_docker(raw, monkeypatch):
    worker = worker_module()
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, raw, "")

    monkeypatch.setattr(worker.subprocess, "run", fake)
    report = worker.ensure_image({})
    assert report["runtime"]["image_id"] == "sha256:" + "a" * 64
    assert calls[0][0] == ["lab-gpu", "image", "ensure"]
    assert "stderr" not in calls[0][1]  # build/SSH errors stay in transport.log


@pytest.mark.parametrize("raw", ["gpu:latest", "", "a" * 64 + "\nextra", "sha256:" + "b" * 64])
def test_broker_image_rejects_invalid_output_or_pin_mismatch(raw, monkeypatch):
    worker = worker_module()
    monkeypatch.setattr(
        worker.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, raw, "")
    )
    with pytest.raises((ValueError, g.Blocked)):
        worker.ensure_image({"image_id": "sha256:" + "a" * 64})


def test_resolved_broker_image_reaches_preflight_and_invalidates_gpu_receipts(
    tmp_path, monkeypatch
):
    steps = (
        g.Step("image", (), 60),
        g.Step("gpu-environment", ("image",), 120, True),
        g.Step("preflight", ("gpu-environment",), 600, True, ("image_id",)),
    )
    calls, _ = driver(monkeypatch, tmp_path, steps)
    selected = ["sha256:" + "a" * 64]

    def fake(argv, log, seconds):
        name, settings = argv[-4:-2]
        cfg = g.read(settings)
        runtime = dict(image_id=selected[0])
        if name != "image":
            assert cfg["image_id"] == selected[0]
        g.write(
            tmp_path / name / "step-result.json",
            dict(result=dict(runtime=runtime, observed_utc=g.time.time())),
        )
        calls.append(name)
        return 0

    monkeypatch.setattr(g, "supervise", fake)
    cfg = {"final_seconds": 600}
    assert g.run(cfg, tmp_path) == 0
    assert "image_id" not in cfg  # caller's optional pin stays separate from resolution
    assert g.run(cfg, tmp_path) == 0
    assert calls == ["image", "gpu-environment", "preflight", "image", "gpu-environment"]
    selected[0] = "sha256:" + "b" * 64
    assert g.run(cfg, tmp_path) == 0
    assert calls[-3:] == ["image", "gpu-environment", "preflight"]


def test_optional_pin_mismatch_blocks_gpu_launch(tmp_path, monkeypatch):
    calls, _ = driver(
        monkeypatch, tmp_path, (g.Step("image", (), 60), g.Step("gpu-smoke", ("image",), 600, True))
    )

    def fake(argv, log, seconds):
        name = argv[-4]
        g.write(
            tmp_path / name / "step-result.json",
            dict(result=dict(runtime=dict(image_id="sha256:" + "b" * 64))),
        )
        calls.append(name)
        return 0

    monkeypatch.setattr(g, "supervise", fake)
    assert g.run({"image_id": "sha256:" + "a" * 64, "final_seconds": 600}, tmp_path) == 1
    assert calls == ["image"]


def test_full_plan_blocks_private_stages_without_spending_budgets(tmp_path, monkeypatch):
    steps = tuple(s for s in g.STEPS if s.name in g.HOST_STEPS)
    monkeypatch.setattr(g, "STEPS", steps)
    monkeypatch.setattr(g, "task_inventory", lambda: [])
    monkeypatch.setattr(g, "source_identity", lambda: "source")
    monkeypatch.setattr(g, "supervise", lambda *a: pytest.fail("no host stage can launch"))
    assert g.run({"final_seconds": 600}, tmp_path) == 1
    receipts = g.read(tmp_path / "receipts.json")
    assert all(r["status"] == "BLOCKED" and not r["issued"] for r in receipts.values())
    assert all("Host launcher integration is required" in r["error"] for r in receipts.values())
    assert not (tmp_path / "final-budget.json").exists()


def test_short_gpu_path_needs_no_release_archive_or_legacy_inputs():
    names = [s.name for s in g.STEPS]
    early = g.STEPS[: names.index("preflight") + 1]
    assert early[:3] == g.STEPS[:3]
    assert [s.name for s in early[:3]] == ["inventory", "image", "gpu-environment"]
    assert all(s.name not in g.HOST_STEPS for s in early)
    assert {k for s in early for k in s.inputs} == {"image_id"}


def test_overflow_accepts_owning_verified_solutions_key():
    worker = worker_module()
    from oeis_learn.evaluation.overflow_diagnostics import STRATA

    strata = {name: [dict(source="fixture")] for name in STRATA}
    worker.require_sampling_strata(strata)
    strata["verified_solutions"] = []
    with pytest.raises(g.Blocked, match="verified stratum"):
        worker.require_sampling_strata(strata)


def test_smolfile_is_parseable_and_gpu_client_is_configured():
    import tomllib

    cfg = tomllib.loads((g.ROOT / "Smolfile").read_text())
    assert cfg["gpu"] is False
    assert "LAB_GPU_IDENTITY=/home/agent/.ssh/lab-gpu" in cfg["env"]


def test_old_null_docker_setting_migrates_without_enabling_an_endpoint(tmp_path):
    path = tmp_path / "settings.json"
    g.write(path, dict(docker_host=None))
    assert "docker_host" not in g.load_settings(path)


@pytest.mark.parametrize("missing", [None, "lab-gpu", "ssh", "user", "key", "unreadable", "cwd"])
def test_vm_client_checks_need_ssh_and_project_identity_but_no_docker(
    tmp_path, monkeypatch, missing
):
    worker = worker_module()
    key = tmp_path / "dedicated-key"
    key.write_text("fixture key contents are never read")
    monkeypatch.setattr(worker, "ROOT", Path("/workspace"))
    monkeypatch.setenv("LAB_GPU_USER", "labgpu-fixture")
    monkeypatch.setenv("LAB_GPU_IDENTITY", str(key))
    calls = []

    def which(name):
        calls.append(name)
        assert name in {"lab-gpu", "ssh"}
        return None if name == missing else "/usr/bin/" + name

    monkeypatch.setattr(worker.shutil, "which", which)
    if missing == "user":
        monkeypatch.delenv("LAB_GPU_USER")
    if missing == "key":
        key.unlink()
    if missing == "unreadable":
        monkeypatch.setattr(worker.os, "access", lambda *a: False)
    if missing == "cwd":
        monkeypatch.setattr(worker, "ROOT", Path("/opt/checkout"))
    if missing:
        with pytest.raises(g.Blocked):
            worker.check_broker_client()
    else:
        worker.check_broker_client()
        assert calls == ["lab-gpu", "ssh"]


def test_future_final_launcher_still_clamps_to_original_wall_budget(tmp_path, monkeypatch):
    # Only a future real host integration can enable this path. Preserve the
    # original lifetime invariant independently of today's capability blocker.
    monkeypatch.setattr(g, "HOST_STEPS", set())
    g.write(tmp_path / "final-budget.json", dict(started_utc=1000, limit_seconds=600))
    monkeypatch.setattr(g.time, "time", lambda: 1190)
    g.write(tmp_path / "cfg.json", {})
    calls = []
    monkeypatch.setattr(g, "supervise", lambda argv, log, seconds: calls.append(seconds) or 0)
    assert g.remote_watchdog("end-to-end", tmp_path / "cfg.json", tmp_path, 1000) == 0
    assert calls == [370]


def test_readonly_vm_owned_lock_still_excludes_a_second_worker(tmp_path, monkeypatch):
    path = tmp_path / "worker.lock"
    path.write_text("stable lock inode")
    path.chmod(0o444)
    inode = path.stat().st_ino
    original = Path.open

    def cross_user_open(self, mode="r", *args, **kwargs):
        # Model a different identity with read access but no write permission.
        # This also reproduces the failure when tests themselves run as root.
        if self == path and any(c in mode for c in "wax+"):
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", cross_user_open)
    held = g.lock(path)
    try:
        assert fcntl.fcntl(held.fileno(), fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        with pytest.raises(g.Blocked, match="Active worker/controller lock"):
            g.lock(path)
        with pytest.raises(g.Blocked):
            g.require_idle(tmp_path)
    finally:
        held.close()
    g.require_idle(tmp_path)
    assert path.stat().st_ino == inode
    assert path.read_text() == "stable lock inode" and stat.S_IMODE(path.stat().st_mode) == 0o444


def test_new_lock_and_settings_remain_readable_with_restrictive_umask(tmp_path):
    old = os.umask(0o077)
    try:
        g.lock(tmp_path / "worker.lock").close()
        g.write(tmp_path / "settings.json", dict(image_id="fixture"))
    finally:
        os.umask(old)
    for name in ("worker.lock", "settings.json"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o644


def test_lock_creation_race_uses_other_creators_inode(tmp_path, monkeypatch):
    path = tmp_path / "worker.lock"
    original = Path.open
    inode = []

    def racing_open(self, mode="r", *args, **kwargs):
        if self == path and mode == "x":
            with original(self, "w") as f:
                f.write("created by the other identity")
            inode.append(path.stat().st_ino)
            raise FileExistsError(str(path))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    g.lock(path).close()
    assert path.stat().st_ino == inode[0]
    assert path.read_text() == "created by the other identity"


def test_unreadable_lock_reports_workspace_access_blocker(tmp_path, monkeypatch):
    original = Path.open

    def deny(self, *args, **kwargs):
        if self.name == "worker.lock":
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(g.Blocked, match="workspace traversal/read"):
        g.lock(tmp_path / "worker.lock")


def test_atomic_report_replaces_foreign_file_without_reusing_foreign_temporary(
    tmp_path, monkeypatch
):
    target = tmp_path / "report.json"
    target.write_text('{"old": true}')
    stale = tmp_path / "report.json.new"
    stale.write_text("interrupted foreign writer")
    for p in (target, stale):
        p.chmod(0o444)
    original = Path.open

    def cross_user_open(self, mode="r", *args, **kwargs):
        if self in (target, stale) and any(c in mode for c in "wax+"):
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", cross_user_open)
    old_inode = target.stat().st_ino
    g.write(target, dict(new=True))
    assert g.read(target) == dict(new=True) and target.stat().st_ino != old_inode
    assert stale.read_text() == "interrupted foreign writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "report.json.new"]


def test_failed_atomic_replace_preserves_original_and_removes_its_temporary(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    g.write(target, dict(old=True))

    def deny(*args):
        raise PermissionError("rename denied")

    monkeypatch.setattr(g.os, "replace", deny)
    with pytest.raises(PermissionError, match="rename denied"):
        g.write(target, dict(new=True))
    assert g.read(target) == dict(old=True)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("missing", ["settings_read", "output_write"])
def test_broker_permission_probe_blocks_before_worker_and_releases_lock(
    tmp_path, monkeypatch, missing
):
    settings = tmp_path / "settings.json"
    g.write(settings, {})
    if missing == "settings_read":
        original = g.read

        def deny(path):
            if path == settings:
                raise PermissionError("settings unreadable")
            return original(path)

        monkeypatch.setattr(g, "read", deny)
        expected = "settings are not readable"
    else:

        def deny(**kwargs):
            raise PermissionError("output unwritable")

        monkeypatch.setattr(g.tempfile, "TemporaryFile", deny)
        expected = "output directory is not writable"
    monkeypatch.setattr(g, "supervise", lambda *args: pytest.fail("must not issue the worker"))
    with pytest.raises(g.Blocked, match=expected):
        g.remote_watchdog("gpu-environment", settings, tmp_path, 120)
    out = tmp_path / "gpu-environment"
    assert not (out / "lease.json").exists()
    g.require_idle(out)


def test_shared_permission_probe_leaves_no_artifacts_or_mode_changes(tmp_path):
    settings = tmp_path / "settings.json"
    g.write(settings, {})
    out = tmp_path / "gpu-environment"
    out.mkdir()
    modes = (settings.stat().st_mode, out.stat().st_mode)
    before = g.tree_hashes(tmp_path)
    g.check_shared_io(out, settings)
    assert g.tree_hashes(tmp_path) == before
    assert modes == (settings.stat().st_mode, out.stat().st_mode)


@pytest.mark.parametrize("umask", [0o022, 0o027, 0o077])
def test_stage_retry_keeps_shared_write_access_under_restrictive_vm_umask(
    tmp_path, monkeypatch, umask
):
    stage = g.Step("gpu-probe", (), 120, True)
    calls, fake = driver(monkeypatch, tmp_path, (stage,))
    attempts = []

    def check_and_run(argv, log, seconds):
        out = tmp_path / stage.name
        assert stat.S_IMODE(out.stat().st_mode) & 0o070 == 0o070
        assert stat.S_IMODE(tmp_path.stat().st_mode) & 0o070 == 0o070
        attempts.append(out.stat().st_ino)
        return 3 if len(attempts) == 1 else fake(argv, log, seconds)

    monkeypatch.setattr(g, "supervise", check_and_run)
    old = os.umask(umask)
    try:
        assert g.run({"final_seconds": 600}, tmp_path) == 1
        assert g.run({"final_seconds": 600}, tmp_path) == 0
    finally:
        os.umask(old)
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert stat.S_IMODE((tmp_path / "history").stat().st_mode) & 0o070 == 0o070
    archived = list((tmp_path / "history").iterdir())
    assert len(archived) == 1 and archived[0].stat().st_ino == attempts[0]


@pytest.mark.parametrize("initial", [0o700, 0o750, 0o755, 0o770, 0o2750])
def test_shared_directory_mask_change_preserves_ownership_and_other_bits(tmp_path, initial):
    path = tmp_path / "stage"
    path.mkdir()
    path.chmod(initial)
    before = path.stat()
    assert g.prepare_shared_directory(path) == path
    after = path.stat()
    assert stat.S_IMODE(after.st_mode) == (stat.S_IMODE(before.st_mode) | 0o070)
    assert (after.st_uid, after.st_gid, after.st_ino) == (
        before.st_uid,
        before.st_gid,
        before.st_ino,
    )


def test_cache_success_does_not_change_retained_output_directory_modes(tmp_path, monkeypatch):
    stage = g.Step("probe", (), 120)
    calls, _ = driver(monkeypatch, tmp_path, (stage,))
    assert g.run({"final_seconds": 600}, tmp_path) == 0
    out = tmp_path / stage.name
    out.chmod(0o750)
    assert g.run({"final_seconds": 600}, tmp_path) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o750
    assert calls == [stage.name]


@pytest.mark.parametrize("silent", [False, True])
def test_shared_mask_failure_prevents_step_or_budget_issuance(tmp_path, monkeypatch, silent):
    path = tmp_path / "state"
    path.mkdir()
    path.chmod(0o750)
    original = Path.chmod

    def fail(self, *args, **kwargs):
        if self == path:
            if silent:
                return
            raise PermissionError("mount rejects chmod")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", fail)
    monkeypatch.setattr(
        g, "supervise", lambda *args: pytest.fail("no work after permission failure")
    )
    with pytest.raises(g.Blocked, match="ACL"):
        g.run({"final_seconds": 600}, path)
    assert not (path / "receipts.json").exists()
    assert not (path / "final-budget.json").exists()


def test_gpu_directory_error_includes_both_acl_types_and_actual_mode(tmp_path, monkeypatch):
    out = tmp_path / "gpu-environment"
    out.mkdir()
    out.chmod(0o750)
    settings = tmp_path / "settings.json"
    g.write(settings, {})

    def deny(**kwargs):
        raise PermissionError("no named-user write permission")

    monkeypatch.setattr(g.tempfile, "TemporaryFile", deny)
    with pytest.raises(g.Blocked) as e:
        g.check_shared_io(out, settings)
    error = str(e.value)
    assert "mode=0750" in error
    assert "both access and default ACLs" in error
    assert "stage directories are recreated" in error


@pytest.mark.parametrize("name", ["transport.log", "worker.log"])
def test_live_logs_are_readable_before_child_launch_and_preserve_appends(
    tmp_path, monkeypatch, name
):
    log = tmp_path / name
    original = Path.open

    def inherited_private_acl(self, mode="r", *args, **kwargs):
        stream = original(self, mode, *args, **kwargs)
        if self == log and mode == "ab":
            # Model inherited other::---, which a permissive umask cannot fix.
            os.fchmod(stream.fileno(), 0o660)
        return stream

    monkeypatch.setattr(Path, "open", inherited_private_acl)
    code = (
        "from pathlib import Path; import sys; "
        "assert Path(sys.argv[1]).stat().st_mode & 0o044 == 0o044; print('live output')"
    )
    argv = [sys.executable, "-c", code, log]
    assert g.supervise(argv, log, 5) == 0
    inode = log.stat().st_ino
    assert g.supervise(argv, log, 5) == 0
    assert log.stat().st_ino == inode
    assert log.read_text().count("\nlive output\n") == 2
    assert stat.S_IMODE(log.stat().st_mode) == 0o664


def test_publish_nested_reports_and_checkpoints_without_granting_write_access(tmp_path):
    out = tmp_path / "stage"
    nested = out / "legacy/0/checkpoints"
    old = os.umask(0o077)
    try:
        nested.mkdir(parents=True)
        (nested / "model.pt").write_bytes(b"checkpoint bytes")
        (out / "tests.xml").write_text("<testsuite/>")
    finally:
        os.umask(old)
    (out / "tests.xml").chmod(0o660)
    before = {p: (p.stat(), p.read_bytes()) for p in out.rglob("*") if p.is_file()}
    g.publish_outputs(out)
    for directory in (out, out / "legacy", out / "legacy/0", nested):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    for path, (info, contents) in before.items():
        after = path.stat()
        assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(info.st_mode) | 0o044
        assert (after.st_uid, after.st_gid, after.st_ino) == (info.st_uid, info.st_gid, info.st_ino)
        assert path.read_bytes() == contents
    assert len(g.tree_hashes(out)) == 2


def test_publish_leaves_foreign_controller_files_untouched(tmp_path, monkeypatch):
    lock = tmp_path / "worker.lock"
    lock.write_text("stable inode")
    lock.chmod(0o444)
    original = Path.lstat
    before = lock.stat()

    def foreign_owner(self):
        info = original(self)
        if self == lock:
            fields = list(info)
            fields[4] = os.geteuid() + 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", foreign_owner)
    g.publish_outputs(tmp_path)
    assert lock.stat() == before
    assert lock.read_text() == "stable inode"


@pytest.mark.parametrize("kind", ["file-symlink", "directory-symlink", "hardlink", "fifo"])
def test_publish_rejects_links_and_special_files_without_changing_external_inputs(tmp_path, kind):
    source = tmp_path / "input"
    source.mkdir()
    target = source / "parent.pt"
    target.write_bytes(b"actual input")
    target.chmod(0o600)
    out = tmp_path / "stage"
    out.mkdir()
    link = out / "linked-output"
    if kind == "file-symlink":
        link.symlink_to(target)
    elif kind == "directory-symlink":
        link.symlink_to(source, target_is_directory=True)
    elif kind == "hardlink":
        os.link(target, link)
    else:
        os.mkfifo(link)
    before = target.stat()
    with pytest.raises(g.Blocked, match="Shared stage output must"):
        g.publish_outputs(out)
    assert target.stat() == before and target.read_bytes() == b"actual input"


@pytest.mark.parametrize("silent", [False, True])
def test_publish_permission_failure_reports_path_and_preserves_artifacts(
    tmp_path, monkeypatch, silent
):
    path = tmp_path / "model.pt"
    path.write_bytes(b"retained checkpoint")
    path.chmod(0o600)
    original = Path.chmod

    def deny(self, mode, *args, **kwargs):
        if self == path:
            if silent:
                return
            raise PermissionError(13, "Permission denied", str(path))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", deny)
    with pytest.raises(g.Blocked) as e:
        g.publish_outputs(tmp_path)
    assert str(path) in str(e.value)
    assert path.read_bytes() == b"retained checkpoint"


@pytest.mark.parametrize("code", [0, 1, 124, 130])
def test_watchdog_publishes_partial_outputs_and_releases_lock_for_all_exits(
    tmp_path, monkeypatch, code
):
    settings = tmp_path / "cfg.json"
    g.write(settings, {})
    out = tmp_path / "gpu-smoke"

    def partial_worker(argv, log, seconds):
        nested = out / "partial"
        nested.mkdir(mode=0o700)
        report = nested / "tests.xml"
        report.write_text("retained test results")
        report.chmod(0o600)
        log.write_text("retained worker output")
        log.chmod(0o660)
        return code

    monkeypatch.setattr(g, "supervise", partial_worker)
    assert g.remote_watchdog("gpu-smoke", settings, tmp_path, 600) == code
    assert (out / "partial").stat().st_mode & 0o055 == 0o055
    for path in (out / "partial/tests.xml", out / "worker.log"):
        assert path.stat().st_mode & 0o044 == 0o044
    g.require_idle(out)


def test_watchdog_releases_lock_when_publication_fails(tmp_path, monkeypatch):
    settings = tmp_path / "cfg.json"
    g.write(settings, {})
    monkeypatch.setattr(g, "supervise", lambda *args: 0)

    def deny(out):
        raise g.Blocked("cannot publish")

    monkeypatch.setattr(g, "publish_outputs", deny)
    with pytest.raises(g.Blocked, match="cannot publish"):
        g.remote_watchdog("gpu-smoke", settings, tmp_path, 600)
    g.require_idle(tmp_path / "gpu-smoke")


@pytest.mark.parametrize("disposition", ["pass", "blocked", "error"])
def test_worker_publishes_third_party_outputs_on_success_and_failure(
    tmp_path, monkeypatch, disposition
):
    worker = worker_module()
    monkeypatch.setitem(sys.modules, "gpu_progress_steps", worker)
    settings = tmp_path / "cfg.json"
    g.write(settings, {})

    def execute(name, cfg, state, out, seconds):
        report = out / "tests.xml"
        report.write_text("retained junit")
        report.chmod(0o600)
        if disposition == "blocked":
            raise g.Blocked("actual device unavailable")
        if disposition == "error":
            raise RuntimeError("actual test failed")
        return dict(value="actual test result")

    monkeypatch.setattr(worker, "execute", execute)
    if disposition == "error":
        with pytest.raises(RuntimeError, match="actual test failed"):
            g.execute_worker("gpu-smoke", settings, tmp_path, 600)
    else:
        assert g.execute_worker("gpu-smoke", settings, tmp_path, 600) == (
            0 if disposition == "pass" else 3
        )
    assert (tmp_path / "gpu-smoke/tests.xml").stat().st_mode & 0o044 == 0o044


def test_unreadable_directory_cannot_be_silently_omitted_from_receipt(tmp_path, monkeypatch):
    def denied_walk(path, *, onerror):
        onerror(PermissionError(13, "Permission denied", str(path / "private")))
        yield  # retain os.walk's generator interface

    monkeypatch.setattr(g.os, "walk", denied_walk)
    with pytest.raises(PermissionError, match="private"):
        g.success_receipt("identity", tmp_path, {}, 1, g.Step("gpu-smoke", (), 1, True))


@pytest.mark.parametrize("publication", [False, True])
def test_controller_can_receipt_and_skip_gpu_owned_outputs(tmp_path, monkeypatch, publication):
    stage = g.Step("gpu-probe", (), 120, True)
    driver(monkeypatch, tmp_path, (stage,))
    if not publication:
        monkeypatch.setattr(g, "publish_outputs", lambda out: None)
    original = Path.open
    in_gpu = False
    issued = []

    def vm_read(self, mode="r", *args, **kwargs):
        if not in_gpu and mode in {"r", "rb"} and self.is_file():
            # The VM's mapped identity cannot use the GPU owner's/group's bits.
            if not self.stat().st_mode & 0o004:
                raise PermissionError(13, "Permission denied", str(self))
        return original(self, mode, *args, **kwargs)

    def transport(argv, log, seconds):
        nonlocal in_gpu
        if argv[0] == "lab-gpu":
            issued.append(stage.name)
            in_gpu = True
            try:
                return g.remote_watchdog(stage.name, Path(argv[-3]), tmp_path, seconds)
            finally:
                in_gpu = False
        out = tmp_path / stage.name
        nested = out / "nested"
        nested.mkdir(mode=0o700)
        (nested / "model.pt").write_bytes(b"checkpoint")
        (nested / "model.pt").chmod(0o600)
        log.write_text("successful GPU worker")
        log.chmod(0o660)
        g.write(out / "step-result.json", dict(result=dict(value="successful GPU check")))
        return 0

    monkeypatch.setattr(Path, "open", vm_read)
    monkeypatch.setattr(g, "supervise", transport)
    assert g.run({"final_seconds": 600}, tmp_path) == (0 if publication else 1)
    receipt = g.read(tmp_path / "receipts.json")[stage.name]
    if publication:
        assert receipt["status"] == "PASS"
        assert {"worker.log", "nested/model.pt"} <= receipt["files"].keys()
        assert g.run({"final_seconds": 600}, tmp_path) == 0
        assert issued == [stage.name]
    else:
        assert receipt["status"] == "FAIL" and "Permission denied" in receipt["error"]


def test_gpu_smoke_keeps_training_on_workspace_storage_without_broker_env_overrides(
    tmp_path, monkeypatch
):
    worker = worker_module()
    monkeypatch.setenv("OEIS_GPU_SMOKE_RUN_DIR", "previous-run")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "fixture-device")
    calls = []

    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        path = Path(argv[-1].removeprefix("--junitxml="))
        path.write_text('<testsuite><testcase name="fixture"/></testsuite>')

    monkeypatch.setattr(worker, "command", launch)
    result = worker.execute("gpu-smoke", {}, tmp_path, tmp_path, 1800)
    argv, kwargs = calls[0]
    assert len(calls) == 1 and result["tests"] == 1
    assert kwargs["env"]["OEIS_GPU_SMOKE_RUN_DIR"] == str(tmp_path / "training-run")
    assert kwargs["env"]["HIP_VISIBLE_DEVICES"] == "fixture-device"
    assert os.environ["OEIS_GPU_SMOKE_RUN_DIR"] == "previous-run"
    assert "tests/integration/test_foundation_gpu.py::test_actual_hip_three_updates" in argv
    assert not any(arg.startswith("--basetemp") for arg in argv)
    assert argv[:4] == [sys.executable, "-m", "pytest", "-q"]


@pytest.mark.parametrize("retained", [False, True])
def test_smoke_training_root_uses_actual_run_filesystem_and_frozen_disk_floor(
    tmp_path, monkeypatch, retained
):
    from types import SimpleNamespace
    from tests.integration import test_foundation_gpu as smoke
    from oeis_learn.tracking import foundation_controller as controller

    temporary = tmp_path / "container-tmp"
    workspace = tmp_path / "workspace/gpu-smoke/training-run"
    temporary.mkdir()
    if retained:
        monkeypatch.setenv("OEIS_GPU_SMOKE_RUN_DIR", str(workspace))
    else:
        monkeypatch.delenv("OEIS_GPU_SMOKE_RUN_DIR", raising=False)
    roots = []

    def prepare(config, pool, root, **kwargs):
        assert config == Path("configs/foundation/wat_smoke.yaml")
        assert kwargs == dict(device="cuda", diagnostic=True)
        root.mkdir(parents=True)
        roots.append(root)

    class ReachedTraining(RuntimeError):
        pass

    def run(root):
        # Model the actual broker: 2 GiB /tmp, adequate workspace storage.
        guard = controller.ResourceGuard(
            root,
            free_floor=100 << 30,
            disk_usage=lambda measured: SimpleNamespace(
                free=(200 if measured == workspace else 2) << 30
            ),
        )
        guard.check()
        raise ReachedTraining("storage guard passed; actual HIP training is not run by this test")

    monkeypatch.setattr(smoke, "prepare_run", prepare)
    monkeypatch.setattr(smoke, "run_training", run)
    monkeypatch.setattr(
        controller, "resource_metrics", lambda device: {"host_available": {"state": "unavailable"}}
    )
    expected = ReachedTraining if retained else OSError
    with pytest.raises(expected):
        smoke.test_actual_hip_three_updates(("unused-config", "fixture-pool"), temporary)
    assert roots == [workspace if retained else temporary / "run"]
