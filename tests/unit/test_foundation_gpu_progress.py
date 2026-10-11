"""Controller safety and recovery checks run without the broker or GPU."""

import importlib.util
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


def test_cleanup_only_targets_its_saved_lease(monkeypatch, tmp_path):
    lease = "a" * 32
    g.write(tmp_path / "lease.json", dict(lease=lease, docker_host="ssh://accepted"))
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, "abcdef123456\n" if "ps" in argv else "", "")

    monkeypatch.setattr(g.subprocess, "run", fake)
    g.reclaim_lease(tmp_path)
    assert calls[0][0][-1] == "label=oeis.progression=" + lease
    assert calls[1][0] == ["docker", "rm", "-f", "abcdef123456"]
    assert calls[0][1]["env"]["DOCKER_HOST"] == "ssh://accepted"
    assert g.read(tmp_path / "reclamation.json")["status"] == "complete"


def test_cleanup_failure_blocks_success(monkeypatch, tmp_path):
    g.write(tmp_path / "lease.json", dict(lease="a" * 32, docker_host=None))
    monkeypatch.setattr(
        g.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "denied")
    )
    with pytest.raises(RuntimeError, match="denied"):
        g.reclaim_lease(tmp_path)
    assert g.read(tmp_path / "reclamation.json")["status"] == "failed"


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


def test_remote_final_deadline_uses_original_wall_budget(tmp_path, monkeypatch):
    settings = tmp_path / "cfg.json"
    g.write(settings, {})
    g.write(tmp_path / "final-budget.json", dict(started_utc=1000, limit_seconds=600))
    monkeypatch.setattr(g.time, "time", lambda: 1190)
    calls = []
    monkeypatch.setattr(g, "supervise", lambda argv, log, seconds: calls.append(seconds) or 0)
    monkeypatch.setattr(g, "reclaim_lease", lambda out: None)
    assert g.remote_watchdog("end-to-end", settings, tmp_path, 1000) == 0
    assert calls == [370]


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
