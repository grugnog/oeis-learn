"""Exercise live quota heartbeats without a container engine or accelerator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "monitor_host", Path(__file__).parents[2] / "scripts/foundation_host.py"
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.fixture
def active_role(tmp_path, monkeypatch):
    root = tmp_path / "job/roles"
    output = root / "train-000025-attempt-test"
    queue = output / "runtime-queue"
    queue.mkdir(parents=True)
    host.write(output / "result.json", {"completed_updates": 2500})
    ledger = SimpleNamespace(remaining=lambda: 300, previous={}, record=Mock())
    monkeypatch.setattr(host.Engine, "from_config", lambda policy: object())
    coordinator = host.Coordinator(root, {"host_free_memory_bytes": 0}, "image", ledger)
    process = SimpleNamespace(returncode=0, poll=Mock(side_effect=[None, 0]))
    monkeypatch.setattr(coordinator, "spawn", lambda *args: ("learner", process))
    reclaimed = Mock()
    monkeypatch.setattr(coordinator, "reclaim", reclaimed)
    monkeypatch.setattr(host.time, "monotonic", lambda: 0)
    monkeypatch.setattr(host.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 << 30))
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return "MemAvailable: 8388608 kB\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    try:
        yield coordinator, output, queue, reclaimed
    finally:
        coordinator.executor.shutdown()


@pytest.mark.parametrize("change", ["unlink", "rename", "remove-parent"])
def test_quota_heartbeat_survives_worker_file_disappearing(active_role, monkeypatch, change):
    coordinator, output, queue, reclaimed = active_role
    transient = queue / "tmp-request"
    transient.write_bytes(b"execution request")
    original_stat = Path.stat
    observed = []

    def stat_after_worker_change(path, *args, **kwargs):
        if path == transient:
            # os.walk has listed the name; model a worker's atomic publication
            # or cleanup before the monitor can stat that directory entry.
            observed.append(path)
            if change == "rename":
                transient.replace(queue / "published")
            else:
                transient.unlink()
                if change == "remove-parent":
                    queue.rmdir()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_after_worker_change)
    assert coordinator.run("train", "learner", output, output, 30) == {
        "completed_updates": 2500
    }
    assert observed == [transient]
    reclaimed.assert_called_once_with("learner")
    assert coordinator.ledger.record.call_args.args == ("train-finished",)


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_quota_heartbeat_preserves_other_filesystem_errors(active_role, monkeypatch, error):
    coordinator, output, queue, reclaimed = active_role
    retained = queue / "retained"
    retained.write_bytes(b"request")
    original_stat = Path.stat

    def inaccessible(path, *args, **kwargs):
        if path == retained:
            raise error("cannot inspect retained file")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible)
    with pytest.raises(error, match="cannot inspect retained file"):
        coordinator.run("train", "learner", output, output, 30)
    reclaimed.assert_called_once_with("learner")
    assert coordinator.ledger.record.call_args.args == ("train-finished",)


def test_quota_still_counts_runtime_queue_files(active_role, monkeypatch):
    coordinator, output, queue, reclaimed = active_role
    retained = queue / "large-retained-request"
    retained.write_bytes(b"request")
    original_stat = Path.stat

    def oversized(path, *args, **kwargs):
        if path == retained:
            return SimpleNamespace(st_size=(200 << 30) + 1)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", oversized)
    with pytest.raises(OSError, match="run disk quota200GiB exhausted"):
        coordinator.run("train", "learner", output, output, 30)
    reclaimed.assert_called_once_with("learner")


def test_workspace_disk_reserve_still_stops_role(active_role, monkeypatch):
    coordinator, output, _, reclaimed = active_role
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="workspace disk reserve exhausted"):
        coordinator.run("train", "learner", output, output, 30)
    reclaimed.assert_called_once_with("learner")
