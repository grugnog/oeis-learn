"""Bounded container removal, foreground-client reaping and retryable failures."""

import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest

from tests.unit.test_fresh_workflow import FakeClock, host


class Process:
    def __init__(self):
        self.returncode = None
        self.kills = 0
        self.waits = []

    def poll(self):
        return self.returncode

    def kill(self):
        self.kills += 1

    def wait(self, timeout):
        self.waits.append(timeout)
        self.returncode = -9 if self.kills else 0
        return self.returncode


def coordinator(root):
    return host.Coordinator(root, {"podman": "podman", "project": "fixture"}, "image", None)


def test_persistent_worker_removal_skips_stop_grace_and_reaps_client(tmp_path):
    controller = coordinator(tmp_path)
    process = Process()
    controller.live["worker"] = process

    def remove(argv, **kwargs):
        # The original controller timeout cannot cover Podman's stop grace.
        if "--time=0" not in argv or kwargs["timeout"] <= 2:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        assert argv[-1] == "worker" and "--force" in argv and "--ignore" in argv
        assert kwargs["check"] and kwargs["capture_output"]
        assert 10 < kwargs["timeout"] <= 30
        return subprocess.CompletedProcess(argv, 0)

    try:
        with patch.object(host.subprocess, "run", side_effect=remove) as removed:
            controller.reclaim("worker")
            controller.reclaim("worker")
        assert removed.call_count == 1
        assert not controller.live
        assert process.kills == 1 and process.returncode == -9
        assert process.waits and all(0 < timeout <= 5 for timeout in process.waits)
    finally:
        controller.executor.shutdown()


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "interrupt"])
def test_removal_failure_remains_tracked_and_can_be_retried(tmp_path, failure):
    controller = coordinator(tmp_path)
    process = Process()
    controller.live["worker"] = process
    error = {
        "timeout": subprocess.TimeoutExpired(["podman"], 30),
        "nonzero": subprocess.CalledProcessError(125, ["podman"]),
        "interrupt": KeyboardInterrupt(),
    }[failure]
    try:
        with patch.object(host.subprocess, "run", side_effect=error), pytest.raises(type(error)):
            controller.reclaim("worker")
        assert controller.live == {"worker": process}
        assert process.returncode == -9 and process.kills == 1
        with patch.object(host.subprocess, "run") as removed:
            controller.close()
        assert removed.call_count == 1 and not controller.live
    finally:
        controller.executor.shutdown()


def test_close_attempts_every_worker_and_retains_only_failed_removals(tmp_path):
    controller = coordinator(tmp_path)
    processes = {f"worker-{number}": Process() for number in range(4)}
    controller.live.update(processes)
    attempted = []

    def remove(argv, **kwargs):
        name = argv[-1]
        attempted.append(name)
        if name in {"worker-0", "worker-2"}:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    try:
        with (
            patch.object(host.subprocess, "run", side_effect=remove),
            pytest.raises(RuntimeError, match="role reclamation failed"),
        ):
            controller.close()
        assert set(attempted) == set(processes)
        assert set(controller.live) == {"worker-0", "worker-2"}
        assert all(process.returncode == -9 for process in processes.values())
        with patch.object(host.subprocess, "run") as retry:
            controller.close()
        assert retry.call_count == 2 and not controller.live
    finally:
        controller.executor.shutdown()


def test_failed_client_reap_remains_tracked(tmp_path):
    controller = coordinator(tmp_path)
    process = Process()
    process.wait = Mock(side_effect=[subprocess.TimeoutExpired("client", 5), 0])
    controller.live["worker"] = process
    try:
        with patch.object(host.subprocess, "run"):
            with pytest.raises(subprocess.TimeoutExpired):
                controller.reclaim("worker")
            assert "worker" in controller.live
            controller.reclaim("worker")
        assert not controller.live and process.wait.call_count == 2
    finally:
        controller.executor.shutdown()


def test_real_foreground_client_is_killed_and_reaped(tmp_path):
    controller = coordinator(tmp_path)
    # Blocking stdin is a real live child, without sleeps or a Podman dependency.
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE
    )
    controller.live["worker"] = process
    try:
        with patch.object(host.subprocess, "run"):
            controller.reclaim("worker")
        assert process.returncode is not None and process.poll() is not None
        assert not controller.live
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        controller.executor.shutdown()


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_probe_reports_pass_only_after_cleanup(tmp_path, capsys, cleanup_error):
    clock = FakeClock()
    reclaimed = []

    class Probe:
        def __init__(self, root, policy, image, ledger):
            self.ledger = ledger

        def run(self, *args):
            clock.n += 10**9
            return {"status": "PASS"}

        def close(self):
            assert not capsys.readouterr().out
            assert self.ledger.previous["status"] == "cleanup"
            reclaimed.append(True)
            clock.n += 10**9
            if cleanup_error:
                raise RuntimeError("probe cleanup failed")

    with (
        patch.object(host, "clock", clock),
        (
            pytest.raises(RuntimeError, match="probe cleanup failed")
            if cleanup_error
            else nullcontext()
        ),
    ):
        host.execute(
            {"workspace": str(tmp_path), "home": str(tmp_path / "private")},
            "image",
            ["check"],
            coordinator_factory=Probe,
        )
    assert reclaimed == [True]
    output = capsys.readouterr().out
    assert not output if cleanup_error else '"status": "PASS"' in output
