"""Failed startup and pool reclamation must not depend on garbage collection."""

import errno
import socket
from threading import Lock
from unittest.mock import Mock, patch

import pytest

from oeis_learn.sandbox import worker_pool as workers


@pytest.mark.parametrize("stage", ["process_creation", "process_start", "ready"])
def test_slot_start_failure_closes_both_socket_endpoints(stage):
    # Hold both sockets and the process alive so refcount/GC cannot close them.
    parent, child = socket.socketpair()
    process = Mock()
    process.pid = None if stage != "ready" else 123
    process.is_alive.return_value = False
    context = Mock()
    context.Process.return_value = process
    failure = OSError(errno.EMFILE, "too many open files")
    if stage == "process_creation":
        context.Process.side_effect = failure
    elif stage == "process_start":
        process.start.side_effect = failure
    try:
        with (
            patch.object(workers.socket, "socketpair", return_value=(parent, child)),
            patch.object(workers.mp, "get_context", return_value=context),
            patch.object(workers, "_receive", side_effect=failure),
            pytest.raises(OSError, match="too many open files"),
        ):
            workers._Slot(0, workers._worker, "python_wasmtime")
        assert parent.fileno() == child.fileno() == -1
        if stage != "process_creation":
            process.close.assert_called_once()
        if stage == "ready":
            process.join.assert_called_once_with(timeout=2)
        else:
            process.join.assert_not_called()
    finally:
        parent.close()
        child.close()


def test_repeated_start_failure_does_not_leak_descriptors():
    sockets = []
    original = socket.socketpair

    def retain_pair():
        pair = original()
        sockets.extend(pair)
        return pair

    context = Mock()
    context.Process.side_effect = OSError(errno.EMFILE, "too many open files")
    try:
        with (
            patch.object(workers.socket, "socketpair", side_effect=retain_pair),
            patch.object(workers.mp, "get_context", return_value=context),
        ):
            for _ in range(128):
                with pytest.raises(OSError):
                    workers._Slot(0, workers._worker, "python_wasmtime")
        assert all(s.fileno() == -1 for s in sockets)
    finally:
        for s in sockets:
            s.close()


def test_pool_reclaims_remaining_slots_when_one_stop_fails():
    pool = workers.WorkerPool.__new__(workers.WorkerPool)
    pool.lock, pool.executor = Lock(), Mock()
    first, second = Mock(), Mock()
    first.stop.side_effect = RuntimeError("worker reclamation exceeded two seconds")
    pool.all_slots = [first, second]
    with pytest.raises(RuntimeError, match="reclamation"):
        pool.close()
    second.stop.assert_called_once()
    pool.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
