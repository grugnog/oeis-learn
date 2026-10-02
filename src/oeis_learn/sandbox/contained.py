"""Killable, resource-bounded workers for trusted solver/proof entry points."""
from __future__ import annotations
import multiprocessing as mp
import resource
import time


def _child(connection, function, args):
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        connection.send(("ok", function(*args)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def run_contained(function, args=(), *, timeout_ms=2000):
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 2000:
        raise ValueError("worker deadline must be an integer in 1..2000 ms")
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(send, function, args), daemon=True)
    try:
        process.start()
        send.close()
        remaining = max(0, timeout_ms / 1000 - (time.monotonic() - started))
        if receive.poll(remaining):
            try:
                return receive.recv()
            except EOFError:
                return "error", "worker exited without a result"
        return "timeout", "worker deadline exceeded"
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join()
        receive.close()
        send.close()
