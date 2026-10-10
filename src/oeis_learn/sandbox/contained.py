"""Killable bounded workers, including startup, IPC and reclamation.

Only trusted internal functions use this service. Pickle never accepts files or
remote input; the child is created by this controller. Result size is bounded.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import resource
import socket
import struct
import time

MAX_MESSAGE = 1 << 20


def _send(channel, value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_MESSAGE:
        raise ValueError("worker result exceeds 1 MiB")
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def _receive(channel, deadline):
    def read(count):
        result = bytearray()
        while len(result) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker deadline exceeded")
            channel.settimeout(remaining)
            part = channel.recv(count - len(result))
            if not part:
                raise EOFError("worker exited without a result")
            result.extend(part)
        return bytes(result)

    size = struct.unpack("!I", read(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("worker result exceeds 1 MiB")
    return pickle.loads(read(size))


def _child(channel, function, args):
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        _send(channel, ("ok", function(*args)))
    except BaseException as exc:
        try:
            _send(channel, ("error", f"{type(exc).__name__}: {str(exc)[:4096]}"))
        except (OSError, ValueError, MemoryError):
            pass
    finally:
        channel.close()


def run_contained(function, args=(), *, timeout_ms=2000):
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 2000:
        raise ValueError("worker deadline must be an integer in 1..2000 ms")
    deadline = time.monotonic() + timeout_ms / 1000
    receive, send = socket.socketpair()
    process = mp.get_context("spawn").Process(
        target=_child, args=(send, function, args), daemon=True
    )
    try:
        process.start()
        send.close()
        try:
            value = _receive(receive, deadline)
            if not isinstance(value, tuple) or len(value) != 2 or value[0] not in ("ok", "error"):
                return "error", "invalid worker response"
            return value
        except TimeoutError:
            return "timeout", "worker deadline exceeded"
        except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
            return "error", str(exc)
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=2)
            if process.is_alive():
                raise RuntimeError("worker reclamation exceeded two seconds")
            process.close()
