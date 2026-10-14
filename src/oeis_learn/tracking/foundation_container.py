"""Bounded host-side Docker/Podman launcher using the existing root Dockerfile image.

No Docker socket enters a workload. This launcher exposes only explicit role
views; the whole workspace is never a qualifying role mount.
Live role coordination uses role_coordinator without sharing a Docker socket.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import uuid
from pathlib import Path

from oeis_learn.container_runtime import resolve_engine


def container_command(
    *,
    role,
    image,
    inputs,
    output,
    name,
    command,
    render_node=None,
    gpu=None,
    engine=None,
    memory_bytes=16 << 30,
    cpus=4,
):
    if role not in ("learner", "preflight", "generation", "evaluator", "worker"):
        raise ValueError("unknown role")
    source, target = Path(inputs).resolve(strict=True), Path(output).resolve(strict=True)
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("distinct nonoverlapping input/output directories required")
    use_gpu = role in ("learner", "preflight") if gpu is None else gpu
    if type(use_gpu) is not bool or (
        use_gpu and role not in ("learner", "preflight", "generation")
    ):
        raise ValueError("invalid role GPU capability")
    labels = {}
    lease = os.environ.get("OEIS_ROLE_LEASE")
    if lease:
        if re.fullmatch(r"[0-9a-f]{32}", lease) is None:
            raise ValueError("invalid progression lease")
        labels["oeis.progression"] = lease
    return resolve_engine(engine).command(
        image=image,
        name=name,
        mounts=[(source, "/work/input", False), (target, "/work/output", True)],
        argv=command,
        gpu=use_gpu,
        render_node=render_node,
        memory_bytes=1 << 30 if role == "worker" else memory_bytes,
        cpus=1 if role == "worker" else cpus,
        pids=64 if role == "worker" else 512,
        labels=labels,
        auto_remove=False,
        environment={
            "PYTHONPATH": "/opt/oeis-build/src",
            "OMP_NUM_THREADS": str(1 if role == "worker" else cpus),
            "HOME": "/tmp",
        },
    )


def launch(*, seconds, **kwargs):
    if type(seconds) is not int or not 0 < seconds <= 600:
        raise ValueError("diagnostic deadline must be within 600 seconds")
    name = "oeis-foundation-" + uuid.uuid4().hex
    engine = resolve_engine(kwargs.pop("engine", None))
    command = container_command(name=name, engine=engine, **kwargs)
    process = subprocess.Popen(command)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        # The engine removes/reaps the cgroup, not just the local engine client.
        try:
            engine.remove(name)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["auto", "docker", "podman"], default="auto")
    parser.add_argument("--role", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-node")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    if args["command"][:1] == ["--"]:
        args["command"] = args["command"][1:]
    return launch(**args)


if __name__ == "__main__":
    raise SystemExit(main())
