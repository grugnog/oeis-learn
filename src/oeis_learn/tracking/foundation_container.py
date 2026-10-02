"""Bounded host-side Docker launcher using the existing root Dockerfile image.

No Docker socket enters a workload. This launcher exposes only explicit role
views; the development VM's shared workspace is not a qualifying mount plan.
Container transport for live evaluator/worker role coordination is not claimed.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import subprocess
import uuid


def container_command(*, role, image, inputs, output, name, command, render_node=None):
    if role not in ("learner", "preflight", "generation", "evaluator", "worker"):
        raise ValueError("unknown role")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("resolved local image ID required; no tag or pull")
    source, target = Path(inputs).resolve(strict=True), Path(output).resolve(strict=True)
    if (
        not source.is_dir()
        or not target.is_dir()
        or source == target
        or source in target.parents
        or target in source.parents
    ):
        raise ValueError("distinct nonoverlapping input/output directories required")
    if any("," in str(p) or "\n" in str(p) for p in (source, target)) or not command:
        raise ValueError("invalid mount/command")
    worker = role == "worker"
    memory, pids, cpus = ("1g", "64", "1") if worker else ("88g", "512", "4")
    argv = [
        "docker",
        "run",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=" + memory,
        "--memory-swap=" + memory,
        "--pids-limit=" + pids,
        "--cpus=" + cpus,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--init",
        "--ipc=private",
        "--log-driver=none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--workdir=/work",
        "--mount",
        f"type=bind,src={source},dst=/work/input,readonly",
        "--mount",
        f"type=bind,src={target},dst=/work/output",
        "--env",
        "PYTHONPATH=/opt/oeis-build/src",
        "--env",
        "OMP_NUM_THREADS=" + cpus,
    ]
    if role in ("learner", "preflight"):
        node = Path(render_node or "/dev/dri/renderD128")
        if re.fullmatch(r"/dev/dri/renderD[0-9]+", str(node)) is None:
            raise ValueError("specific render node required")
        for path in (Path("/dev/kfd"), node):
            if not path.is_char_device():
                raise ValueError(f"GPU device unavailable: {path}")
            argv.extend(["--device", str(path), "--group-add", str(path.stat().st_gid)])
    elif render_node is not None:
        raise ValueError("GPU device forbidden for this role")
    return argv + ["--entrypoint", command[0], image, *command[1:]]


def launch(*, seconds, **kwargs):
    if type(seconds) is not int or not 0 < seconds <= 600:
        raise ValueError("diagnostic deadline must be within 600 seconds")
    name = "oeis-foundation-" + uuid.uuid4().hex
    command = container_command(name=name, **kwargs)
    process = subprocess.Popen(command)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        # Docker removes/reaps the cgroup, not just the local docker client.
        try:
            removed = subprocess.run(
                ["docker", "rm", "--force", name],
                check=False,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if removed.returncode != 0:
                raise RuntimeError("Docker workload reclamation failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
