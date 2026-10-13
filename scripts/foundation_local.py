#!/usr/bin/env python3
"""Run fresh foundation diagnostics on the lab host, without a VM or SSH."""

import argparse
import json
import pwd
import re
import stat
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("/etc/lab/gpu-runner.json")


def relative(value):
    path = (ROOT / value).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        raise ValueError("paths must remain inside the registered host checkout") from None


def load_policy(project):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", project):
        raise ValueError("invalid project name")
    if not POLICY.is_file():
        raise ValueError(
            "run on the lab host with /etc/lab/gpu-runner.json; use foundation_run.py in the VM"
        )
    info = POLICY.stat()
    if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("host policy must be root-owned and not group/world writable")
    policy = json.loads(POLICY.read_bytes())[project]
    if not policy.get("foundation", False):
        raise ValueError("foundation host integration is disabled in Nix policy")
    if Path(policy["workspace"]).resolve() != ROOT:
        raise ValueError("this checkout differs from the registered project workspace")
    if policy["project"] != project or policy["user"] != "labgpu-" + project:
        raise ValueError("project account identity mismatch")
    for key in ("python", "script", "env", "podman"):
        if not policy[key].startswith("/nix/store/") or ".." in Path(policy[key]).parts:
            raise ValueError(f"{key} must refer to the installed immutable Nix executable")
    return policy


class Host:
    def __init__(self, policy):
        self.policy = policy
        self.account = pwd.getpwnam(policy["user"])
        if self.account.pw_uid == 0:
            raise ValueError("the project container account must be rootless")
        self.image = None

    def broker(self, operation, *argv, capture=False):
        # Root imports only the installed Nix broker, never this checkout.
        request = {"version": 1, "op": operation, "cwd": ".", "argv": list(argv)}
        return subprocess.run(
            [
                "sudo",
                "-n",
                "--",
                self.policy["python"],
                "-I",
                self.policy["script"],
                "broker",
                self.policy["project"],
            ],
            input=(json.dumps(request) + "\n").encode(),
            stdout=subprocess.PIPE if capture else None,
            check=True,
        )

    def ensure(self):
        if self.image is None:
            value = self.broker("ensure", capture=True).stdout.decode().strip()
            if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value):
                raise ValueError("broker did not return an immutable image ID")
            self.image = value if value.startswith("sha256:") else "sha256:" + value
        return self.image

    def as_project(self, *argv):
        policy, account = self.policy, self.account
        return [
            "sudo",
            "-n",
            "-u",
            policy["user"],
            "--",
            policy["env"],
            "-i",
            "HOME=" + policy["home"],
            "USER=" + policy["user"],
            "LOGNAME=" + policy["user"],
            "PATH=" + policy["path"],
            "XDG_RUNTIME_DIR=/run/user/" + str(account.pw_uid),
            *argv,
        ]

    def cpu(self, *argv, network=False, timeout=10800):
        policy, account, image = self.policy, self.account, self.ensure()
        name = "lab-foundation-cpu-" + policy["project"] + "-" + uuid.uuid4().hex
        command = self.as_project(
            policy["podman"],
            "--runtime=crun",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--label",
            "lab.foundation.cpu=" + policy["project"],
            "--network=" + ("slirp4netns" if network else "none"),
            "--read-only",
            "--read-only-tmpfs=false",
            "--image-volume=ignore",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=2g,mode=1777",
            "--tmpfs",
            "/run:rw,nosuid,nodev,size=16m,mode=755",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--userns=keep-id",
            "--user",
            f"{account.pw_uid}:{account.pw_gid}",
            "--group-add=keep-groups",
            "--memory=" + policy["buildMemory"],
            "--memory-swap=" + policy["buildMemory"],
            "--cpus=" + str(policy["cpus"]),
            "--pids-limit=1024",
            "--shm-size=1g",
            "--workdir=/workspace",
            "--mount",
            "type=bind,src=" + policy["workspace"] + ",target=/workspace,rw",
            "--mount",
            "type=bind,src=" + policy["home"] + "/cache,target=/cache,rw",
            "--env",
            "HOME=/cache/home",
            "--env",
            "XDG_CACHE_HOME=/cache",
            "--env",
            "CARGO_TARGET_DIR=/cache/cargo-target",
            "--env",
            "PYTHONPATH=/workspace/src",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint",
            "python",
            image,
            "/workspace/scripts/foundation_run.py",
            *argv,
        )
        try:
            return subprocess.run(command, check=True, timeout=timeout, stdin=subprocess.DEVNULL)
        finally:
            cleanup = subprocess.run(
                self.as_project(policy["podman"], "rm", "--force", "--ignore", name),
                timeout=30,
                check=False,
            )
            if cleanup.returncode:
                raise RuntimeError(f"CPU container cleanup failed: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="oeis-learn")
    commands = parser.add_subparsers(dest="command", required=True)
    image = commands.add_parser("image")
    image.add_argument("operation", choices=["ensure"])
    data = commands.add_parser("data")
    data.add_argument("operation", choices=["fetch"])
    data.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prepare.add_argument("--output", default="tmp-fresh-inputs-host")
    check = commands.add_parser("check")
    check.add_argument("--inputs", required=True)
    for operation in ("train", "smoke"):
        command = commands.add_parser(operation)
        command.add_argument("--inputs", required=True)
        command.add_argument("--run", required=True)
    for operation in ("resume", "status", "cancel", "evaluate", "finalize", "pack"):
        command = commands.add_parser(operation)
        command.add_argument("--run", required=True)
        if operation == "evaluate":
            command.add_argument("--split", choices=["development", "final"], default="development")
        if operation == "pack":
            command.add_argument("--output", required=True)
            command.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        host = Host(load_policy(args.project))
        if args.command == "image":
            print(host.ensure())
        elif args.command == "data":
            host.cpu(
                "data", "fetch", "--config", relative(args.config), network=True, timeout=14400
            )
        elif args.command == "prepare":
            host.cpu(
                "prepare", "--config", relative(args.config), "--output", relative(args.output)
            )
        elif args.command == "pack":
            host.cpu(
                "pack",
                "--run",
                relative(args.run),
                "--output",
                relative(args.output),
                *(["--checkpoint"] if args.checkpoint else []),
                timeout=600,
            )
        elif args.command in {"check", "train", "smoke"}:
            inputs = relative(args.inputs)
            host.cpu("check", "--inputs", inputs, "--inputs-only", timeout=600)
            if args.command == "check":
                host.broker("foundation", "check")
            else:
                run = relative(args.run)
                try:
                    host.broker(
                        "foundation",
                        "run",
                        inputs,
                        run,
                        "smoke" if args.command == "smoke" else "fresh-pilot-1h",
                    )
                finally:
                    if (ROOT / run / "run.json").is_file():
                        host.cpu(
                            "pack",
                            "--run",
                            run,
                            "--output",
                            "tmp-analysis/" + Path(run).name + ".zip",
                            timeout=600,
                        )
        else:
            extra = (
                [args.split, "latest"]
                if args.command == "evaluate"
                else ["best-development"] if args.command == "finalize" else []
            )
            # In particular status/cancel do not build images or acquire the GPU.
            host.broker("foundation", args.command, relative(args.run), *extra)
        return 0
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: host container deadline exceeded: {exc}", file=sys.stderr)
        return 124
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
