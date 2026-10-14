#!/usr/bin/env python3
"""Run fresh foundation diagnostics on the lab host, without a VM or SSH."""

import argparse
import json
import os
import pwd
import re
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("/etc/lab/gpu-runner.json")


def pack_run(run, output, *, checkpoint=False):
    sys.path.insert(0, str(ROOT / "src"))
    from oeis_learn.fresh.bundle import pack

    result = pack(ROOT / relative(run), ROOT / relative(output), checkpoint=checkpoint)
    print(json.dumps(result, indent=2))
    return result


def relative(value):
    path = (ROOT / value).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        raise ValueError("paths must remain inside the registered host checkout") from None


def extended_settings(benchmark, output):
    root = ROOT / relative(benchmark)
    run = json.loads((root / "run.json").read_bytes())
    status = json.loads((root / "status.json").read_bytes())
    if run["settings"]["profile"] != "fresh-benchmark" or status["status"] != "complete":
        raise ValueError("settings require a successfully completed hardware benchmark")
    measured = json.loads((root / "roles/benchmark/result.json").read_bytes())["recommended"]
    if (
        measured["batch_size"] not in {4, 8, 16, 32, 64}
        or type(measured["length_buckets"]) is not bool
    ):
        raise ValueError("invalid measured batch recommendation")
    cfg = json.loads((root / "input-manifest.json").read_bytes())["config"]
    cfg["run"].pop("model", None)
    cfg["archive"].update(
        training_prompts=2048, development_structures=128, seconds=7200, sampling_shards=64
    )
    cfg["run"].update(
        profile="fresh-extended-12h",
        training_seconds=36000,
        batch_size=measured["batch_size"],
        length_buckets=measured["length_buckets"],
    )
    cfg["run"]["evaluation"].update(tokens=1023, per_group_seconds=60)
    target = ROOT / relative(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("settings output already exists; keep immutable experiment evidence")
    with target.open("x") as stream:
        json.dump(cfg, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"output": str(target), "recommended": measured, "qualified": False}, indent=2)
    )


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

    def authenticate(self):
        # Image builds and input verification can outlast the terminal's sudo ticket.
        # Renew immediately before a transition, while stdin still belongs to the user.
        if sys.stdin.isatty():
            subprocess.run(["sudo", "-v"], check=True)

    def broker(self, operation, *argv, capture=False):
        # Root imports only the installed Nix broker, never this checkout.
        request = {"version": 1, "op": operation, "cwd": ".", "argv": list(argv)}
        self.authenticate()
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
        if network != (list(argv[:2]) == ["data", "fetch"]):
            raise ValueError("only bulk fetch may receive networking")
        command = self.as_project(
            self.policy["python"],
            "-I",
            str(ROOT / "scripts/foundation_local.py"),
            "_cpu-worker",
            self.policy["project"],
            self.ensure(),
            str(timeout),
            json.dumps(argv),
        )
        self.authenticate()
        # The running project-account worker
        # owns its Podman cleanup even after the administrator's ticket expires.
        return subprocess.run(command, check=True, timeout=timeout + 45, stdin=subprocess.DEVNULL)

    def cpu_worker(self, image, argv, timeout):
        if os.getuid() != self.account.pw_uid:
            raise ValueError("CPU worker must run as the registered rootless project account")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("immutable CPU image required")
        if not argv or argv[0] not in {"data", "prepare", "check", "seal-inputs"}:
            raise ValueError("invalid CPU operation")
        if argv[0] == "data" and argv[1:2] != ["fetch"]:
            raise ValueError("only bulk fetch is a CPU data operation")
        if argv[0] == "check" and "--inputs-only" not in argv:
            raise ValueError("CPU check cannot launch nested GPU requests")
        if argv[0] == "seal-inputs" and "--accept-shortfall" not in argv:
            raise ValueError("partial sealing requires explicit acceptance")
        if not 1 <= timeout <= 14400:
            raise ValueError("CPU deadline must be 1..14400 seconds")
        policy, account = self.policy, self.account
        name = "lab-foundation-cpu-" + policy["project"] + "-" + uuid.uuid4().hex
        command = [
            policy["podman"],
            "--runtime=crun",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--label",
            "lab.foundation.cpu=" + policy["project"],
            "--network=" + ("slirp4netns" if argv[:2] == ["data", "fetch"] else "none"),
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
        ]
        failure = None
        try:
            return subprocess.run(command, check=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                subprocess.run(
                    [policy["podman"], "rm", "--force", "--ignore", name],
                    timeout=30,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                message = f"CPU container cleanup failed: {name}: {exc}"
                if failure is None:
                    raise RuntimeError(message) from exc
                print(message, file=sys.stderr)


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
    seal = commands.add_parser("seal-inputs")
    seal.add_argument("--inputs", required=True)
    seal.add_argument("--accept-shortfall", action="store_true", required=True)
    check = commands.add_parser("check")
    check.add_argument("--inputs", required=True)
    plan = commands.add_parser("extended-settings")
    plan.add_argument("--benchmark", required=True)
    plan.add_argument("--output", default="tmp-analysis/fresh-extended-settings.json")
    for operation in ("train", "smoke", "benchmark"):
        command = commands.add_parser(operation)
        command.add_argument("--inputs", required=True)
        command.add_argument("--run", required=True)
        if operation == "train":
            command.add_argument(
                "--profile",
                default="fresh-pilot-1h",
                choices=["fresh-pilot-1h", "fresh-extended-12h"],
            )
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
        if args.command == "pack":
            pack_run(args.run, args.output, checkpoint=args.checkpoint)
            return 0
        if args.command == "extended-settings":
            extended_settings(args.benchmark, args.output)
            return 0
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
        elif args.command == "seal-inputs":
            host.cpu("seal-inputs", "--inputs", relative(args.inputs), "--accept-shortfall")
        elif args.command in {"check", "train", "smoke", "benchmark"}:
            inputs = relative(args.inputs)
            host.cpu("check", "--inputs", inputs, "--inputs-only", timeout=600)
            if args.command == "check":
                host.broker("foundation", "check")
            else:
                run = relative(args.run)
                failure = None
                try:
                    host.broker(
                        "foundation",
                        "run",
                        inputs,
                        run,
                        {"smoke": "smoke", "benchmark": "fresh-benchmark"}.get(
                            args.command, getattr(args, "profile", "fresh-pilot-1h")
                        ),
                    )
                except BaseException as exc:
                    failure = exc
                    raise
                finally:
                    if (ROOT / run / "run.json").is_file():
                        try:
                            pack_run(run, "tmp-analysis/" + Path(run).name + ".zip")
                        except (ValueError, OSError, RuntimeError) as exc:
                            if failure is None:
                                raise
                            print(f"Analysis packaging also failed: {exc}", file=sys.stderr)
        else:
            extra = (
                [args.split, "latest"]
                if args.command == "evaluate"
                else ["best-development"] if args.command == "finalize" else []
            )
            # In particular status/cancel do not build images or acquire the GPU.
            host.broker("foundation", args.command, relative(args.run), *extra)
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted; use status/cancel to inspect any still-active GPU run.", file=sys.stderr
        )
        return 130
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: host container deadline exceeded: {exc}", file=sys.stderr)
        return 124
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    if sys.argv[1:2] == ["_cpu-worker"]:
        try:
            project, image, seconds, payload = sys.argv[2:]
            if len(payload.encode()) > 65536:
                raise ValueError("CPU argument payload exceeds64KiB")
            arguments = json.loads(payload)
            if (
                not isinstance(arguments, list)
                or len(arguments) > 32
                or any(not isinstance(a, str) or "\x00" in a for a in arguments)
            ):
                raise ValueError("invalid bounded CPU argv")
            Host(load_policy(project)).cpu_worker(image, arguments, int(seconds))
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        except (
            ValueError,
            OSError,
            RuntimeError,
            KeyError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            print(f"ERROR: CPU worker: {exc}", file=sys.stderr)
            raise SystemExit(
                124
                if isinstance(exc, subprocess.TimeoutExpired)
                else exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
            ) from None
    else:
        raise SystemExit(main())
