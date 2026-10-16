#!/usr/bin/env python3
"""Portable foundation host wrapper; foundation_run.py is the public entrypoint."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oeis_learn.container_runtime import Engine  # noqa: E402
from oeis_learn.fresh.common import digest, file_hash, read  # noqa: E402

CAPS = {
    "smoke": 1800,
    "fresh-pilot-1h": 3600,
    "fresh-benchmark": 600,
    "fresh-extended-12h": 43200,
    "bootstrap-learnability": 3600,
}


def relative(value):
    path = (ROOT / value).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        raise ValueError("artifact paths must remain inside the workspace") from None


def pack_run(run, output, *, checkpoint=False):
    from oeis_learn.fresh.bundle import pack

    result = pack(ROOT / relative(run), ROOT / relative(output), checkpoint=checkpoint)
    print(json.dumps(result, indent=2))
    return result


def state_path(run):
    return ROOT / "run-state" / digest(relative(run))[7:]


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


class Host:
    def __init__(self, args):
        self.args = args
        self.engine = Engine(args.engine)
        facts = self.engine.facts()
        available = facts["memory_bytes"] or 8 << 30
        memory = (
            int(args.memory_gib * (1 << 30))
            if args.memory_gib is not None
            else min(16 << 30, available // 2)
        )
        memory = memory // (1 << 20) * (1 << 20)
        cpus = args.cpus if args.cpus is not None else min(4, facts["cpus"])
        if memory < 1 << 30 or memory > available or not 0 < cpus <= facts["cpus"]:
            raise ValueError("requested memory/CPU limits exceed the engine's available resources")
        details = (
            {
                "host": {
                    "security": {"rootless": facts["rootless"], "selinuxEnabled": facts["selinux"]},
                    "ociRuntime": {"name": facts["runtime"]},
                }
            }
            if self.engine.name == "podman"
            else {
                "SecurityOptions": (["rootless"] if facts["rootless"] else [])
                + (["selinux"] if facts["selinux"] else [])
            }
        )
        self.policy = {
            "workspace": str(ROOT),
            "state_root": str(ROOT / "run-state"),
            "engine": self.engine.name,
            "engine_executable": self.engine.executable,
            "engine_info": details,
            "render_node": args.render_node,
            "memory_bytes": memory,
            "cpu_memory_bytes": min(memory, 12 << 30),
            "cpus": cpus,
            "workers": max(1, min(4, int((available - memory) / (2 << 30)))),
            "free_disk_floor_bytes": args.free_disk_gib << 30,
            "host_free_memory_bytes": int(args.host_free_memory_gib * (1 << 30)),
            "runtime": {
                "engine": self.engine.name,
                "engine_version": facts["version"],
                "memory_bytes": memory,
                "cpus": cpus,
                "render_node": args.render_node,
                "coordinator_sha256": file_hash(ROOT / "scripts/foundation_host.py"),
                "adapter_sha256": file_hash(ROOT / "src/oeis_learn/container_runtime.py"),
            },
        }
        self.policy["runtime"].update(
            {
                key: self.policy[key]
                for key in (
                    "workers",
                    "cpu_memory_bytes",
                    "free_disk_floor_bytes",
                    "host_free_memory_bytes",
                )
            }
        )
        self.image = None

    def ensure(self):
        if self.image is None:
            if self.args.image:
                self.image = self.engine.image(self.args.image)
            else:
                self.image = self.engine.build(ROOT, base_image=self.args.gpu_base_image)
        return self.image

    def cpu(self, *argv, timeout=10800, network=False):
        if network != (list(argv[:2]) == ["data", "fetch"]):
            raise ValueError("only bulk acquisition may receive network access")
        cache = ROOT / "container-cache"
        cache.mkdir(exist_ok=True)
        name = "oeis-cpu-" + uuid.uuid4().hex
        command = self.engine.command(
            image=self.ensure(),
            name=name,
            mounts=[(ROOT, "/workspace", True), (cache, "/cache", True)],
            auto_remove=False,
            argv=["python", "/opt/oeis-build/scripts/foundation_tasks.py", *argv],
            memory_bytes=self.policy["cpu_memory_bytes"],
            cpus=self.policy["cpus"],
            network=network,
            workdir="/workspace",
            tmp_bytes=2 << 30,
            environment={
                "HOME": "/cache/home",
                "XDG_CACHE_HOME": "/cache",
                "OEIS_WORKSPACE": "/workspace",
                "PYTHONPATH": "/opt/oeis-build/src",
            },
        )
        try:
            return subprocess.run(command, check=True, timeout=timeout, stdin=subprocess.DEVNULL)
        finally:
            self.engine.remove(name)

    @staticmethod
    def control(action, run):
        job = state_path(run)
        if action == "status":
            print(json.dumps(read(job / "status.json"), indent=2))
            return
        from foundation_host import process_identity

        active = read(job / "active.json")
        if process_identity(active["pid"]) != active:
            raise ValueError("controller is not active; use resume to reclaim retained containers")
        os.kill(active["pid"], signal.SIGTERM)

    @contextmanager
    def gpu_lock(self):
        state = ROOT / "run-state"
        state.mkdir(exist_ok=True)
        with (state / "gpu.lock").open("a+") as held:
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "another foundation GPU operation is active in this workspace"
                ) from None
            yield

    def select_gpu(self):
        node = self.policy["render_node"] or str(self.engine.render_node())
        self.policy["render_node"] = node
        self.policy["runtime"]["render_node"] = node

    def pipeline(self, action, *argv):
        self.select_gpu()
        run = argv[1] if action == "run" else argv[0] if argv else None
        image = (
            self.ensure()
            if action in {"run", "check", "diagnose"}
            else self.engine.image(read(state_path(run) / "run.json")["image"])
        )
        policy = dict(self.policy)
        token = digest([str(ROOT), argv[1] if action == "diagnose" else run or "probe"])[7:39]
        policy["run_token"] = token
        cap = (
            CAPS[argv[2]]
            if action == "run"
            else 300
            if action == "check"
            else 600
            if action == "diagnose"
            else 43200
        )
        with self.gpu_lock():
            self.engine.cleanup(token)
            command = [
                sys.executable,
                str(ROOT / "scripts/foundation_host.py"),
                json.dumps(policy),
                image,
                action,
                *argv,
            ]
            process = subprocess.Popen(command, start_new_session=True)
            try:
                code = process.wait(timeout=cap + 45)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                self.engine.cleanup(token)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engine",
        choices=["auto", "docker", "podman"],
        default=os.environ.get("OEIS_CONTAINER_ENGINE", "auto"),
    )
    p.add_argument("--image", help="use an existing image instead of building the gpu target")
    p.add_argument("--gpu-base-image", help="override the ROCm base image for a new build")
    p.add_argument(
        "--render-node", help="specific /dev/dri/renderD* device; required with multiple GPUs"
    )
    p.add_argument("--memory-gib", type=float)
    p.add_argument("--cpus", type=int)
    p.add_argument("--free-disk-gib", type=int, default=5)
    p.add_argument("--host-free-memory-gib", type=float, default=1)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("image").add_argument("operation", choices=["ensure"])
    data = commands.add_parser("data")
    data.add_argument("operation", choices=["fetch"])
    data.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", default="configs/foundation/fresh_pilot.yaml")
    prep.add_argument("--output", default="tmp-fresh-inputs-host")
    seal = commands.add_parser("seal-inputs")
    seal.add_argument("--inputs", required=True)
    seal.add_argument("--accept-shortfall", action="store_true", required=True)
    commands.add_parser("check").add_argument("--inputs", required=True)
    plan = commands.add_parser("extended-settings")
    plan.add_argument("--benchmark", required=True)
    plan.add_argument("--output", default="tmp-analysis/fresh-extended-settings.json")
    for op in ("train", "smoke", "benchmark", "learnability"):
        cmd = commands.add_parser(op)
        cmd.add_argument("--inputs", required=True)
        cmd.add_argument("--run", required=True)
        cmd.add_argument(
            "--encoder",
            choices=["tri_stream", "hybrid", "integer_bytes"],
            help="explicit v2 comparison on the same admitted input bytes",
        )
        if op == "train":
            cmd.add_argument(
                "--profile",
                choices=["fresh-pilot-1h", "fresh-extended-12h"],
                default="fresh-pilot-1h",
            )
    for op in ("resume", "status", "cancel", "evaluate", "finalize", "pack"):
        cmd = commands.add_parser(op)
        cmd.add_argument("--run", required=True)
        if op == "evaluate":
            cmd.add_argument("--split", choices=["development", "final"], default="development")
        if op == "pack":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--checkpoint", action="store_true")
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("--run", required=True)
    diagnostic.add_argument("--output", required=True)
    validation = commands.add_parser("validate")
    validation.add_argument("--suite", choices=["cpu", "gpu"], default="gpu")
    validation.add_argument("--output", default="tmp-validation")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "pack":
            pack_run(args.run, args.output, checkpoint=args.checkpoint)
            return 0
        if args.command == "extended-settings":
            extended_settings(args.benchmark, args.output)
            return 0
        if args.free_disk_gib < 1 or not args.host_free_memory_gib > 0:
            raise ValueError("disk and host memory reserves must be positive")
        if args.command in {"status", "cancel"}:
            Host.control(args.command, relative(args.run))
            return 0
        host = Host(args)
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
        elif args.command == "diagnose":
            host.pipeline("diagnose", relative(args.run), relative(args.output))
        elif args.command == "validate":
            from oeis_learn.container_validation import validate

            validate(host, args.suite, ROOT / relative(args.output))
        elif args.command in {"check", "train", "smoke", "benchmark", "learnability"}:
            inputs = relative(args.inputs)
            host.cpu("check", "--inputs", inputs, "--inputs-only", timeout=1800)
            if args.command == "check":
                host.pipeline("check")
            else:
                run = relative(args.run)
                try:
                    host.pipeline(
                        "run",
                        inputs,
                        run,
                        {
                            "smoke": "smoke",
                            "benchmark": "fresh-benchmark",
                            "learnability": "bootstrap-learnability",
                        }.get(args.command, getattr(args, "profile", "fresh-pilot-1h")),
                    )
                finally:
                    if (ROOT / run / "run.json").is_file():
                        pack_run(run, "tmp-analysis/" + Path(run).name + ".zip")
        else:
            extra = (
                [args.split, "latest"]
                if args.command == "evaluate"
                else ["best-development"]
                if args.command == "finalize"
                else []
            )
            try:
                host.pipeline(args.command, relative(args.run), *extra)
            finally:
                if (ROOT / args.run / "run.json").is_file():
                    pack_run(args.run, "tmp-analysis/" + Path(args.run).name + ".zip")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; retained run state can be inspected or resumed.", file=sys.stderr)
        return 130
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: container deadline exhausted: {exc}", file=sys.stderr)
        return 124
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


def entrypoint():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
