"""Standard-library Docker/Podman adapter for host-side orchestration.

All engine calls use the caller's normal engine connection and permissions.
Workloads never receive the engine executable, connection credentials or socket.
"""

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE_ID = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def private_runtime_environment(output):
    """Put compiler scratch/cache in this role's existing writable output mount.

    ROCm/LLVM can exceed the small /tmp tmpfs during the first backward pass.
    Use disk-backed, role-private directories without exposing shared caches or
    any additional host paths. These files are disposable, not training evidence.
    """
    output = Path(output).resolve(strict=True)
    root = output / ".runtime"
    for path in (root, root / "tmp", root / "home", root / "cache"):
        if path.is_symlink():
            raise ValueError(f"runtime directory cannot be a symlink: {path}")
        path.mkdir(mode=0o700, exist_ok=True)
    base = "/work/output/.runtime"
    return {
        "TMPDIR": base + "/tmp",
        "TMP": base + "/tmp",
        "TEMP": base + "/tmp",
        "HOME": base + "/home",
        "XDG_CACHE_HOME": base + "/cache",
    }


def normalized_image(value):
    value = value.strip()
    if not IMAGE_ID.fullmatch(value):
        raise ValueError("an inspected immutable image ID is required")
    return value if value.startswith("sha256:") else "sha256:" + value


class Engine:
    @classmethod
    def from_config(cls, config):
        name = config.get("engine", "docker")
        return cls(
            name,
            executable=config.get("engine_executable", name),
            details=config.get("engine_info", {}),
        )

    def __init__(self, name="auto", *, executable=None, details=None):
        if name not in {"auto", "docker", "podman"}:
            raise ValueError("engine must be auto, docker or podman")
        if name == "auto":
            failures = []
            for candidate in ("docker", "podman"):
                if shutil.which(candidate):
                    try:
                        selected = Engine(candidate)
                        selected.info()
                        self.__dict__.update(selected.__dict__)
                        return
                    except (OSError, RuntimeError) as exc:
                        failures.append(str(exc))
            raise RuntimeError("No usable Docker/Podman connection. " + "; ".join(failures))
        self.name = name
        self.executable = executable or shutil.which(name)
        if not self.executable:
            raise RuntimeError(f"{name} CLI is unavailable; select an installed container engine")
        self.details = details

    def call(self, args, **kwargs):
        return subprocess.run([self.executable, *map(str, args)], check=True, **kwargs)

    def info(self):
        if self.details is None:
            try:
                result = self.call(
                    ["info", "--format", "{{json .}}"], capture_output=True, timeout=30
                )
                self.details = json.loads(result.stdout)
            except (subprocess.SubprocessError, ValueError) as exc:
                raise RuntimeError(
                    f"Cannot use {self.name} with this account/connection: {exc}"
                ) from exc
        return self.details

    def facts(self):
        info = self.info()
        if self.name=='docker':
            if isinstance(info,dict) and {'host','store','version'}<=set(info):
                raise RuntimeError('docker returned Podman metadata; a compatibility command cannot provide independent Docker portability evidence')
            memory=info.get('MemTotal') if isinstance(info,dict) else None
            if type(memory) is not int or memory<=0:
                raise RuntimeError('Docker did not report a usable MemTotal; independent Docker resource/portability qualification is unavailable')
        if self.name == "podman":
            host = info.get("host", {})
            security = host.get("security", {})
            return {
                "engine": self.name,
                "memory_bytes": host.get("memTotal", 0),
                "cpus": host.get("cpus", 1),
                "rootless": security.get("rootless", False),
                "selinux": security.get("selinuxEnabled", False),
                "runtime": host.get("ociRuntime", {}).get("name"),
                "version": info.get("version", {}).get("Version", "unknown"),
            }
        return {
            "engine": self.name,
            "memory_bytes": info.get("MemTotal", 0),
            "cpus": info.get("NCPU", 1),
            "rootless": any("rootless" in str(v) for v in info.get("SecurityOptions", [])),
            "selinux": any("selinux" in str(v) for v in info.get("SecurityOptions", [])),
            "runtime": info.get("DefaultRuntime"),
            "version": info.get("ServerVersion", "unknown"),
        }

    def image(self, reference):
        result = self.call(
            ["image", "inspect", reference, "--format", "{{.Id}}"], capture_output=True, timeout=30
        )
        return normalized_image(result.stdout.decode())

    def build(self, root, *, target="gpu", tag="oeis-learn:gpu", base_image=None):
        args = ["build", "--target", target, "--tag", tag]
        if base_image:
            args += ["--build-arg", "GPU_BASE_IMAGE=" + base_image]
        self.call([*args, str(Path(root).resolve())], timeout=3600)
        return self.image(tag)

    def user_args(self):
        if self.name == "docker":
            # User mapping needs security options, not measured memory. Keep
            # resource qualification strict in facts(), independently of argv
            # construction for explicitly bounded roles.
            info = self.info()
            facts = {
                "rootless": any("rootless" in str(v) for v in info.get("SecurityOptions", [])),
                "selinux": any("selinux" in str(v) for v in info.get("SecurityOptions", [])),
            }
        else:
            facts = self.facts()
        args = ["--user", f"{os.getuid()}:{os.getgid()}"]
        if self.name == "docker" and facts["rootless"]:
            args = ["--user", "0:0"]  # Namespace root maps to the unprivileged engine owner.
        if self.name == "podman" and facts["rootless"]:
            args += ["--userns=keep-id"]
            if facts["runtime"] == "crun":
                args += ["--group-add=keep-groups"]
        if facts["selinux"]:
            # Explicit mounts remain the access boundary; do not relabel a user's checkout.
            args += ["--security-opt=label=disable"]
        return args

    def command(
        self,
        *,
        image,
        name,
        mounts,
        argv,
        gpu=False,
        render_node=None,
        memory_bytes=4 << 30,
        cpus=2,
        pids=512,
        network=False,
        workdir="/work",
        environment=None,
        labels=None,
        tmp_bytes=64 << 20,
        auto_remove=True,
        startup_diagnostics=False,
    ):
        image = normalized_image(image)
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", name):
            raise ValueError("invalid container name")
        if type(memory_bytes) is not int or memory_bytes <= 0 or not 0 < cpus <= 256:
            raise ValueError("positive bounded memory/CPU limits required")
        if not argv or type(gpu) is not bool or type(network) is not bool:
            raise ValueError("explicit command and device/network capabilities required")
        args = [
            self.executable,
            *(["--log-level=debug"] if startup_diagnostics and self.name == "podman" else []),
            "run",
            "--name",
            name,
            "--pull=never",
            "--network=" + ("bridge" if network else "none"),
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=" + str(memory_bytes),
            "--memory-swap=" + str(memory_bytes),
            "--cpus=" + str(cpus),
            "--pids-limit=" + str(pids),
            "--ipc=private",
            "--shm-size=1g",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={tmp_bytes},mode=1777",
            "--tmpfs",
            "/run:rw,nosuid,nodev,size=16m,mode=755",
            "--workdir",
            workdir,
            *self.user_args(),
        ]
        if auto_remove:
            args += ["--rm"]
        for source, destination, writable in mounts:
            source = Path(source).resolve(strict=True)
            if not source.is_dir() or any(c in str(source) + destination for c in (",", "\n")):
                raise ValueError("regular directory mount with representable paths required")
            if not destination.startswith("/") or type(writable) is not bool:
                raise ValueError("explicit absolute mount target and write capability required")
            args += [
                "--mount",
                f"type=bind,src={source},dst={destination}" + ("" if writable else ",readonly"),
            ]
        if gpu:
            args += self.device_args(render_node)
        elif render_node:
            raise ValueError("GPU device is forbidden for a CPU container")
        for key, value in (environment or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("invalid environment name")
            args += ["--env", f"{key}={value}"]
        for key, value in (labels or {}).items():
            args += ["--label", f"{key}={value}"]
        return [*args, "--entrypoint", argv[0], image, *argv[1:]]

    def device_args(self, render_node=None):
        args = []
        node = Path(render_node) if render_node else self.render_node()
        if not re.fullmatch(r"/dev/dri/renderD[0-9]+", str(node)):
            raise ValueError("specific ROCm render device required")
        for path in (Path("/dev/kfd"), node):
            if not path.is_char_device():
                raise RuntimeError(f"ROCm device unavailable on this host: {path}")
            args += ["--device", str(path)]
            if not (self.name == "podman" and self.facts()["rootless"]):
                args += ["--group-add", str(path.stat().st_gid)]
        return args

    @staticmethod
    def render_node():
        nodes = sorted(p for p in Path("/dev/dri").glob("renderD*") if p.is_char_device())
        if len(nodes) != 1:
            raise RuntimeError(
                "Select --render-node explicitly when zero or multiple render devices are visible"
            )
        return nodes[0]

    def remove(self, name, *, timeout=30):
        args = ["rm", "--force"]
        if self.name == "podman":
            args += ["--time=0", "--ignore"]
        try:
            self.call([*args, name], capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or b"").decode(errors="replace").lower()
            if not any(v in error for v in ("no such container", "does not exist")):
                raise RuntimeError(f"Container reclamation failed for {name}: {error}") from exc

    def cleanup(self, label):
        result = self.call(
            ["ps", "--all", "--filter", "label=oeis.run=" + label, "--format", "{{.Names}}"],
            capture_output=True,
            timeout=30,
        )
        names = result.stdout.decode().splitlines()
        if names:
            failures = []
            with ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
                for future in [pool.submit(self.remove, name) for name in names]:
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(str(exc))
            if failures:
                raise RuntimeError("run container cleanup failed: " + "; ".join(failures))


def resolve_engine(engine=None):
    if engine is None or isinstance(engine, str):
        return Engine(engine or os.environ.get("OEIS_CONTAINER_ENGINE", "auto"))
    return engine
