"""Bounded, opt-in whole-role telemetry. No torch import or subprocess sensors."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import stat
import threading
import time


def character_device(path):
    info = Path(path).stat()
    if not stat.S_ISCHR(info.st_mode):
        raise ValueError(f"character device required: {path}")
    return dict(major=os.major(info.st_rdev), minor=os.minor(info.st_rdev))


def device_view(root=Path("/dev")):
    """Attest the actual role-visible accelerator nodes, independent of inspect."""
    paths = [root / "kfd", *sorted((root / "dri").glob("*")), *sorted(root.glob("nvidia*"))]
    found = {}
    for path in paths:
        if path.is_dir() or not path.exists():
            continue
        found["/dev/" + str(path.relative_to(root))] = character_device(path)
    return dict(version="pipeline-device-view/v1", devices=found)


def audit_device_view(reported, view, expected):
    if (
        not isinstance(view, dict)
        or view.get("version") != "pipeline-device-view/v1"
        or view.get("devices") != expected
    ):
        raise ValueError("actual role device nodes differ from expected devices")
    # Rootless Podman can omit Devices entirely. Never accept missing metadata
    # without the independently captured role view and host major/minor match.
    if reported and {d.get("PathOnHost") for d in reported} != set(expected):
        raise ValueError("reported device paths differ from expected devices")
    return "inspect-and-role-character-devices" if reported else "role-character-devices"


def interval_union(intervals):
    total, end = 0.0, float("-inf")
    for left, right in sorted(intervals):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def exclusive_seconds(rows):
    result = {}
    for row in rows:
        children = [
            (max(row["start"], x["start"]), min(row["end"], x["end"]))
            for x in rows
            if x["depth"] == row["depth"] + 1
            and row["start"] <= x["start"]
            and x["end"] <= row["end"]
        ]
        result[row["name"]] = result.get(row["name"], 0) + max(
            0, row["end"] - row["start"] - interval_union(children)
        )
    return result


def read_text(path):
    try:
        return Path(path).read_text()[:65536].strip()
    except (OSError, ValueError):
        return None


def cpu_capacity(affinity, cpu_max, cpuset):
    limits = [float(affinity)]
    if cpu_max and cpu_max.split()[0] != "max":
        quota, period = map(int, cpu_max.split())
        limits.append(quota / period)
    if cpuset:
        count = 0
        for part in cpuset.split(","):
            ends = part.split("-")
            count += int(ends[-1]) - int(ends[0]) + 1
        limits.append(float(count))
    return min(limits)


def gpu_sensor_path(render_node, drm=Path("/sys/class/drm")):
    if not render_node or not Path(render_node).name.startswith("renderD"):
        return None
    candidate = Path(drm) / Path(render_node).name / "device"
    return candidate.resolve() if candidate.is_dir() else None


def cgroup_path():
    root = Path("/sys/fs/cgroup")
    value = read_text("/proc/self/cgroup") or ""
    names = [x[3:] for x in value.splitlines() if x.startswith("0::")]
    if names and ".." not in Path(names[0]).parts:
        candidate = root / names[0].lstrip("/")
        if (candidate / "cpu.stat").is_file():
            return candidate
    return root if (root / "cpu.stat").is_file() else None


def proc_tree():
    """Sample live descendants with start ticks to distinguish PID reuse."""
    found, pending = [], [os.getpid()]
    seen = set()
    ticks = os.sysconf("SC_CLK_TCK")
    while pending and len(seen) < 512:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        base = Path("/proc") / str(pid)
        text = read_text(base / "stat")
        if text is None:
            continue
        fields = text[text.rfind(")") + 2 :].split()
        status = read_text(base / "status") or ""
        metrics = {}
        for line in status.splitlines():
            key, _, val = line.partition(":")
            if key in {
                "VmRSS",
                "VmHWM",
                "Threads",
                "voluntary_ctxt_switches",
                "nonvoluntary_ctxt_switches",
            }:
                metrics[key] = int(val.split()[0]) * (1024 if key.startswith("Vm") else 1)
        found.append(
            dict(
                pid=pid,
                start_ticks=int(fields[19]),
                cpu_seconds=(int(fields[11]) + int(fields[12])) / ticks,
                **metrics,
            )
        )
        # Include children of all threads, not just the Python leader.
        for children in base.glob("task/*/children"):
            pending.extend(int(x) for x in (read_text(children) or "").split())
    return found


class Recorder:
    MAX_EVENTS = 20000
    MAX_SAMPLES = 5402

    def __init__(self, output, *, instrumented, render_node=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.instrumented = instrumented
        self.gpu = gpu_sensor_path(render_node)
        self.cgroup = cgroup_path()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.depth = 0
        self.events = []
        self.samples = 0
        self.missed = 0
        self.collector_cpu = 0.0
        self.started = time.monotonic()
        self.overflow = 0
        self.collector_error = None

    def capacity(self):
        cg = self.cgroup
        quota = read_text(cg / "cpu.max") if cg else None
        cpuset = read_text(cg / "cpuset.cpus.effective") if cg else None
        return dict(
            affinity=sorted(os.sched_getaffinity(0)),
            cpu_max=quota,
            cpuset=cpuset,
            effective_cpus=cpu_capacity(len(os.sched_getaffinity(0)), quota, cpuset),
            cgroup=str(cg) if cg else None,
            gpu_sensor=str(self.gpu) if self.gpu else None,
            missing_gpu_reason=None
            if self.gpu
            else "selected render-node sysfs mapping unavailable",
            cpu_topology=read_text("/proc/cpuinfo"),
            kernel=list(os.uname()),
            amdgpu_module_version=read_text("/sys/module/amdgpu/version"),
            gpu_units={
                "gpu_busy_percent": "percent",
                "mem_busy_percent": "percent",
                "mem_info_vram_used": "bytes",
                "power1_average": "microwatts; may include APU CPU",
            },
            ancestor_quota_reason="container-visible limits only; host ancestor restrictions may be hidden",
        )

    def snapshot(self):
        started = time.monotonic()
        cg = self.cgroup
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        gpu = {}
        if self.gpu:
            for name in (
                "gpu_busy_percent",
                "mem_busy_percent",
                "mem_info_vram_used",
                "pp_dpm_sclk",
                "pp_dpm_mclk",
            ):
                gpu[name] = read_text(self.gpu / name)
            for p in self.gpu.glob("hwmon/hwmon*/*"):
                if p.name in {"temp1_input", "power1_average", "power1_input"}:
                    gpu[str(p.relative_to(self.gpu))] = read_text(p)
        return dict(
            time=started,
            capture_seconds=time.monotonic() - started,
            sensor_age_seconds=None,
            sensor_age_reason="sysfs firmware cadence not exposed; timestamp is host read time",
            role_cpu_seconds=usage.ru_utime + usage.ru_stime,
            reaped_children_cpu_seconds=children.ru_utime + children.ru_stime,
            role_peak_rss_bytes=usage.ru_maxrss * 1024,
            processes=proc_tree(),
            gpu=gpu or None,
            gpu_reason=None if gpu else "no readable mapped counters",
            host_cpu=read_text("/proc/stat"),
            host_memory=read_text("/proc/meminfo"),
            host_load=read_text("/proc/loadavg"),
            cgroup={
                name: read_text(cg / name) if cg else None
                for name in (
                    "cpu.stat",
                    "memory.stat",
                    "memory.current",
                    "memory.peak",
                    "memory.events",
                    "cpu.pressure",
                    "memory.pressure",
                    "io.stat",
                    "io.pressure",
                )
            },
        )

    def _collect(self):
        due = time.monotonic()
        with (self.output / "samples.jsonl").open("a") as handle:
            while not self.stop.is_set() and self.samples < self.MAX_SAMPLES:
                begin = time.thread_time()
                try:
                    row = self.snapshot()
                    row["sample_lateness_seconds"] = max(0, time.monotonic() - due)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    self.samples += 1
                except (OSError, ValueError, IndexError) as exc:
                    try:
                        handle.write(json.dumps(dict(time=time.monotonic(), error=str(exc))) + "\n")
                    except OSError:
                        self.collector_error = f"{type(exc).__name__}: {exc}"
                        self.ready.set()
                        return
                    self.missed += 1
                self.ready.set()
                self.collector_cpu += time.thread_time() - begin
                due += 1
                now = time.monotonic()
                if now > due:
                    skipped = int(now - due) + 1
                    self.missed += skipped
                    due += skipped
                self.stop.wait(max(0, due - time.monotonic()))

    def __enter__(self):
        self.output.joinpath("capability.json").write_text(json.dumps(self.capacity(), indent=2))
        if self.instrumented:
            self.thread = threading.Thread(target=self._collect, daemon=True)
            self.thread.start()
            self.ready.wait(timeout=2)
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("telemetry collector failed bounded shutdown")
        self.output.joinpath("telemetry.json").write_text(
            json.dumps(
                dict(
                    samples=self.samples,
                    missed=self.missed,
                    collector_cpu_seconds=self.collector_cpu,
                    collector_native_tid=None if self.thread is None else self.thread.native_id,
                    seconds=time.monotonic() - self.started,
                    event_overflow=self.overflow,
                    collector_error=self.collector_error,
                )
            )
        )

    @contextmanager
    def phase(self, name, **metadata):
        depth = self.depth
        self.depth += 1
        start = self.snapshot() if depth == 0 else dict(time=time.monotonic())
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.depth -= 1
            finish = self.snapshot() if depth == 0 else dict(time=time.monotonic())
            row = dict(
                name=name,
                start=start["time"],
                end=finish["time"],
                depth=depth,
                error=error,
                metadata=metadata,
            )
            if depth == 0:
                row["boundaries"] = dict(start=start, end=finish)
            if len(self.events) < self.MAX_EVENTS:
                self.events.append(row)
                with (self.output / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            else:
                self.overflow += 1
