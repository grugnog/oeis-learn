"""Typed measurements and bounded, append-only controller telemetry.

One controller owns a log. Durable records are never evicted to make room.
An incomplete final line is quarantined before continuing after a crash.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

from oeis_learn.experiments.artifacts import canonical_bytes, load_json


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_json(path, value):
    """Durable mutable pointer/snapshot; immutable evidence uses atomic_write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def metric(value, unit, source, *, state="measured", reason=None):
    if state not in ("measured", "disabled", "unavailable") or not unit or not source:
        raise ValueError("invalid metric metadata")
    if state == "measured":
        if type(value) not in (int, float) or not math.isfinite(value) or reason is not None:
            raise ValueError("measured metric requires a finite value")
    elif value is not None or not reason:
        raise ValueError("nonmeasured metric requires null value and reason")
    return dict(state=state, value=value, unit=unit, source=source, reason=reason)


class EventLog:
    def __init__(self, root, *, segment_bytes=64 << 20, quota_bytes=2 << 30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not 0 < segment_bytes <= 64 << 20 or not segment_bytes <= quota_bytes <= 2 << 30:
            raise ValueError("invalid log caps")
        self.segment_bytes, self.quota_bytes = segment_bytes, quota_bytes
        self.events = {}
        paths = sorted(self.root.glob("*.jsonl"))
        for index, path in enumerate(paths):
            if path.name != f"{index:08d}.jsonl" or path.stat().st_size > segment_bytes:
                raise ValueError("invalid log segment")
            with path.open("rb") as stream:
                offset = 0
                for line in stream:
                    if not line.endswith(b"\n"):
                        if path != paths[-1]:
                            raise ValueError("incomplete historical log segment")
                        # Retain failed-write evidence before removing only the torn tail.
                        with (self.root / (path.name + ".partial")).open("xb") as out:
                            out.write(line)
                            out.flush()
                            os.fsync(out.fileno())
                        with path.open("r+b") as out:
                            out.truncate(offset)
                            out.flush()
                            os.fsync(out.fileno())
                        fsync_directory(self.root)
                        break
                    record = load_json(line)
                    if set(record) != {"sequence", "event_id", "data"} or record["sequence"] != len(
                        self.events
                    ):
                        raise ValueError("invalid event sequence")
                    if record["event_id"] in self.events:
                        raise ValueError("duplicate durable event")
                    self.events[record["event_id"]] = record["data"]
                    offset += len(line)
        self.index = max(0, len(paths) - 1)

    def append(self, event_id, data):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event ID required")
        if event_id in self.events:
            if self.events[event_id] != data:
                raise ValueError("conflicting duplicate event")
            return False
        raw = canonical_bytes(dict(sequence=len(self.events), event_id=event_id, data=data)) + b"\n"
        if len(raw) > self.segment_bytes:
            raise OSError("event exceeds segment quota")
        if (
            sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()) + len(raw)
            > self.quota_bytes
        ):
            raise OSError("required evidence log quota exhausted")
        path = self.root / f"{self.index:08d}.jsonl"
        if path.exists() and path.stat().st_size + len(raw) > self.segment_bytes:
            self.index += 1
            path = self.root / f"{self.index:08d}.jsonl"
        with path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(self.root)
        self.events[event_id] = data
        return True


def resource_metrics(device):
    """RSS, host availability and allocator bytes are overlapping measurements."""
    import torch

    def unavailable(source):
        return metric(None, "bytes", source, state="unavailable", reason="not exposed")

    result = {"rss": unavailable("proc/self/statm"), "host_available": unavailable("proc/meminfo")}
    try:
        result["rss"] = metric(
            int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
            "bytes",
            "proc/self/statm",
        )
        rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["host_available"] = metric(
            int(rows["MemAvailable"].split()[0]) * 1024, "bytes", "proc/meminfo"
        )
    except (OSError, ValueError, KeyError, IndexError):
        pass
    for name, fn in (
        ("gpu_allocated", torch.cuda.memory_allocated),
        ("gpu_reserved", torch.cuda.memory_reserved),
        ("gpu_peak_allocated", torch.cuda.max_memory_allocated),
    ):
        result[name] = (
            metric(fn(device), "bytes", "torch allocator")
            if torch.device(device).type == "cuda"
            else metric(
                None, "bytes", "torch allocator", state="unavailable", reason="CPU diagnostic"
            )
        )
    return result
