"""Supervisor evidence for round029 timing preparation and image acquisition."""

from contextlib import contextmanager
from pathlib import Path
import time

from oeis_learn.generalization.progressive.contracts import load, seal
from .identity import confined
from .preparation_recovery import retain
from .progressive_pilot_capacity import _mount_space
from .supervision import log_tail


def storage(root, engine):
    details = getattr(engine, "details", None) or {}
    graph_root = details.get("store", {}).get("graphRoot") or details.get("DockerRootDir")
    result = {}
    for name, path in (("workspace", root), ("engine_store", graph_root)):
        try:
            if not path:
                raise ValueError("engine did not report a local storage path")
            device, available, inodes = _mount_space(Path(path))
            result[name] = dict(device=device, free_bytes=available, free_inodes=inodes)
        except (ValueError, OSError) as exc:
            result[name] = dict(free_bytes=None, free_inodes=None, reason=str(exc))
    return result


@contextmanager
def preparation_boundary(out, deadline):
    """Child operations keep their own termination evidence, including on errors."""
    try:
        yield
    finally:
        stopped = True
        for name in ("image-build", "package", "custodian", "native"):
            child = out / name
            if child.exists():
                try:
                    cleanup = load(confined(child, "cleanup.json"))
                    stopped &= cleanup["acknowledged"] is True and cleanup["within_bound"] is True
                except (OSError, ValueError, KeyError):
                    stopped = False
        retain(
            out / "cleanup.json",
            dict(acknowledged=stopped, within_bound=time.monotonic_ns() <= deadline),
        )


def failure_details(root, grant, problem):
    """Export bounded originals for this grant and attempt into the private operator file."""
    from oeis_learn.generalization.progressive.velocity_records import SESSION

    if not problem or not problem.get("attempt_path"):
        return None
    area = confined(root, f"run-state/discovery/{SESSION}/qualification/" + grant[7:])
    out = confined(root, problem["attempt_path"])
    if not out.is_relative_to(area / "operations"):
        raise ValueError("failure attempt belongs to another grant")
    detail = dict(attempt_path=problem["attempt_path"], steps={})
    for name, parent in (
        ("operation", out),
        ("image_build", out / "image-build"),
        ("package", out / "package"),
        ("custodian", out / "custodian"),
        ("native_preparation", out / "native"),
    ):
        if not parent.is_dir():
            continue
        step = {}
        for filename in (
            "process.json",
            "cleanup.json",
            "storage-before.json",
            "storage-after.json",
        ):
            if (parent / filename).exists():
                step[filename[:-5]] = load(confined(area, parent / filename))
        for stream in ("stdout", "stderr"):
            tail = log_tail(confined(area, parent), stream)
            if tail:
                step[stream + "_tail"] = tail
        detail["steps"][name] = step
    return detail
