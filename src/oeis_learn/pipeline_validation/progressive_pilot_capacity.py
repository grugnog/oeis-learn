"""Incremental peak storage for the privately retained pilot roles."""

from collections import defaultdict
import re
from pathlib import Path

from oeis_learn.generalization.contracts import artifact_refs
from oeis_learn.generalization.progressive.contracts import load, read_bound_record, seal
from .identity import confined


def paid_allocations(area):
    """Use sealed manifests and current stat metadata. Actual reuse reopens content."""
    allocations = defaultdict(lambda: [0, 0])
    seen = set()
    finished = set()
    for path in sorted(area.glob("roles/*/boundary.json")):
        role = confined(area, path).parent
        boundary = load(path)
        if boundary != seal(boundary):
            raise ValueError("changed pilot role boundary")
        request = load(confined(role, "request.json"))
        if request != seal(request) or request["identity"] != boundary["request"]:
            raise ValueError("storage credit request changed")
        if request["role"] not in {"train", "search", "score"}:
            continue
        data = load(confined(role, "data.json"), maximum=128 << 20)
        if (
            request != seal(request)
            or data != seal(data)
            or request["data"] != data["identity"]
            or request["identity"] != boundary["request"]
        ):
            raise ValueError("storage credit request changed")
        match = re.fullmatch(
            r"((?:ordinary|first_divergence|incumbent)-(?:0|1000000|4000000))(?:-resume-[0-9]+)?(?:-scoreretry-[0-9]+)?",
            data["block"],
        )
        if not match:
            continue  # Qualification and preparation have separate retained allocations.
        producer = read_bound_record(role, boundary["producer"])
        if producer["request"] != boundary["request"]:
            raise ValueError("storage credit belongs to another request")
        output = confined(role, boundary["producer"]["path"]).parent
        result = producer["result"]
        if request["role"] == "train" or data["block"] == match[1]:
            finished.add((match[1], request["role"]))
        entries = []
        if "checkpoint" in result:
            checkpoint = output / "checkpoint"
            ref = result["checkpoint"]
            manifest = read_bound_record(checkpoint, ref)
            entries.extend(
                ("checkpoint", checkpoint, item)
                for item in [ref, *artifact_refs(manifest["states"])]
            )
        for key in ("candidates", "scores", "rows"):
            if key in result:
                entries.append(("producer", output, result[key]))
        for kind, parent, ref in entries:
            file = confined(parent, ref["path"])
            stat = file.stat()
            inode = (stat.st_dev, stat.st_ino)
            if not file.is_file() or stat.st_size != ref["bytes"]:
                raise ValueError("retained allocation differs in size")
            if inode not in seen:
                seen.add(inode)
                value = allocations[(stat.st_dev, kind)]
                value[0] += min(stat.st_size, stat.st_blocks * 512)
                value[1] += 1
    return allocations, finished


def capacity(area, host, checkpoint_bytes, *, scale, share, cached, measure, outstanding=None):
    """Report each backing filesystem once, including independent image and report stores."""
    required = {}
    floor = host.policy["free_disk_floor_bytes"]

    def reserve(path, bytes_needed, nodes_needed):
        path = Path(path)
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        if device not in required:
            required[device] = dict(
                device=device, **measure(path), required_bytes=floor, required_inodes=0
            )
        required[device]["required_bytes"] += bytes_needed
        required[device]["required_inodes"] += nodes_needed

    if cached:
        return []
    paid, finished = paid_allocations(area)
    pending = None if outstanding is None else set(outstanding) - finished
    dev = area.stat().st_dev
    checkpoints, cp_nodes = paid[(dev, "checkpoint")]
    producers, producer_nodes = paid[(dev, "producer")]
    producer_peak = scale["candidate_score_bytes"] + scale["campaign_bytes"]
    # One simultaneous replacement checkpoint and one producer staging area remain available.
    ntrain = 6 if pending is None else sum(role == "train" for _, role in pending)
    have_producers = pending is None or any(role in ("search", "score") for _, role in pending)
    cp_bytes = (
        (
            checkpoint_bytes
            + min(ntrain * checkpoint_bytes, max(0, 6 * checkpoint_bytes - checkpoints))
        )
        if ntrain
        else 0
    )
    cp_inodes = (13 + min(ntrain * 13, max(0, 6 * 13 - cp_nodes))) if ntrain else 0
    shard_bytes = producer_peak + max(0, producer_peak - producers) if have_producers else 0
    shard_inodes = 160 * 8 * 4 + max(0, 160 * 8 * 4 - producer_nodes) if have_producers else 0
    reserve(area, cp_bytes + shard_bytes, cp_inodes + shard_inodes)
    reserve(share, scale["campaign_bytes"], 352)
    engine = getattr(host, "engine", None)
    from oeis_learn.container_runtime import Engine

    details = engine.info() if isinstance(engine, Engine) else getattr(engine, "details", None)
    if isinstance(details, dict):
        storage = details.get("store", {}).get("graphRoot") or details.get("DockerRootDir")
        if storage and Path(storage).is_dir():
            reserve(
                storage, 0, 0
            )  # Existing images reside here. Building rechecks the engine floor.
    for record in required.values():
        record.update(
            deficit_bytes=max(0, record["required_bytes"] - record["free_bytes"]),
            deficit_inodes=max(0, record["required_inodes"] - record["free_inodes"]),
        )
    return list(required.values())
