"""Incremental peak storage for the privately retained pilot roles."""

from collections import defaultdict
import re
from pathlib import Path

from oeis_learn.generalization.contracts import artifact_refs
from oeis_learn.generalization.progressive.contracts import load, read_bound_record, seal
from .identity import confined


def paid_allocations(area, verified, rosters=None):
    """Credit verified bytes once per inode and backing filesystem."""
    allocations = defaultdict(lambda: [0, 0])
    from .progressive_research_inputs import checked_file
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
        key = (match[1], request["role"])
        expected = (rosters or {}).get(match[1])
        whole_roster = expected is None or {r["id"] for r in data.get("rows", [])} == expected
        if "checkpoint" in result or data["block"] == match[1] and whole_roster:
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
        for field in ("candidates", "scores", "rows"):
            if field in result:
                entries.append(("producer", output, result[field]))
        for kind, parent, ref in entries:
            file = confined(parent, ref["path"])
            stat = file.stat()
            inode = (stat.st_dev, stat.st_ino)
            if not file.is_file() or stat.st_size != ref["bytes"]:
                raise ValueError("retained allocation differs in size")
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, ref["sha256"])
            if verified.get(inode) != signature:
                checked_file(parent, ref)
                verified[inode] = signature
            if inode not in seen:
                seen.add(inode)
                value = allocations[(key, stat.st_dev, kind)]
                value[0] += min(stat.st_size, stat.st_blocks * 512)
                value[1] += 1
    return allocations, finished


def capacity(area, host, checkpoint_bytes, *, scale, share, cached, measure, outstanding=None, verified=None, rosters=None):
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
        return device

    if cached:
        return []
    from .progressive_pilot_recovery import assigned_request
    from oeis_learn.generalization.progressive.program_search_pilot import ARMS, LOOKS

    paid, finished = paid_allocations(area, {} if verified is None else verified, rosters)
    if outstanding is None:
        outstanding = {(f"{arm}-{target}", role) for arm in ARMS for target in LOOKS
                       for role in ("train", "search", "score")}
        outstanding.update((f"incumbent-{LOOKS[-1]}", role) for role in ("search", "score"))
    pending = set(outstanding) - finished
    resident = scale.get("resident")
    staging = {}
    reserve(area, 0, 0)
    for block, role in sorted(pending):
        slots = len(rosters[block]) if rosters and block in rosters else 160
        path = area / "roles"
        claim = area / "assignments" / f"{block}-{role}.json"
        if claim.is_file():
            folder = path / assigned_request(area, claim)[7:]
            attempt = max(folder.glob("attempt-*"), default=folder / "attempt-0000")
            path = attempt / "output" / {"train": "checkpoint", "search": "generation", "score": "scored"}[role]
        device = reserve(path, 0, 0)
        if role == "train":
            size, nodes, kind = checkpoint_bytes, 13, "checkpoint"
        else:
            if resident:
                field = "candidates_bytes" if role == "search" else "scored_bytes"
                observed_bytes = resident[field]
                if role == "score" and resident.get("score_control"):
                    observed_bytes = max(observed_bytes, resident["score_control"]["scored_bytes"])
                size = (observed_bytes * slots + resident["cases"] - 1) // resident["cases"]
            else:
                # Before cold qualification the donor's full producer envelope bounds one role.
                size = scale["candidate_score_bytes"] + scale["campaign_bytes"]
            attempts = 8 if block.endswith("-4000000") else 1
            # Search retains case boundaries alongside the assembled candidate file.
            size *= 2 if role == "search" else 1
            nodes = 12 + (slots * (2 + attempts) if role == "search" else 2)
            kind = "producer"
        occupied, inodes = paid[((block, role), device, kind)]
        reserve(path, max(0, size - occupied), max(0, nodes - inodes))
        # Roles run sequentially. Only one atomic replacement is staged on each device.
        prior_bytes, prior_nodes = staging.get(device, (0, 0))
        staging[device] = max(prior_bytes, size), max(prior_nodes, nodes)
    for device, (size, nodes) in staging.items():
        required[device]["required_bytes"] += size
        required[device]["required_inodes"] += nodes
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
