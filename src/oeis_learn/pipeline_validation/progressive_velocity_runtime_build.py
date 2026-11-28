"""Acquire the original initialization image after a settled, cache-only failure."""

from pathlib import Path
import subprocess

from oeis_learn.container_runtime import Engine, normalized_image
from oeis_learn.generalization.progressive.budgets import KEYS
from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_records import operation_key
from .identity import confined
from .preparation_recovery import retain
from .progressive_pilot_capacity import mount_demand
from .progressive_velocity_broad_host import CapacityShortfall
from .progressive_velocity_image import storage
from .progressive_velocity_operations import input_record


def protected_time(quote, phase):
    """Leave the still-unissued roles, publication read and priced follow-up funded."""
    protected = integer(quote["followup_reserve_us"], 1)
    if phase != "measure":
        protected += max(121_000_000, 120_000_000 + integer(quote["initial_role_us"], 1))
    if phase in {"build_runtime", "package_runtime"}:
        protected += integer(quote["checkpoint_read_us"], 1)
    if phase == "build_runtime":
        protected += 121_000_000
    return protected


def context_size(root, revision):
    """Size the tracked build context, allowing for both source and staging."""
    listing = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rlz", revision])
    size = count = 0
    directories = {b""}
    for entry in filter(None, listing.split(b"\0")):
        metadata, name = entry.split(b"\t", 1)
        length = int(metadata.split()[3])
        size += ((length + 4095) // 4096) * 4096
        count += 1
        while b"/" in name:
            name = name.rsplit(b"/", 1)[0]
            if name in directories:
                break
            directories.add(name)
    return dict(bytes=2 * (size + 4096 * len(directories)), inodes=2 * (count + len(directories)))


def request(root, area, coords, grant, binding, quote, prefix, state):
    """Only a paid lookup with no package child or installed output can switch acquisition."""
    logical = operation_key(coords, 0)["logical"]
    block = confined(area, "operations/" + logical[7:], exists=False)
    if not (block / "first-error.json").is_file() or (block / "receipt.json").exists():
        return None
    error = load(confined(block, "first-error.json"))
    if error != seal(error):
        raise ValueError("original package failure identity differs")
    if (error["error"], error["reason"], error["stage"]) != (
        "ValueError",
        "settled compatible runtime required",
        coords["stage"],
    ):
        return None
    keys = {k for k in state["committed"] if k.startswith(logical + "/dispatch/")}
    if not 1 <= len(keys) <= 2 or any(
        k.startswith(logical + "/dispatch/") for k in state["pending"]
    ):
        return None
    if list(block.glob("boundary-*.json")) or list(block.glob("recovering-*.json")):
        return None
    if load(confined(block, "input.json")) != input_record(
        coords, binding, dict.fromkeys(KEYS, 0), grant
    ):
        raise ValueError("original cache lookup registration differs")
    if error["attempt"] != 0:
        return None
    for attempt in range(len(keys)):
        key = f"{logical}/dispatch/{attempt}"
        if key not in keys or any(
            v for k, v in state["reserved_work"][key].items() if k != "elapsed_us"
        ):
            raise ValueError("original lookup requires a settled workstation-only charge")
        out = confined(block, f"attempt-{attempt:04d}")
        if {p.name for p in out.iterdir()} != {"cleanup.json"}:
            return None
        clean = load(confined(out, "cleanup.json"))
        if clean.get("acknowledged") is not True or clean.get("within_bound") is not True:
            return None
    owners = [owner for row in prefix for owner in row["owners"]]
    if sorted(o["owner"] for o in owners) != ["first_account", "learner_00", "worker_staging"]:
        raise ValueError("incomplete frozen initialization storage requirements")
    return seal(
        dict(
            binding=binding["identity"],
            quote=quote["identity"],
            target="gpu",
            candidate=binding["candidate"],
            cache_error=error["identity"],
            lookups=sorted(keys),
            context=context_size(root, binding["candidate"]),
            workspace={
                u: sum(integer(o["incremental"][u]) for o in owners) for u in ("bytes", "inodes")
            },
            reserve={
                u: max(integer(m["reserved_" + u]) for m in prefix) for u in ("bytes", "inodes")
            },
        )
    )


def capacity(area, engine, terms, floor):
    """Combine demand on a shared backing device and guard both mounts throughout the build."""
    if isinstance(engine, Engine):
        engine.info()
    if (getattr(engine, "details", None) or {}).get("host", {}).get("serviceIsRemote") is True:
        raise ValueError("local image store capacity unavailable: remote engine")
    mounts = {}
    for name, sample in storage(area, engine).items():
        device = sample.get("device")
        if device is None:
            raise ValueError("local image store capacity unavailable: " + name)
        units = ("bytes", "inodes")
        free = {u: integer(sample["free_" + u]) for u in units}
        live, pending = mounts.setdefault(device, (dict(free), []))
        live.update({u: min(live[u], free[u]) for u in units})
        pending.append(terms["workspace" if name == "workspace" else "context"])
    reserve = dict(terms["reserve"], bytes=max(floor, terms["reserve"]["bytes"]))
    rows = [
        dict(device=dev, **mount_demand(available=live, reserve=reserve, pending=pending))
        for dev, (live, pending) in sorted(mounts.items())
    ]
    if any(row["deficit_bytes"] or row["deficit_inodes"] for row in rows):
        raise CapacityShortfall("build_runtime", rows, None)
    return rows


def tag(terms, out):
    return "oeis-learn:init-" + terms["identity"][7:31] + "-" + out.name


def acquire(area, source, out, engine, terms, floor, seconds):
    logs = out / "image-build"
    iid = logs / "image-id"
    lowest = {}
    samples = 0

    def observe():
        nonlocal samples
        for row in capacity(area, engine, terms, floor):
            dev = row["device"]
            prior = lowest.setdefault(dev, dict(row))
            for name in ("available_bytes", "available_inodes"):
                prior[name] = min(prior[name], row[name])
        samples += 1

    logs.mkdir()
    image = engine.build(
        source,
        target=terms["target"],
        tag=tag(terms, out),
        seconds=seconds(),
        evidence=logs,
        iidfile=iid,
        check_resources=observe,
    )
    observe()
    retain(
        out / "producer.json",
        seal(
            dict(
                request=terms["identity"],
                image=image,
                samples=samples,
                lowest=list(lowest.values()),
            )
        ),
    )


def verify(source, out, terms, dispatch):
    logs = confined(out, "image-build")
    iid = confined(logs, "image-id")
    command = load(confined(logs, "command.json"))["argv"]
    proc = load(confined(logs, "process.json"))
    cleanup = load(confined(logs, "cleanup.json"))
    result = load(confined(out, "producer.json"))
    integer(result["samples"], 2)
    if (
        set(result) != {"request", "image", "samples", "lowest", "identity"}
        or Path(command[0]).name not in {"docker", "podman"}
        or command[1:]
        != [
            "build",
            "--target",
            terms["target"],
            "--tag",
            tag(terms, out),
            "--iidfile",
            str(iid),
            str(source),
        ]
        or proc["outcome"] != "COMPLETE"
        or proc["returncode"] != 0
        or proc["cleanup_failure"]
        or cleanup.get("acknowledged") is not True
        or cleanup.get("within_bound") is not True
        or result != seal(result)
        or result["request"] != terms["identity"]
        or result["image"] != normalized_image(iid.read_text())
    ):
        raise ValueError("original image build, immutable ID or supervisor termination differs")
    if [r["device"] for r in result["lowest"]] != [r["device"] for r in dispatch]:
        raise ValueError("image store or workspace backing device changed during the build")
    for row, opened in zip(result["lowest"], dispatch, strict=True):
        if row.keys() != opened.keys():
            raise ValueError("image build storage observation has unexpected fields")
        for unit in ("bytes", "inodes"):
            need, reserved, free = (
                integer(row[k + "_" + unit]) for k in ("required", "reserved", "available")
            )
            if (need, reserved) != (opened["required_" + unit], opened["reserved_" + unit]) or (
                free < need + reserved or row["deficit_" + unit] != 0
            ):
                raise ValueError("image acquisition breached its frozen workspace or store reserve")
    return result
