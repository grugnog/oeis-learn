"""Read paid training and publication measurements for the supported round029 launch."""

import hashlib
import json
import zipfile

from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.program_search_costing import trajectory_overheads
from oeis_learn.generalization.progressive.velocity_records import SESSION
from .identity import confined
from . import progressive_velocity_broad_host as broad
from . import progressive_velocity_checkpoint_host as checkpoints
from .progressive_pilot_capacity import velocity_mounts
from .progressive_velocity_donor import ARCHIVES, _archives


def paid_training(root, fresh):
    folder = confined(root, "run-state/discovery/" + SESSION)
    pointer = checkpoints.origin(folder)
    area = confined(folder, "qualification/" + pointer["grant"][7:])
    grant = load(confined(area, "grant.json"))
    out = broad.attempt(area, broad.coordinates(grant, "measure"), grant)
    producer = load(confined(out, "producer.json"))
    request = load(confined(area, "device-request.json"))
    if (fresh != seal(fresh) or pointer["receipt"] != fresh["identity"]
            or producer != seal(producer) or producer["identity"] != fresh["device"]["producer"]
            or request != seal(request) or producer["request"] != request["identity"]):
        raise ValueError("original fresh optimizer samples and request required")
    samples = producer["matrix"]["methods"]
    starts = {m: samples[m]["startup_us"] for m in fresh["device"]["methods"]}
    counts = {m: sum(c["updates_at_look"] for c in request["design"]["cells"][m]) for m in starts}
    return starts, counts


def release_measurement(root, policy, info):
    """Read the source-pinned successful release already checked by donor preflight."""
    if info["archives"]["release"].get("reader") != policy["release_reader"]:
        return None
    paths, _ = _archives(root, policy)
    with paths["release"].open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != ARCHIVES["release"][1]:
            raise ValueError("original release archive changed")
        raw.seek(0)
        with zipfile.ZipFile(raw) as archive:
            report = json.loads(archive.read("release.json"))
    if report != seal(report) or report["candidate"] != policy["release_reader"]:
        raise ValueError("original release measurement or pinned reader differs")
    result = report["stages"]["scale"]["result"]
    return dict(receipt=report["identity"], candidate=report["candidate"],
                floor_bytes=report["admission"]["floor_bytes"],
                **{k: result[k] for k in ("corpora", "journal", "peak_allocated_bytes", "peak_inodes")})


def publication_proxy(release):
    """One production-size write/read and ledger replay at the original measured scale."""
    if release is None:
        return None
    corpora, ledger = release["corpora"], release["journal"]
    spans = [c["cold_us"] + c["warm_us"] for c in corpora.values()]
    return dict(receipt=release["receipt"],
                corpora={k: dict(c) for k, c in corpora.items()}, journal=dict(ledger),
                expected_us=(sum(spans) + 1) // 2 + ledger["cold_us"],
                tail_us=max(spans) + ledger["cold_us"],
                scope="One historical-scale write, expanded-member verification and a cold journal replay. "
                      "The two original encodings supply the midpoint and slower observed price.")


def checkpoint_space(root, cycles, trajectory, floor):
    """Compare all retained treatment cuts and one staged write to current free space."""
    path = "run-state/discovery-learning"
    owners = []
    for method, row in cycles["methods"].items():
        # The measured tree already contains the initial and both updated generations.
        # Charge a whole observed tree per extra hourly generation until a single
        # generation's allocated size is measured. One future write may be pending.
        copies = 1 + trajectory["methods"][method]["hourly_saves"]["tail"]
        observed = {u: row["checkpoint_" + u] for u in ("bytes", "inodes")}
        owners.append(dict(owner="learner_" + method, path=path, retained=[], observed=observed,
                           projected={u: n * copies for u, n in observed.items()},
                           revision=cycles["candidate"], scope="retained treatment checkpoints",
                           harm="loss of a resumable learner cut", exhaustion="price one treatment",
                           rehearsal="T060 peak coexistence"))
    staged = dict(owners[0], owner="pending_checkpoint", scope="one concurrent checkpoint write")
    staged["observed"] = {u: 0 for u in ("bytes", "inodes")}
    staged["projected"] = {u: max(o["observed"][u] for o in owners) for u in ("bytes", "inodes")}
    floors = [dict(path=path, bytes=floor, inodes=0)]
    full = velocity_mounts(root, [*owners, staged], floors)
    first = dict(owners[0], projected=owners[0]["observed"])
    first_stage = dict(staged, projected=first["observed"])
    prefix = velocity_mounts(root, [first, first_stage], floors)
    return dict(checkpoint_receipt=cycles["receipt"], wave=full, first_ordinary_cut=prefix,
                scope="Four retained checkpoint trees plus a single pending write, with provisional headroom. "
                      "The first ordinary cut has its own smaller storage check.")


def launch_preflight(root, policy, info, *, floor):
    prices = info["prices"]
    cycles, fresh = prices["checkpoint_cycles"], prices["fresh_start"]
    result = dict(checkpoint_receipt=cycles["receipt"], fresh_receipt=fresh["identity"],
                  trajectory=None, publication_proxy=None, checkpoint_space=None,
                  registered_mounts=info["capacity"], missing={})
    try:
        release = release_measurement(root, policy, info)
        result["publication_proxy"] = publication_proxy(release)
        if release is None:
            result["missing"]["publication_proxy"] = "original verified round028 release required"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        result["missing"]["publication_proxy"] = type(exc).__name__
    try:
        if cycles.get("device_charge_us") is None:
            raise ValueError("original successful checkpoint dispatch has no proven wall interval")
        starts, counts = paid_training(root, fresh)
        result["updates_to_4M"] = counts
        trajectory = trajectory_overheads(fresh["device"]["methods"], starts, cycles)
        result["trajectory"] = trajectory
        result["checkpoint_space"] = checkpoint_space(root, cycles, trajectory, floor)
    except (OSError, ValueError, KeyError) as exc:
        result["missing"]["trajectory_or_capacity"] = str(exc)
    return result
