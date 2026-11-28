"""Custodian projections, isolated donor roles and private comparative evidence replay."""

import os
from pathlib import Path
import runpy
import time

from oeis_learn.generalization.contracts import artifact_refs
from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.campaign import learned_search_policy
from oeis_learn.generalization.progressive.contracts import (
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_trials import exposure
from oeis_learn.generalization.state import restore
from . import progressive_velocity_checkpoint_host as checkpoint
from . import progressive_velocity_trial_preparation as preparation
from . import progressive_velocity_wave as wave
from .identity import confined
from .preparation_recovery import retain
from .progressive_evaluation import model_receipt
from .progressive_handoff import stage
from .progressive_panel_sources import read_inputs
from .progressive_research_inputs import checked_file
from .progressive_round028_audit import ORIGINAL
from .progressive_round028_pilot import outcome_channels
from .progressive_round028_receipts import verify_role_mounts
from .progressive_round028_rehearsal import physical_scope
from .progressive_velocity_broad_host import attempt
from .progressive_velocity_operations import InsufficientAuthority
from .progressive_velocity_preparation import funded_inventory
from .progressive_velocity_runtime import wheel_sources


def runtime(root, folder, accepted):
    marker = load(confined(folder, preparation.INITIAL_MARKER))
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant = load(confined(area, "grant.json"))
    record = load(confined(area, "receipt.json"))
    phase = "package_runtime" if "image_build" in record else "package"
    directory = attempt(area, checkpoint.coordinates(grant, phase), grant)
    item = load(confined(directory, "runtime.json"))
    path = checked_file(area, item["wheel"])
    if (
        record != seal(record)
        or record["identity"] != accepted["initialization"]
        or item != seal(item)
        or item["wheel"]["sha256"] != accepted["wheel"]
        or item["parent"]["image"] != accepted["image"]
    ):
        raise ValueError("registered science wheel or image differs from the paid zero cut")
    wheel_sources(path, accepted["source"])
    return path


def panel(root, folder, registration):
    ptr = load(confined(folder, "preparation/supply/families/funded.json"))
    _, families = funded_inventory(root, "families", ptr, with_original=True)
    wanted = set()
    for role, name in (("development", "development"), ("retention", "development_retention")):
        rows = [
            dict(task=r["task"], group=r["family"], stratum=r["stratum"])
            for r in families["panels"][name]
        ]
        if rows != registration[role]:
            raise ValueError("registered development or retention roster differs")
        wanted.update(r["task"] for r in rows)
    campaign_path = confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json")
    campaign = read_bound_record(
        root, reference(root, campaign_path), identity=learned_search_policy()["campaign"]
    )
    refs = {r["name"]: r for r in campaign["payload"]["references"]}
    candidates = read_inputs(root, refs, velocity=True)["candidates"]
    selected = {r["id"]: r for r in candidates if r["id"] in wanted}
    if set(selected) != wanted:
        raise ValueError("original development witnesses missing")
    return selected


def capacity(area, accepted, floor, limits=None, retained=None, owner="comparative wave"):
    """Credit stable, authenticated files once, capped by their original owner's peak."""
    # All writable roles are confined to operations; the journal and archive can be mounted separately.
    mounts = {}
    for subdir in ("", "operations", "share", "accounting", "inputs", "projections"):
        path = area / subdir
        if path.exists():
            mounts[path.stat().st_dev] = path
    rows = {r["device"]: r for r in accepted["capacity"]}
    if mounts.keys() != rows.keys():
        raise InsufficientAuthority(f"{owner}: actual mount IDs {sorted(mounts)} differ from "
                                    f"accepted capacity {sorted(rows)}")
    seen = set()
    credits = {dev: dict(bytes=0, inodes=0) for dev in rows}
    for name, paths in (retained or {}).items():
        device = limits[name].get("device", area.stat().st_dev)
        amount = dict(bytes=0, inodes=0)
        for path, original in paths.items():
            try:
                current = path.lstat()
            except FileNotFoundError:
                continue
            key = current.st_dev, current.st_ino
            if (current.st_dev == device and key not in seen
                    and (current.st_ino, current.st_ctime_ns, current.st_size) ==
                        (original.st_ino, original.st_ctime_ns, original.st_size)):
                amount["bytes"] += current.st_blocks * 512
                amount["inodes"] += 1
                seen.add(key)
        for unit in amount:
            credits[device][unit] += min(amount[unit], limits[name][unit])
    for device, row in rows.items():
        stat = os.statvfs(mounts[device])
        for unit, available, reserve in (
            ("bytes", stat.f_bavail * stat.f_frsize, max(floor, row["reserved_bytes"])),
            ("inodes", stat.f_favail, row["reserved_inodes"]),
        ):
            required = row["required_" + unit] - credits[device][unit]
            deficit = required + reserve - available
            if deficit > 0:
                raise InsufficientAuthority(
                    f"{owner} on mount {device}: available {available} {unit}, "
                    f"pending {required}, reserve {reserve}, deficit {deficit} {unit}"
                )


def next_files(directory, pointer):
    store = confined(directory, "checkpoint")
    manifest = load(checked_file(store, pointer))
    names = [
        "checkpoint/generalization-latest.json",
        "checkpoint/" + pointer["path"],
        "accounting/budget-binding.json",
    ]
    names += ["checkpoint/" + r["path"] for r in artifact_refs(manifest["states"])]
    if (directory / "accounting/budget.jsonl").is_file():
        names.append("accounting/budget.jsonl")
    return [reference(directory, confined(directory, rel)) for rel in sorted(set(names))]


def restart_mounts(out, request, inputs, *, readonly):
    """Reopen the last stopped attempt's full state and all subsequent debits."""
    number = int(out.name.removeprefix("attempt-"))
    if not number or request["role"] != "train":
        return inputs
    prior = confined(out.parent, f"attempt-{number - 1:04d}")
    pointer = confined(prior, "learner/checkpoint/generalization-latest.json", exists=False)
    if not pointer.exists():
        return inputs
    clean = load(confined(prior, "cleanup.json"))
    if clean.get("acknowledged") is not True:
        raise InsufficientAuthority(
            "previous comparative learner needs verified engine termination"
        )
    original = confined(prior, "learner")
    store = confined(original, "checkpoint")
    selected = load(pointer)
    manifest = load(checked_file(store, selected))
    contract = digest([request["learner"], request["cap"]])
    restore(manifest, contract=contract, cap=request["cap"], root=store)
    files = next_files(original, selected)
    binding = seal(
        dict(campaign=contract, caps=dict(dict.fromkeys(KEYS, 0), tokens=request["cap"]))
    )
    if load(confined(original, "accounting/budget-binding.json")) != binding:
        raise ValueError("interrupted learner debit binding differs")
    record = seal(dict(request=request["identity"], prior=number - 1, files=files))
    publication = confined(out.parent, f"restart-{number:04d}.json", exists=readonly)
    if not readonly:
        retain(publication, record)
    if load(publication) != record:
        raise ValueError("original interrupted checkpoint or issued token charges differ")
    mounts = [(p, d) for p, d in inputs if not d.startswith("/work/inputs/parent/")]
    return [
        *mounts,
        (publication, "/work/inputs/restart.json"),
        *((checked_file(original, ref), "/work/inputs/parent/" + ref["path"]) for ref in files),
    ]


def check_cut(out, request, package, producer, expected):
    cut = producer["cut"]
    directory = confined(out, "learner")
    store = confined(directory, "checkpoint")
    pointer = load(confined(store, "generalization-latest.json"))
    contract = digest([package["identity"], request["cap"]])
    if cut["checkpoint"] != pointer or cut["target"] != request["target"]:
        raise ValueError("training target or installed checkpoint differs")
    manifest = load(checked_file(store, pointer))
    restored = restore(manifest, contract=contract, cap=request["cap"], root=store)
    caps = dict(dict.fromkeys(KEYS, 0), tokens=request["cap"])
    accounting = confined(directory, "accounting")
    if (accounting / "budget.jsonl").exists():
        balance = BudgetJournal.inspect_existing(accounting, contract, caps)
    elif expected["updates"] == 0 and load(confined(accounting, "budget-binding.json")) == seal(
        dict(campaign=contract, caps=caps)
    ):
        balance = dict(pending=[], consumed=dict(tokens=0))
    else:
        raise ValueError("missing learner debit journal")
    ledger = restored["ledger"]
    replay = exposure(
        package,
        package["rows"],
        updates=expected["updates"],
        batch=package["policy"]["effective_batch"],
        discarded=balance["consumed"]["tokens"] - expected["useful_tokens"],
    )
    if (
        balance["pending"]
        or ledger.pending
        or ledger.consumed != cut["charged_tokens"]
        or ledger.updates != cut["updates"]
        or any(cut[k] != v for k, v in {**expected, **replay}.items())
        or restored["states"]["data"] != dict(cursor=cut["cursor"], pending_batch=None)
        or producer["weight"] != manifest["states"]["weights"]
        or request["target"] == 0
        and cut["weights"] != package["initial_weights"]
    ):
        raise ValueError("checkpoint exposure, optimizer ledger or initial tensors differ")
    worker_look = dict(
        registration=package["registration"],
        method=request["method"],
        seed=request["seed"],
        look=request["target"],
    )
    weight = manifest["states"]["weights"]
    model_terms = dict(
        profile=digest(package["profile"]),
        weights=cut["weights"],
        checkpoint=weight["sha256"] if request["target"] else None,
        origin=digest([package["identity"], pointer["sha256"]]),
    )
    if cut["model"] != model_receipt(worker_look, **model_terms):
        raise ValueError("worker model differs from the authenticated checkpoint")
    model = model_receipt(dict(worker_look, registration=request["registration"]), **model_terms)
    return dict(cut=dict(cut, model=model), weight=weight)


def custody(root, folder, accepted, reg, source):
    """Only the custodian opens the source census, incumbent and paid acquisition."""
    original = load(confined(folder, preparation.INITIAL_MARKER))
    quote = load(confined(folder, "qualification/" + original["grant"][7:] + "/quote.json"))
    fresh = checkpoint.origin(folder)
    measurement = load(confined(folder, "qualification/" + fresh["grant"][7:] + "/receipt.json"))
    if (
        quote != seal(quote)
        or quote["sources"] != accepted["sources"]
        or measurement != seal(measurement)
        or measurement["identity"] != fresh["receipt"]
        or fresh["receipt"] != accepted["sources"]["fresh"]
    ):
        raise ValueError("comparative role prices differ from the paid pre-score quote")
    from .progressive_velocity_census import training_source
    from .progressive_velocity_donor import inventory

    freeze, _, _ = training_source(root)
    original_weight = freeze["reference"]
    weight = checked_file(root, original_weight)
    (cut,) = [c for c in inventory(root)["cuts"] if c["method"] == "D"]
    parent = confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json")
    campaign = read_bound_record(
        root, reference(root, parent), identity=learned_search_policy()["campaign"]
    )
    (recipe_ref,) = [r for r in campaign["payload"]["references"] if r["name"] == "recipe"]
    if (recipe_ref["role"], recipe_ref["kind"]) != ("custodian", "recipe"):
        raise ValueError("original incumbent recipe authority differs")
    recipe = load(checked_file(root, recipe_ref["artifact"]))
    if (
        not cut["inference"]
        or cut["weights"] != original_weight["sha256"]
        or cut["recipe"] != digest(recipe)
    ):
        raise ValueError("original incumbent weights or recipe differ")
    witnesses = {
        k: {f: v[f] for f in ("id", "indices", "values")}
        for k, v in panel(root, folder, reg).items()
    }
    from .progressive_round029 import preflight

    prices = preflight(root, source)["inventory"]["prices"]["launch_preparation"]
    mounts = prices["wave_space"]["wave"]
    keys = ("device", "required_bytes", "required_inodes")
    if [[m[k] for k in keys] for m in mounts] != [[m[k] for k in keys] for m in accepted["capacity"]]:
        raise ValueError("reopened owner peaks differ from the accepted wave capacity")
    limits = {o["owner"]: dict(o["incremental"], device=m["device"]) for m in mounts for o in m["owners"]}
    return seal(
        dict(
            accepted=accepted["identity"],
            capacity_owners=limits,
            wheel=reference(root, runtime(root, folder, accepted)),
            startup_us=integer(quote["initial_role_us"], 1),
            training_tail_us={
                m: integer(v["tail_us"], 1) for m, v in measurement["device"]["methods"].items()
            },
            witnesses=witnesses,
            incumbent=dict(
                recipe=recipe,
                weight=reference(root, weight),
                origin=digest([cut["registration"], cut["lineage"], cut["weights"]]),
            ),
        )
    )


class Donor:
    def __init__(
        self, root, source, area, accepted, basis, revision, args, host_factory, *, readonly=False
    ):
        if basis != seal(basis) or basis["accepted"] != accepted["identity"]:
            raise ValueError("private comparative role inputs differ")
        self.accepted, self.source, self.area = accepted, source, area
        self.startup_us, self.training_tail_us = basis["startup_us"], basis["training_tail_us"]
        self.wheel = checked_file(root, basis["wheel"])
        self.witnesses = basis["witnesses"]
        self.capacity_owners = basis["capacity_owners"]
        self.retained = {}
        self.incumbent = basis["incumbent"]
        self.incumbent_path = checked_file(root, self.incumbent["weight"])
        self.role = confined(source, wave.DRIVER)
        operator = confined(
            source, "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
        )
        self.operator = runpy.run_path(str(operator))
        self.readonly = readonly
        self.host = (
            None
            if readonly
            else host_factory(args, source_root=source, workspace=root, revision=revision)
        )

    def credit_cut(self, method, out, cut):
        directory = confined(out, "learner")
        name = "learner_" + method
        if name not in self.capacity_owners:
            return
        items = self.retained.setdefault(name, {})
        for ref in next_files(directory, cut["checkpoint"]):
            if ref["path"].startswith("checkpoint/") and not ref["path"].endswith("latest.json"):
                path = checked_file(directory, ref)
                items[path] = path.stat()

    def channels(self, look):
        return outcome_channels(look, self.witnesses)

    def supervise(self, request, data, extra, proof):
        rp = self.area / "inputs" / (request["identity"][7:] + ".json")
        dp = self.area / "projections" / (request["data"][7:] + ".json")
        if self.readonly:
            if (
                load(confined(self.area, rp)) != request
                or load(confined(self.area, dp), maximum=wave.MAX_INPUT) != data
            ):
                raise ValueError("private role projection differs")
        else:
            retain(rp, request)
            retain(dp, data)
        wheel_dest = "/work/inputs/" + self.wheel.name
        inputs = [
            (self.role, "/work/inputs/role.py"),
            (self.wheel, wheel_dest),
            (rp, "/work/inputs/request.json"),
            (dp, "/work/inputs/data.json"),
            *extra,
        ]
        image = self.accepted["image"]

        def guard():
            capacity(self.area, self.accepted, self.host.policy["free_disk_floor_bytes"],
                     self.capacity_owners, self.retained, request["role"] + " " + request["method"])
            if self.host.engine.image(image) != image:
                raise ValueError("accepted image is missing from the donor cache")
            if (
                request["gpu"]
                and physical_scope(self.host, "hip", machine=True) != self.accepted["physical"]
            ):
                raise ValueError("current HIP device differs from the accepted hardware")

        def execute(out, allowance):
            start = time.monotonic_ns() // 1000
            seconds = allowance // 1_000_000 - 120
            if seconds < 1:
                raise InsufficientAuthority("role and cleanup exceed remaining time")
            stage(
                self.host,
                self.source,
                image,
                out,
                [
                    "python",
                    "-I",
                    "-B",
                    "/work/inputs/role.py",
                    wheel_dest,
                    request["wheel"],
                    request["source"],
                ],
                gpu=request["gpu"],
                seconds=seconds,
                readonly_inputs=restart_mounts(out, request, inputs, readonly=False),
                record_lease=request["gpu"],
            )
            end = time.monotonic_ns() // 1000
            device = self.host.policy["render_node"] if request["gpu"] else None
            retain(
                out / "dispatch.json",
                seal(
                    dict(
                        request=request["identity"],
                        start_us=start,
                        end_us=end,
                        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        render_node=device,
                    )
                ),
            )

        def verify(out):
            mounts = restart_mounts(out, request, inputs, readonly=True)
            inspection = load(confined(out, "container-inspect.json"))
            verify_role_mounts(
                inspection,
                image,
                {**{dst: (p, False) for p, dst in mounts}, "/work/output": (out, True)},
            )
            process, cleanup, span, producer = [
                load(confined(out, name + ".json"))
                for name in ("process", "cleanup", "dispatch", "producer")
            ]
            elapsed = span["end_us"] - span["start_us"]
            if (
                span != seal(span)
                or span["request"] != request["identity"]
                or producer != seal(producer)
                or producer["request"] != request["identity"]
                or process["outcome"] != "COMPLETE"
                or process["returncode"] != 0
                or process["cleanup_failure"]
                or cleanup["acknowledged"] is not True
                or cleanup["within_bound"] is not True
                or not 0 < producer["worker_us"] <= elapsed
                or inspection[0]["HostConfig"].get("DeviceRequests")
            ):
                raise ValueError("original comparative supervision or producer differs")
            device = {}
            if request["gpu"]:
                device = self.operator["conditioning_device_evidence"](
                    out, inspection[0], producer, elapsed, span["render_node"]
                )
            elif inspection[0]["HostConfig"].get("Devices"):
                raise ValueError("CPU role received device access")
            if request["role"] != "train":
                from oeis_learn.generalization.progressive.velocity_donation_report import hashes

                hashes(producer["native"])
            return seal(
                dict(
                    producer=producer["identity"],
                    start_us=span["start_us"],
                    end_us=span["end_us"],
                    boot=span["boot"],
                    worker_cpu_us=integer(producer["worker_cpu_us"]),
                    **device,
                    **proof(out, producer),
                )
            )

        return execute, verify, guard
