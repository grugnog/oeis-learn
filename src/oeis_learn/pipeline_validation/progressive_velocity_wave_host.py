"""Custodian projections, isolated donor roles and private comparative evidence replay."""

import os
from pathlib import Path
import runpy
import time

from oeis_learn.generalization.contracts import artifact_refs
from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.campaign import learned_search_policy, velocity_policy
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
from .progressive_evaluation import model_receipt, velocity_probe
from .progressive_handoff import stage
from .progressive_panel_sources import read_inputs
from .progressive_pilot_capacity import owned_allocations
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


def capacity(area, accepted, floor):
    stat = os.statvfs(area)
    owned = owned_allocations([area])
    device = area.stat().st_dev
    # The accepted donor colocates the registered owners on one physical mount.
    if [r["device"] for r in accepted["capacity"]] != [device]:
        raise ValueError("comparative mount layout differs from the accepted donor")
    (row,) = accepted["capacity"]
    for unit, available, reserve in (
        ("bytes", stat.f_bavail * stat.f_frsize, max(floor, row["reserved_bytes"])),
        ("inodes", stat.f_favail, row["reserved_inodes"]),
    ):
        if available < reserve + max(0, row["required_" + unit] - owned[unit]):
            raise InsufficientAuthority("insufficient live comparative capacity in " + unit)


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


class Donor:
    def __init__(
        self, root, folder, source, area, accepted, reg, args, host_factory, *, readonly=False
    ):
        self.accepted = accepted
        self.source, self.area = source, area
        self.role = confined(source, wave.DRIVER)
        self.wheel = runtime(root, folder, accepted)
        self.witnesses = panel(root, folder, reg)
        operator = confined(
            source, "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
        )
        self.operator = runpy.run_path(str(operator))
        from .progressive_velocity_donor import inventory

        models, paths = self.operator["paired_model_inputs"](root, inventory(root)["cuts"])
        self.incumbent, self.incumbent_path = models["D"], paths["D"]
        self.readonly = readonly
        self.host = (
            None
            if readonly
            else host_factory(args, source_root=source, workspace=root, revision=source.name)
        )

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
            capacity(self.area, self.accepted, self.host.policy["free_disk_floor_bytes"])
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
                readonly_inputs=inputs,
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
            inspection = load(confined(out, "container-inspect.json"))
            verify_role_mounts(
                inspection,
                image,
                {**{dst: (p, False) for p, dst in inputs}, "/work/output": (out, True)},
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
