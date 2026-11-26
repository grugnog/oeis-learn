"""One CPU workstation collection for generator panel supply, calibration and historical pricing."""

import runpy
import time
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_costing import PROXIES, clock_cells
from oeis_learn.generalization.progressive.program_search_panels import (
    RESERVATIONS,
    allocate_roles,
    calibration_families,
    descriptor,
    freeze_panels,
    fresh_groups,
)

from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_panel_sources import extra_source, read_inputs, require
from .progressive_panel_worker import CHUNK, CHUNKS, METHODS
from .progressive_research_inputs import ExposureIndex, checked_file
from .progressive_round028_audit import ORIGINAL, Donor
from .progressive_round028_rehearsal import candidate_image, space

PARENT = "sha256:92d1ae2220ebbd2140376a5611269eb740e6eedc0635cd7de49598ba85fe5102"
PRODUCER = "46594ef8ba9e4098b1f5759b28635a4537d6d399"


def prior_archive(root, source, folder, store, area):
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
        publish,
        verify_transport,
    )

    from .progressive_round028 import ARCHIVE
    from .progressive_round028_receipts import recover

    local = area / "original-rehearsal.zip"
    paths = (
        [local]
        if local.exists()
        else sorted(
            confined(root, "run-state/discovery-tools").glob(
                "round028-receipts-*/round028-rehearsal.zip"
            )
        )
    )
    for path in reversed(paths):
        try:
            receipt = verify_transport(confined(root, path))
        except (KeyError, ValueError, OSError, zipfile.BadZipFile):
            if path == local:
                raise
            continue
        if receipt.get("evidence") == PARENT and receipt.get("audit") == store.state["publication"]:
            return path, receipt
    prior = runpy.run_path(
        str(source / "specs/015-learned-program-search/validation/candidate_source.py")
    )["materialize"](root, PRODUCER)
    evidence = recover(root, folder, prior, PRODUCER, store)
    require(evidence["identity"] == PARENT, "original training forecast parent changed")
    return local, publish(local, evidence, confined(folder, ARCHIVE))


def original_costs(root, cut):
    members = {r["task"]: r["stratum"] for r in cut["cases"]}

    def producers(names):
        for name in names:
            pair = cut["producers"].get(name)
            if pair is None:
                continue
            try:
                gen = read_bound_record(root, pair["generation"], version="discovery-candidates/v1")
                scores = read_bound_record(root, pair["scoring"], version="discovery-scoring/v1")
            except FileNotFoundError:
                continue
            yield gen, scores, pair["generation"]["sha256"]

    return {
        PROXIES[0]: clock_cells(producers([PROXIES[0]]), members),
        PROXIES[1]: clock_cells(
            producers([f"capability-retained_D-{i:02d}" for i in range(8)]), members
        ),
    }


class NativeRoles:
    """One immutable CPU image, separate read-only role requests and recoverable completed outputs."""

    phase = "cpu"
    worker = "oeis_learn.pipeline_validation.progressive_panel_worker"
    request_version = "round028-panel-role/v1"
    producer_version = "round028-panel-output/v1"
    fixed_attempt = False

    def __init__(self, args, source, root, revision, area, registration, host_factory):
        self.args, self.source, self.root, self.revision = args, source, root, revision
        self.area, self.registration, self.host_factory = area, registration, host_factory
        self.host = self.native = self.image = None
        self.build_us = 0
        self.charges = []

    def __call__(self, role, data, *, seconds, inputs=()):
        from .progressive_handoff import stage

        if self.host is None:
            self.host = self.host_factory(
                self.args, source_root=self.source, workspace=self.root, revision=self.revision
            )
            require(
                space(self.area)["free_bytes"] >= self.host.policy["free_disk_floor_bytes"],
                "workstation free-space floor",
            )
            self.image, self.build_us, _ = candidate_image(
                self.host, self.source, self.area, self.revision, self.phase
            )
        if self.fixed_attempt:
            from .progressive_round028_rehearsal import physical_scope

            physical = physical_scope(self.host, self.phase)
            require(
                space(self.area)["free_bytes"] >= self.host.policy["free_disk_floor_bytes"],
                "candidate timing workstation free-space floor",
            )
        req = seal(
            {
                "version": self.request_version,
                **({"physical": physical} if self.fixed_attempt else {}),
                "registration": self.registration["identity"],
                "source": self.registration["source"],
                "engine": self.host.engine.name,
                "image": self.image,
                "candidate_native": self.native,
                "role": role,
                "data": data["identity"],
            }
        )
        if self.fixed_attempt:
            # A restart or changed physical scope cannot issue a second draw allowance.
            claim = self.area / "assignments" / (data["block"] + "-" + role + ".json")
            retain(claim, seal({"request": req["identity"]}))
        folder = confined(self.area, "roles/" + req["identity"][7:], exists=False)
        folder.mkdir(parents=True, exist_ok=True)
        req_path, data_path = folder / "request.json", folder / "data.json"
        retain(req_path, req)
        retain(data_path, data)
        mounts = [
            (req_path, "/work/inputs/request.json"),
            (data_path, "/work/inputs/data.json"),
            *inputs,
        ]
        receipt = folder / "boundary.json"
        # Recover a role that finished and passed engine inspection before its boundary was written.
        if not receipt.exists():
            for attempt in sorted(folder.glob("attempt-*")):
                end = attempt / "end.json"
                if end.is_file():
                    ended = load(confined(folder, end))
                    require(
                        seal(ended) == ended and ended["request"] == req["identity"],
                        "panel role timing changed",
                    )
                    if ended["success"]:
                        self.bind(folder, attempt, req)
                        break
        if not receipt.exists():
            if self.fixed_attempt and any(folder.glob("attempt-*")):
                raise ValueError("original timing attempt lacks a completed inspected receipt")
            require(self.host.engine.image(self.image) == self.image, "candidate image changed")
            ordinal = len(list(folder.glob("attempt-*")))
            attempt = confined(folder, f"attempt-{ordinal:04d}", exists=False)
            attempt.mkdir()
            output = attempt / "output"
            began = time.monotonic_ns()
            retain(
                attempt / "start.json",
                seal({"request": req["identity"], "seconds": seconds, "wall_ns": time.time_ns()}),
            )
            success = False
            try:
                stage(
                    self.host,
                    self.source,
                    self.image,
                    output,
                    [
                        "python",
                        "-I",
                        "-m",
                        self.worker,
                    ],
                    gpu=self.phase == "hip",
                    seconds=seconds,
                    readonly_inputs=mounts,
                    **({"record_lease": True} if self.phase == "hip" else {}),
                )
                success = True
            finally:
                retain(
                    attempt / "end.json",
                    seal(
                        {
                            "request": req["identity"],
                            "success": success,
                            "elapsed_us": (time.monotonic_ns() - began) // 1000,
                        }
                    ),
                )
            self.bind(folder, attempt, req)
        boundary = load(confined(folder, receipt))
        require(
            seal(boundary) == boundary and boundary["request"] == req["identity"],
            "panel CPU boundary changed",
        )
        producer = read_bound_record(folder, boundary["producer"], version=self.producer_version)
        from .progressive_round028_receipts import verify_role_mounts

        inspected = load(
            checked_file(folder, boundary["inspection"]), maximum=CAMPAIGN_DOCUMENT_BYTES
        )
        output = confined(folder, Path(boundary["producer"]["path"]).parent)
        verify_role_mounts(
            inspected,
            req["image"],
            {
                "/work/output": (output, True),
                **{dest: (path, False) for path, dest in mounts},
            },
        )
        timing = read_bound_record(folder, boundary["timing"])
        require(
            timing["success"] is True and timing["request"] == req["identity"],
            "unexpected role timing",
        )
        require(
            producer["request"] == req["identity"]
            and (self.native is None or self.native == producer["native"]),
            "panel role producer registration differs",
        )
        self.native = producer["native"]
        self.last_output, self.last_producer = output, producer
        self.last_wall_us, self.last_lease_us = timing["elapsed_us"], 0
        self.last_boundary = receipt
        if self.phase == "hip":
            lease = load(checked_file(folder, boundary["lease"]))
            self.last_lease_us = lease["elapsed_us"]
            require(
                type(self.last_lease_us) is int and 0 < self.last_lease_us <= self.last_wall_us,
                "original GPU lease differs from the role interval",
            )
        for p in sorted(folder.glob("attempt-*/start.json")):
            start = load(confined(folder, p))
            require(
                seal(start) == start and start["request"] == req["identity"],
                "panel role start changed",
            )
            end = p.with_name("end.json")
            if end.exists():
                value = load(confined(folder, end))
                require(
                    seal(value) == value and value["request"] == req["identity"],
                    "panel role debit changed",
                )
                self.charges.append(value["elapsed_us"])
            else:
                self.charges.append(None)
        return producer["result"]

    def bind(self, folder, attempt, req):
        output = attempt / "output"
        producer = load(confined(folder, output / "producer.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
        end = load(confined(folder, attempt / "end.json"))
        require(
            seal(producer) == producer
            and producer["request"] == req["identity"]
            and seal(end) == end
            and end["success"] is True
            and end["request"] == req["identity"],
            "completed inspected panel role required",
        )
        retain(
            folder / "boundary.json",
            seal(
                {
                    "request": req["identity"],
                    "producer": reference(folder, output / "producer.json"),
                    "inspection": reference(folder, output / "container-inspect.json"),
                    "timing": reference(folder, attempt / "end.json"),
                    **(
                        {"lease": reference(folder, output / "gpu-lease.json")}
                        if self.phase == "hip"
                        else {}
                    ),
                }
            ),
        )


def preallocate(inputs, registration):
    with ExitStack() as stack:
        indexes = [stack.enter_context(ExposureIndex(p, readonly=True)) for p in inputs["indexes"]]
        groups, counts = fresh_groups(inputs["candidates"], inputs["historical"], indexes=indexes)
    seed = registration["seed"]
    samples = calibration_families(groups, seed, count=registration["calibration_limit"])
    unknown = {f: [dict(r, band="unknown") for r in rows] for f, rows in groups.items()}

    def select(held):
        return allocate_roles({f: rows for f, rows in unknown.items() if f not in held}, seed=seed)

    allocation = select(samples)
    if allocation["deficits"] and samples:
        # Precommit the family hash order, then retain a fitting prefix
        # alongside the design. An arbitrary calibration count cannot create a deficit.
        baseline = select({})

        def preserves(candidate):
            return all(
                cell["missing_cases"] <= baseline["deficits"].get(key, {}).get("missing_cases", 0)
                and cell["missing_families"]
                <= baseline["deficits"].get(key, {}).get("missing_families", 0)
                for key, cell in candidate["deficits"].items()
            )

        if not preserves(allocation):
            ordered = list(samples.items())
            low, high = 0, len(ordered)
            allocation = baseline
            while low + 1 < high:
                mid = (low + high) // 2
                trial = select(dict(ordered[:mid]))
                if preserves(trial):
                    low, allocation = mid, trial
                else:
                    high = mid
            samples = dict(ordered[:low])
    return groups, counts, samples, allocation


def calibrate(samples, area, native):
    retained = []
    for i in range(0, len(samples), 8):
        batch = list(samples.items())[i : i + 8]
        truth = [{k: r[k] for k in ("id", "indices", "values")} for _, r in batch]
        prompts = [
            {"id": r["id"], "indices": r["indices"][:20], "values": r["values"][:20]} for r in truth
        ]
        choices = native("choose", seal({"prompts": prompts}), seconds=8 * 3 * 60 + 120)
        scores = native(
            "score",
            seal(
                {
                    "truth": truth,
                    "choices": choices,
                    "witnesses": {r["id"]: r["source"] for _, r in batch},
                }
            ),
            seconds=8 * 3 * 60 + 120,
        )
        for family, row in batch:
            methods = scores[row["id"]]
            require(
                set(methods) == set(METHODS)
                and all(
                    set(v) == {"exact", "measured"}
                    and all(type(b) is bool for b in v.values())
                    and (not v["exact"] or v["measured"])
                    for v in methods.values()
                ),
                "incomplete calibration producer",
            )
            retained.append(
                {
                    "family": family,
                    "descriptor": descriptor(row),
                    "denominator": 3,
                    "exact": sum(v["exact"] for v in methods.values()),
                    "measured": sum(v["measured"] for v in methods.values()),
                }
            )
    retain(area / "calibration-observations.json", seal({"observations": retained}))
    return retained


def progress(area, phase, **counts):
    path = area / "progress.json"
    state = seal(dict(phase=phase, **counts))
    if not path.exists() or load(path) != state:
        revise(path, state)


def prepare(root, area, inputs, registration, native):
    from oeis_learn.generalization.progressive.program_search_panels import calibrate_bands

    groups, counts, samples, allocation = preallocate(inputs, registration)
    deficits = allocation["deficits"]
    progress(
        area,
        "retained_supply",
        eligible=counts["eligible"],
        components=counts["components"],
        deficient_cells=len(deficits),
        calibration_families=len(samples),
    )
    supplement = {
        "native_draws": 0,
        "native_accepted": 0,
        "source_selected": 0,
        "source_remaining": None,
    }
    if any(k.endswith("/oeis") for k in deficits):
        additional, source = extra_source(root, inputs["data"], area)
        inputs["candidates"].extend(additional)
        supplement.update(source_selected=source["selected"], source_remaining=source["remaining"])
    if any(not k.endswith("/oeis") for k in deficits) or not samples:
        for start in range(0, CHUNK * CHUNKS, CHUNK):
            output = native(
                "draws",
                seal(
                    {
                        "seed": registration["seed"],
                        "support": inputs["package"],
                        "start": start,
                        "count": CHUNK,
                    }
                ),
                seconds=CHUNK * 4 + 120,
            )
            require(
                output["attempted"]
                == CHUNK
                == sum(output["rejected"].values()) + len(output["rows"])
                and all(
                    seal(r) == r
                    and r["native_verified"] is True
                    and r["membership"] == "evaluation_candidate"
                    for r in output["rows"]
                ),
                "native extension witnesses differ",
            )
            inputs["candidates"].extend(output["rows"])
            supplement["native_draws"] += CHUNK
            supplement["native_accepted"] += len(output["rows"])
            progress(
                area,
                "native_supply",
                attempted=supplement["native_draws"],
                accepted=supplement["native_accepted"],
            )
    if supplement["native_accepted"] or supplement["source_selected"]:
        groups, counts, samples, allocation = preallocate(inputs, registration)
    retain(
        area / "calibration-reservation.json",
        seal(
            {
                "registration": registration["identity"],
                "witnesses": {f: digest(r) for f, r in samples.items()},
            }
        ),
    )
    observed = []
    if not allocation["deficits"] and samples:
        progress(area, "conventional_calibration", calibration_families=len(samples))
        observed = calibrate(samples, area, native)
        bands = calibrate_bands(observed)
        retain(area / "difficulty-map.json", bands)
        allocation = freeze_panels(groups, samples, observed, seed=registration["seed"])
    calibrated = bool(observed) and any(r["measured"] for r in observed)
    panels = allocation["panels"]
    chosen = {r["id"] for rows in panels.values() for r in rows}
    private = seal(
        {
            "version": "round028-private-panel-supply/v1",
            "registration": registration["identity"],
            "calibrated": calibrated,
            "allocations": allocation,
            "witnesses": {
                r["id"]: r for rows in groups.values() for r in rows if r["id"] in chosen
            },
        }
    )
    retain(area / "supply.json", private)
    return {
        "history": inputs["counts"],
        "supply": counts,
        "extension": supplement,
        "panels": panel_counts(panels),
        "calibration": {
            "reserved": len(samples),
            "observed": len(observed),
            "exact": sum(r["exact"] for r in observed),
            "measured": sum(r["measured"] for r in observed),
        },
        "calibrated": calibrated,
    }


def panel_counts(panels):
    aggregate = {}
    for name in RESERVATIONS:
        aggregate[name] = {}
        for stratum in ("retention",) if name.endswith("_retention") else STRATA:
            rows = [r for r in panels[name] if r["stratum"] == stratum]
            bands = Counter(r["band"] for r in rows)
            aggregate[name][stratum] = {
                "cases": len(rows),
                "families": len({r["group"] for r in rows}),
                "bands": dict(bands),
            }
    return aggregate


def bind_private(area, report):
    reg = report["registration"]
    names = ["supply.json", "calibration-reservation.json", "historical-clocks.json", "public.json"]
    if report["summary"]["calibration"]["observed"]:
        names.extend(["calibration-observations.json", "difficulty-map.json"])
    prior = area / "boundary.json"
    if prior.exists():
        cut = load(confined(area, prior))
    else:
        cut = seal(
            {
                "registration": reg,
                "report": report["identity"],
                "files": [reference(area, area / name) for name in names],
            }
        )
    require(
        seal(cut) == cut
        and cut["registration"] == reg
        and cut["report"] == report["identity"]
        and [ref["path"] for ref in cut["files"]] == names,
        "private supply boundary differs",
    )
    records = {ref["path"]: read_bound_record(area, ref) for ref in cut["files"]}
    supply = records["supply.json"]
    held = records["calibration-reservation.json"]
    panels = supply["allocations"]["panels"]
    require(
        supply["registration"] == held["registration"] == reg
        and records["public.json"] == report
        and records["historical-clocks.json"]["cells"] == report["historical_clocks"]
        and panel_counts(panels) == report["summary"]["panels"]
        and len(held["witnesses"]) == report["summary"]["calibration"]["reserved"],
        "private panel commitments differ from the public supply counts",
    )
    seen, owners = set(), {}
    for name, members in panels.items():
        for case in members:
            require(
                case["id"] not in seen
                and case["component"] not in held["witnesses"]
                and owners.setdefault(case["component"], name) == name,
                "private panel member or component reused",
            )
            seen.add(case["id"])
    require(seen == set(supply["witnesses"]), "private panel witnesses missing")
    retain(prior, cut)


def panels_command(args, source, root, folder, revision, store, host_factory):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.program_search_panel_report import (
        make_report,
        publish,
        verify_panels,
    )

    from .progressive_round028 import reopen

    require(
        store.state["publication"] and store.state["route"] == "MATCHED_GENERATOR",
        "published matched-generator audit required",
    )
    area = confined(folder, "private/panel-supply/" + revision, exists=False)
    area.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior, trainer = prior_archive(root, source, folder, store, area)
    require(
        trainer["evidence"] == PARENT
        and all(v == "VERIFIED" for v in trainer["checkpoint_closures"].values()),
        "verified matched checkpoints and original training forecast required",
    )
    registration = seal(
        {
            "version": "round028-panel-supply-registration/v1",
            "audit": store.state["publication"],
            "rehearsal": PARENT,
            "source": digest(collect_snapshot(source)),
            "candidate": revision,
            "seed": int(digest([store.campaign["identity"], "fresh-family-supply/v1"])[7:15], 16),
            "roles": list(RESERVATIONS),
            "cases_per_stratum": 32,
            "calibration_limit": 48,
            "methods": list(METHODS),
            "native_extension": CHUNK * CHUNKS,
            "source_extension": 4096,
            "comparative_dispatches": 0,
        }
    )
    retain(area / "registration.json", registration)
    archive = confined(
        folder, "share/panel-supply/" + revision + "/round028-panels.zip", exists=False
    )
    if archive.exists():
        receipt = verify_panels(archive)
        require(
            receipt["candidate"] == revision
            and receipt["registration"] == registration["identity"],
            "panel supply publication belongs to a different candidate",
        )
        report = load(confined(area, "public.json"))
        require(
            report["identity"] == receipt["evidence"] and (area / "boundary.json").is_file(),
            "published panel archive lost its private boundary",
        )
        bind_private(area, report)
    else:
        began = time.monotonic_ns()
        local = area / "public.json"
        if local.is_file():
            report = load(confined(area, local))
        else:
            donor = Donor(root, folder, store.campaign)
            original = donor.saved(
                ORIGINAL + "/prepare-research/frozen/campaign.json",
                donor.policy["campaign"],
                maximum=CAMPAIGN_DOCUMENT_BYTES,
            )
            refs = {r["name"]: r for r in original["payload"]["references"]}
            cut = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
            clock_path = area / "historical-clocks.json"
            if not clock_path.exists():
                retain(
                    clock_path,
                    seal({"parent": cut["identity"], "cells": original_costs(root, cut)}),
                )
            clocks = load(confined(area, clock_path))
            require(
                seal(clocks) == clocks and clocks["parent"] == cut["identity"],
                "historical clock parent differs",
            )
            progress(
                area,
                "authenticated_historical_clocks",
                timed=sum(v["timed"] for p in clocks["cells"].values() for v in p.values()),
            )
            inputs = read_inputs(root, refs)
            progress(
                area,
                "connecting_historical_families",
                candidates=len(inputs.get("candidates", [])),
                historical_rows=len(inputs.get("historical", [])),
            )
            native = NativeRoles(args, source, root, revision, area, registration, host_factory)
            summary = prepare(root, area, inputs, registration, native)
            report = make_report(
                registration,
                trainer,
                summary,
                clocks["cells"],
                timings={
                    "elapsed_us": (time.monotonic_ns() - began) // 1000,
                    "role_us": sum(n for n in native.charges if n is not None),
                    "unknown_role_intervals": native.charges.count(None),
                    "image_build_us": native.build_us,
                },
            )
            retain(local, report)
        bind_private(area, report)
        archive.parent.mkdir(parents=True, exist_ok=True)
        receipt = publish(archive, report, prior)
    progress(area, "complete", deficient_cells=len(receipt["deficits"]))
    return {
        "mechanical_status": "COMPLETE",
        "scientific_status": "PILOT_UNMEASURED",
        "reason": receipt["action"],
        "archive": str(archive.relative_to(folder)),
        "receipt": receipt,
        "gpu_dispatches": 0,
    }, 0
