"""Own the eleven-hour child pilot on the original workstation and its GPU lease."""

import io
import json
import time
import zipfile
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.graph import generator_look
from oeis_learn.generalization.progressive.program_search_outcome import (
    archive_bytes,
    verify_outcome,
    verify_workstation,
    workstation_receipt,
)
from oeis_learn.generalization.progressive.program_search_pilot import (
    ARMS,
    LOOKS,
    SHIFT_US,
    family_counts,
    pilot_archive,
    pilot_registration,
    summary,
    verify_pilot,
)
from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
from oeis_learn.generalization.progressive.program_search_timing import PANEL, PANEL_REVISION
from oeis_learn.generalization.progressive.training_rows import pair_seeds
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_panel_sources import require
from .progressive_research_inputs import checked_file
from .progressive_round028_panels import NativeRoles, bind_private
from .progressive_round028_rehearsal import space

AREA = "private/generator-pilot-11h"
SHARE = "share/generator-pilot-11h/round028-pilot.zip"
PARENT_WORKSTATION = "sha256:1901e97e71d146db415743f380715f4ad9e96da07f16424bd628663379a80ce9"


class PilotRole(NativeRoles):
    worker = "oeis_learn.pipeline_validation.progressive_pilot_worker"
    request_version = "round028-pilot-role/v1"
    producer_version = "round028-pilot-output/v1"
    fixed_attempt = True

    def stage_options(self, role):
        return {"tmp_bytes": 16 << 20} if role == "storage" else {}

    def __call__(self, role, data, *, seconds, inputs=()):
        # Pin each phase's measured native identity independently of which cached stages ran.
        self.native = None
        value = super().__call__(role, data, seconds=seconds, inputs=inputs)
        retain(self.area / ("native-" + self.phase + ".json"), seal({"native": self.native}))
        return value


def research_rosters(supply):
    # Only membership commitments for the two reserved panels reach the graph compiler.
    return {
        name: [dict(task=r["id"], stratum=r["stratum"], group=r["group"]) for r in supply[name]]
        for name in ("development", "development_retention", "replication", "replication_retention")
    }


def outcome_channels(look, witnesses):
    from oeis_learn.generalization.progressive.cohorts import validate_prompt

    visible, truth = [], []
    for member in look["members"]:
        key = member["task"]
        row = witnesses[key]
        require(
            row["id"] == key and len(row["indices"]) == len(row["values"]) == 100,
            "frozen scoring roster differs",
        )
        visible.append(
            validate_prompt(dict(id=key, indices=row["indices"][:20], values=row["values"][:20]))
        )
        truth.append({k: row[k] for k in ("id", "indices", "values")})
    return visible, truth


def parent_evidence(folder, store):
    from .progressive_round028 import ARCHIVE

    term = store.state.get("terminal")
    require(
        term and store.state["route"] == "MATCHED_GENERATOR",
        "verified original terminal decision required",
    )
    closure = read_bound_record(folder, term["closure"], version="round028-terminal-boundary/v1")
    original = confined(folder, ARCHIVE).read_bytes()
    receipt = verify_outcome(io.BytesIO(original))
    require(
        receipt["report"] == term["report"] == closure["report"],
        "original journal and publication differ",
    )
    workstation = workstation_receipt(receipt, closure["replay"]["identity"], store.tip)
    require(workstation["identity"] == PARENT_WORKSTATION, "original workstation receipt differs")
    raw = archive_bytes({"workstation.json": encode(workstation), "round028.zip": original})
    return raw, verify_workstation(io.BytesIO(raw))


def original_inputs(root, folder, store):
    from .progressive_round028_audit import ORIGINAL, Donor
    from .progressive_round028_timing import PANEL_ARCHIVE
    from .progressive_research_inputs import _json

    donor = Donor(root, folder, store.campaign)
    original = donor.saved(
        ORIGINAL + "/prepare-research/frozen/campaign.json",
        donor.policy["campaign"],
        maximum=CAMPAIGN_DOCUMENT_BYTES,
    )
    refs = {r["name"]: r for r in original["payload"]["references"]}
    recipe, baseline = refs["recipe"], refs["retained-D-evaluation-weights"]
    require(
        (recipe["role"], recipe["kind"], baseline["role"], baseline["kind"])
        == ("custodian", "recipe", "generator", "checkpoint"),
        "frozen D recipe and weights required",
    )
    area = confined(folder, "private/panel-supply/" + PANEL_REVISION)
    with zipfile.ZipFile(confined(folder, PANEL_ARCHIVE)) as z:
        panel = json.loads(z.read("panels.json"))
    require(panel["identity"] == PANEL, "unknown parent panel supply")
    bind_private(area, panel)
    supply = load(confined(area, "supply.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(
        supply["calibrated"] and not supply["allocations"]["deficits"],
        "complete calibrated reservations required",
    )
    return (
        supply,
        refs,
        _json(checked_file(root, recipe["artifact"])),
        checked_file(root, baseline["artifact"]),
        area,
    )


def calibration_rows(area):
    rows = []
    for path in sorted(area.glob("roles/*/request.json")):
        req = load(confined(area, path))
        if req["role"] != "score":
            continue
        data = load(confined(area, path.with_name("data.json")), maximum=CAMPAIGN_DOCUMENT_BYTES)
        require(
            req == seal(req) and data == seal(data) and data["identity"] == req["data"],
            "calibration source changed",
        )
        for r in data["truth"]:
            rows.append(dict(r, source=data["witnesses"][r["id"]], stratum="calibration"))
    require(
        len({r["id"] for r in rows}) == len(rows) == 48,
        "complete disjoint calibration closure required",
    )
    return rows


def prepare_training(root, area, registration, cpu, refs, supply, panel_area):
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matched_stream
    from oeis_learn.generalization.progressive.generator import default_policy, draw
    from oeis_learn.generalization.progressive.prepared_panels import composition
    from .progressive_panel_sources import read_inputs
    from .progressive_research_inputs import ExposureIndex
    from .progressive_round028_support import training_package
    from .progressive_round028_audit import Donor
    from .progressive_round028 import SESSION

    target = area / "training.json"
    cut = area / "training-boundary.json"
    if cut.exists():
        saved = load(confined(area, cut))
        require(
            saved == seal(saved) and saved["registration"] == registration["identity"],
            "wrong sealed training cut",
        )
        return read_bound_record(area, saved["training"], version="round028-pilot-training/v1")
    inputs = read_inputs(root, refs, include_protected=True)
    protected = [*inputs["protected"], *supply["witnesses"].values(), *calibration_rows(panel_area)]
    deny = set()
    for row in protected:
        keys = set(ExposureIndex.keys(row))
        if row.get("stratum") in ("parameter", "retention"):
            keys = {p for p in keys if p[0] != "structure"}
        elif row.get("source"):
            keys.add(("composition", composition(row, allow_parameters=True)))
        deny.update(keys)
    seed = pair_seeds(registration["identity"], 0)["training_families"]
    law = default_policy(seed=seed)
    rows = []
    # Same native mechanical source as the historical strict training reservoir, with a new seed.
    for start in range(0, 8192, 512):
        result = cpu(
            "prepare",
            seal(dict(block=f"training-input-{start}", seed=seed, start=start, count=512)),
            seconds=512 * 4 + 120,
        )
        batch = read_bound_record(cpu.last_output, result["rows"], version="round028-pilot-rows/v1")
        require(
            (batch["seed"], batch["start"], batch["count"]) == (seed, start, 512)
            and len(batch["rows"]) + sum(batch["rejected"].values()) == 512,
            "training counter range differs",
        )
        for row in batch["rows"]:
            generated = draw(law, row["ordinal"])
            require(
                start <= row["ordinal"] < start + 512
                and all(
                    row[k] == generated[k]
                    for k in ("source", "source_policy", "draw_identity", "structure")
                ),
                "wrong seed-specific mechanical row",
            )
            candidate = dict(row, id=row["draw_identity"], stratum="training")
            if (
                not (
                    set(ExposureIndex.keys(candidate))
                    | {("composition", composition(candidate, allow_parameters=True))}
                )
                & deny
            ):
                rows.append(row)
    require(
        len({r["draw_identity"] for r in rows}) == len(rows) and len(rows) >= 32,
        "insufficient disjoint training rows",
    )
    probes = {}
    for row in rows:
        kind = "frontier" if {"loop", "select"} & set(row["operations"]) else "familiar"
        probes.setdefault(kind, dict(id=row["draw_identity"], role="training"))
    require(len(probes) == 2, "both strict training probe bins required")
    stream = matched_stream([rows], {"probes": probes}, registration["identity"], 0, law=law)
    require(
        stream["eligible_rows"] > 0, "no disjoint independently verified first-divergence labels"
    )
    folder = confined(root, SESSION)
    base = training_package(Donor(root, folder, load(confined(folder, "campaign.json"))), refs)
    profile = resolve_profile(
        base["profile"]["base_recipe"],
        shape="reference",
        seed=stream["seeds"]["initialization"],
        dropout=0,
    )
    package = seal(
        dict(
            version="round028-pilot-training/v1",
            stream=stream,
            rows=rows,
            profile=profile,
            microbatch=base["update_policy"]["microbatch_size"],
            exposure_cut=digest(sorted(deny)),
        )
    )
    retain(target, package)
    retain(cut, seal(dict(registration=registration["identity"], training=reference(area, target))))
    return package


def checkpoint_inputs(output, result):
    from oeis_learn.generalization.contracts import artifact_refs
    from oeis_learn.generalization.state import restore

    base = confined(output, "checkpoint")
    ref = result["checkpoint"]
    manifest = read_bound_record(base, ref)
    restored = restore(manifest, contract=result["contract"], cap=result["cap"], root=base)
    require(
        restored["ledger"].consumed == result["milestone"]["charged"]
        and restored["states"]["weights"] == result["weights"]
        and ref["sha256"] == result["milestone"]["parent"],
        "original full checkpoint closure differs",
    )
    mounts = [
        (checked_file(base, r), "/work/inputs/previous/" + r["path"])
        for r in [ref, *artifact_refs(manifest["states"])]
    ]
    return mounts, checked_file(base, result["weights"])


def hardware_qualification(args, root, source, folder, store, revision, host_factory, area):
    from .progressive_round028_rehearsal import rehearse

    result, code = rehearse(
        args,
        root,
        source,
        folder,
        store,
        revision,
        {"decision": {"action": "MATCHED_GENERATOR"}},
        host_factory=host_factory,
        output_area=area / "qualification",
    )
    require(
        not code and result["mechanical_status"] == "COMPLETE",
        "candidate CPU/native or HIP qualification failed",
    )
    records = {}
    for path in sorted((area / "qualification/stages").glob("*/receipt.json")):
        request = load(confined(area, path.with_name("request.json")))
        boundary = load(confined(area, path))
        if request["image"] == result[request["stage"]]["image"]:
            records[request["stage"]] = boundary["identity"]
    require(set(records) == {"cpu", "hip"}, "missing original candidate qualification receipts")
    return records


def accept_pilot(root, folder, store, area, cpu, gpu, package, freeze, recipe, weights):
    from .pilot_clocks import CORRECT_FULL
    from .progressive_round028 import reopen
    from .progressive_round028_timing import original_channels, timing_cases

    checks = {}
    entry = package["stream"]["pairs"][0]
    rows = {r["draw_identity"]: r for r in package["rows"]}
    data = seal(
        dict(
            block="qualify-training-signal",
            profile=package["profile"],
            sample=dict(
                positive=rows[entry["positive_row"]],
                negative=rows[entry["negative_row"]],
                label=entry["label"],
            ),
        )
    )
    cpu("gradient", data, seconds=900)
    checks["training_signal"] = cpu.last_producer["identity"]
    cpu("storage", seal({"block": "qualify-disposable-storage"}), seconds=120)
    checks["storage"] = cpu.last_producer["identity"]
    cpu("recovery", seal({"block": "qualify-recovery-fixtures"}), seconds=3600)
    checks["recovery"] = cpu.last_producer["identity"]
    historical = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    prompts, truth = original_channels(root, timing_cases(historical)[:1])
    weight = dict(freeze["reference"], path="inputs/weights.pt")
    data = seal(
        dict(
            block="qualify-resident-search",
            arm="incumbent",
            target=LOOKS[-1],
            rows=prompts,
            recipe=recipe,
            weights=weight,
            seconds=180,
        )
    )
    output = gpu("search", data, seconds=300, inputs=[(weights, "/work/inputs/weights.pt")])
    candidate = output["candidates"]
    generated = read_bound_record(gpu.last_output, candidate, version="discovery-candidates/v1")
    checks["search"] = gpu.last_producer["identity"]
    ref = dict(candidate, path="inputs/candidates.json")
    out = cpu(
        "score",
        seal(dict(block="qualify-resident-search", rows=truth, candidates=ref, seconds=180)),
        seconds=300,
        inputs=[(checked_file(gpu.last_output, candidate), "/work/inputs/candidates.json")],
    )
    scored = read_bound_record(cpu.last_output, out["scores"], version="discovery-scoring/v1")
    _, closed = score_projection(
        generated, scored, candidate["sha256"], prompts, weight, recipe, CORRECT_FULL
    )
    require(closed, "candidate-bound exposed qualification portfolio did not close")
    checks["score"] = cpu.last_producer["identity"]
    return checks


def check_space(area, host, checkpoint_bytes):
    measured = space(area)
    # Six pinned model cuts and one simultaneous save/restore. Include the observed original report workload.
    needed = (
        host.policy["free_disk_floor_bytes"] + 7 * checkpoint_bytes + 2 * (209010758 + 56568256)
    )
    inodes = 7 * 13 + 2 * 160 * 8 * 4 + 352
    if measured["free_bytes"] < needed or measured["free_inodes"] < inodes:
        raise ValueError(
            f"pilot storage deficit: {max(0, needed - measured['free_bytes'])} bytes, "
            f"{max(0, inodes - measured['free_inodes'])} inodes"
        )
    return dict(**measured, required_bytes=needed, required_inodes=inodes)


def score_projection(generated, scored, candidates_sha, prompts, weights, recipe, policy):
    from .pilot_clocks import clock_record
    from oeis_learn.generalization.progressive.program_search_costing import CLOCK_FIELDS

    require(
        generated["checkpoint"] == weights["sha256"]
        and generated["recipe"] == digest(recipe)
        and generated["prompts"] == digest(sorted(prompts, key=lambda r: r["id"]))
        and scored["candidates"] == candidates_sha
        and scored["denominator"] == len(prompts)
        and set(generated["records"]) == set(scored["outcomes"]) == {r["id"] for r in prompts},
        "candidate, truth, weight or ordered prompt substitution",
    )
    clock = generated["clock"]
    require(
        seal(clock) == clock
        and clock["policy"] == policy
        and clock["work"] == generated["model"]
        and len(clock["clocks"]) == len(prompts)
        and {r["prompt"] for r in clock["clocks"]} == set(scored["outcomes"])
        and all(
            r == clock_record(**{k: r[k] for k in CLOCK_FIELDS}) and r["policy"] == policy
            for r in clock["clocks"]
        ),
        "original registered case clocks required",
    )
    results = {}
    for key, modes in scored["outcomes"].items():
        row = modes["equal_budget"]
        solved = row["success"] and row["scoring"]["complete"]
        complete = bool(solved or row["complete"])
        results[key] = {
            "outcome": "success" if solved else "failure" if complete else "unresolved",
            "complete": complete,
        }
    return results, clock["cost_complete"]


def run_pilot(args, source, root, folder, revision, store, host_factory):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import _atomic_write
    from .progressive_study import generator_observation
    from .progressive_round028_rehearsal import candidate_image, physical_scope

    area = confined(folder, AREA, exists=False)
    area.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_raw, parent = parent_evidence(folder, store)
    registration = pilot_registration(parent, revision, digest(collect_snapshot(source)))
    retain(area / "registration.json", registration)

    def progress(phase, **info):
        revise(
            area / "progress.json",
            seal(dict(phase=phase, registration=registration["identity"], **info)),
        )

    archive = confined(folder, SHARE, exists=False)
    progress("check_original_inputs")
    supply, refs, d_recipe, d_weights, panel_area = original_inputs(root, folder, store)
    panels = research_rosters(supply["allocations"]["panels"])
    graph = {
        str(target): {
            arm: generator_look(
                panels,
                registration=registration["identity"],
                pair=0,
                arm=arm,
                target=target,
                split="development",
            )
            for arm in ((*ARMS, "incumbent") if target == LOOKS[-1] else ARMS)
        }
        for target in LOOKS
    }
    cpu, gpu = (
        PilotRole(args, source, root, revision, area, registration, host_factory) for _ in range(2)
    )
    gpu.phase = "hip"
    host = host_factory(args, source_root=source, workspace=root, revision=revision)
    with (
        zipfile.ZipFile(io.BytesIO(parent_raw)) as outer,
        zipfile.ZipFile(io.BytesIO(outer.read("round028.zip"))) as inner,
    ):
        old_arms = json.loads(inner.read("rehearsal.json"))["hip"]["arms"]
    largest = max(v["checkpoint_bytes"] for v in old_arms.values())
    check_space(area, host, largest)
    progress("qualify_candidate")
    qualified = hardware_qualification(
        args, root, source, folder, store, revision, host_factory, area
    )
    for runner in (cpu, gpu):
        runner.host = host
        runner.image, runner.build_us, _ = candidate_image(
            host, source, area, revision, runner.phase
        )
    progress("prepare_strict_training")
    package = prepare_training(root, area, registration, cpu, refs, supply, panel_area)
    training = reference(area, area / "training.json")
    freeze = seal(
        dict(
            version="round028-pilot-freeze/v1",
            registration=registration["identity"],
            training=training,
            panels=digest(panels),
            roster=reference(panel_area, panel_area / "supply.json"),
            graph=graph,
            stream=package["stream"]["identity"],
            seeds=pair_seeds(registration["identity"], 0),
            images={"cpu": cpu.image, "hip": gpu.image},
            reference=reference(root, d_weights),
            recipe=digest(d_recipe),
        )
    )
    retain(area / "freeze.json", freeze)
    progress("accept_frozen_pilot")
    checks = accept_pilot(root, folder, store, area, cpu, gpu, package, freeze, d_recipe, d_weights)
    acceptance = seal(
        dict(
            version="round028-pilot-acceptance/v1",
            freeze=freeze["identity"],
            qualification=digest(qualified),
            checks=checks,
            images=freeze["images"],
            physical={r.phase: physical_scope(host, r.phase) for r in (cpu, gpu)},
        )
    )
    retain(area / "acceptance.json", acceptance)
    budgets = BudgetJournal(
        area / "accounting",
        registration["identity"],
        dict(dict.fromkeys(KEYS, 0), elapsed_us=SHIFT_US),
    )

    def settle(block, role, elapsed):
        key = block + "/" + role + "/0"
        saved = load(confined(area, "reservations/" + block + "-" + role + ".json"))
        require(
            saved == seal(saved) and saved["key"] == key and 0 <= elapsed <= saved["us"],
            "original pilot lease differs",
        )
        original = budgets.reserve(
            block,
            role,
            0,
            dict(dict.fromkeys(KEYS, 0), elapsed_us=saved["us"]),
            boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            monotonic_us=time.monotonic_ns() // 1000,
        )
        if not original["committed"]:
            start = original["reserved"]
            budgets.commit(
                key,
                boot=start["boot"],
                monotonic_us=start["start_us"] + elapsed,
                termination_proven=True,
            )

    def run(runner, role, data, mounts=()):
        # Persist original time reservations. Completed boundaries can be reverified after a reboot.
        check_space(area, host, largest)
        key = data["block"] + "/" + role + "/0"
        saved = area / "reservations" / (data["block"] + "-" + role + ".json")
        if saved.exists():
            limit = load(confined(area, saved))["us"]
        else:
            limit = SHIFT_US - budgets.snapshot()["consumed"]["elapsed_us"] if runner is gpu else 0
            require(runner is cpu or limit > 0, "eleven-hour allocation spent")
            retain(saved, seal(dict(key=key, us=limit)))
        now = time.monotonic_ns() // 1000
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        reservation = budgets.reserve(
            data["block"],
            role,
            0,
            dict(dict.fromkeys(KEYS, 0), elapsed_us=limit),
            boot=boot,
            monotonic_us=now,
        )
        stage_seconds = (
            max(1, limit // 1_000_000) if runner is gpu else data.get("seconds", 3600) + 120
        )
        result = runner(role, seal(data), seconds=stage_seconds, inputs=mounts)
        if not reservation["committed"]:
            original = reservation["reserved"]
            elapsed = runner.last_lease_us if runner is gpu else 0
            require(elapsed <= limit, "measured GPU lease exceeded remaining allocation")
            budgets.commit(
                key,
                boot=original["boot"],
                monotonic_us=original["start_us"] + elapsed,
                termination_proven=True,
            )
        return result

    previous = {}
    published = []
    for target in LOOKS:
        looks = graph[str(target)]
        cuts, outcomes = {}, {}
        comparable = True
        for arm, look in looks.items():
            block = f"{arm}-{target}"
            progress("pilot", arm=arm, target=target, role="train" if arm in ARMS else "search")
            if arm in ARMS:
                prior, prior_mounts = previous.get(arm, (None, []))
                trained = run(
                    gpu,
                    "train",
                    dict(
                        block=block,
                        arm=arm,
                        target=target,
                        previous=prior,
                        registration=registration["identity"],
                        training=dict(training, path="inputs/training.json"),
                        package=package["identity"],
                    ),
                    [(area / "training.json", "/work/inputs/training.json"), *prior_mounts],
                )
                cuts[arm] = trained["milestone"]
                mounts, weights = checkpoint_inputs(gpu.last_output, trained)
                previous[arm] = (trained["checkpoint"], mounts)
                recipe = package["profile"]
                weight = dict(trained["weights"], path="inputs/weights.pt")
            else:
                recipe, weights = d_recipe, d_weights
                weight = dict(freeze["reference"], path="inputs/weights.pt")
            prompts, truth = outcome_channels(look, supply["witnesses"])
            from .progressive_pilot_recovery import measured_search

            outcomes[arm], closed = measured_search(
                run, settle, progress, cpu, gpu, look, recipe, weight, weights, prompts, truth
            )
            comparable &= closed
        retained = generator_observation(
            panels,
            outcomes,
            cuts,
            registration=registration["identity"],
            pair=0,
            target=target,
            funded=target != LOOKS[-1],
            comparable_cost=comparable,
        )
        retain(area / f"observation-{target}.json", retained)
        counts = family_counts(looks[ARMS[0]]["members"], outcomes, list(looks))
        published.append(
            dict(
                target=target,
                cuts=cuts,
                counts=counts,
                contrasts=summary(counts, len(looks)),
                comparable=comparable,
                action=retained["action"],
            )
        )
        require(
            target == LOOKS[-1] or retained["action"] == "CONTINUE_TO_RESERVED_ENDPOINT",
            "intermediate training exposures diverged",
        )
    report = seal(
        dict(
            version="round028-generator-pilot/v1",
            registration=registration,
            freeze=freeze["identity"],
            acceptance=acceptance["identity"],
            looks=published,
            action=published[-1]["action"],
            research_gpu_us=budgets.snapshot()["consumed"]["elapsed_us"],
            verification="private_choices_and_checkpoints",
        )
    )
    content = pilot_archive(report, parent_raw)
    receipt = verify_pilot(io.BytesIO(content))
    if archive.exists():
        require(archive.read_bytes() == content, "pilot publication differs on replay")
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(archive, content)
    progress("complete", report=receipt["report"], action=receipt["next_action"])
    return dict(
        mechanical_status="COMPLETE",
        scientific_status="PILOT_MEASURED",
        archive=SHARE,
        receipt=receipt,
    ), 0
