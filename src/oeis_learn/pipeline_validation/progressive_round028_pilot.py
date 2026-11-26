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
    RESIDENT_SUPERVISOR_GRACE_SECONDS,
    family_counts,
    frozen_family_sizes,
    pilot_archive,
    pilot_registration,
    resident_measurement_gaps,
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

LEGACY_AREA = "private/generator-pilot-11h"
AREA = "private/generator-pilot-11h-v2"
SHARE = "share/generator-pilot-11h-v2/round028-pilot.zip"
PARENT_WORKSTATION = "sha256:1901e97e71d146db415743f380715f4ad9e96da07f16424bd628663379a80ce9"


class PilotDeficit(ValueError):
    def __init__(self, code, **counts):
        super().__init__(code)
        self.code = code
        if not all(type(n) is int and n >= 0 for n in counts.values()):
            raise ValueError("diagnostic counts must be nonnegative integers")
        self.counts = counts


class PilotRole(NativeRoles):
    worker = "oeis_learn.pipeline_validation.progressive_pilot_worker"
    request_version = "round028-pilot-role/v1"
    producer_version = "round028-pilot-output/v1"
    fixed_attempt = True
    machine_bound_attempts = True
    repeatable_preparation = frozenset({"prepare", "gradient", "storage", "recovery"})

    def bind_physical(self, physical):
        from .progressive_round028_rehearsal import physical_scope

        machine = physical_scope(self.host, self.phase, machine=True)
        path = confined(self.area, "physical-" + self.phase + ".json", exists=False)
        if path.exists() and load(path) != seal({"scope": machine}):
            raise PilotDeficit(f"PILOT_{self.phase.upper()}_MACHINE_CHANGED")
        super().bind_physical(machine)

    def stage_options(self, role):
        return {"tmp_bytes": 16 << 20} if role == "storage" else {}

    def __call__(self, role, data, *, seconds, inputs=()):
        # Pin each phase's measured native identity independently of which cached stages ran.
        self.native = None
        value = super().__call__(role, data, seconds=seconds, inputs=inputs)
        bound = load(self.last_boundary)
        timing = read_bound_record(self.last_boundary.parent, bound["timing"])
        if timing.get("supervisor_wall_unknown"):
            self.last_wall_us = None
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
    stream = None
    initial_draws = max(512, (inputs["counts"]["native_training"] + 511) // 512 * 512)
    for start in range(0, law["max_draws"], 512):
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
        revise(
            area / "progress.json",
            seal(
                dict(
                    phase="prepare_strict_training",
                    registration=registration["identity"],
                    native_draws=start + 512,
                    disjoint_rows=len(rows),
                )
            ),
        )
        # Extend the same prespecified counter stream on a supply gap, before opening any pilot scores.
        if start + 512 < initial_draws or (
            start + 512 < law["max_draws"] and (start + 512) % initial_draws != 0
        ):
            continue
        require(len({r["draw_identity"] for r in rows}) == len(rows), "duplicate strict draw")
        probes = {}
        for row in rows:
            kind = "frontier" if {"loop", "select"} & set(row["operations"]) else "familiar"
            probes.setdefault(kind, dict(id=row["draw_identity"], role="training"))
        if len(rows) >= 32 and len(probes) == 2:
            stream = matched_stream(
                [rows[i : i + law["shard_rows"]] for i in range(0, len(rows), law["shard_rows"])],
                {"probes": probes},
                registration["identity"],
                0,
                law=law,
            )
            if stream["eligible_rows"]:
                break
    if not stream or not stream["eligible_rows"]:
        raise PilotDeficit(
            "STRICT_TRAINING_SIGNAL_DEFICIT",
            native_draws=start + 512,
            disjoint_rows=len(rows),
            eligible_rows=0,
            training_bins=len(probes),
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
            exposure_cut=digest(
                [
                    inputs["census"],
                    inputs["data"]["identity"],
                    supply["identity"],
                    reference(panel_area, panel_area / "calibration-reservation.json"),
                ]
            ),
            native_draws=start + 512,
            excluded_keys=len(deny),
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
    if code or result["mechanical_status"] != "COMPLETE":
        counts = {k: result[k] for k in ("required_bytes", "deficit_bytes") if k in result}
        if "free_bytes" in result.get("filesystem", {}):
            counts["free_bytes"] = result["filesystem"]["free_bytes"]
        raise PilotDeficit(result["reason"], **counts)
    return result["qualification_receipts"]


def accept_pilot(root, folder, store, area, cpu, gpu, package, freeze, recipe, weights, progress):
    from .pilot_clocks import CORRECT_FULL
    from .progressive_round028 import reopen
    from .progressive_round028_timing import original_channels, timing_cases
    from .progressive_pilot_recovery import qualification_role, resident_seconds

    from .progressive_pilot_qualification import exposed_scorer_control, resident_scale, score_choices

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
    progress("accept_frozen_pilot", role="gradient")
    cpu("gradient", data, seconds=900)
    checks["training_signal"] = cpu.last_producer["identity"]
    progress("accept_frozen_pilot", role="storage")
    cpu("storage", seal({"block": "qualify-disposable-storage"}), seconds=120)
    checks["storage"] = cpu.last_producer["identity"]
    progress("accept_frozen_pilot", role="recovery")
    cpu("recovery", seal({"block": "qualify-recovery-fixtures"}), seconds=3600)
    checks["recovery"] = cpu.last_producer["identity"]
    historical = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    prompts, truth = original_channels(root, timing_cases(historical)[:1])
    weight = dict(freeze["reference"], path="inputs/weights.pt")
    data = seal(
        dict(
            block="qualify-resident-search",
            arm="incumbent",
            inference_seed=freeze["seeds"]["initialization"],
            target=LOOKS[-1],
            rows=prompts,
            recipe=recipe,
            weights=weight,
            seconds=resident_seconds(len(prompts)),
        )
    )
    progress("accept_frozen_pilot", role="search")
    output = qualification_role(gpu, "search", data, seconds=data["seconds"] + RESIDENT_SUPERVISOR_GRACE_SECONDS, inputs=[(weights, "/work/inputs/weights.pt")])
    candidate = output["candidates"]
    generated = read_bound_record(gpu.last_output, candidate, version="discovery-candidates/v1")
    checks["search"] = gpu.last_producer["identity"]
    progress("accept_frozen_pilot", role="score")
    scored, _, _ = score_choices(cpu, data["block"], truth, generated,
                                 checked_file(gpu.last_output, candidate), data["seconds"])
    _, closed = score_projection(
        generated, scored, candidate["sha256"], prompts, weight, recipe, CORRECT_FULL,
        inference_seed=data["inference_seed"],
    )
    require(closed, "candidate-bound exposed qualification portfolio did not close")
    checks["score"] = cpu.last_producer["identity"]

    # Use only previously exposed historical terms for the cold, endpoint-sized
    # qualification. Alias repeated inputs so all 160 registered clocks are exercised.
    from oeis_learn.generalization.progressive.runtime_reservation import TRAINING_SECONDS
    exposed, exposed_truth = original_channels(root, timing_cases(historical))
    progress("accept_frozen_pilot", role="full_size_learner")
    training = freeze["training"]
    trained = qualification_role(gpu, "train", seal(dict(
        block="qualify-zero-state", arm=ARMS[0], target=0, previous=None, resume_charge=None,
        registration=freeze["registration"], training=dict(training, path="inputs/training.json"),
        package=package["identity"])), seconds=TRAINING_SECONDS,
        inputs=[(area / training["path"], "/work/inputs/training.json")])
    _, weights = checkpoint_inputs(gpu.last_output, trained)
    retain_training_lineage(area, freeze, trained["milestone"])
    weight = dict(trained["weights"], path="inputs/weights.pt")
    scale = resident_scale(cpu, gpu, exposed, exposed_truth, package["profile"], weight, weights,
                           freeze["seeds"]["initialization"], progress)
    if scale["selected_choices"] == 0:
        progress("accept_frozen_pilot", role="exposed_scorer_control")
        scale = seal(dict(scale, score_control=exposed_scorer_control(root, historical, cpu, scale["cases"])))
    checks["resident_scale"] = retain_resident_scale(area, freeze["registration"], scale)
    return checks


def retain_training_lineage(area, freeze, milestone):
    from oeis_learn.generalization.progressive.program_search_pilot import LINEAGE_FIELDS

    require(milestone["target"] == 0 and milestone["stream"] == freeze["stream"],
            "qualification checkpoint differs from registered training")
    retain(area / "qualified-lineage.json", {k: milestone[k] for k in LINEAGE_FIELDS})


def check_space(area, host, checkpoint_bytes, *, scale, share, cached=False, outstanding=None, verified=None, rosters=None):
    from .progressive_pilot_capacity import capacity

    records = capacity(area, host, checkpoint_bytes, scale=scale, share=share, cached=cached, measure=space,
                       outstanding=outstanding, verified=verified, rosters=rosters)
    for record in records:
        if record["deficit_bytes"] or record["deficit_inodes"]:
            raise PilotDeficit("PILOT_STORAGE_DEFICIT", **record)
    return records


def score_projection(generated, scored, candidates_sha, prompts, weights, recipe, policy, *, inference_seed=None):
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
        and clock.get("inference_seed") == inference_seed
        and clock.get("source") == digest(recipe)
        and clock.get("expected_portfolios") == len(prompts)
        and clock.get("missing_portfolios") == 0
        and type(clock["cost_complete"]) is bool
        and clock["cost_complete"] == all(r["cost_complete"] for r in clock["clocks"])
        and len(clock["clocks"]) == len(prompts)
        and {r["prompt"] for r in clock["clocks"]} == set(scored["outcomes"])
        and all(
            r == clock_record(**{k: r[k] for k in CLOCK_FIELDS}) and r["policy"] == policy
            and r["mode"] == "correct" and r["snapshot"] == generated["model"]
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
    full_allowance = inference_seed is None or all(
        c["effective_deadline"] == c["original_deadline"] for c in clock["clocks"]
    )
    return results, clock["cost_complete"] and full_allowance


# Kept public for callers that price a single role from original measurements.
from oeis_learn.generalization.progressive.program_search_timing import estimate_role


def cost_status(area):
    costs = [load(confined(area, p)) for p in sorted((area / "costs").glob("*.json"))]
    require(all(c == seal(c) for c in costs), "changed pilot cost")
    spent = sum(c["charged_us"] for c in costs)
    retries = [load(confined(area, path)) for path in sorted((area / "qualification-costs").glob("*.json"))]
    require(all(r == seal(r) for r in retries), "changed qualification retry cost")
    return dict(
        registered_us=SHIFT_US,
        qualification_retries=dict(roles=len(retries), gpu_lease_us=sum(r["lease_us"] for r in retries),
            known_wall_us=sum(r["wall_us"] or 0 for r in retries),
            unknown_wall_intervals=sum(r["wall_us"] is None for r in retries)),
        settled_role_us=spent,
        gpu_lease_us=sum(c["lease_us"] for c in costs),
        remaining_before_open_roles_us=max(0, SHIFT_US - spent),
        overrun_us=max(0, spent - SHIFT_US),
        settled_roles=len(costs),
        outstanding_roles=len(list((area / "reservations").glob("*.json"))) - len(costs),
    )


def publication_inputs(area, root=None):
    paths = [
        "registration.json",
        "freeze.json",
        "acceptance.json",
        "accounting/budget-binding.json",
        "accounting/budget.jsonl",
        "public-freeze.json",
        *(f"trials/pilot-{target}.json" for target in LOOKS if (area / f"trials/pilot-{target}.json").exists()),
    ]
    for pattern in (
        "costs/*.json",
        "roles/*/boundary.json",
        "qualification/stages/*/receipt.json",
        "admissions/*.json",
    ):
        paths.extend(str(p.relative_to(area)) for p in sorted(area.glob(pattern)))
    paths.extend(n for n in ("qualified-lineage.json", "prescore.json") if (area / n).is_file())
    files = [reference(area, confined(area, name)) for name in paths]
    # A boundary authenticates its producer, which can itself refer to another
    # file. Hash these private leaves while streaming large tensors from disk.
    for entry in sorted(area.glob("roles/*/boundary.json")):
        role = confined(area, entry).parent
        bound = load(entry)
        require(bound == seal(bound), "private role boundary differs")
        result = read_bound_record(role, bound["producer"])
        output = checked_file(role, bound["producer"]).parent
        for field in ("inspection", "lease", "timing"):
            if field in bound:
                checked_file(role, bound[field])
        for field in ("rows", "scores", "candidates"):
            if field in result.get("result", {}):
                read_bound_record(output, result["result"][field])
        if "checkpoint" in result.get("result", {}):
            checkpoint_inputs(output, result["result"])
    if root is not None:
        frozen = load(confined(area, "freeze.json"))
        checked_file(root, frozen["reference"])
        checked_file(root, frozen["roster_source"])
        checked_file(area, frozen["training"])
    leaves = []
    for pattern in ("training*.json", "predecessor.json", "reservations/*.json",
                    "qualification-costs/*.json", "qualification-scale.json", "resident-measurements.json",
                    "resident-clock-margins.json", "resident-score-control.json", "exposed-scorer-control/candidates.json",
                    "recovered/**/*", "completed/*.json", "shared-zero.json", "assignment-recovery/*.json", "roles/*/request.json",
                    "roles/*/data.json", "roles/*/attempt-*/start.json", "roles/*/attempt-*/end.json",
                    "roles/*/attempt-*/output/producer.json", "roles/*/attempt-*/output/case-boundaries/**/*",
                    "roles/*/attempt-*/output/accounting/**/*", "roles/*/attempt-*/output/checkpoint/**/*",
                    "roles/*/attempt-*/output/training-origin.json", "roles/*/attempt-*/output/scored/scores.json",
                    "roles/*/attempt-*/output/generation/candidates.json", "roles/*/attempt-*/output/generation/attempts/**/*"):
        leaves.extend(reference(area, file) for file in sorted(area.glob(pattern)) if file.is_file())
    return dict(boundaries=files, private_files=len(leaves), private_digest=digest(leaves))


def published_pilot(receipt):
    return dict(
        mechanical_status="COMPLETE",
        scientific_status="PILOT_INCOMPLETE" if receipt.get("unmeasured_looks") else "PILOT_MEASURED",
        archive=SHARE,
        receipt=receipt,
    ), 0


def retain_resident_scale(area, registration, scale):
    path = confined(area, "qualification-scale.json", exists=False)
    if not path.is_file():
        retain(path, scale)
        return scale["identity"]
    original = load(path)
    expected = original["identity"]
    previous = accepted_scale(area, {"identity": registration}, {"checks": {"resident_scale": expected}})
    if previous != scale:
        # Only the first segmented measurement can be replaced by a new cold attempt.
        # Later controllers can add measurements derived from those same retained clocks.
        if "cold_attempt" in previous:
            require(all(scale.get(k) == v for k, v in previous.items() if k != "identity"),
                    "original cold measurement changed")
        supplement = ("resident-score-control.json" if "minimum_case_allowance_us" in previous
                      else "resident-clock-margins.json")
        retain(area / supplement,
               seal(dict(registration=registration, original=expected, scale=scale)))
    return expected


def accepted_scale(area, registration, acceptance):
    original = load(confined(area, "qualification-scale.json"))
    require(original == seal(original) and original["identity"] == acceptance["checks"]["resident_scale"]
            and original["cases"] == 160, "qualification scale or acceptance differs")
    scale = original
    for name in ("resident-measurements.json", "resident-clock-margins.json", "resident-score-control.json"):
        extra = confined(area, name, exists=False)
        if not extra.is_file():
            continue
        measured = load(extra)
        require(measured == seal(measured) and measured["registration"] == registration["identity"]
                and measured["original"] == original["identity"]
                and measured["scale"] == seal(measured["scale"]), "supplemental resident measurement differs")
        supplemental = measured["scale"]
        if "cold_attempt" in scale:
            require(all(supplemental.get(k) == v for k, v in scale.items() if k != "identity"),
                    "supplement differs from the retained cold measurement")
        scale = supplemental
    return scale


def qualification_result(area, registration, acceptance, prices, controller, workstation, scientific_status=None, *, terminal=None):
    from oeis_learn.generalization.progressive.program_search_timing import projected_funding
    from .progressive_pilot_funding import funding_state

    last = terminal["looks"][-1]["target"] if terminal else None
    balance = funding_state(area, registration["identity"], prices, SHIFT_US, stopped_after=last)
    projected = projected_funding(prices, balance, registration["identity"])
    if scientific_status is None:
        scientific_status = "PILOT_INCOMPLETE" if balance["charges"] or balance["opened"] else "PILOT_UNMEASURED"
    scale = accepted_scale(area, registration, acceptance)
    require_resident_measurements(scale, registration)
    if projected > SHIFT_US:
        raise PilotDeficit("PILOT_ROLE_TIME_DEFICIT", required_us=projected, registered_us=SHIFT_US,
                           deficit_us=projected - SHIFT_US)
    return dict(mechanical_status="QUALIFIED", scientific_status=scientific_status,
                controller=controller, workstation=workstation,
                registration=registration["identity"], worker=registration["candidate"], acceptance=acceptance["identity"],
                **({"prescore": load(confined(area, "prescore.json"))} if (area / "prescore.json").is_file() else {}),
                **({"terminal": terminal} if terminal else {}),
                balance=balance, scale=scale, workspace_space=space(area), projected_us=projected, registered_us=SHIFT_US), 0


def require_resident_measurements(scale, registration):
    gaps = resident_measurement_gaps(scale, registration["portfolio_seconds"])
    if gaps:
        raise PilotDeficit("QUALIFICATION_MEASUREMENTS_INCOMPLETE", **gaps)


def registered_pilot_source(root, controller_source, controller_revision, area, parent):
    """Reopen the registered worker tree before deriving any seeds, images or requests."""
    import re
    import runpy

    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .progressive_round028 import git

    path = confined(area, "registration.json", exists=False)
    revision = controller_revision
    source = controller_source
    if path.is_file():
        saved = load(path)
        require(
            saved == seal(saved)
            and isinstance(saved.get("candidate"), str)
            and re.fullmatch("[0-9a-f]{40}", saved["candidate"]),
            "invalid saved pilot registration",
        )
        revision = saved["candidate"]
        if revision != controller_revision:
            require(
                git(root, "merge-base", "--is-ancestor", revision, controller_revision) == "",
                "the controller must descend from the registered pilot source",
            )
            helper = (
                controller_source
                / "specs/015-learned-program-search/validation/candidate_source.py"
            )
            source = runpy.run_path(str(helper))["materialize"](root, revision)
    registration = pilot_registration(parent, revision, digest(collect_snapshot(source)))
    retain(path, registration)
    return source, registration


def run_pilot(args, source, root, folder, revision, store, host_factory):
    from oeis_learn.generalization.progressive.contracts import _atomic_write
    from .progressive_study import generator_observation
    from .progressive_round028_rehearsal import candidate_image, physical_scope

    area = confined(folder, AREA, exists=False)
    area.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_raw, parent = parent_evidence(folder, store)
    old = confined(folder, LEGACY_AREA, exists=False)
    if (old / "registration.json").is_file():
        # A scientific worker change needs a separate freeze. Never grant a second research budget.
        _, predecessor = registered_pilot_source(root, source, revision, old, parent)
        if any((old / key).exists() for key in ("accounting", "reservations", "trials", "publication.json")):
            raise PilotDeficit("PREDECESSOR_RESEARCH_ALREADY_OPENED")
        retain(area / "predecessor.json", seal(dict(registration=predecessor["identity"],
                                                    original_parent=parent["report"])))
    controller = revision
    source, registration = registered_pilot_source(root, source, controller, area, parent)
    revision = registration["candidate"]
    if controller != revision:
        retain(
            area / "controllers" / (controller + ".json"),
            seal(
                dict(
                    version="round028-pilot-controller/v1",
                    controller=controller,
                    candidate=revision,
                    registration=registration["identity"],
                )
            ),
        )

    def progress(phase, **info):
        revise(
            area / "progress.json",
            seal(
                dict(
                    phase=phase,
                    registration=registration["identity"],
                    controller=controller,
                    candidate=revision,
                    **info,
                )
            ),
        )

    archive = confined(folder, SHARE, exists=False)
    publication = confined(area, "publication.json", exists=False)
    if publication.is_file():
        boundary = load(publication)
        require(
            boundary == seal(boundary)
            and boundary["registration"] == registration["identity"]
            and boundary["original_receipt"] == parent["workstation"]["identity"]
            and boundary["evidence"] == publication_inputs(area, root)
            and boundary["archive"] == reference(folder, archive),
            "changed pilot publication boundary",
        )
        receipt = verify_pilot(archive)
        require(receipt["report"] == boundary["report"], "published pilot receipt differs")
        progress("complete", report=receipt["report"], action=receipt["next_action"])
        if getattr(args, "action", None) == "qualify-pilot":
            freeze = load(confined(area, "freeze.json"))
            acceptance = load(confined(area, "acceptance.json"))
            prices = {(block, role): price for block, role, price in freeze["forecast"]}
            with zipfile.ZipFile(archive) as z:
                terminal = json.loads(z.read("pilot.json"))
            return qualification_result(area, registration, acceptance, prices, controller, parent["workstation"],
                                        published_pilot(receipt)[0]["scientific_status"], terminal=terminal)
        return published_pilot(receipt)
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
        rehearsal = json.loads(inner.read("rehearsal.json"))
        timing = json.loads(inner.read("timing.json"))
        from oeis_learn.generalization.progressive.program_search_timing import parent_plan

        plan = parent_plan(json.loads(inner.read("panels.json")), rehearsal)
    old_arms = rehearsal["hip"]["arms"]
    strata = {
        r["task"]: r["stratum"]
        for name in ("development", "development_retention")
        for r in panels[name]
    }
    from oeis_learn.generalization.progressive.program_search_timing import pilot_prices, remaining_price

    prices = pilot_prices(graph, strata, timing, rehearsal, plan)
    from .progressive_pilot_funding import funding_state
    from oeis_learn.generalization.progressive.program_search_timing import projected_funding

    carried = funding_state(area, registration["identity"], prices, SHIFT_US)
    projected_funding(prices, carried, registration["identity"])
    completed = set(map(tuple, carried["completed"]))
    if carried["shared_zero"]:
        completed.update((ARMS[1] + "-0", role) for role in ("search", "score"))
    largest = max(v["checkpoint_bytes"] for v in old_arms.values())
    rosters = {f"{arm}-{target}": {r["task"] for r in look["members"]}
               for target, looks in graph.items() for arm, look in looks.items()}
    scale = dict(rehearsal["original_scale"])
    if (area / "acceptance.json").is_file():
        scale["resident"] = accepted_scale(area, registration, load(confined(area, "acceptance.json")))
    space_args = dict(scale=scale, share=archive.parent, verified={}, rosters=rosters)
    check_space(area, host, largest, outstanding=set(prices) - completed, **space_args)
    progress("qualify_candidate")
    qualified = hardware_qualification(
        args, root, source, folder, store, revision, host_factory, area
    )
    for runner in (cpu, gpu):
        runner.host = host
        runner.image, runner.build_us, _ = candidate_image(
            host, source, area / "qualification", revision, runner.phase
        )
        require(
            runner.image == qualified[runner.phase]["image"],
            "research image differs from qualified image",
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
            roster_source=reference(root, panel_area / "supply.json"),
            predecessor=load(area / "predecessor.json")["registration"] if (area / "predecessor.json").is_file() else None,
            graph=graph,
            forecast=sorted([block, role, price] for (block, role), price in prices.items()),
            stream=package["stream"]["identity"],
            seeds=pair_seeds(registration["identity"], 0),
            images={"cpu": cpu.image, "hip": gpu.image},
            reference=reference(root, d_weights),
            recipe=digest(d_recipe),
        )
    )
    retain(area / "freeze.json", freeze)
    design = frozen_family_sizes(graph, registration["identity"], freeze["identity"], prices)
    retain(area / "public-freeze.json", design)
    progress("accept_frozen_pilot")
    check_space(area, host, largest, outstanding=set(prices) - completed, **space_args)
    checks = accept_pilot(
        root, folder, store, area, cpu, gpu, package, freeze, d_recipe, d_weights, progress
    )
    scale["resident"] = accepted_scale(area, registration, {"checks": checks})
    require_resident_measurements(scale["resident"], registration)
    # A reboot remeasures the candidate on the same machine. It cannot rewrite the original acceptance.
    current = seal(
        dict(
            version="round028-pilot-qualification/v1",
            freeze=freeze["identity"],
            qualification=qualified,
            checks=checks,
            images=freeze["images"],
            physical={r.phase: physical_scope(host, r.phase) for r in (cpu, gpu)},
        )
    )
    retain(area / "admissions" / (current["identity"][7:] + ".json"), current)
    facts = dict(
        version="round028-pilot-acceptance/v1",
        freeze=freeze["identity"],
        checks=checks,
        images=freeze["images"],
        physical={r.phase: physical_scope(host, r.phase, machine=True) for r in (cpu, gpu)},
    )
    path = area / "acceptance.json"
    if path.exists():
        acceptance = load(confined(area, path))
        accepted = read_bound_record(
            area, acceptance["qualification"], version="round028-pilot-qualification/v1"
        )
        require(
            acceptance == seal(dict(facts, qualification=acceptance["qualification"]))
            and all(accepted[k] == facts[k] for k in ("freeze", "checks", "images")),
            "accepted pilot scope differs",
        )
    else:
        acceptance = seal(
            dict(
                facts,
                qualification=reference(
                    area, area / "admissions" / (current["identity"][7:] + ".json")
                ),
            )
        )
        retain(path, acceptance)
    # A prior pilot with research work already recorded keeps its original evidence scope.
    began_research = any(any((area / key).glob("*")) for key in ("reservations", "trials", "accounting"))
    prescore_path = area / "prescore.json"
    prescore = None
    if prescore_path.exists() or not began_research:
        prescore = seal(dict(version="round028-pre-score/v1", registration=registration["identity"],
                             acceptance=acceptance["identity"], design=design["identity"],
                             lineage=load(confined(area, "qualified-lineage.json"))))
        retain(prescore_path, prescore)
    check_space(area, host, largest, outstanding=set(prices) - completed, **space_args)
    if getattr(args, "action", None) == "qualify-pilot":
        progress("qualified", acceptance=acceptance["identity"])
        return qualification_result(area, registration, acceptance, prices, controller, parent["workstation"])
    budgets = BudgetJournal(
        area / "accounting",
        registration["identity"],
        dict(dict.fromkeys(KEYS, 0), elapsed_us=SHIFT_US),
    )

    def settle(block, role, elapsed, wall_us=None, telemetry=None):
        key = block + "/" + role + "/0"
        saved = load(confined(area, "reservations/" + block + "-" + role + ".json"))
        charged = wall_us if wall_us is not None else saved["us"]
        require(
            saved == seal(saved) and saved["key"] == key and 0 <= elapsed <= charged,
            "original pilot lease differs",
        )
        telemetry = telemetry or {}
        retain(
            area / "costs" / (block + "-" + role + ".json"),
            seal(
                dict(
                    block=block,
                    role=role,
                    lease_us=elapsed,
                    wall_us=wall_us,
                    charged_us=charged,
                    covered_us=telemetry.get("covered_us", 0),
                    busy_us=telemetry.get("busy_us"),
                )
            ),
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
                monotonic_us=start["start_us"] + min(charged, saved["us"]),
                termination_proven=True,
            )

    def run(runner, role, data, mounts=()):
        # Persist original time reservations. Completed boundaries can be reverified after a reboot.
        assignment = area / "assignments" / (data["block"] + "-" + role + ".json")
        cached = False
        if assignment.is_file():
            from .progressive_pilot_recovery import assigned_request
            key = assigned_request(area, assignment)
            cached = (confined(area, "roles/" + key[7:], exists=False) / "boundary.json").is_file()
        check_space(area, host, largest, cached=cached, outstanding=set(prices)-completed, **space_args)
        key = data["block"] + "/" + role + "/0"
        saved = area / "reservations" / (data["block"] + "-" + role + ".json")
        if saved.exists():
            limit = load(confined(area, saved))["us"]
        else:
            limit = SHIFT_US - max(budgets.snapshot()["consumed"]["elapsed_us"], cost_status(area)["settled_role_us"])
            estimate = estimate_role(role, data, strata, timing, rehearsal, plan)
            base = "-".join(data["block"].split("-")[:2])
            needed = remaining_price(prices, completed, (base, role), estimate)
            if needed > limit or limit <= 0:
                raise PilotDeficit(
                    "PILOT_ROLE_TIME_DEFICIT",
                    registered_us=SHIFT_US,
                    required_us=needed,
                    next_role_us=estimate,
                    remaining_us=max(0, limit),
                    deficit_us=max(0, needed - limit),
                )
            retain(saved, seal(dict(key=key, us=limit, estimate_us=estimate)))
        # Each original supervisor interval has its own elapsed-time origin. Retain device occupancy separately.
        budgets.reserve(
            data["block"],
            role,
            0,
            dict(dict.fromkeys(KEYS, 0), elapsed_us=limit),
            boot="round028-supervised-role-duration/v1",
            monotonic_us=0,
        )
        # The shift admits new work. An admitted case or update keeps its own service clock.
        from oeis_learn.generalization.progressive.runtime_reservation import TRAINING_SECONDS

        stage_seconds = TRAINING_SECONDS if role == "train" else data["seconds"] + RESIDENT_SUPERVISOR_GRACE_SECONDS
        result = runner(role, seal(data), seconds=stage_seconds, inputs=mounts)
        if role == "search":
            completed.add(("-".join(data["block"].split("-")[:2]), role))
        settle(
            data["block"],
            role,
            runner.last_lease_us if runner is gpu else 0,
            runner.last_wall_us,
            result.get("telemetry"),
        )
        return result

    from oeis_learn.generalization.progressive.trials import TrialJournal

    trials = TrialJournal(
        area / "trials",
        study=registration["identity"],
        token_cap=LOOKS[-1],
        elapsed_cap_us=SHIFT_US,
    )
    from .progressive_pilot_funding import finish_role, sharing_receipt

    previous = {}
    published = []
    zero_executions = {}
    for target in LOOKS:
        looks = graph[str(target)]
        cuts, outcomes = {}, {}
        comparable = True
        for arm, look in looks.items():
            progress("pilot", arm=arm, target=target, role="train" if arm in ARMS else "search")
            if arm in ARMS:
                from .progressive_pilot_recovery import measured_training

                trained = measured_training(
                    run,
                    settle,
                    gpu,
                    arm,
                    target,
                    previous.get(arm, (None, [])),
                    training,
                    package["identity"],
                    registration["identity"],
                )
                completed.add((f"{arm}-{target}", "train"))
                cuts[arm] = trained["milestone"]
                mounts, weights = checkpoint_inputs(gpu.last_output, trained)
                finish_role(area, registration["identity"], f"{arm}-{target}", "train")
                previous[arm] = (trained["checkpoint"], mounts)
                recipe = package["profile"]
                weight = dict(trained["weights"], path="inputs/weights.pt")
            else:
                recipe, weights = d_recipe, d_weights
                weight = dict(freeze["reference"], path="inputs/weights.pt")
            prompts, truth = outcome_channels(look, supply["witnesses"])
            if target == 0:
                from oeis_learn.generalization.progressive.trials import reusable_zero_look

                cut = cuts[arm]
                zero_executions[arm] = dict(
                    weights=cut["initial"],
                    ordered_prompts=digest(prompts),
                    codec=cut["codec"],
                    inference_seed=look["seed"],
                    runtime=digest([freeze["images"], acceptance["physical"]]),
                    source=registration["source"],
                    profile=digest(recipe),
                    fit_and_choice=digest([weight["sha256"], look["profile"]]),
                    budgets=digest(
                        [look["attempts"], look["portfolio_us"], look["fit_us"], look["max_tokens"]]
                    ),
                    support=freeze["stream"],
                    comparator=digest("strict-random"),
                    role=digest("visible-only"),
                )
                if arm == ARMS[1] and reusable_zero_look(
                    cuts[ARMS[0]], cut, looks, zero_executions
                ):
                    retain(area / "shared-zero.json", sharing_receipt(cuts, looks, zero_executions))
                    outcomes[arm] = {}
                    completed.update((f"{arm}-{target}", role) for role in ("search", "score"))
                    continue
            from .progressive_pilot_recovery import measured_search

            outcomes[arm], closed = measured_search(
                run, settle, progress, cpu, gpu, look, recipe, weight, weights, prompts, truth
            )
            completed.update((f"{arm}-{target}", role) for role in ("search", "score"))
            for role in ("search", "score"):
                if (area / "costs" / f"{arm}-{target}-{role}.json").is_file() or any(
                    (area / "costs").glob(f"{arm}-{target}-*-{role}.json")
                ):
                    finish_role(area, registration["identity"], f"{arm}-{target}", role)
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
            zero_executions=zero_executions if target == 0 else None,
        )
        trials.observe_pilot(retained)
        if retained["zero_reused"]:
            outcomes[ARMS[1]] = outcomes[ARMS[0]]
        counts = family_counts(looks[ARMS[0]]["members"], outcomes, list(looks))
        published.append(
            dict(
                target=target,
                cuts=cuts,
                counts=counts,
                contrasts=summary(counts, len(looks)),
                comparable=comparable,
                zero_reused=bool(retained["zero_reused"]),
                action=retained["action"],
            )
        )
        if retained["action"] == "ENDPOINT_NOT_ADMITTED":
            break
    costs = [load(confined(area, path)) for path in sorted((area / "costs").glob("*.json"))]
    report = seal(
        dict(
            version="round028-generator-pilot/v3" if prescore else "round028-generator-pilot/v2",
            **({"prescore": prescore} if prescore else {}),
            registration=registration,
            freeze=design["identity"],
            design=design,
            acceptance=acceptance["identity"],
            looks=published,
            action=published[-1]["action"],
            research_gpu_us=sum(c["lease_us"] for c in costs),
            research_role_us=sum(c["charged_us"] for c in costs),
            costs=costs,
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
    retain(
        publication,
        seal(
            dict(
                version="round028-pilot-publication/v1",
                registration=registration["identity"],
                original_receipt=parent["workstation"]["identity"],
                evidence=publication_inputs(area, root),
                archive=reference(folder, archive),
                report=receipt["report"],
            )
        ),
    )
    progress("complete", report=receipt["report"], action=receipt["next_action"])
    return published_pilot(receipt)
