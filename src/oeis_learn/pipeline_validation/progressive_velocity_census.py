"""Custodian-only CPU preparation from the original donor. No learner or device imports."""

import json
from pathlib import Path
import sys
import zipfile

from oeis_learn.generalization.progressive.campaign import learned_search_policy, velocity_policy
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.program_search_panels import reserve_velocity_families
from oeis_learn.generalization.progressive.shared_support import validated_rows
from oeis_learn.generalization.progressive.training_rows import matching_pairs
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import historical_row, read_inputs, require
from .progressive_research_inputs import checked_file
from .progressive_round028_audit import ORIGINAL
from .progressive_velocity_donor import AREA, PARENT, PILOT_REPORT, _archives, _verify_with_pin

SEED = 29
KINDS = ("matching_pairs", "families")


def quotas():
    synthetic = {s: 1 for s in STRATA if s != "oeis"}
    return {
        **{
            role: dict.fromkeys(STRATA, 32) for role in ("development", "replication", "assessment")
        },
        **{role: {"retention": 32} for role in ("development_retention", "replication_retention")},
        **{role: dict(synthetic) for role in ("practice", "teacher", "calibration")},
    }


def training_source(root):
    policy = velocity_policy()
    files, _ = _archives(root, policy)
    require(
        {"pilot", "qualification"} <= files.keys(),
        "pinned pilot and qualification archives required",
    )
    report = _verify_with_pin(root, policy["pilot_reader"], files["pilot"], files["qualification"])
    require(report.get("report") == PILOT_REPORT, "historical pilot replay differs")
    with zipfile.ZipFile(files["pilot"]) as archive:
        design = json.loads(archive.read("pilot.json"))["design"]
    area = confined(root, f"{PARENT}/{AREA}")
    freeze_path = confined(area, "freeze.json")
    freeze = load(freeze_path, maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(
        freeze == seal(freeze) and freeze["identity"] == design["private_freeze"],
        "pilot freeze differs from public commitment",
    )
    package = read_bound_record(area, freeze["training"], version="round028-pilot-training/v1")
    stream = package["stream"]
    require(
        stream == seal(stream)
        and stream["identity"] == freeze["stream"]
        and stream["seeds"] == freeze["seeds"],
        "ordinary training stream differs",
    )
    sources = [reference(root, p) for p in (files["pilot"], files["qualification"], freeze_path)]
    sources.append(
        dict(freeze["training"], path=str((area / freeze["training"]["path"]).relative_to(root)))
    )
    return freeze, package, sources


def practice_prompts():
    from oeis_learn.decoder.bigint_codec import encode_body
    from .progressive_velocity_worker import REHEARSAL_PROGRAMS, _disposable_rows, literal_controls

    rows, _ = _disposable_rows()
    sources = {tuple(encode_body(p)): p for p in REHEARSAL_PROGRAMS}
    prompts = [
        dict(source=sources[tuple(r["tokens"])], values=r["values"], stratum="practice")
        for r in rows.values()
    ]
    source, _, controls = literal_controls()
    prompts.extend(
        dict(source=source, values=list(v), stratum="practice") for v in controls.values()
    )
    return prompts


def prior_exposures(inputs, supply, calibration, training):
    require(
        calibration == seal(calibration) and calibration["registration"] == supply["registration"],
        "original calibration registration differs",
    )
    wanted = set(calibration["witnesses"].values())
    require(len(wanted) == len(calibration["witnesses"]), "duplicate calibration witness")
    held = {}
    for row in inputs["candidates"]:
        identity = digest(row)
        if identity in wanted:
            held[identity] = row
    require(held.keys() == wanted, "original calibration witnesses missing")
    history = list(inputs["historical"])
    support = inputs["support"]
    from oeis_learn.generalization.progressive.program_search_panels import reserved_family_rows

    selected = [*supply["witnesses"].values(), *held.values()]
    for row in reserved_family_rows(inputs["candidates"], selected):
        history.append(historical_row(row, support, "reservation", velocity=True))
    for row in training:
        history.append(
            historical_row(
                dict(row, id=row.get("id", row["draw_identity"])),
                support,
                "training",
                velocity=True,
            )
        )
    return history


def measure(root, kind):
    if kind not in KINDS:
        raise ValueError("unknown donor supply scope")
    freeze, training, sources = training_source(root)
    rows = training["rows"]
    if kind == "matching_pairs":
        # Use the original positive order, including its deliberate held-out omissions.
        shards = [rows[i : i + 4096] for i in range(0, len(rows), 4096)]
        indexed = validated_rows(shards, native_verified=True)
        stream = training["stream"]
        order = stream["order"]
        require(
            digest([indexed[key]["identity"] for key in order]) == stream["row_identities"],
            "original positive order differs",
        )
        measured = matching_pairs(shards, order, seed=stream["seeds"]["training_families"])
    else:
        campaign_path = confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json")
        campaign_ref = reference(root, campaign_path)
        original = read_bound_record(
            root, campaign_ref, identity=learned_search_policy()["campaign"]
        )
        refs = {r["name"]: r for r in original["payload"]["references"]}
        inputs = read_inputs(root, refs, velocity=True)
        stock = freeze["roster_source"]
        supply = read_bound_record(root, stock, version="round028-private-panel-supply/v1")
        panel_area = confined(root, stock["path"]).parent
        cal_path = confined(panel_area, "calibration-reservation.json")
        calibration_ref = reference(panel_area, cal_path)
        require(
            training["exposure_cut"]
            == digest(
                [inputs["census"], inputs["data"]["identity"], supply["identity"], calibration_ref]
            ),
            "pilot training exposure cut differs from original census and reservations",
        )
        selected = {r["id"] for roster in supply["allocations"]["panels"].values() for r in roster}
        require(selected == supply["witnesses"].keys(), "prior reserved panel witnesses missing")
        calibration = read_bound_record(panel_area, calibration_ref)
        history = prior_exposures(inputs, supply, calibration, rows)
        measured = reserve_velocity_families(
            inputs["candidates"],
            history,
            indexes=(),
            quotas=quotas(),
            seed=SEED,
            exposed_prompts=practice_prompts(),
        )
        sources += [campaign_ref, stock, reference(root, cal_path), *inputs["originals_used"]]
    originals = {r["path"]: r for r in sources}
    return measured, [originals[p] for p in sorted(originals)]


def worker(root, out, kind):
    measured, sources = measure(root, kind)
    retain(out / "measurement.json", measured)
    retain(out / "inputs.json", seal(dict(kind=kind, originals=sources)))


if __name__ == "__main__":
    try:
        worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    except Exception as exc:
        retain(
            Path(sys.argv[2]) / "failure.json",
            seal(dict(error=type(exc).__name__, reason=str(exc))),
        )
        raise
