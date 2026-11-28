"""Read outcome-blind round028 supply from the authenticated round027 preparation."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import (
    read_bound_record,
    reference,
    seal,
    validate,
)
from oeis_learn.generalization.progressive.generator import structure_id
from oeis_learn.generalization.progressive.prepared_support import transfer_support
from oeis_learn.generalization.progressive.research_sampling import source_eligibility

from .identity import confined
from .progressive_research_census import classify, validate_request
from .progressive_research_inputs import _json, checked_file, validate_manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def files_for(root, ref):
    path = checked_file(root, ref)
    census = _json(path)
    require(
        seal(census) == census
        and census["version"] == "progressive-research-census/v1"
        and census["diagnostic"] is False,
        "original native research census required",
    )
    files = {Path(r["path"]).name: (path.parent, r) for r in census["artifacts"]}
    require(
        len(files) == len(census["artifacts"]) == 4
        and set(files)
        == {
            "training-reservoir.json",
            "panel-reservations.json",
            "census.sqlite",
            "training-exposure.sqlite",
        },
        "incomplete native census artifacts",
    )
    return census, files


def reservations(census, files, request):
    roster = _json(checked_file(*files["panel-reservations.json"]))
    require(
        set(roster) == set(request["panels"] + request["retention_panels"])
        and "assessment" in roster
        and "development" in roster,
        "original panel roles differ",
    )
    result = []
    for panel, rows in roster.items():
        counts = Counter(row["stratum"] for row in rows)
        expected = ("parameter",) if panel in request["retention_panels"] else STRATA
        require(
            all(counts[s] == census["panels"][panel + "/" + s]["tasks"] for s in expected)
            and not counts.keys() - set(expected)
            and all(r["role"] == panel for r in rows),
            "reservation count or role differs",
        )
        result.extend(rows)
    require(len(result) == census["reserved_panel_rows"], "retained reservations differ")
    return result


def native_candidates(path, request, census, training, support):
    counters = Counter()
    result = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        require(
            json.loads(db.execute("SELECT value FROM binding").fetchone()[0]) == request
            and db.execute("SELECT value FROM native").fetchone()[0] == census["native"],
            "original native database registration differs",
        )
        for i, (ordinal, status, raw) in enumerate(
            db.execute("SELECT ordinal,status,row FROM draws ORDER BY ordinal")
        ):
            require(
                i == ordinal and status in {"TRAINING", "EVALUATION", "REJECTED"},
                "unfinished native census",
            )
            counters[status] += 1
            if status == "REJECTED":
                continue
            row = json.loads(raw)
            require(
                seal(row) == row
                and row["native_verified"] is True
                and (ordinal < request["training_draws"]) == (status == "TRAINING"),
                "invalid original native witness",
            )
            if status == "TRAINING":
                require(
                    row == training[counters[status] - 1],
                    "training reservoir differs from the native ledger",
                )
            else:
                require(
                    row["membership"] == "evaluation_candidate"
                    and classify(row, support) == row["stratum"],
                    "original evaluation witness or transfer stratum differs",
                )
                result.append(row)
    require(
        sum(counters.values()) == request["training_draws"] + request["evaluation_draws"]
        and len(training) == counters["TRAINING"]
        and all(n == census["counts"][k] for k, n in counters.items()),
        "native census count mismatch",
    )
    return result


def historical_row(row, support, role, *, legacy=False, velocity=False):
    result = {
        k: row[k]
        for k in (
            "id",
            "indices",
            "values",
            "source",
            "structure",
            "group",
            "family",
            "declared_groups",
        )
        if k in row
    }
    if velocity and "composition" in row:
        result["composition"] = row["composition"]
    if result.get("source"):
        structure = structure_id(result["source"])
        require(
            result.get("structure", structure) == structure or legacy, "structure witness changed"
        )
        result["structure"] = structure
    # Only the common witnessed gradient structures are transferable from prior training.
    # Legacy development/confirmation/assessment with no transfer declaration stays protected.
    result["stratum"] = (
        "parameter"
        if role == "training" and result.get("structure") in support["structures"]
        else row.get("stratum")
        if not legacy and row.get("stratum") in STRATA
        else "historical"
    )
    return result


def verify_origins(path, counts):
    """Only complete, authenticated originals may replace an older relation index."""
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        require(
            dict(db.execute("SELECT name, rows FROM origins")) == counts,
            "historical index origins differ from the reopened source rosters",
        )


def read_inputs(root, refs, *, include_protected=False, velocity=False):
    def authority(name, kind):
        ref = refs[name]
        require(
            (ref["role"], ref["kind"]) == ("custodian", kind), "incorrect historical source role"
        )
        return ref["artifact"]

    data = read_bound_record(
        root,
        authority("registered-research-data", "source"),
        version="progressive-registered-data/v1",
    )
    exposure_ref = authority("exposure", "groups")
    exposure_path = checked_file(root, exposure_ref)
    exposure = validate("exposure", read_bound_record(root, exposure_ref))
    require(
        exposure["purpose"] == "research" and exposure["track"] == "synthetic_strict",
        "research exposure required",
    )
    # The frozen exposure commits to the census. Authenticate the sibling materials
    # against that parent and independently match its training-support projection.
    materials_path = confined(root, exposure_path.parent.parent / "materials.json")
    materials = _json(materials_path)
    require(
        seal(materials) == materials
        and materials["version"] == "progressive-research-materials/v1"
        and materials["status"] == "COMPLETE"
        and materials["diagnostic"] is False
        and materials["exposure"] == exposure_ref
        and data["status"] == "COMPLETE",
        "registered corrected materials required",
    )
    census, files = files_for(root, materials["census"])
    request = validate_request(read_bound_record(root, materials["request"]))
    require(
        exposure["parents"] == [census["identity"]]
        and census["request"] == request["identity"]
        and census["native"] == materials["native"]
        and request["candidates"] == data["candidates"]
        and request["exposure"] == data["exposure"]
        and request["registration"] == data["identity"],
        "historical exposure or source parent mismatch",
    )
    prior, originals = files_for(root, census["panel_revision"]["original"])
    require(
        prior["request"] == census["request"]
        and prior["native"] == census["native"]
        and all(
            originals[k][1]["sha256"] == files[k][1]["sha256"]
            for k in files
            if k != "panel-reservations.json"
        ),
        "corrected census changed original witnesses",
    )
    old = reservations(prior, originals, request)
    corrected = reservations(census, files, request)
    training = _json(checked_file(*files["training-reservoir.json"]))
    package = read_bound_record(
        root, materials["support"], version="progressive-prepared-support/v1"
    )
    support, binding = transfer_support(package)
    require(binding == census["transfer_support"], "registered common training support changed")
    candidates = native_candidates(
        checked_file(*files["census.sqlite"]), request, census, training, support
    )
    sources = _json(checked_file(root, data["candidates"]), maximum=128 << 20)
    candidates.extend(dict(r, stratum="oeis") for r in sources if source_eligibility(r) is None)
    historical = [historical_row(r, support, "reservation", velocity=velocity) for r in [*old, *corrected]]
    protected = list(historical) if include_protected else []
    historical.extend(historical_row(r, support, "training", velocity=velocity) for r in training)
    manifest = validate_manifest(read_bound_record(root, data["manifest"]))
    old_data = manifest["prior_inputs"]
    sources = [("original." + item["name"], item["ref"], item["role"]) for item in old_data["originals"]]
    sources += [
        ("prior.probes", old_data["probes"], "training"),
        ("prior.observed", old_data["observed"], "legacy"),
        ("fresh-002.targets", manifest["fresh_002"]["targets"], "legacy"),
    ]
    origins = {}
    for name, ref, role in sources:
        additions = [
            historical_row(
                r,
                support,
                "training" if role == "training" else "legacy",
                legacy=True,
                velocity=velocity,
            )
            for r in _json(checked_file(root, ref))
        ]
        origins[name] = len(additions)
        historical.extend(additions)
        if include_protected and role != "training":
            protected.extend(additions)
    indexes = [
        checked_file(root, data["exposure"]),
        checked_file(*files["training-exposure.sqlite"]),
    ]
    require(
        exposure["payload"]["indexes"] == [reference(root, p) for p in indexes],
        "frozen exposure indexes differ",
    )
    originals_used = []
    if velocity:
        verify_origins(indexes[0], origins)
        verify_origins(indexes[1], {"prospective-support": len(training)})
        originals_used = [
            authority("registered-research-data", "source"),
            exposure_ref,
            reference(root, materials_path),
            materials["census"],
            materials["request"],
            materials["support"],
            census["panel_revision"]["original"],
            data["candidates"],
            data["manifest"],
            data["exposure"],
            *(ref for _, ref, _ in sources),
        ]
        for namespace, ref in [*files.values(), originals["panel-reservations.json"]]:
            originals_used.append(dict(ref, path=str((namespace / ref["path"]).relative_to(root))))
    return {
        "candidates": candidates,
        "historical": historical,
        **({"protected": protected} if include_protected else {}),
        **({"originals_used": originals_used} if velocity else {}),
        "indexes": indexes,
        "package": package,
        "support": support,
        "native": census["native"],
        "data": data,
        "census": census["identity"],
        "counts": {
            "original_reservations": len(old),
            "corrected_reservations": len(corrected),
            "native_training": len(training),
            "older_exposure_rows": len(historical) - len(old) - len(corrected) - len(training),
        },
    }


def extra_source(root, data, area):
    """Continue the original fixed source order once, retaining earlier work verbatim."""
    from .progressive_research_source import project

    original_ref = data["candidates"]
    parent = checked_file(root, original_ref).parent.parent
    record = _json(confined(root, parent / "source-projection.json"))
    require(
        seal(record) == record
        and record["status"] == "COMPLETE"
        and record["snapshot"] == data["source"]
        and reference(root, checked_file(parent, record["candidates"])) == original_ref,
        "original source selection changed",
    )
    start = record["selection"]["selected"]
    source = checked_file(root, data["source"])
    cut = _json(source, maximum=128 << 20)
    require(
        start == min(len(cut["records"]), record["selection"]["limit"])
        and record["selection"]["seed"] == 20261110,
        "original source interval differs",
    )
    if start == len(cut["records"]):
        return [], {"selected": 0, "remaining": 0}
    output = area / "extended-source"
    receipt = output / "source-projection.json"
    extended = _json(receipt) if receipt.is_file() else project(root, source, output, start=start)
    require(
        seal(extended) == extended
        and extended["status"] == "COMPLETE"
        and extended["snapshot"] == data["source"]
        and extended["selection"]["start"] == start,
        "incomplete additional indexed source cut",
    )
    rows = _json(checked_file(output, extended["candidates"]), maximum=128 << 20)
    selected = extended["selection"]["selected"]
    return [dict(r, stratum="oeis") for r in rows if source_eligibility(r) is None], {
        "selected": selected,
        "remaining": len(cut["records"]) - selected - start,
    }
