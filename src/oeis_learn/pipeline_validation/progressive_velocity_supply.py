"""Separate custodian measurements from the read-only round029 preflight."""

from contextlib import ExitStack

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.program_search_panels import reserve_velocity_families
from oeis_learn.generalization.progressive.training_rows import matching_pairs
from oeis_learn.generalization.progressive.velocity_records import SESSION, open_stored, physical_root_id
from .identity import confined, validate
from .preparation_recovery import retain, revise
from .progressive_research_inputs import ExposureIndex, checked_file


FORMATS = {
    "families": ("round029-census-input/v1", "round029-family-reservations/v1"),
    "matching_pairs": ("round029-pair-input/v1", "round029-training-pairs/v1"),
}


def _area(root, kind):
    if kind not in FORMATS:
        raise ValueError("unknown supply measurement")
    return confined(root, f"run-state/discovery/{SESSION}/preparation/supply/{kind}", exists=False)


def _registration(root, kind, source):
    return seal(dict(version="round029-supply-registration/v1", study=SESSION, kind=kind,
                     state_root_id=physical_root_id(root), policy=digest(velocity_policy()), source=source))


def _source(root, kind, registration):
    if registration != _registration(root, kind, registration["source"]):
        raise ValueError("supply registration, policy or physical root differs")
    return open_stored(root, registration["source"], owner="custodian", format=FORMATS[kind][0])


def register_supply(root, kind, source):
    """Commit custodian input bytes before measuring them. A new scope has its own receipt."""
    area = _area(root, kind)
    registration = _registration(root, kind, source)
    _source(root, kind, registration)
    area.mkdir(parents=True, exist_ok=True)
    with RunStore.owner(area):
        path = confined(area, "registration.json", exists=False)
        if not path.exists() or load(path) != registration:
            revise(path, registration)
    return registration


def _measure(root, kind, data, client):
    if kind == "families":
        validate(data, {"version", "candidates", "historical", "indexes", "quotas", "seed"}, FORMATS[kind][0])
        with ExitStack() as stack:
            indexes = [stack.enter_context(ExposureIndex(checked_file(root, ref), readonly=True))
                       for ref in data["indexes"]]
            return reserve_velocity_families(data["candidates"], data["historical"], indexes=indexes,
                                             quotas=data["quotas"], seed=data["seed"])
    validate(data, {"version", "shards", "order", "seed"}, FORMATS[kind][0])
    shards = [load(checked_file(root, ref), maximum=ref["bytes"]) for ref in data["shards"]]
    return matching_pairs(shards, data["order"], seed=data["seed"], client=client)


def measure_supply(root, kind, *, client=None):
    """Run in funded preparation. Reopening a completed measurement never repeats fits."""
    area = _area(root, kind)
    with RunStore.owner(area):
        reg = load(confined(area, "registration.json"))
        path = _source(root, kind, reg)
        target = confined(area, reg["identity"][7:], exists=False)
        receipt = confined(area, target / "receipt.json", exists=False)
        if not receipt.exists():
            data = load(path, maximum=reg["source"]["artifact"]["bytes"])
            measurement = _measure(root, kind, data, client)
            artifact = confined(area, target / (measurement["identity"][7:] + ".json"), exists=False)
            retain(artifact, measurement)
            record = seal(dict(version="round029-supply-measurement/v1", registration=reg["identity"],
                               result=dict(format=FORMATS[kind][1], owner="custodian", artifact=reference(root, artifact))))
            retain(receipt, record)
        return _read_measurement(root, area, kind, reg)


def _read_measurement(root, area, kind, reg):
    record = load(confined(area, reg["identity"][7:] + "/receipt.json"))
    validate(record, {"version", "registration", "result"}, "round029-supply-measurement/v1")
    if record["registration"] != reg["identity"]:
        raise ValueError("supply measurement belongs to another registered scope")
    result = record["result"]
    path = open_stored(root, result, owner="custodian", format=FORMATS[kind][1])
    measured = load(path, maximum=result["artifact"]["bytes"])
    expected = area / reg["identity"][7:] / (measured["identity"][7:] + ".json")
    if path != expected or measured != seal(measured) or measured["version"] != FORMATS[kind][1]:
        raise ValueError("registered supply measurement differs")
    return measured, record["identity"]


def _counts(kind, measured):
    if kind == "matching_pairs":
        pairs = measured["pairs"]
        counts = measured["dispositions"]
        eligible = len(pairs)
        families = len({pair["family"] for pair in pairs.values()})
        if (eligible != measured["eligible_rows"] or families != measured["eligible_families"]
                or counts.get("eligible", 0) != eligible or sum(counts.values()) != measured["positive_rows"]):
            raise ValueError("pair supply receipt disagrees with measured rows")
        return dict(count=eligible, eligible_families=families, positive_rows=measured["positive_rows"],
                    excluded_rows=measured["positive_rows"] - eligible, dispositions=counts)
    by_role = {}
    families = set()
    for role, quotas in measured["quotas"].items():
        cases = measured["panels"][role]
        families.update(r["family"] for r in cases)
        by_role[role] = {}
        for stratum in quotas:
            members = [row for row in cases if row["stratum"] == stratum]
            by_role[role][stratum] = dict(cases=len(members), families=len({r["family"] for r in members}))
    return dict(count=len(families), by_role=by_role, eligible_cases=measured["eligible"],
                eligible_components=measured["components"], excluded_cases=measured["excluded"],
                deficits=measured["deficits"], provenance_unavailable=measured["provenance_unavailable"],
                provenance_reason=measured["provenance_reason"])


def supply_inventory(root):
    """Report committed counts as of their input identities. No lock, fit, write or device."""
    inventory = {}
    for kind in FORMATS:
        try:
            area = _area(root, kind)
            if area.is_dir() and not (area / "registration.json").exists() and (area / "funded.json").is_file():
                from .progressive_velocity_preparation import funded_inventory

                inventory[kind] = funded_inventory(root, kind, load(confined(area, "funded.json")))
                continue
            path = confined(area, "registration.json") if area.is_dir() else None
            if path is None:
                raise FileNotFoundError
            reg = load(path)
            source = _source(root, kind, reg)
            inputs = load(source, maximum=reg["source"]["artifact"]["bytes"])
            if inputs != seal(inputs) or inputs["version"] != FORMATS[kind][0]:
                raise ValueError("registered custodian source differs")
            # Hash the referenced originals without repeating a fit or opening an index for writing.
            for ref in inputs["shards" if kind == "matching_pairs" else "indexes"]:
                checked_file(root, ref)
            measured, receipt = _read_measurement(root, area, kind, reg)
            inventory[kind] = dict(**_counts(kind, measured), registration=reg["identity"],
                                   measurement=receipt, reason=None)
        except FileNotFoundError:
            inventory[kind] = dict(count=None, reason="registered custodian input or completed measurement unavailable")
        except (KeyError, TypeError, ValueError, OSError):
            inventory[kind] = dict(count=None, reason="custodian input or measurement failed verification")
    return inventory
