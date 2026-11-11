"""Append-only relation access. New names/seeds never remove historical exposure."""

from .contracts import validate, record

ROLES = ("training", "development", "confirmation", "assessment")


def groups(events, group, *, endpoint, structure_sharing=False):
    # Relations are declared detector identities, not an all-domain equivalence proof.
    related = {group}
    changed = True
    while changed:
        changed = False
        for event in events:
            links = set(event["relations"]) | {event["group"]}
            if structure_sharing and endpoint == "parameter":
                links = {x for x in links if not x.startswith("structure:")}
            if links & related and not links <= related:
                related |= links
                changed = True
    return related


def exposed(registry, group, *, endpoint, role, structure_sharing=False):
    r = validate("exposure", registry)
    events = r["payload"]["events"]
    related = groups(events, group, endpoint=endpoint, structure_sharing=structure_sharing)
    matches = [e for e in events if (set(e["relations"]) | {e["group"]}) & related]
    if "fresh-002" in group or any("fresh-002" in e["source"] for e in matches):
        return True
    # Development is intentionally reusable, never described as fresh.
    if role == "development":
        return bool(matches)
    return bool(matches)


def append(registry, events, *, created_at):
    previous = validate("exposure", registry)
    payload = previous["payload"]
    return record(
        "exposure",
        dict(
            previous=previous["identity"],
            detector_limits=payload["detector_limits"],
            events=payload["events"] + list(events),
            **({'indexes':payload['indexes']} if 'indexes' in payload else {}),
        ),
        created_at=created_at,
        parents=[previous["identity"]],
        purpose=previous["purpose"],
        track=previous["track"],
    )


def validate_extension(previous, current):
    old = validate("exposure", previous)
    new = validate("exposure", current)
    n = len(old["payload"]["events"])
    if (
        new["payload"]["previous"] != old["identity"]
        or new["payload"]["events"][:n] != old["payload"]["events"]
        or new["payload"]["detector_limits"] != old["payload"]["detector_limits"]
        or new['payload'].get('indexes',[])[:len(old['payload'].get('indexes',[]))] != old['payload'].get('indexes',[])
    ):
        raise ValueError("exposure history cannot be erased")
    return new


def detect(records, *, window=20):
    """Custodian-only exact finite windows, canonical structures and source links.

    Every row has a complete indexed finite series. Overlap includes prefix/shift
    matches, not approximate or algebraic semantic equivalence. The same detectors
    apply to historical exposure and newly prepared panels.
    """
    from oeis_learn.synthesis.contracts import canonical_integer
    from .contracts import digest

    if type(window) is not int or not 20 <= window <= 100:
        raise ValueError("finite overlap window")
    result = []
    for row in records:
        if set(row) != {"id", "indices", "values", "structure", "composition", "relations"}:
            raise ValueError("closed detector row")
        if not len(row["indices"]) == len(row["values"]) == 100:
            raise ValueError("100 indexed terms required")
        for value in row["indices"] + row["values"]:
            canonical_integer(value)
        keys = {
            "structure:" + row["structure"],
            "composition:" + row["composition"],
            "behavior:" + digest(row["values"]),
        } | {"source:" + x for x in row["relations"]}
        keys |= {"window:" + digest(row["values"][i : i + window]) for i in range(101 - window)}
        result.append(
            dict(
                id=row["id"],
                groups=sorted(keys),
                limits="exact finite windows and supplied structure/source relations; not all-domain semantic equivalence",
            )
        )
    return result
