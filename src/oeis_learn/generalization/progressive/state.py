"""Complete update cuts separate learning compatibility from task entitlement."""

from .contracts import record, validate, verify_refs

CUTS = (
    "stream",
    "allocation",
    "archive",
    "library",
    "cost_journal",
    "selection",
    "milestones",
    "consumed",
)


def snapshot(
    *,
    compatibility,
    campaign,
    task,
    full_state,
    model_update,
    cuts,
    created_at,
    purpose,
    track,
    kind="full_state",
):
    if set(cuts) != set(CUTS):
        raise ValueError("complete dynamic cut required")
    return record(
        "state",
        {
            "compatibility": compatibility,
            "campaign": campaign,
            "task": task,
            "full_state": full_state,
            "model_update": model_update,
            "cuts": cuts,
            "kind": kind,
        },
        created_at=created_at,
        purpose=purpose,
        track=track,
    )


def restore(record, *, root, compatibility, campaign, task):
    p = validate("state", record)["payload"]
    if p["kind"] != "full_state":
        raise ValueError("weights-only is a new initialization, not resume")
    if (p["compatibility"], p["campaign"], p["task"]) != (compatibility, campaign, task):
        raise ValueError("incompatible learning lineage or task entitlement")
    verify_refs(root, [p["full_state"]], max_bytes=64 << 20)
    return p
