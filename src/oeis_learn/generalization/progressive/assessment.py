"""Final selection and conservative one-way assessment exposure."""

from pathlib import Path
from .contracts import seal, verify_refs, load, atomic_write, digest
from .campaign import validate_campaign, validate_task


def validate_selection(root, selection, campaign):
    validate_campaign(campaign)
    if (
        set(selection)
        != {
            "version",
            "identity",
            "campaign",
            "task",
            "checkpoint",
            "library",
            "panel",
            "candidates",
        }
        or selection["version"] != "progressive-locked-selection/v1"
        or seal(selection) != selection
        or selection["campaign"] != campaign["identity"]
    ):
        raise ValueError("final locked selection required")
    task = validate_task(selection["task"], campaign=campaign)
    if task["payload"]["kind"] != "packaging":
        raise ValueError("reserved final assessment task required")
    refs = [selection["checkpoint"], selection["candidates"], *selection["panel"]]
    if selection["library"] is not None:
        refs.append(selection["library"])
    verify_refs(root, refs, max_bytes=64 << 30)
    frozen = [r["artifact"] for r in campaign["payload"]["references"] if r["role"] == "assessment"]
    if not selection["panel"] or any(r not in frozen for r in selection["panel"]):
        raise ValueError("assessment panel was not frozen")
    from oeis_learn.pipeline_validation.identity import confined

    channel = load(confined(root, selection["candidates"]["path"]), maximum=512 << 20)
    if (
        seal(channel) != channel
        or channel.get("kind") != "candidate"
        or channel["data"].get("checkpoint") != selection["checkpoint"]["sha256"]
    ):
        raise ValueError("candidate/final checkpoint binding differs")
    component_costs = [
        r.get("costs", {}).get("components") for r in channel["data"].get("records", {}).values()
    ]
    if selection["library"] is None:
        if any(component_costs):
            raise ValueError("final component candidates require the selected library")
    else:
        from .contracts import validate

        library = validate("library", load(confined(root, selection["library"]["path"])))
        if (
            library["track"] != campaign["track"]
            or library["purpose"] != campaign["purpose"]
            or not component_costs
            or any(not c or c.get("library") != library["identity"] for c in component_costs)
        ):
            raise ValueError("candidate/final library binding differs")
    for ref in selection["panel"]:
        truth = load(confined(root, ref["path"]))
        if seal(truth) != truth or truth.get("kind") != "assessment_truth":
            raise ValueError("assessment-only truth required")
    return selection


def expose(root, campaign, selection):
    """A durable STARTED event prevents an interrupted panel from becoming untouched."""
    control = Path(root) / "run-state/discovery-control" / campaign["identity"].split(":")[1]
    control.mkdir(parents=True, exist_ok=True)
    import fcntl

    path = control / "assessment.lock"
    if path.is_symlink():
        raise ValueError("assessment lock alias")
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event = control / "assessment-exposure.json"
        if event.exists():
            raise ValueError("assessment already exposed; adaptive continuation forbidden")
        atomic_write(control / "locked-selection.json", selection)
        value = seal(
            dict(
                version="progressive-assessment-exposure/v1",
                campaign=campaign["identity"],
                selection=selection["identity"],
                panel=digest(selection["panel"]),
                status="STARTED",
                untouched=False,
            )
        )
        atomic_write(event, value)
        return value


def check_derived(value, *, campaign, task, operation, root):
    if (
        operation != "assess"
        or set(value) != {"version", "identity", "selection", "artifacts"}
        or value["version"] != "progressive-assessment-derived/v1"
        or seal(value) != value
        or value["selection"]["campaign"] != campaign["identity"]
        or value["selection"]["task"] != task
    ):
        raise ValueError("locked assessment derivation required")
    validate_selection(root, value["selection"], campaign)
    control = Path(root) / "run-state/discovery-control" / campaign["identity"].split(":")[1]
    if load(control / "locked-selection.json") != value["selection"]:
        raise ValueError("original durable selection lock required")
    allowed = value["selection"]["panel"] + [value["selection"]["candidates"]]
    from oeis_learn.pipeline_validation.identity import confined

    extras = [r for r in value["artifacts"] if r not in allowed]
    if len(extras) != 1 or set(r["path"] for r in value["artifacts"]) != set(
        r["path"] for r in allowed + extras
    ):
        raise ValueError("exact assessment derivation closure required")
    locked = load(confined(root, extras[0]["path"]))
    if (
        locked.get("kind") != "locked_selection"
        or locked.get("data") != value["selection"]
        or seal(locked) != locked
    ):
        raise ValueError("original selection channel required")
    return value["artifacts"]
