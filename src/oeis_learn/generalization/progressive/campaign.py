"""Discovery authority validation, separate from old PAC execution policy."""

from dataclasses import dataclass
import json
from .contracts import validate, verify_refs
from .graph import validate_graph


def validate_task(task, *, campaign=None, track=None, compatibility=None):
    task = validate("task", task)
    p = task["payload"]
    if track is not None and task["track"] != track:
        raise ValueError("wrong task track")
    if p["kind"] == "training" and p["caps"]["elapsed_us"] > 86_400_000_000:
        raise ValueError("24 hour task ceiling")
    if p["startup_us"] + p["finalization_us"] >= p["caps"]["elapsed_us"]:
        raise ValueError("no useful work after reserves")
    if task["purpose"] == "research" and p["finalization_us"] < 1_800_000_000:
        raise ValueError("research finalization reserve below 30 minutes")
    init = p["initializer"]
    if (init["mode"] == "random") != (init["artifact"] is None):
        raise ValueError("explicit compatible initializer required")
    if compatibility is not None and init["compatibility"] != compatibility:
        raise ValueError("incompatible initializer")
    if campaign is not None:
        c = validate_campaign(campaign)
        if (
            p["campaign"] != c["identity"]
            or task["track"] != c["track"]
            or task["purpose"] != c["purpose"]
        ):
            raise ValueError("task campaign binding")
        slots = {n["id"]: n for n in c["payload"]["graph"]["nodes"]}
        if p["slot"] not in slots or slots[p["slot"]]["stage"] != p["stage"]:
            raise ValueError("task not reserved")
        if any(p["caps"][k] > slots[p["slot"]]["caps"][k] for k in p["caps"]):
            raise ValueError("task exceeds frozen child ceiling")
        plan = {k: v for k, v in p.items() if k != "campaign"}
        if plan not in c["payload"]["tasks"]:
            raise ValueError("task is absent from frozen roster")
    return task


def validate_campaign(campaign):
    c = validate("campaign", campaign)
    p = c["payload"]
    validate_graph(p["graph"], caps=p["caps"])
    plans = p["tasks"]
    slots = {n["id"]: n for n in p["graph"]["nodes"]}
    if len({t["id"] for t in plans}) != len(plans) or len({t["output"] for t in plans}) != len(
        plans
    ):
        raise ValueError("duplicate task identity/output")
    for plan in plans:
        if plan["slot"] not in slots or plan["stage"] != slots[plan["slot"]]["stage"]:
            raise ValueError("unresolved task slot")
        if plan["kind"] == "training" and plan["caps"]["elapsed_us"] > 86_400_000_000:
            raise ValueError("training ceiling")
        if plan["initializer"]["mode"] != "random" and plan["initializer"]["artifact"] is None:
            raise ValueError("missing initializer")
    for slot in slots:
        children = [t for t in plans if t["slot"] == slot]
        if any(
            sum(t["caps"][k] for t in children) > slots[slot]["caps"][k]
            for k in slots[slot]["caps"]
        ):
            raise ValueError("task child sums exceed slot")
    if c["purpose"] == "research":
        training = sum(t["caps"]["elapsed_us"] for t in plans if t["kind"] == "training")
        ancillary = sum(t["caps"]["elapsed_us"] for t in plans if t["kind"] != "training")
        if (
            training > 536 * 3600 * 1_000_000
            or ancillary > 136 * 3600 * 1_000_000
            or p["caps"]["elapsed_us"] > 672 * 3600 * 1_000_000
        ):
            raise ValueError("maximum research campaign reservation exceeded")
    s = p["storage"]
    needed = (
        s["checkpoint_bytes"] * (2 + s["milestone_pins"]) + s["shard_bytes"] + s["package_bytes"]
    )
    if (
        s["required_bytes"] != needed
        or needed > s["available_bytes"]
        or s["required_inodes"] > s["available_inodes"]
    ):
        raise ValueError("storage deficit or inconsistent peak-live graph")
    if needed > p["caps"]["disk_bytes"]:
        raise ValueError("storage graph exceeds campaign disk cap")
    return c


@dataclass(frozen=True)
class VerifiedRuntime:
    """In-process result of original-producer validation, never serialized authority."""

    identity: str
    payload: dict
    purpose: str
    track: str
    producers: tuple[str, ...]
    execution_ready: bool = False


def validate_runtime(runtime, *, root, current_facts, required_capabilities=(), inspected=False):
    r = validate("runtime", runtime)
    p = r["payload"]
    if r["purpose"] == "research" and p["fixture"]:
        raise ValueError("fixtures cannot authorize research")
    from oeis_learn.pipeline_validation.progressive_roles import ROLE_POLICY

    if p["role_policy"] != ROLE_POLICY:
        raise ValueError("role policy differs")
    refs = p["qualification"]
    payloads = verify_refs(root, refs)
    from oeis_learn.generalization.qualification import consume_public

    producers = []
    for ref, raw in zip(refs, payloads):
        q = json.loads(raw)
        from oeis_learn.pipeline_validation.identity import confined

        producer_root = confined(root, ref["path"]).parent
        original = consume_public(
            q,
            current_facts,
            root=producer_root,
            required_suites=["bigint-cpu", "bigint-gpu"],
            required_capabilities=list(required_capabilities),
        )
        scope = p["scope"]
        if (
            scope["images"] != current_facts["images"]
            or scope["source"] != current_facts["source_closure"]["sha256"]
            or scope["execution_policy"] != current_facts["runtime"]["execution_policy"]
            or scope["precision"] != current_facts["precision"]
        ):
            raise ValueError("runtime source/image/precision mismatch")
        # Shape/language capabilities must be explicitly covered by original bytes.
        requested = ["model:" + scope["model"], "language:" + scope["language"]]
        if not set(requested) <= set(q["covered_capabilities"]):
            raise ValueError("model/language scope uncovered")
        producers.append(original)
    if set(p["operations"]) - set(
        __import__(
            "oeis_learn.pipeline_validation.progressive_roles", fromlist=["REGISTRY"]
        ).REGISTRY
    ):
        raise ValueError("unknown runtime operation")
    return VerifiedRuntime(r["identity"], p, r["purpose"], r["track"], tuple(producers), inspected)
