"""Diagnostic qualification/prior-stage doubles; native cycle roles stay real."""

import copy
import json
from types import SimpleNamespace

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)


def freeze_inputs(root, monkeypatch, *, source=False, low=False, build=None):
    from tests.unit.test_progressive_freeze import (
        test_complete_diagnostic_freeze_binds_original_producer_and_task_roster,
    )

    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(root, monkeypatch)
    b = load(root / "test-blueprint.json")
    prepared = load(root / "test-prepared.json")
    atomic_write(root / "authority-base-recipe.json", load(root / "recipe.json"))
    for ref in prepared["payload"]["references"]:
        if ref["kind"] == "recipe":
            ref["artifact"] = reference(root, "authority-base-recipe.json")
    if build:
        c = build(root)
    elif source:
        from tests.integration.test_progressive_short_cycle import fixture

        c, _, _ = fixture(root)
    else:
        from tests.helpers.progressive_oeis_fixture import fixture

        c, _, _, _ = fixture(root, low=low)
    if build:
        channel = next(
            load(root / r["artifact"]["path"])
            for r in c["payload"]["references"]
            if r["kind"] == "prepared"
            and load(root / r["artifact"]["path"]).get("kind") == "training_request"
        )
    else:
        cycle = load(root / ("short-plan.json" if source else "cycle-plan.json"))
        channel = load(root / cycle["training"][0]["channel"])
    package = load(root / channel["data"]["artifact"]["path"])
    # Freeze measures real weights for this selected model, rather than the
    # tiny unrelated placeholder from the authority-only unit fixture.
    import torch

    from oeis_learn.generalization.progressive.models import construct

    encoder, decoder, _ = construct(package["profile"], device="cpu")
    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
        root / "measured-cycle-weights.pt",
    )
    prepared["payload"]["checkpoint"] = reference(root, "measured-cycle-weights.pt")
    model = digest({k: v for k, v in package["profile"]["model"].items() if k != "seed"})

    def put(path, value):
        path.write_text(json.dumps(value))

    q = load(root / "qualification/qualification.json")
    producer = load(root / "qualification/producer.json")
    producer["capabilities"] = q["covered_capabilities"] = [
        "model:" + model,
        "language:bigint-body/v1",
        "progressive-lifecycle",
    ]
    put(root / "qualification/producer.json", seal(producer))
    q["producer"]["receipt"] = reference(root / "qualification", "producer.json")
    put(root / "qualification/qualification.json", seal(q))
    runtime = load(root / "runtime.json")
    runtime["track"] = c["track"]
    runtime["payload"]["scope"]["model"] = model
    runtime["payload"]["qualification"] = [reference(root, "qualification/qualification.json")]
    runtime["payload"]["operations"] = ["generate", "admit-training", "train", "score"]
    put(root / "runtime.json", seal(runtime))
    b["track"] = prepared["track"] = c["track"]
    b["payload"]["selected_stages"] = c["payload"]["graph"]["selected_stages"]
    prefix = []
    for node in c["payload"]["graph"]["nodes"][:-1]:
        p = copy.deepcopy(c["payload"]["tasks"][0])
        p.update(
            id="prior-" + node["id"],
            stage=node["stage"],
            slot=node["id"],
            kind="evaluation",
            token_target=0,
            output="prior-" + node["id"],
        )
        p["caps"] = {k: 0 for k in p["caps"]}
        p["caps"].update(elapsed_us=10000000, disk_bytes=1 << 20)
        p["startup_us"] = p["finalization_us"] = 1000
        # A frozen inert policy is historical evidence, not a runnable bypass.
        p["policy"] = prepared["payload"]["target_policy"]["sha256"]
        node["caps"] = dict(p["caps"])
        prefix.append(p)
    plans = prefix + c["payload"]["tasks"]
    caps = dict(c["payload"]["caps"])
    caps["disk_bytes"] += 64 << 20
    caps["elapsed_us"] += 100000000
    prepared["payload"].update(
        runtime=reference(root, "runtime.json"),
        graph=c["payload"]["graph"],
        task_plans=plans,
        caps=caps,
        package_bytes=4 << 20,
        # Canonical cycles retain per-phase projections, original receipts and
        # restore generations. Reserve this before freeze, for every child.
        required_inodes=max(prepared['payload']['required_inodes'],16384+8192*len(plans)),
    )
    b["payload"]["proposed_caps"] = caps
    refs = []
    for item in prepared["payload"]["references"]:
        if item["kind"] == "groups":
            continue
        if item["kind"] == "qualification":
            item["artifact"] = reference(root, "qualification/qualification.json")
        if item["kind"] == "checkpoint":
            item["artifact"] = reference(root, "measured-cycle-weights.pt")
        refs.append(item)
    refs += c["payload"]["references"]
    if not source and not build:
        refs += [
            {
                "name": "cycle-" + name,
                "role": "custodian",
                "kind": "groups",
                "artifact": reference(root, name + ".json"),
            }
            for name in ("exposure", "transfer", "retention")
        ]
    else:
        groups = load(root / "groups.json")
        groups["track"] = c["track"]
        put(root / "groups.json", seal(groups))
        refs.append(
            {
                "name": "groups",
                "role": "custodian",
                "kind": "groups",
                "artifact": reference(root, "groups.json"),
            }
        )
    prepared["payload"]["references"] = refs
    b = seal(b)
    prepared["payload"]["blueprint"] = b["identity"]
    prepared = seal(prepared)
    atomic_write(root / "cycle-blueprint.json", b)
    plan_payload = dict(prepared["payload"])
    plan_payload.pop("blueprint")
    atomic_write(
        root / "cycle-preparation-plan.json",
        {"version": "progressive-preparation-plan/v1", "payload": plan_payload},
    )
    b["payload"]["references"].append(
        {
            "name": "preparation_plan",
            "role": "custodian",
            "kind": "policy",
            "artifact": reference(root, "cycle-preparation-plan.json"),
        }
    )
    b = seal(b)
    put(root / "cycle-blueprint.json", b)
    return prefix


def accept_prior_stages(root, campaign, prefix, monkeypatch):
    """Only eligibility/qualification boundaries are doubled, never device PASS."""
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.pipeline_validation import progressive_roles as roles

    control = root / "run-state/discovery-control" / campaign["identity"][7:]
    store = RunStore(control, campaign)
    certificates = {
        e["source"]: {
            "source_slot": e["source"],
            "target_slot": e["target"],
            "verdict": e["condition"],
            "action": e["action"],
        }
        for e in campaign["payload"]["graph"]["edges"]
        if e["condition"] == "CONFIRMED_GAIN"
    }
    store.append(
        dict(
            store.state,
            tasks={p["id"]: "COMPLETE" for p in prefix},
            decisions=certificates,
            status="PAUSED",
        )
    )
    atomic_write(
        root / "accepted/acceptance.json",
        seal(
            {
                "version": "discovery-acceptance/v1",
                "scope": "progressive-lifecycle",
                "qualified": True,
                "campaign": campaign["identity"],
                "runtime": campaign["payload"]["runtime"],
                "status": "COMPLETE",
            }
        ),
    )
    # The sealed acceptance-purpose authority was constructed before freeze.
    # Never patch the shared artifact verifier: modules first imported while a
    # patch is active can retain it after pytest restores the original module.
    # Real research still requires original installed device producers.
    monkeypatch.setattr(
        roles, "current_runtime", lambda *a, **kw: SimpleNamespace(execution_ready=True)
    )
