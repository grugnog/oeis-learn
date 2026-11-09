"""Outcome-blind preparation assembly and installed public capability reconciliation."""

import json
from .identity import confined
from oeis_learn.generalization.progressive.contracts import (
    load,
    validate,
    verify_refs,
    record,
    reference,
    atomic_write,
    seal,
)
from .progressive_evidence import publish


def prepare(args, root, host_factory=None):
    from .progressive_commands import _output

    blueprint = validate("blueprint", load(confined(root, args.blueprint)))
    output = _output(root, args.output)
    selected = [
        r["artifact"] for r in blueprint["payload"]["references"] if r["name"] == "preparation_plan"
    ]
    if len(selected) != 1:
        return publish(
            output,
            operation="prepare",
            mechanical_status="BLOCKED",
            reason="PREPARATION_PLAN_DEFICIT",
            missing=["preparation_plan"],
        ), 2
    plan = json.loads(verify_refs(root, selected, max_bytes=32 << 20)[0])
    if set(plan) != {"version", "payload"} or plan["version"] != "progressive-preparation-plan/v1":
        raise ValueError("closed prospective preparation plan required")
    if "blueprint" in plan["payload"]:
        raise ValueError("preparation plan must avoid circular blueprint identity")
    payload = dict(plan["payload"], blueprint=blueprint["identity"])
    prepared = record(
        "prepared",
        payload,
        created_at=blueprint["created_at"],
        purpose=blueprint["purpose"],
        track=blueprint["track"],
    )
    validate("prepared", prepared)
    refs = [r["artifact"] for r in payload["references"]] + [
        payload[k] for k in ("runtime", "checkpoint", "target_policy")
    ]
    unique = {r["path"]: r for r in refs}
    if any(unique[r["path"]] != r for r in refs):
        raise ValueError("conflicting preparation reference")
    verify_refs(root, list(unique.values()), max_bytes=64 << 30)
    runtime = validate("runtime", load(confined(root, payload["runtime"]["path"])))
    producer = reference(root, confined(root, args.qualification) / "qualification.json")
    if producer not in runtime["payload"]["qualification"]:
        raise ValueError("preparation must bind original qualification route")
    # Assembly never changes a cohort in response to scores or creates evidence.
    deficits = list(payload["deficits"])
    deficits += sorted(
        set(blueprint["payload"]["unresolved"]) - {r["name"] for r in payload["references"]}
    )
    if payload["graph"]["selected_stages"] != blueprint["payload"]["selected_stages"]:
        raise ValueError("preparation stages differ from prospective blueprint")
    if any(payload["caps"][k] > blueprint["payload"]["proposed_caps"][k] for k in payload["caps"]):
        raise ValueError("preparation cannot enlarge reservation")
    if deficits:
        prepared = seal(dict(prepared, payload=dict(payload, deficits=sorted(set(deficits)))))
    else:
        from .progressive_calibration import prepare as calibrate

        prepared = calibrate(root, blueprint, prepared, output, args, host_factory)
        deficits = prepared['payload']['deficits']
    atomic_write(output / "prepared.json", prepared)
    return publish(
        output,
        operation="prepare",
        mechanical_status="PARTIAL" if deficits else "COMPLETE",
        reason="BOUND_OUTCOME_BLIND_PREPARATION",
        missing=deficits,
        metrics={"prepared": reference(output, output / "prepared.json")},
    ), 2 if deficits else 0


def publish_readiness(values, output):
    from oeis_learn.generalization.progressive.readiness import from_request

    data = [v['data']['record'] if isinstance(v['data'],dict) and v['data'].get('version')=='discovery-owned-output/v1' else v['data'] for v in values]

    def one(version):
        rows = [v for v in data if isinstance(v, dict) and v.get("version") == version]
        if len(rows) != 1:
            raise ValueError("one original " + version + " required")
        return rows[0]

    cohort = one("discovery-cohort/v1")
    exposure = one("discovery-exposure/v1")
    audits = [a for v in data if isinstance(v, list) for a in v]
    for a in audits:
        validate("coverage", a)
    candidates = [
        v for v in data if isinstance(v, dict) and v.get("version") == "discovery-candidates/v1"
    ]
    scores = [v for v in data if isinstance(v, dict) and v.get("version") == "discovery-scoring/v1"]

    def write(name, value):
        path = output / "public" / name
        atomic_write(path, value)
        return reference(output, path)

    evaluations = []
    for i, c in enumerate(candidates):
        ref = write(f"candidates-{i}.json", c)
        matching = [s for s in scores if s["candidates"] == ref["sha256"]]
        if len(matching) != 1:
            raise ValueError("original score/checkpoint binding differs")
        evaluations.append(dict(candidates=ref, scores=write(f"scores-{i}.json", matching[0])))
    if len(scores) != len(candidates):
        raise ValueError("unmatched public scoring producer")
    request = seal(
        dict(
            version="capability-readiness-request/v1",
            cohort=write("cohort.json", cohort),
            exposure=write("exposure.json", exposure),
            coverage=[write(f"coverage-{i}.json", a) for i, a in enumerate(audits)],
            evaluations=evaluations,
        )
    )
    result = from_request(output, request)
    atomic_write(output / "capability-inputs.json", request)
    atomic_write(output / "readiness.json", result)
    return publish(
        output,
        operation="publish",
        mechanical_status="COMPLETE" if result["status"] == "COMPLETE" else "PARTIAL",
        reason="RECONCILED_ORIGINAL_CAPABILITY_PRODUCERS",
        missing=result["missing"],
        metrics={"readiness": reference(output, output / "readiness.json")},
    )
