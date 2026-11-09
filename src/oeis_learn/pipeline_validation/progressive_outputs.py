"""Typed, committed predecessor output slots; no arbitrary path authority."""

import json
from pathlib import Path
from oeis_learn.generalization.progressive.contracts import (
    seal,
    load,
    verify_refs,
    atomic_write,
    reference,
    record,
)
from oeis_learn.pipeline_validation.identity import confined
from .progressive_oeis import _Derived

SLOTS = {
    ("generate", "candidates"): "candidate",
    ("score", "scores"): "public_evidence",
    ("census", "census"): "public_evidence",
    ("audit-language", "coverage"): "public_evidence",
    ("train", "checkpoint_record"): "full_state",
}


def slot(value, *, plans, consumer, graph):
    if (
        type(value) is not dict
        or set(value) != {"task", "operation", "output", "kind"}
        or (value["operation"], value["output"]) not in SLOTS
        or SLOTS[value["operation"], value["output"]] != value["kind"]
    ):
        raise ValueError("closed typed predecessor output slot required")
    source = next((p for p in plans if p["id"] == value["task"]), None)
    if (
        source is None
        or source["id"] == consumer["id"]
        or (value["kind"]!="public_evidence" and (source["arm"] != consumer["arm"] or source["seed"] != consumer["seed"]))
    ):
        raise ValueError("paired original predecessor task required")
    nodes = {n["id"]: n for n in graph["nodes"]}
    ancestors = set()

    def walk(name):
        for p in nodes[name]["dependencies"]:
            if p not in ancestors:
                ancestors.add(p)
                walk(p)

    walk(consumer["slot"])
    if source["slot"] not in ancestors:
        raise ValueError("output producer is not a frozen ancestor")
    return source


def commit(root, store, task, role_output, operation):
    role_output = confined(root, str(Path(role_output).relative_to(root)))
    result = load(role_output / "result.json")
    if (
        seal(result) != result
        or result.get("operation") != operation
        or result.get("mechanical_status") != "COMPLETE"
    ):
        raise ValueError("completed original producer result required")
    if result.get("task") not in (None, task["identity"]) or result.get("campaign") not in (
        None,
        store.campaign["identity"],
    ):
        raise ValueError("producer task/campaign mismatch")
    cut = dict(
        task=task["identity"],
        operation=operation,
        result=reference(root, role_output / "result.json"),
    )
    outputs = dict(store.state.get("outputs", {}))
    prior = outputs.get(task["payload"]["id"])
    if prior is not None and prior != cut:
        raise ValueError("original committed output cannot change")
    outputs[task["payload"]["id"]] = cut
    store.append(dict(store.state, outputs=outputs))


def resolve(root, store, task, request, output):
    from .progressive_roles import REGISTRY
    from oeis_learn.generalization.progressive.projections import describe, CHANNELS

    artifacts = {}
    derived = []
    for channel, values in request["artifacts"].items():
        artifacts[channel] = []
        for i, value in enumerate(values):
            if isinstance(value, str):
                artifacts[channel].append(describe(root, value))
                continue
            source = slot(
                value,
                plans=store.campaign["payload"]["tasks"],
                consumer=task["payload"],
                graph=store.campaign["payload"]["graph"],
            )
            if store.state["tasks"].get(source["id"]) != "COMPLETE":
                raise ValueError("output predecessor has not committed")
            cut = store.state.get("outputs", {}).get(source["id"])
            if cut is None or cut["operation"] != value["operation"]:
                raise ValueError("original output producer missing")
            expected = record(
                "task",
                dict(source, campaign=store.campaign["identity"]),
                created_at=store.campaign["created_at"],
                purpose=task["purpose"],
                track=task["track"],
            )
            if cut["task"] != expected["identity"]:
                raise ValueError("original output task differs")
            result = json.loads(verify_refs(root, [cut["result"]])[0])
            if seal(result) != result or result["mechanical_status"] != "COMPLETE":
                raise ValueError("committed producer changed")
            ref = result["metrics"].get(value["output"])
            if type(ref) is not dict:
                raise ValueError("producer output has no original artifact reference")
            base = confined(root, cut["result"]["path"]).parent
            data = json.loads(verify_refs(base, [ref], max_bytes=512 << 20)[0])
            kind = value["kind"]
            if kind not in CHANNELS[channel]:
                raise ValueError("producer output cannot enter this role channel")
            # Preserve original content and binding; candidate wrapper gains no
            # permission to expose scorer/assessment truth to another role.
            wrapped = seal(
                dict(
                    version="discovery-channel/v1",
                    kind=kind,
                    role=REGISTRY[request["operation"]].role,
                    data=data,
                )
            )
            path = Path(output) / "derived" / channel / f"{i}.json"
            atomic_write(path, wrapped)
            descriptor = describe(root, reference(root, path)["path"])
            artifacts[channel].append(descriptor)
            derived.append(descriptor.ref)
    authority = (
        _Derived(
            store.campaign["identity"],
            task["identity"],
            request["operation"],
            task["payload"]["policy"],
            tuple(derived),
        )
        if derived
        else None
    )
    return artifacts, authority


def validate_requests(root,plans,refs,graph):
    policies={r['artifact']['sha256']:r['artifact'] for r in refs if r['kind']=='policy'}
    for plan in plans:
        ref=policies.get(plan['policy'])
        if ref is None:continue
        request=json.loads(verify_refs(root,[ref])[0])
        if type(request) is not dict or 'operation' not in request:continue
        from .progressive_roles import REGISTRY
        if set(request)!={'operation','artifacts','image','seconds','reservation'} or request['operation'] not in REGISTRY or set(request['artifacts'])!=set(REGISTRY[request['operation']].channels):
            raise ValueError('closed frozen installed request required')
        for channel,items in request['artifacts'].items():
            if not isinstance(items,list) or len(items)>4096:raise ValueError('bounded frozen channel slots required')
            for value in items:
                if isinstance(value,dict):slot(value,plans=plans,consumer=plan,graph=graph)
                elif not isinstance(value,str):raise ValueError('literal artifact path or typed predecessor slot required')
