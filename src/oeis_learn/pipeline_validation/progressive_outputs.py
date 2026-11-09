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
    ("extract-components", "library"): "library",
    ("verify-components", "library"): "library",
    ("generate", "component_proposals"): "component_proposal",
    ("admit-training", "target"): "admitted_target",
    ("generate", "evaluation"): "public_evidence",
    ("train", "weights"): "checkpoint",
    ("score", "scores"): "public_evidence",
    ("census", "census"): "public_evidence",
    ("audit-language", "coverage"): "public_evidence",
}


def slot(value, *, plans, consumer, graph):
    if (
        type(value) is not dict
        or set(value) - {'member'} != {"task", "operation", "output", "kind"}
        or (value["operation"], value["output"]) not in SLOTS
        or SLOTS[value["operation"], value["output"]] != value["kind"]
    ):
        raise ValueError("closed typed predecessor output slot required")
    if ('member' in value) != (value['kind']=='component_proposal') or ('member' in value and (type(value['member']) is not str or not value['member'])):
        raise ValueError('frozen component proposal member selector required')
    source = next((p for p in plans if p["id"] == value["task"]), None)
    if (
        source is None
        or source["id"] == consumer["id"]
        or (
            value["kind"] != "public_evidence"
            and (source["arm"] != consumer["arm"] or source["seed"] != consumer["seed"])
        )
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
    same_slot=(source['slot']==consumer['slot'] and next(i for i,p in enumerate(plans) if p['id']==source['id'])<next(i for i,p in enumerate(plans) if p['id']==consumer['id']))
    if source["slot"] not in ancestors and not same_slot:
        raise ValueError("output producer is not a frozen ancestor")
    return source


def commit(root, store, task, role_output, operation):
    role_output = confined(root, str(Path(role_output).relative_to(root)))
    result = load(role_output / "result.json")
    if (
        seal(result) != result
        or result.get("operation") != operation
        or (result.get('mechanical_status')!='COMPLETE' and not (operation=='census' and result.get('mechanical_status')=='PARTIAL'))
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
            source=store.resolved_plan(source)
            if source is None:raise ValueError('producer trial choice has not committed')
            if store.state["tasks"].get(source["id"]) not in ({'COMPLETE','PARTIAL'} if value['kind']=='public_evidence' else {'COMPLETE'}):
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
            if seal(result) != result or result['mechanical_status'] not in ({'COMPLETE','PARTIAL'} if value['kind']=='public_evidence' else {'COMPLETE'}):
                raise ValueError("committed producer changed")
            ref = result["metrics"].get(value["output"])
            if type(ref) is not dict:
                raise ValueError("producer output has no original artifact reference")
            base = confined(root, cut["result"]["path"]).parent
            data = json.loads(verify_refs(base, [ref], max_bytes=512 << 20)[0])
            kind = value["kind"]
            if kind == "checkpoint":
                from oeis_learn.generalization.progressive.state import restore

                state_root = (
                    root
                    / "run-state/discovery-learning"
                    / store.campaign["identity"].split(":")[1]
                    / cut["task"].split(":")[1]
                    / "checkpoints"
                )
                p = restore(
                    data,
                    root=state_root,
                    compatibility=source["initializer"]["compatibility"],
                    campaign=store.campaign["identity"],
                    task=cut["task"],
                )
                if p["compatibility"] != task["payload"]["initializer"]["compatibility"]:
                    raise ValueError("evaluation checkpoint compatibility differs")
                full = json.loads(verify_refs(state_root, [p["full_state"]], max_bytes=64 << 20)[0])
                weights = full["states"]["weights"]
                verify_refs(state_root, [weights], max_bytes=64 << 30)
                data = dict(
                    artifact=reference(root, confined(state_root, weights["path"])),
                    compatibility=p["compatibility"],
                    profile="bigint-body/v1",
                )
            if kind == 'library':
                from oeis_learn.generalization.progressive.component_search import validate_input
                validate_input(data)
                if data['library']['purpose'] != task['purpose'] or data['library']['track'] != task['track']:
                    raise ValueError('committed component library scope differs')
                data=dict(artifact=reference(root,confined(base,ref['path'])),compatibility=task['payload']['initializer']['compatibility'],profile='progressive-component-search/v1')
            if kind == 'admitted_target':
                data=dict(membership='training',artifact=reference(root,confined(base,ref['path'])),profile='bigint-body/v1',compatibility=task['payload']['initializer']['compatibility'])
            if kind == 'component_proposal':
                if set(data) != {'version','identity','members'} or seal(data)!=data:
                    raise ValueError('closed generated component proposal inventory required')
                data=data['members'].get(value['member'])
                if data is None:raise ValueError('no eligible component proposal for frozen member')
            if kind not in CHANNELS[channel]:
                raise ValueError("producer output cannot enter this role channel")
            if kind=='public_evidence':
                data=seal(dict(version='discovery-owned-output/v1',task=source['id'],task_record=cut['task'],operation=value['operation'],output=value['output'],artifact=reference(root,confined(base,ref['path'])),record=data))
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


def validate_requests(root, plans, refs, graph, *, purpose="acceptance", track="synthetic_strict"):
    policies = {r["artifact"]["sha256"]: r["artifact"] for r in refs if r["kind"] == "policy"}
    for plan in plans:
        ref = policies.get(plan["policy"])
        if ref is None:
            continue
        request = json.loads(verify_refs(root, [ref])[0])
        if type(request) is not dict or "operation" not in request:
            continue
        from .progressive_roles import REGISTRY

        if request.get('operation')=='source-cycle':
            from .progressive_source_cycle import validate_request
            validate_request(root,request,plans=plans,refs=refs,graph=graph,purpose=purpose,track=track)
            continue
        if request.get("operation") == "oeis-cycle":
            from .progressive_oeis import validate_frozen_request
            validate_frozen_request(root, request, plans=plans, refs=refs, graph=graph,
                                    purpose=purpose, track=track)
            continue
        if (
            set(request) != {"operation", "artifacts", "image", "seconds", "reservation"}
            or request["operation"] not in REGISTRY
            or set(request["artifacts"]) != set(REGISTRY[request["operation"]].channels)
        ):
            raise ValueError("closed frozen installed request required")
        for channel, items in request["artifacts"].items():
            if not isinstance(items, list) or len(items) > 4096:
                raise ValueError("bounded frozen channel slots required")
            for value in items:
                if isinstance(value, dict):
                    slot(value, plans=plans, consumer=plan, graph=graph)
                elif isinstance(value,str) and request['operation']=='propose-trial':
                    envelope=json.loads(verify_refs(root,[reference(root,value)])[0])
                    if envelope.get('data',{}).get('version')=='progressive-verified-proposal/v1':raise ValueError('verified feedback must come from owned study journal')
                elif not isinstance(value, str):
                    raise ValueError("literal artifact path or typed predecessor slot required")
