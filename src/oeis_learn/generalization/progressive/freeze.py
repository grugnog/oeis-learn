"""Resolve complete prospective inputs and publish an immutable campaign cut."""

import json
import shutil
from pathlib import Path

from oeis_learn.pipeline_validation.identity import confined

from .campaign import validate_campaign, validate_runtime, validate_task
from .contracts import CAMPAIGN_DOCUMENT_BYTES, digest, load, record, reference, validate, verify_refs
from .targets import validate_policy


def inventory(root, reference_campaign):
    """Audit explicit retained files only; collect all missing refs in one pass."""
    base = confined(root, reference_campaign, exists=False)
    if not base.is_dir():
        return {"references": [], "unresolved": ["reference_campaign"]}
    names = (
        "campaign.json",
        "study.json",
        "runtime.json",
        "recipe.json",
        "inputs.json",
        "exposure.json",
    )
    refs = []
    missing = []
    supplied = base / "retained-inputs.json"
    if supplied.is_file():
        from oeis_learn.pipeline_validation.identity import seal

        selected = load(supplied)
        if (
            set(selected) != {"version", "identity", "references"}
            or selected["version"] != "discovery-retained-inputs/v1"
            or seal(selected) != selected
        ):
            raise ValueError("closed retained input inventory required")
        for item in selected["references"]:
            try:
                from .contracts import _check, schema

                _check(schema("common")["$defs"]["reference"], item, schema("common"))
                verify_refs(root, [item["artifact"]], max_bytes=64 << 30)
                refs.append(item)
            except (ValueError, OSError):
                missing.append("invalid:" + item.get("name", "reference"))
    for name in names:
        path = base / name
        if not path.is_file():
            continue
        try:
            r = reference(root, path)
            verify_refs(root, [r])
            value = load(path)
            refs.append(
                {
                    "name": name,
                    "role": "custodian",
                    "kind": "recipe" if name == "recipe.json" else "source",
                    "artifact": r,
                }
            )
        except (ValueError, OSError):
            missing.append(name + ":invalid")
    # Walk only hash/size/path references explicitly supplied by these producers.
    from oeis_learn.generalization.contracts import artifact_refs

    seen = {r["artifact"]["path"] for r in refs}
    queue = [(r["artifact"], base) for r in refs]
    while queue:
        ref, namespace = queue.pop(0)
        if len(seen) > 10000:
            raise ValueError("retained input inventory cap")
        try:
            value = json.loads(verify_refs(root, [ref])[0])
        except (ValueError, UnicodeError):
            continue
        if isinstance(value,dict):
            if value.get('version') in {'progressive-qualification-plan/v1','progressive-acceptance-plan/v1','progressive-preflight-plan/v1'}:
                namespace=Path(root)
            elif value.get('version')=='generalization-qualification/v2':
                namespace=confined(root,ref['path']).parent
        for child in artifact_refs(value):
            # Retained producers use paths relative to their campaign directory.
            try:
                path = confined(namespace, child["path"])
                resolved = reference(root, path)
                if any(resolved[k] != child[k] for k in ("bytes", "sha256")):
                    raise ValueError("retained identity changed")
                if resolved["path"] in seen:
                    continue
                verify_refs(root, [resolved])
                seen.add(resolved["path"])
                queue.append((resolved, namespace))
                refs.append(
                    {
                        "name": resolved["path"],
                        "role": "custodian",
                        "kind": "source",
                        "artifact": resolved,
                    }
                )
            except (OSError, ValueError):
                missing.append("artifact:" + child["path"])
    for required in (
        "retained_checkpoint",
        "original_qualification",
        "fresh-002_exposure",
        "prepared_groups",
        "measured_feasibility",
    ):
        if not any(r["name"] == required for r in refs):
            missing.append(required)
    return {"references": refs, "unresolved": sorted(set(missing))}


def freeze(
    root, blueprint, prepared, *, qualification_root, available_bytes=None, available_inodes=None
):
    b = validate("blueprint", blueprint)
    p = validate("prepared", prepared)
    if (
        p["payload"]["blueprint"] != b["identity"]
        or b["track"] != p["track"]
        or b["purpose"] != p["purpose"]
    ):
        raise ValueError("preparation scope differs")
    q = p["payload"]
    if q["deficits"]:
        raise IncompleteFreeze(q["deficits"])
    if q["graph"]["selected_stages"] != b["payload"]["selected_stages"]:
        raise ValueError("selected stages changed")
    if any(q["caps"][k] > b["payload"]["proposed_caps"][k] for k in q["caps"]):
        raise ValueError("prepared caps exceed proposal")
    refs = q["references"]
    if len({r["name"] for r in refs}) != len(refs):
        raise ValueError("conflicting named input")
    unresolved = set(b["payload"]["unresolved"]) - {r["name"] for r in refs}
    if unresolved:
        raise IncompleteFreeze(sorted(unresolved))
    required = {
        "checkpoint",
        "recipe",
        "source",
        "image",
        "groups",
        "policy",
        "qualification",
        "prepared",
    }
    if not required <= {r["kind"] for r in refs}:
        raise IncompleteFreeze(
            ["missing_reference:" + k for k in sorted(required - {r["kind"] for r in refs})]
        )
    # Distinct named refs may point at the same legitimate policy: verify once,
    # while still rejecting conflicting identity or physical aliases.
    unique = {}
    for ref in [r["artifact"] for r in refs] + [q["runtime"], q["checkpoint"], q["target_policy"]]:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("conflicting input identity")
        unique[ref["path"]] = ref
    verify_refs(root, list(unique.values()), max_bytes=64 << 30)
    runtime = validate("runtime", json.loads(verify_refs(root, [q["runtime"]])[0]))
    if runtime["track"] != b["track"] or runtime["purpose"] != b["purpose"]:
        raise ValueError("wrong runtime track/purpose")
    qual_root = confined(root, qualification_root)
    original = load(qual_root / "qualification.json")
    from oeis_learn.generalization.qualification import FACTS

    if not FACTS <= set(original):
        raise ValueError("original qualification facts required")
    facts = {k: original[k] for k in FACTS}
    # Freeze has no engine side effects. It reconciles the original producer and
    # local source bytes; execution must re-inspect current device/image facts.
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    source = load(confined(qual_root, facts["source_closure"]["path"]), maximum=64 << 20)
    if source != collect_snapshot(root):
        raise ValueError("stale qualification source closure")
    # Runtime qualification references use workspace-relative exact producer refs.
    if reference(root, qual_root / "qualification.json") not in runtime["payload"]["qualification"]:
        raise ValueError("qualification producer route differs")
    validate_runtime(runtime, root=root, current_facts=facts)
    execution_contracts=[r['artifact'] for r in refs if r['name']=='measured_evaluation_feasibility']
    if any(r['name']=='research-reservation-v2' for r in refs) and len(execution_contracts)!=1:
        raise ValueError('expanded reservation requires its measured execution contract')
    if execution_contracts:
        from oeis_learn.pipeline_validation.progressive_research_feasibility import verify_contract
        if len(execution_contracts)!=1:raise ValueError('one research execution contract required')
        execution_contract=verify_contract(root,execution_contracts[0],runtime)
        from .research_budget import verify_reservation
        verify_reservation(root,refs,q['task_plans'],execution_contract)
    policy = json.loads(verify_refs(root, [q["target_policy"]])[0])
    validate_policy(policy)
    if not policy["payload"]["selected"]:
        raise ValueError("proposed target defaults cannot execute")
    from .trial_scopes import variants
    all_plans=[variant for plan in q['task_plans'] for variant in variants(plan)]
    compatibility = _scope_inputs(
        root,
        refs,
        runtime,
        qualification_root=qual_root,
        checkpoint=q["checkpoint"],
        target_policy=q["target_policy"],
        track=b["track"],
        task_plans=all_plans,
    )
    from oeis_learn.pipeline_validation.progressive_calibration import validate_prepared
    validate_prepared(root, b, p)
    from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
    validate_requests(root,all_plans,refs,q["graph"], purpose=b["purpose"], track=b["track"])
    from oeis_learn.pipeline_validation.progressive_study import validate_frozen as validate_studies
    validate_studies(root,dict(purpose=b['purpose'],track=b['track'],payload=dict(references=refs,tasks=q['task_plans'])))
    known_policies = {r["artifact"]["sha256"] for r in refs if r["kind"] == "policy"} | {
        policy["identity"]
    }
    declared_checkpoints = [r["artifact"] for r in refs if r["kind"] == "checkpoint"]
    initializer_closure = {}
    calibrated_bytes=calibrated_inodes=0
    for plan in all_plans:
        selected_runtime=trial_runtime(root,plan,runtime)
        if 'trial_runtime' in plan and not any(r['artifact']==plan['trial_runtime'] and r['kind']=='qualification' for r in refs):
            raise ValueError('per-trial runtime must be frozen in original references')
        from .iteration import margin
        work=margin(plan,research=b['purpose']=='research' and plan['kind']=='training')
        if work:
            from .iteration import validate_producer
            calibration=validate_producer(root,plan,selected_runtime,refs)
            calibrated_bytes=max(calibrated_bytes,calibration.get('checkpoint_bytes',0));calibrated_inodes=max(calibrated_inodes,calibration.get('checkpoint_inodes',0))
        if work+plan['startup_us']+plan['finalization_us']>=plan['caps']['elapsed_us']:raise IncompleteFreeze(['complete_iteration:'+plan['id']])
        if (
            plan["policy"] not in known_policies
            or plan["initializer"]["compatibility"] not in compatibility
        ):
            raise ValueError("unresolved task policy or incompatible initializer")
        initializer = plan["initializer"]
        from .campaign import initializer_origin

        _, closure = initializer_origin(root, initializer, track=b["track"], purpose=b["purpose"])
        initializer_closure.update({r["path"]: r for r in closure})
        if initializer["mode"] == "full_state" and not (
            {"parent_slot", "parent_task"} & set(initializer)
        ):
            candidate = validate("state", load(confined(root, initializer["artifact"]["path"])))
            if (
                candidate["payload"]["compatibility"] != plan["initializer"]["compatibility"]
                or candidate["payload"]["kind"] != "full_state"
            ):
                raise ValueError("incompatible discovery full-state initializer")
        if initializer["artifact"] is not None:
            if initializer["artifact"] not in declared_checkpoints:
                raise ValueError("initializer is not a permitted checkpoint")
            verify_refs(root, [initializer["artifact"]], max_bytes=64 << 30)
    slots = {n["id"]: n["caps"] for n in q["graph"]["nodes"]}
    validate_policy(policy, reservations=slots)
    if policy["payload"]["diagnostics"]:
        from oeis_learn.pipeline_validation.progressive_decision_control import frozen_evidence

        # Use original referenced bytes, never a post-outcome replacement registry.
        selected, panels, designs = frozen_evidence(
            root, {"payload": {"references": refs, "target_policy": policy["identity"]}}
        )
        nodes = {n["id"]: n for n in q["graph"]["nodes"]}
        for branch in selected["payload"]["diagnostics"]:
            if any(
                nodes[s]["kind"] != "diagnostic" or not any(t["slot"] == s for t in q["task_plans"])
                for s in branch["task_slots"]
            ):
                raise ValueError("concrete frozen diagnostic tasks required")
            panel = panels.get(branch["panel"])
            if (
                panel is None
                or panel["payload"]["role"] != "confirmation"
                or panel["track"] != b["track"]
                or panel["purpose"] != b["purpose"]
            ):
                raise ValueError("original unexposed confirmation panel required for diagnostic")
            design = designs.get(branch["design"])
            if design is None:
                raise ValueError("original diagnostic design reference required")
            if branch["signal"] == "seed_disagreement" and (
                design.get("version") != "progressive-third-seed-design/v1"
                or design.get("branch") != branch["id"]
            ):
                raise ValueError("frozen third-seed registration required")
    from .storage import checkpoint_size

    measured, checkpoint_inodes = checkpoint_size(
        root,
        q["checkpoint"],
        checkpoint_root=q.get("checkpoint_root"),
        research=b["purpose"] == "research",
    )
    measured=max(measured,calibrated_bytes);checkpoint_inodes=max(checkpoint_inodes,calibrated_inodes)
    task_store_bytes = sum(t["caps"]["disk_bytes"] for t in q["task_plans"])
    for plan in q["task_plans"]:
        if (
            plan["kind"] == "training"
            and measured * (2 + q["milestone_pins"]) > plan["caps"]["disk_bytes"]
        ):
            raise IncompleteFreeze(["task_checkpoint_storage:" + plan["id"]])
    # Frozen disjoint task partitions include all retained predecessor stores,
    # archive history and task metadata. Separate reserves cover shared journal,
    # the largest projection (conservatively all inputs) and final packaging.
    from .storage import projection_size, JOURNAL_BYTES

    projection_bytes, projection_inodes = projection_size(root, list(unique.values()))
    # Retained strict origins are copied into the learner's isolated training view.
    projection_bytes += sum(r["bytes"] for r in initializer_closure.values()) + measured
    projection_inodes += sum(len(Path(r["path"]).parts) + 1 for r in initializer_closure.values())
    journal_bytes = JOURNAL_BYTES if b["purpose"] == "research" else 128 << 20
    required_bytes = (
        task_store_bytes + projection_bytes + journal_bytes + q["shard_bytes"] + q["package_bytes"]
    )
    required_inodes = max(
        q["required_inodes"],
        len(q["task_plans"]) * (checkpoint_inodes * (2 + q["milestone_pins"]) + 256)
        + projection_inodes
        + 16384,
    )
    usage = shutil.disk_usage(root)
    import os

    stat = os.statvfs(root)
    free = usage.free if available_bytes is None else min(usage.free, available_bytes)
    inodes = stat.f_favail if available_inodes is None else min(stat.f_favail, available_inodes)
    if required_bytes > free or required_inodes > inodes:
        raise IncompleteFreeze(["storage_bytes" if required_bytes > free else "storage_inodes"])
    campaign = record(
        "campaign",
        {
            "blueprint": b["identity"],
            "runtime": runtime["identity"],
            "graph": q["graph"],
            "tasks": q["task_plans"],
            "references": refs,
            "caps": q["caps"],
            "storage": {
                "checkpoint_bytes": measured,
                "atomic_generations": 2,
                "milestone_pins": q["milestone_pins"],
                "shard_bytes": q["shard_bytes"],
                "package_bytes": q["package_bytes"],
                "required_bytes": required_bytes,
                "available_bytes": free,
                "required_inodes": required_inodes,
                "task_store_bytes": task_store_bytes,
                "projection_bytes": projection_bytes,
                "journal_bytes": journal_bytes,
                "checkpoint_inodes": checkpoint_inodes,
                "available_inodes": inodes,
            },
            "target_policy": policy["identity"],
        },
        created_at=p["created_at"],
        parents=[b["identity"], p["identity"], runtime["identity"]],
        purpose=b["purpose"],
        track=b["track"],
    )
    validate_campaign(campaign)
    cycle_plans = set()
    for item in refs:
        if item["kind"] != "policy":
            continue
        request = load(confined(root, item["artifact"]["path"]),maximum=CAMPAIGN_DOCUMENT_BYTES)
        if request.get("operation") == "oeis-cycle":
            from oeis_learn.pipeline_validation.progressive_oeis import validate_cycle

            if (
                set(request.get("artifacts", {})) != {"cycle"}
                or len(request["artifacts"]["cycle"]) != 1
            ):
                raise ValueError("one frozen OEIS cycle plan required")
            cycle = load(confined(root, request["artifacts"]["cycle"][0]), maximum=256 << 20)
            validate_cycle(root, campaign, cycle)
            cycle_plans.add(cycle["identity"])
    if len(cycle_plans) > 1:
        raise ValueError("a further OEIS cycle requires a new study")
    tasks = []
    for plan in q["task_plans"]:
        tasks.append(
            validate_task(
                record(
                    "task",
                    dict(plan, campaign=campaign["identity"]),
                    created_at=p["created_at"],
                    parents=[campaign["identity"]],
                    purpose=b["purpose"],
                    track=b["track"],
                ),
                campaign=campaign,
            )
        )
    return campaign, runtime, tasks


def _scope_inputs(root, refs, runtime, *, qualification_root, checkpoint, target_policy, track,task_plans=()):
    """A valid file hash alone cannot substitute for its declared input scope."""
    scope = runtime["payload"]["scope"]
    grouped = {kind: [r for r in refs if r["kind"] == kind] for kind in {r["kind"] for r in refs}}
    if not any(r["artifact"] == checkpoint for r in grouped["checkpoint"]):
        raise ValueError("measured checkpoint is not a declared checkpoint")
    original = reference(root, qualification_root / "qualification.json")
    if not any(r["artifact"] == original for r in grouped["qualification"]):
        raise ValueError("declared qualification differs from original producer")
    if not any(r["artifact"] == target_policy for r in grouped["policy"]):
        raise ValueError("selected target policy is not a declared input")
    from oeis_learn.fresh.bigint_config import resolve

    learner_packages = []
    packages_by_scope = {}
    plans_by_scope = {}
    targets_by_channel = {}
    policies={r['artifact']['sha256']:r['artifact'] for r in refs if r['kind']=='policy'}
    requests={}
    for plan in task_plans:
        plans_by_scope.setdefault(plan['initializer']['compatibility'],[]).append(plan)
        if plan['kind']!='training' or plan['policy'] not in policies:
            continue
        if plan['policy'] not in requests:
            requests[plan['policy']]=json.loads(verify_refs(root,[policies[plan['policy']]])[0])
        request=requests[plan['policy']]
        if request.get('operation')=='train':
            for path in request.get('artifacts',{}).get('training',[]):
                if isinstance(path,str):targets_by_channel.setdefault(path,[]).append(plan)
    for item in grouped.get("prepared", []):
        envelope = load(confined(root, item["artifact"]["path"]), maximum=256 << 20)
        if (envelope.get("version") == "discovery-channel/v1"
                and envelope.get("kind") == "training_request"):
            from oeis_learn.pipeline_validation.identity import seal
            from oeis_learn.pipeline_validation.progressive_learner import compatibility, validate_input
            from .projections import _validate_channel
            from .trial_scopes import materialize
            from .supply import plan as supply_plan
            if seal(envelope)!=envelope or envelope.get('role')!='learner' or item['role']!='learner':
                raise ValueError('frozen training request role/identity differs')
            data=envelope['data']
            _validate_channel('training_request',data)
            from .shared_support import shared
            package=shared(json.loads(verify_refs(root,[data['artifact']],max_bytes=256<<20)[0]))
            if data['compatibility']!=compatibility(package):
                raise ValueError('declared training compatibility differs')
            targets=targets_by_channel.get(item['artifact']['path'],[])
            # The same finite law occurs at every look. Expand each exact
            # seed/law once, without rereading or rehashing all other tasks.
            expanded_by_key={}
            for plan in targets:
                params=plan['trial_choices'][plan['trial_choice']]['parameters'] if 'trial_choice' in plan else None
                key=(plan['seed'],plan['initializer']['compatibility'],digest(params),
                     plan.get('trial',{}).get('phase')=='confirmation')
                if key not in expanded_by_key:
                    expanded_by_key[key]=materialize(package,plan)
            candidates=list(expanded_by_key.values()) or [package]
            for expanded in candidates:
                key=compatibility(expanded)
                matched=plans_by_scope.get(key,[])
                validate_input(expanded,{'purpose':runtime['purpose'],'track':track,
                    'payload':{'arm':matched[0]['arm'] if matched else 'without_source',
                        'seed':expanded['profile']['model']['seed'],
                        'initializer':dict(compatibility=key,mode='full_state' if 'full_state' in expanded
                            else 'weights_only' if 'weights' in expanded else 'random',
                            artifact=expanded.get('weights',expanded.get('full_state',{}).get('artifact')))}},
                    allow_pending=True)
                for target,arm in {(p['token_target'],p['arm']) for p in targets
                                   if p['initializer']['compatibility']==key}:
                    try:supply_plan(expanded,target,arm=arm)
                    except ValueError as exc:raise IncompleteFreeze([str(exc)]) from exc
                learner_packages.append(expanded)
                packages_by_scope[key]=expanded
    recipes = []
    for item in grouped["recipe"]:
        recipe = load(confined(root, item["artifact"]["path"]))
        if resolve(recipe) != recipe:
            raise ValueError("fully resolved retained recipe required")
        if recipe["execution"]["language"] != scope["language"]:
            raise ValueError("recipe/runtime language differs")
        model = digest({k: v for k, v in recipe["model"].items() if k != "seed"})
        qualified_models = (
            {
                digest({k: v for k, v in p["profile"]["model"].items() if k != "seed"})
                for p in learner_packages
            }
            if learner_packages
            else {model}
        )
        runtime_scopes={}
        for key,package in packages_by_scope.items():
            matched=plans_by_scope.get(key,[])
            scopes=[]
            for plan in matched:
                selected=plan.get('trial_runtime',{}).get('sha256',runtime['identity'])
                if selected not in runtime_scopes:
                    runtime_scopes[selected]=trial_runtime(root,plan,runtime)['payload']['scope']
                scopes.append(runtime_scopes[selected])
            scopes=scopes or [scope]
            model_id=digest({k:v for k,v in package['profile']['model'].items() if k!='seed'})
            if any(model_id!=candidate['model'] or candidate['language']!=scope['language'] or candidate['source']!=scope['source'] for candidate in scopes):
                raise ValueError('per-trial model/update qualification differs')
        if not learner_packages and qualified_models != {scope['model']}:
            raise ValueError('recipe/model qualification differs')
        if any(p["profile"]["base_recipe"] != recipe for p in learner_packages):
            raise ValueError("research profiles require the declared resolved base recipe")
        if track == "synthetic_strict" and recipe["track"]["kind"] != "generic_only":
            raise ValueError("assisted recipe cannot be strict")
        recipes.append(recipe)
    for item in grouped["image"]:
        images = load(confined(root, item["artifact"]["path"]))
        if images != {"version": "discovery-images/v1", "images": scope["images"]}:
            raise ValueError("declared image pair differs")
    panels, exposures = [], {}
    for item in grouped["groups"]:
        value = load(confined(root, item["artifact"]["path"]), maximum=512 << 20)
        kind = {"discovery-exposure/v1": "exposure", "discovery-cohort/v1": "cohort"}.get(
            value.get("version")
        )
        if kind is None:
            raise ValueError("closed grouping/exposure authority required")
        validate(kind, value)
        if value["track"] != track or value["purpose"] != runtime["purpose"]:
            raise ValueError("wrong grouping track/purpose")
        if kind == "cohort":
            panels.append(value)
        else:
            exposures[value["identity"]] = value
    from .cohorts import validate_frozen_panels

    validate_frozen_panels(root, panels, exposures)
    if len(recipes) != 1:
        raise ValueError("one resolved initial recipe required")
    if learner_packages:
        identities = set(packages_by_scope)
        return identities
    return {learning_compatibility(recipes[0], refs, scope, track)}


def learning_compatibility(recipe, refs, scope, track):
    """Learning semantics exclude task output paths, elapsed caps and seed choice."""
    return digest(
        {
            "model": {k: v for k, v in recipe["model"].items() if k != "seed"},
            "learning": recipe["learning"],
            "inputs": recipe["inputs"],
            "execution": recipe["execution"],
            "execution_policy": scope["execution_policy"],
            "track": track,
            "support": sorted(
                r["artifact"]["sha256"] for r in refs if r["kind"] in {"source", "policy"}
            ),
        }
    )


class IncompleteFreeze(ValueError):
    def __init__(self, deficits):
        self.deficits = list(deficits)
        super().__init__("freeze prerequisites incomplete: " + ", ".join(self.deficits))


def trial_runtime(root,plan,default):
    ref=plan.get('trial_runtime')
    if ref is None:return default
    value=validate('runtime',json.loads(verify_refs(root,[ref])[0]))
    if value['purpose']!=default['purpose'] or value['track']!=default['track']:
        raise ValueError('per-trial runtime purpose/track differs')
    from oeis_learn.generalization.qualification import FACTS
    producers=[json.loads(blob) for blob in verify_refs(root,value['payload']['qualification'])]
    if not producers or any(not FACTS<=set(q) for q in producers):
        raise ValueError('original per-trial qualification required')
    validate_runtime(value,root=root,current_facts={k:producers[0][k] for k in FACTS})
    return value
