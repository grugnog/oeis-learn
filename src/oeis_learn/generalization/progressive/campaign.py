"""Discovery authority validation, separate from old PAC execution policy."""

import json
from dataclasses import dataclass

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
    trial=p.get('trial')
    shared=trial is not None and trial['phase'] not in {'confirmation','selection'}
    if task['purpose']=='research' and p['finalization_us']<1800000000 and not (shared and trial.get('shared_finalization_us',0)>=1800000000):
        raise ValueError('research finalization reserve below 30 minutes')
    from .iteration import margin
    work=margin(p,research=task["purpose"]=="research" and p["kind"]=="training")
    if work+p["startup_us"]+p["finalization_us"]>=p["caps"]["elapsed_us"]:raise ValueError("complete iteration cannot fit frozen task")
    init = p["initializer"]
    if (
        task["track"] == "synthetic_strict"
        and init["artifact"] is not None
        and "origin" not in init
    ):
        raise ValueError("strict imported initializer requires original origin evidence")
    future = (
        init["mode"] == "full_state"
        and init["artifact"] is None
        and ("parent_slot" in init or "parent_task" in init)
    )
    if ("parent_slot" in init or "parent_task" in init) and (
        not future or {"parent_slot", "parent_task"} <= set(init)
    ):
        raise ValueError("parent slot requires an unresolved full-state successor")
    if not future and (init["mode"] == "random") != (init["artifact"] is None):
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
        if plan not in c['payload']['tasks']:
            from .trial_scopes import resolve
            templates=[t for t in c['payload']['tasks'] if t['id']==plan['id']]
            if len(templates)!=1 or 'trial_choice' not in plan or resolve(templates[0],plan['trial_choice'])!=plan:
                raise ValueError('task is absent from frozen finite choice roster')
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
        if 'trial_choices' in plan:
            from .trial_scopes import variants
            if plan.get('trial',{}).get('phase') not in {'training','generation','scoring','confirmation','retention'}:
                raise ValueError('trial alternatives require a bound finite study phase')
            identities=[dict(v['parameters'],shape=v['parameters'].get('shape','reference')) for v in plan['trial_choices']]
            if len({json.dumps(v,sort_keys=True) for v in identities})!=len(identities):raise ValueError('duplicate trial alternative')
            variants(plan)
        init = plan["initializer"]
        if (
            c["track"] == "synthetic_strict"
            and init["artifact"] is not None
            and "origin" not in init
        ):
            raise ValueError("strict imported initializer requires original origin evidence")
        future = (
            init["mode"] == "full_state"
            and init["artifact"] is None
            and ("parent_slot" in init or "parent_task" in init)
        )
        if "parent_task" in init:
            parents = [
                t
                for t in plans[: plans.index(plan)]
                if t["id"] == init["parent_task"]
                and (t["slot"] == plan["slot"] or t["slot"] in slots[plan["slot"]]["dependencies"])
                and t["arm"] == plan["arm"]
                and t["seed"] == plan["seed"]
            ]
            if (
                not future
                or "parent_slot" in init
                or len(parents) != 1
                or parents[0]["initializer"]["compatibility"] != init["compatibility"]
            ):
                raise ValueError("one earlier compatible task in the same finite slot required")
        if "parent_slot" in init and (
            not future or init["parent_slot"] not in slots[plan["slot"]]["dependencies"]
        ):
            raise ValueError("successor initializer must name a frozen predecessor slot")
        if future and "parent_slot" in init:
            parents = [
                t
                for t in plans
                if t["slot"] == init["parent_slot"]
                and t["arm"] == plan["arm"]
                and t["seed"] == plan["seed"]
            ]
            if (
                len(parents) != 1
                or parents[0]["initializer"]["compatibility"] != init["compatibility"]
            ):
                raise ValueError("one compatible paired predecessor required")
        elif not future and init["mode"] != "random" and init["artifact"] is None:
            raise ValueError("missing initializer")
    held={}
    if c['purpose']=='research':
        for plan in plans:
            trial=plan.get('trial')
            if trial and trial['phase'] not in {'confirmation','selection'}:
                key=trial['study'],trial['ordinal']
                value=(plan['slot'],trial.get('shared_finalization_us',0))
                if value[1]<1800000000 or (key in held and held[key]!=value):raise ValueError('one shared frozen trial finalization envelope required')
                held[key]=value
    for slot in slots:
        children = [t for t in plans if t["slot"] == slot]
        if any(
            sum(t["caps"][k] for t in children)+(sum(n for selected,n in held.values() if selected==slot) if k=='elapsed_us' else 0) > slots[slot]["caps"][k]
            for k in slots[slot]["caps"]
        ):
            raise ValueError("task child sums exceed slot")
    if c["purpose"] == "research":
        training = sum(t["caps"]["elapsed_us"] for t in plans if t["kind"] == "training")
        ancillary = sum(t["caps"]["elapsed_us"] for t in plans if t["kind"] != "training")
        expanded=any(r['name']=='research-reservation-v2' for r in p['references'])
        from .runtime_reservation import LIMITS_SECONDS,TOTAL_SECONDS
        if (
            training > 536 * 3600 * 1_000_000
            or ancillary > (LIMITS_SECONDS['evaluation_analysis'] if expanded else 136*3600)*1_000_000
            or p["caps"]["elapsed_us"] > (TOTAL_SECONDS if expanded else 672*3600)*1_000_000
        ):
            raise ValueError("maximum research campaign reservation exceeded")
    s = p["storage"]
    planned = max((t["token_target"] for t in plans if t["kind"] == "training"), default=0)
    if s["milestone_pins"] < sum(m <= planned for m in (25000000, 100000000, 400000000)):
        raise ValueError("frozen milestone pin storage is insufficient for planned exposure")
    needed = (
        s["checkpoint_bytes"] * (2 + s["milestone_pins"]) + s["shard_bytes"] + s["package_bytes"]
    )
    complete_storage = {
        "task_store_bytes",
        "projection_bytes",
        "journal_bytes",
        "checkpoint_inodes",
    }
    if 'task_checkpoints' in s:
        from .storage_plan import checkpoint_bound
        rows=s['task_checkpoints']
        if (not complete_storage <= set(s) or len({r['task'] for r in rows})!=len(rows) or
                {r['task'] for r in rows}!={t['id'] for t in plans if t['kind']=='training'} or
                not any(r['name']=='research_resource_policy' for r in p['references'])):
            raise ValueError('complete measured task checkpoint inventory required')
        for t in plans:
            if t['kind']!='training':continue
            b=checkpoint_bound(s,t)
            pins=min(s['milestone_pins'],sum(m<=t['token_target'] for m in (25000000,100000000,400000000)))
            if (b['checkpoint_bytes']>s['checkpoint_bytes'] or b['checkpoint_inodes']>s['checkpoint_inodes']
                    or b['milestone_pins']!=pins):
                raise ValueError('task checkpoint scope exceeds frozen storage')
    if complete_storage & set(s):
        if not complete_storage <= set(s) or s["task_store_bytes"] != sum(
            t["caps"]["disk_bytes"] for t in plans
        ):
            raise ValueError("complete simultaneous task storage partitions required")
        from .storage_plan import checkpoint_bound
        if any(
            t["kind"] == "training"
            and checkpoint_bound(s,t)['checkpoint_bytes'] * (2 + checkpoint_bound(s,t)['milestone_pins']) > t["caps"]["disk_bytes"]
            for t in plans
        ):
            raise ValueError("task checkpoint storage partition insufficient")
        task_inodes = len(plans) * (s["checkpoint_inodes"] * (2 + s["milestone_pins"]) + 256)
        if s["journal_bytes"] < 128 << 20 or s["required_inodes"] < 16384 + task_inodes:
            raise ValueError("host journal/task storage reserve insufficient")
        needed = (
            s["task_store_bytes"]
            + s["projection_bytes"]
            + s["journal_bytes"]
            + s["shard_bytes"]
            + s["package_bytes"]
        )
    elif c["purpose"] == "research":
        raise ValueError("complete peak-live research storage graph required")
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
        producer = json.loads(verify_refs(producer_root,[q['producer']['receipt']])[0])
        if not p['fixture']:
            from oeis_learn.pipeline_validation.progressive_handoff import validate_device_proofs
            from .contracts import digest
            source=json.loads(verify_refs(producer_root,[q['source_closure']],max_bytes=64<<20)[0])
            validate_device_proofs(producer_root,q,source=digest(source),image=current_facts['images'][1],
                                   engine=current_facts['engine_security']['engine'])
        producers.append(original)
    if set(p["operations"]) - set(
        __import__(
            "oeis_learn.pipeline_validation.progressive_roles", fromlist=["REGISTRY"]
        ).REGISTRY
    ):
        raise ValueError("unknown runtime operation")
    return VerifiedRuntime(r["identity"], p, r["purpose"], r["track"], tuple(producers), inspected)


def initializer_origin(root, initializer, *, track, purpose):
    """Verify retained strict weights against original full-state producers.

    A hash of an imported tensor is not origin evidence. Each bounded lineage
    link binds an original campaign/task, full checkpoint and prior exposure.
    Returned references are the explicit learner-safe projection closure; source
    campaign reference inventories are metadata, never recursively mounted.
    """
    from .contracts import digest, reference, seal
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.generalization.contracts import artifact_refs
    from oeis_learn.generalization.state import restore

    if track != "synthetic_strict" or initializer["artifact"] is None:
        return None, []
    if root is None or initializer.get("origin") is None:
        raise ValueError("strict imported initializer requires original origin evidence")
    refs, seen = {}, set()

    def read(ref):
        if ref["path"] in refs and refs[ref["path"]] != ref:
            raise ValueError("conflicting initializer origin reference")
        refs[ref["path"]] = ref
        return json.loads(verify_refs(root, [ref], max_bytes=64 << 20)[0])

    def visit(ref, expected, mode):
        if ref["sha256"] in seen or len(seen) >= 64:
            raise ValueError("bounded acyclic strict origin required")
        seen.add(ref["sha256"])
        origin = read(ref)
        if (
            set(origin)
            != {
                "version",
                "identity",
                "checkpoint",
                "campaign",
                "task",
                "state",
                "state_root",
                "exposure",
                "parent",
            }
            or origin["version"] != "progressive-initializer-origin/v1"
            or seal(origin) != origin
            or origin["checkpoint"] != expected
        ):
            raise ValueError("closed original initializer origin binding required")
        campaign = validate_campaign(read(origin["campaign"]))
        task = validate_task(read(origin["task"]), campaign=campaign)
        state = validate("state", read(origin["state"]))
        exposure = validate("exposure", read(origin["exposure"]))
        if any(
            v["track"] != "synthetic_strict" or v["purpose"] != purpose
            for v in (campaign, task, state, exposure)
        ):
            raise ValueError("assisted or cross-purpose origin cannot enter strict track")
        if not any(
            r["kind"] == "groups"
            and all(r["artifact"][k] == origin["exposure"][k] for k in ("sha256", "bytes"))
            for r in campaign["payload"]["references"]
        ):
            raise ValueError("original campaign exposure producer required")
        p = state["payload"]
        if (
            p["kind"] != "full_state"
            or p["campaign"] != campaign["identity"]
            or p["task"] != task["identity"]
            or p["compatibility"] != task["payload"]["initializer"]["compatibility"]
            or any(p["cuts"][k] not in (None, digest(None)) for k in ("archive", "library"))
        ):
            raise ValueError("strict mechanical full-state lineage differs")
        base = confined(root, origin["state_root"])
        full_ref = p["full_state"]
        raw = verify_refs(base, [full_ref], max_bytes=64 << 20)[0]
        full = json.loads(raw)
        bound = reference(root, confined(base, full_ref["path"]))
        refs[bound["path"]] = bound
        # Verify the entire tensor closure without importing torch on the host.
        for child in artifact_refs(full["states"]):
            verify_refs(base, [child], max_bytes=64 << 30)
            bound = reference(root, confined(base, child["path"]))
            if bound["path"] in refs and refs[bound["path"]] != bound:
                raise ValueError("conflicting initializer tensor reference")
            refs[bound["path"]] = bound
        restore(full, contract=p["compatibility"], cap=full["ledger"]["cap"], root=base)
        if full["ledger"]["updates"] != p["model_update"] or full["ledger"]["pending"] is not None:
            raise ValueError("origin requires complete committed model boundary")
        weights = full["states"]["weights"]
        if mode == "weights_only":
            if any(expected[k] != weights[k] for k in ("sha256", "bytes")):
                raise ValueError("imported weights differ from original full-state tensor")
        elif expected != origin["state"]:
            raise ValueError("full-state import differs from original checkpoint")
        prior = task["payload"]["initializer"]
        if prior["mode"] == "random":
            if origin["parent"] is not None:
                raise ValueError("random origin cannot hide a parent")
        else:
            if origin["parent"] is None:
                raise ValueError("missing strict random-origin ancestor")
            parent_origin = read(origin["parent"])
            parent_task = visit(
                origin["parent"],
                parent_origin["checkpoint"],
                "full_state" if prior["artifact"] is None else prior["mode"],
            )
            if prior["artifact"] is not None:
                if prior["artifact"]["sha256"] != parent_origin["checkpoint"]["sha256"]:
                    raise ValueError("imported ancestor checkpoint differs")
            elif (
                parent_task["payload"]["campaign"] != campaign["identity"]
                or parent_task["payload"]["arm"] != task["payload"]["arm"]
                or parent_task["payload"]["seed"] != task["payload"]["seed"]
                or (
                    prior.get("parent_task") != parent_task["payload"]["id"]
                    and prior.get("parent_slot") != parent_task["payload"]["slot"]
                )
            ):
                raise ValueError("successor origin differs from frozen parent")
        return task

    visit(initializer["origin"], initializer["artifact"], initializer["mode"])
    return read(initializer["origin"]), list(refs.values())
