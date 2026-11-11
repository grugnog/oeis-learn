"""Canonical finite study orchestration; only installed proposal roles import Optuna."""

import json
from pathlib import Path
from oeis_learn.generalization.progressive.contracts import (
    load,
    verify_refs,
    reference,
    atomic_write,
    seal,
    digest,
)
from oeis_learn.generalization.progressive.trials import TrialJournal
from oeis_learn.generalization.progressive.trial_evidence import objective
from oeis_learn.generalization.progressive.tuning import (
    preflight,
    validate_screening_panel,
    pruning,
)
from .identity import confined


def policy(root, campaign, study):
    matches = []
    for item in campaign["payload"]["references"]:
        if item["kind"] != "policy":
            continue
        doc = json.loads(verify_refs(root, [item["artifact"]], max_bytes=32 << 20)[0])
        if (
            isinstance(doc, dict)
            and doc.get("version") == "progressive-study-policy/v1"
            and doc.get("identity") == study
        ):
            matches.append(doc)
    if len(matches) != 1 or seal(matches[0]) != matches[0]:
        raise ValueError("one original frozen study policy required")
    p = matches[0]
    if set(p) - {"diagnostic_pruning_coordinate"} != {
        "version",
        "identity",
        "token_cap",
        "target_tokens",
        "elapsed_cap_us",
        "screening",
        "panel",
        "evaluations",
        "confirmation",
    }:
        raise ValueError("closed finite study policy required")
    if "diagnostic_pruning_coordinate" in p and (
        campaign["purpose"] != "acceptance"
        or p["diagnostic_pruning_coordinate"] not in (1000000, 2000000)
    ):
        raise ValueError("short pruning coordinates are diagnostic-only")
    preflight(**p["screening"], cap_us=p["elapsed_cap_us"])
    if campaign["purpose"] == "research":
        if p["target_tokens"] != 4000000 or p["elapsed_cap_us"] != 7200000000:
            raise ValueError("fixed research screening horizon required")
        validate_screening_panel(p["panel"])
    if not 0 < p["target_tokens"] <= p["token_cap"] <= 2**63 - 1:
        raise ValueError("frozen token ceiling required")
    return p


def journal(root, campaign, p):
    return TrialJournal(
        Path(root)
        / "run-state/discovery-studies"
        / campaign["identity"].split(":")[1]
        / p["identity"].split(":")[1],
        study=p["identity"],
        token_cap=p["token_cap"],
        elapsed_cap_us=p["elapsed_cap_us"],
        provenance_root=root,
    )


def costs(journal, campaign, trial):
    total = dict.fromkeys(("startup", "training", "inference", "search", "verification"), 0)
    tokens = 0
    for plan in campaign["payload"]["tasks"]:
        meta = plan.get("trial")
        if (
            meta is None
            or (meta["study"], meta["ordinal"]) != (trial["study"], trial["ordinal"])
            or meta["phase"] in {"confirmation", "selection"}
        ):
            continue
        with journal.lock():
            events = journal._read()[0]
        selected = [e for e in events if e["operation"].startswith(plan["id"] + "/")]
        spent = dict.fromkeys(plan["caps"], 0)
        reservations = {e["operation"]: e for e in selected if e["event"] == "RESERVE"}
        for e in selected:
            if e["event"] == "RESERVE":
                for k, n in e["cost"].items():
                    spent[k] += n
            else:
                spent["elapsed_us"] -= (
                    reservations[e["operation"]]["cost"]["elapsed_us"] - e["cost"]["elapsed_us"]
                )
        category = {
            "proposal": "search",
            "training": "training",
            "generation": "inference",
            "scoring": "verification",
            "retention": "verification",
        }[meta["phase"]]
        host_keys = {key for key in reservations if "/host-input/" in key}
        host_us = sum(
            e["cost"]["elapsed_us"]
            for e in selected
            if e["operation"] in host_keys and e["event"] == "COMMIT"
        )
        total["startup"] += host_us
        total[category] += spent["elapsed_us"] - host_us
        tokens += spent["tokens"]
    return total, tokens


def prepare_proposal(root, store, task, artifacts, output):
    from oeis_learn.generalization.progressive.projections import describe
    from .progressive_oeis import _Derived

    meta = task["payload"]["trial"]
    p = policy(root, store.campaign, meta["study"])
    j = journal(root, store.campaign, p)
    if meta["phase"] != "proposal" or meta["ordinal"] != len(j.trials):
        # A proposal whose process finished before host publication is replayed
        # from the same durable proposal; no new ask or sampler state exists.
        if str(meta["ordinal"]) not in j.trials:
            raise ValueError("serial original trial roster required")
    history = []
    for i in range(meta["ordinal"]):
        state = j.trials[str(i)]
        if state["result"] is None:
            raise ValueError("prior trial must terminate before next ask")
        result = {k: v for k, v in state["result"].items() if k != "evidence"}
        history.append(dict(proposal=state["proposal"], status=state["status"], result=result))
    selected = None
    if meta["ordinal"] >= 12:
        from oeis_learn.generalization.progressive.selection import rank

        rows = [dict(s["result"], id=k) for k, s in j.trials.items() if int(k) < 12]
        ranked = rank(rows, target=p["target_tokens"], root=root)["ordered"]
        if not ranked or "0" not in ranked:
            raise ValueError("feasible unchanged baseline required before sizing")
        selected = j.trials[ranked[0]]["proposal"]["parameters"]
    data = dict(
        version="progressive-verified-proposal/v1",
        study=p["identity"],
        ordinal=meta["ordinal"],
        history=history,
        selected_learning=selected,
    )
    path = output / "verified-proposal.json"
    atomic_write(
        path,
        seal(dict(version="discovery-channel/v1", kind="search_space", role="proposal", data=data)),
    )
    descriptor = describe(root, reference(root, path)["path"])
    return {"search": [descriptor]}, _Derived(
        store.campaign["identity"],
        task["identity"],
        "propose-trial",
        task["payload"]["policy"],
        (descriptor.ref,),
    )


def original(root, store, task_id, operation, name):
    cut = store.state.get("outputs", {}).get(task_id)
    if (
        store.state["tasks"].get(task_id) != "COMPLETE"
        or cut is None
        or cut["operation"] != operation
    ):
        raise ValueError("committed original trial producer required")
    from oeis_learn.generalization.progressive.contracts import record

    source = store.resolved_plan(
        next(p for p in store.campaign["payload"]["tasks"] if p["id"] == task_id)
    )
    if source is None:
        raise ValueError("original producer trial choice required")
    expected = record(
        "task",
        dict(source, campaign=store.campaign["identity"]),
        created_at=store.campaign["created_at"],
        purpose=store.campaign["purpose"],
        track=store.campaign["track"],
    )
    if cut["task"] != expected["identity"]:
        raise ValueError("original producer task binding differs")
    result = json.loads(verify_refs(root, [cut["result"]])[0])
    if (
        seal(result) != result
        or result.get("task") not in (None, expected["identity"])
        or result.get("campaign") not in (None, store.campaign["identity"])
    ):
        raise ValueError("trial producer changed")
    base = confined(root, cut["result"]["path"]).parent
    ref = result["metrics"][name]
    verify_refs(base, [ref], max_bytes=64 << 30)
    return reference(root, confined(base, ref["path"]))


def weight_ref(root, store, task_id):
    state_ref = original(root, store, task_id, "train", "checkpoint_record")
    state = json.loads(verify_refs(root, [state_ref])[0])
    base = (
        Path(root)
        / "run-state/discovery-learning"
        / store.campaign["identity"].split(":")[1]
        / state["payload"]["task"].split(":")[1]
        / "checkpoints"
    )
    from oeis_learn.generalization.progressive.state import restore

    source = store.resolved_plan(
        next(p for p in store.campaign["payload"]["tasks"] if p["id"] == task_id)
    )
    restore(
        state,
        root=base,
        compatibility=source["initializer"]["compatibility"],
        campaign=store.campaign["identity"],
        task=store.state["outputs"][task_id]["task"],
    )
    full = json.loads(verify_refs(base, [state["payload"]["full_state"]], max_bytes=64 << 20)[0])
    ref = full["states"]["weights"]
    verify_refs(base, [ref], max_bytes=64 << 30)
    return reference(root, confined(base, ref["path"]))


def before(root, store, task):
    meta = task["payload"].get("trial")
    if meta is None or meta["phase"] not in {"training", "generation", "scoring"}:
        return
    p = policy(root, store.campaign, meta["study"])
    j = journal(root, store.campaign, p)
    state = j.trials.get(str(meta["ordinal"]))
    if state is None:
        raise ValueError("durable ask required before trial dispatch")
    if state["status"] in {"PROPOSED", "RESERVED"} or state["recovery_pending"]:
        j.dispatch(meta["ordinal"])
    elif state["status"] != "RUNNING":
        raise ValueError("terminal trial cannot perform more work")


def skip(store, meta):
    tasks = dict(store.state["tasks"])
    for p in store.campaign["payload"]["tasks"]:
        t = p.get("trial")
        if (
            t
            and (t["study"], t["ordinal"]) == (meta["study"], meta["ordinal"])
            and t["phase"] not in {"proposal", "confirmation", "selection"}
            and tasks.get(p["id"]) not in {"COMPLETE", "FAILED", "PARTIAL"}
        ):
            tasks[p["id"]] = "SKIPPED"
    store.append(dict(store.state, tasks=tasks))


def after(root, store, task, result, budget):
    meta = task["payload"].get("trial")
    if meta is None or meta["phase"] in {"confirmation", "selection"}:
        return
    p = policy(root, store.campaign, meta["study"])
    j = journal(root, store.campaign, p)
    ordinal = meta["ordinal"]
    if meta["phase"] == "proposal":
        ref = original(root, store, task["payload"]["id"], "propose-trial", "proposal")
        proposal = json.loads(verify_refs(root, [ref])[0])
        if (proposal["study"], proposal["ordinal"]) != (meta["study"], ordinal):
            raise ValueError("proposal study/ordinal differs")
        actual = j.propose(ordinal, parameters=proposal["parameters"], seed=proposal["seed"])
        if actual != proposal:
            raise ValueError("proposal changed before dispatch")
        try:
            store.choose_trial(proposal)
        except ValueError as exc:
            if str(exc) != "proposal lacks a frozen prequalified trial alternative":
                raise
            j.dispatch(ordinal)
            timing, tokens = costs(budget, store.campaign, meta)
            j.tell(
                ordinal,
                dict(
                    status="INFEASIBLE",
                    tokens=tokens,
                    elapsed_us=sum(timing.values()),
                    value=None,
                    role="development",
                ),
            )
            skip(store, meta)
        return
    timing, tokens = costs(budget, store.campaign, meta)
    elapsed = sum(timing.values())
    status = result.get("metrics", {}).get("task_status", result["mechanical_status"])
    if status == "PAUSED":
        j.interrupt(ordinal, tokens=tokens, elapsed_us=elapsed)
        return
    if status in {"FAILED", "INVALID", "BLOCKED", "PARTIAL"}:
        j.tell(
            ordinal,
            dict(
                status="FAILED" if status in {"FAILED", "INVALID", "BLOCKED"} else "PARTIAL",
                tokens=tokens,
                elapsed_us=elapsed,
                value=None,
                role="development",
            ),
        )
        skip(store, meta)
        return
    if meta["phase"] != "scoring":
        return
    spec = p["evaluations"].get(task["payload"]["id"])
    if spec is None or set(spec) != {
        "generation_task",
        "checkpoint_task",
        "qualification",
        "roster",
        "retention_baseline_generation",
        "retention_baseline_scoring",
        "retention_candidate_generation",
        "retention_candidate_scoring",
    }:
        raise ValueError("closed bound development evaluation required")
    producers = dict(
        generation=original(root, store, spec["generation_task"], "generate", "candidates"),
        scoring=original(root, store, task["payload"]["id"], "score", "scores"),
        checkpoint=weight_ref(root, store, spec["checkpoint_task"]),
        qualification=spec["qualification"],
    )
    for key in (
        "retention_baseline_generation",
        "retention_baseline_scoring",
        "retention_candidate_generation",
        "retention_candidate_scoring",
    ):
        gen = key.endswith("generation")
        producers[key] = original(
            root,
            store,
            spec[key],
            "generate" if gen else "score",
            "candidates" if gen else "scores",
        )
    if store.campaign["purpose"] == "research":
        checkpoint_plan = store.resolved_plan(
            next(
                t for t in store.campaign["payload"]["tasks"] if t["id"] == spec["checkpoint_task"]
            )
        )
        qualified = checkpoint_plan["iteration_budget"]["producer"]
        if spec["qualification"] is not None and spec["qualification"] != qualified:
            raise ValueError("trial qualification differs from original selected model/update case")
        producers["qualification"] = qualified
    docs = {
        k: json.loads(verify_refs(root, [ref])[0])
        for k, ref in producers.items()
        if k != "checkpoint"
    }

    def outcome(scores, task):
        row = scores.get("outcomes", {}).get(task, {}).get("equal_budget", {})
        return (
            "success"
            if row.get("success") is True
            else "failure"
            if row.get("complete") is True
            else "unresolved"
        )

    roster = spec["roster"]
    retention_roster = docs["retention_baseline_generation"]["records"]
    evidence = seal(
        dict(
            version="progressive-trial-evidence/v1",
            role="development",
            checkpoint=producers["checkpoint"]["sha256"],
            qualification=docs["qualification"]["identity"],
            scope="NATIVE_VERIFIED"
            if store.campaign["purpose"] == "research"
            else "DIAGNOSTIC_ONLY",
            roster=roster,
            outcomes=[dict(r, outcome=outcome(docs["scoring"], r["task"])) for r in roster],
            retention=[
                dict(
                    task=t,
                    baseline=outcome(docs["retention_baseline_scoring"], t),
                    candidate=outcome(docs["retention_candidate_scoring"], t),
                )
                for t in retention_roster
            ],
            generator=docs["generation"]["identity"],
            scorer=docs["scoring"]["identity"],
            costs=timing,
            cost_scope="supervised task intervals; startup and cleanup enclosed in phase costs",
            producer_refs=producers,
        )
    )
    try:
        value = objective(evidence, elapsed_us=elapsed, root=root)
    except ValueError:
        j.tell(
            ordinal,
            dict(
                status="INFEASIBLE",
                tokens=tokens,
                elapsed_us=elapsed,
                value=None,
                role="development",
                evidence=evidence,
            ),
        )
        skip(store, meta)
        return
    coordinate = meta["coordinate"]
    if tokens < coordinate:
        j.tell(
            ordinal,
            dict(
                status="PARTIAL",
                tokens=tokens,
                elapsed_us=elapsed,
                value=value,
                role="development",
                evidence=evidence,
            ),
        )
        skip(store, meta)
        return
    path = j.root / f"intermediate-{ordinal:02d}-{coordinate:012d}.json"
    observation = seal(
        dict(
            ordinal=ordinal,
            coordinate=coordinate,
            value=value,
            status="COMPLETE",
            role="development",
            evidence=evidence,
        )
    )
    if path.exists() and load(path) != observation:
        raise ValueError("intermediate observation cannot change")
    if not path.exists():
        atomic_write(path, observation)
    references = []
    for f in sorted(j.root.glob("intermediate-*.json")):
        value_record = load(f)
        prior = value_record["ordinal"]
        if prior >= ordinal or j.trials[str(prior)]["status"] != "COMPLETE":
            continue
        if seal(value_record) != value_record or value_record["role"] != "development":
            raise ValueError("original completed comparable observation required")
        proof = value_record["evidence"]
        actual = objective(proof, elapsed_us=sum(proof["costs"].values()), root=root)
        if actual != value_record["value"]:
            raise ValueError("intermediate pruning feedback differs from original outcomes")
        references.append(value_record)
    diagnostic = p.get("diagnostic_pruning_coordinate")
    prune = pruning(
        ordinal=ordinal,
        value=value,
        coordinate=diagnostic or coordinate,
        references=[
            dict(r, coordinate=diagnostic) if diagnostic and r["coordinate"] == coordinate else r
            for r in references
        ],
    )
    if prune or coordinate >= p["target_tokens"]:
        status = "PRUNED" if prune else "COMPLETE" if tokens >= p["target_tokens"] else "PARTIAL"
        j.tell(
            ordinal,
            dict(
                status=status,
                tokens=tokens,
                elapsed_us=elapsed,
                value=value,
                role="development",
                evidence=evidence,
            ),
        )
        if prune:
            skip(store, meta)


def prepare_selection(root, store, task, output):
    from oeis_learn.generalization.progressive.selection import confirmation
    from oeis_learn.generalization.progressive.projections import describe
    from .progressive_oeis import _Derived

    p = policy(root, store.campaign, task["payload"]["trial"]["study"])
    j = journal(root, store.campaign, p)
    expected = {
        t["trial"]["ordinal"]
        for t in store.campaign["payload"]["tasks"]
        if t.get("trial", {}).get("study") == p["identity"] and t["trial"]["phase"] == "proposal"
    }
    if set(map(int, j.trials)) != expected or any(s["result"] is None for s in j.trials.values()):
        raise ValueError("finite screening roster must terminate before selection")
    rows = [
        dict(s["result"], id=k, parameters=s["proposal"]["parameters"]) for k, s in j.trials.items()
    ]
    config = p["confirmation"]
    if set(config) != {"panels", "seeds"}:
        raise ValueError("closed preallocated confirmation design required")
    panels = []
    for ref in config["panels"]:
        from oeis_learn.generalization.progressive.contracts import validate

        panel = validate("cohort", json.loads(verify_refs(root, [ref])[0]))
        if (
            panel["purpose"] != store.campaign["purpose"]
            or panel["track"] != store.campaign["track"]
        ):
            raise ValueError("confirmation panel scope differs")
        panels.append(
            dict(
                identity=panel["identity"],
                role=panel["payload"]["role"],
                exposed=False,
                groups=sorted({g for row in panel["payload"]["rows"] for g in row["groups"]}),
            )
        )
    from oeis_learn.generalization.progressive.selection import rank

    ordered = rank(rows, target=p["target_tokens"], root=root)["ordered"]
    by_id = {r["id"]: r for r in rows}
    seen = set()
    count = 0
    for name in ["0", *ordered]:
        if name not in ordered:
            continue
        params = by_id[name]["parameters"]
        key = json.dumps(dict(params, shape=params.get("shape", "reference")), sort_keys=True)
        if key not in seen:
            seen.add(key)
            count += 1
    count = min(4, count)
    selected = confirmation(
        rows,
        baseline="0",
        panels=panels[:count],
        seeds=config["seeds"],
        target=p["target_tokens"],
        root=root,
    )
    if store.campaign["purpose"] == "research" and expected != set(range(16)):
        raise ValueError("research requires the frozen twelve-plus-four roster")
    path = output / "finalists.json"
    atomic_write(
        path,
        seal(
            dict(
                version="discovery-channel/v1",
                kind="public_evidence",
                role="verifier",
                data=selected,
            )
        ),
    )
    descriptor = describe(root, reference(root, path)["path"])
    return {"public": [descriptor]}, _Derived(
        store.campaign["identity"],
        task["identity"],
        "publish",
        task["payload"]["policy"],
        (descriptor.ref,),
    )


def commit_selection(root, store, task, result):
    p = policy(root, store.campaign, task["payload"]["trial"]["study"])
    j = journal(root, store.campaign, p)
    cut = store.state["outputs"][task["payload"]["id"]]
    base = confined(root, cut["result"]["path"]).parent
    selected = json.loads(verify_refs(base, [result["metrics"]["finalists"]])[0])
    choices = dict(store.state.get("choices", {}))
    tasks = dict(store.state["tasks"])
    for plan in store.campaign["payload"]["tasks"]:
        meta = plan.get("trial")
        if not meta or meta["study"] != p["identity"] or meta["phase"] != "confirmation":
            continue
        position = meta["ordinal"]
        if position >= len(selected["selected"]):
            tasks[plan["id"]] = "SKIPPED"
            continue
        if (
            plan["initializer"]["mode"] != "random"
            or plan["initializer"]["artifact"] is not None
            or plan["seed"] not in selected["seeds"]
            or plan["caps"]["elapsed_us"] > 43200000000
        ):
            raise ValueError("fresh paired confirmation entitlement required")
        source = j.trials[selected["selected"][position]]["proposal"]["parameters"]
        matching = [
            i
            for i, c in enumerate(plan.get("trial_choices", []))
            if dict(c["parameters"], shape=c["parameters"].get("shape", "reference"))
            == dict(source, shape=source.get("shape", "reference"))
        ]
        if len(matching) != 1:
            raise ValueError("finalist lacks original prequalified confirmation scope")
        if plan["id"] in choices and choices[plan["id"]] != matching[0]:
            raise ValueError("locked finalist cannot change")
        choices[plan["id"]] = matching[0]
    for position in range(len(selected["selected"])):
        seeds = {
            plan["seed"]
            for plan in store.campaign["payload"]["tasks"]
            if plan.get("trial", {}).get("study") == p["identity"]
            and plan["trial"]["phase"] == "confirmation"
            and plan["trial"]["ordinal"] == position
        }
        if seeds != set(selected["seeds"]):
            raise ValueError("both fresh paired confirmation seeds must be preallocated")
    path = j.root / "selection.json"
    if path.exists() and load(path) != selected:
        raise ValueError("locked finalists cannot change")
    if not path.exists():
        atomic_write(path, selected)
    store.append(dict(store.state, choices=choices, tasks=tasks))


def validate_frozen(root, campaign):
    """All adaptive alternatives are finite, paired and paid for before outcomes."""
    from oeis_learn.generalization.progressive.trial_scopes import variants

    refs = {
        r["artifact"]["sha256"]: r["artifact"]
        for r in campaign["payload"]["references"]
        if r["kind"] == "policy"
    }
    studies = {t["trial"]["study"] for t in campaign["payload"]["tasks"] if "trial" in t}
    for identity in studies:
        p = policy(root, campaign, identity)
        plans = [
            t for t in campaign["payload"]["tasks"] if t.get("trial", {}).get("study") == identity
        ]
        for ordinal in {t["trial"]["ordinal"] for t in plans if t["trial"]["phase"] == "proposal"}:
            members = [
                t
                for t in plans
                if t["trial"]["ordinal"] == ordinal
                and t["trial"]["phase"] not in {"confirmation", "selection"}
            ]
            if campaign["purpose"] == "research" and (
                p["screening"]["finalization_us"] < 1800000000
                or any(
                    t["trial"].get("shared_finalization_us") != p["screening"]["finalization_us"]
                    for t in members
                )
            ):
                raise ValueError("one protected original trial finalization reserve required")
            if (
                sum(t["caps"]["elapsed_us"] for t in members)
                + (p["screening"]["finalization_us"] if campaign["purpose"] == "research" else 0)
                > p["elapsed_cap_us"]
                or sum(t["caps"]["tokens"] for t in members) > p["token_cap"]
            ):
                raise ValueError("trial children exceed original finite screening reservation")
            proposals = [t for t in members if t["trial"]["phase"] == "proposal"]
            if len(proposals) != 1:
                raise ValueError("one durable serial ask per trial required")
            seed = int(digest([identity, ordinal]).split(":")[1][:8], 16)
            if any(t["seed"] != seed for t in members):
                raise ValueError("preallocated trial seed differs from deterministic ask")
            if campaign["purpose"] == "research":
                if {
                    t["trial"]["coordinate"] for t in members if t["trial"]["phase"] == "scoring"
                } != {1000000, 2000000, 4000000}:
                    raise ValueError("all three original screening looks must be reserved")
        members_by_id = {t["id"]: t for t in plans}
        for score_id, evaluation in p["evaluations"].items():
            scored = members_by_id.get(score_id)
            if scored is None or scored["trial"]["phase"] != "scoring":
                raise ValueError("development tell requires its frozen scoring task")
            ordinal = scored["trial"]["ordinal"]
            expected = {
                "generation_task": "generation",
                "checkpoint_task": "training",
                "retention_baseline_generation": "retention",
                "retention_baseline_scoring": "retention",
                "retention_candidate_generation": "retention",
                "retention_candidate_scoring": "retention",
            }
            for name, phase in expected.items():
                source = members_by_id.get(evaluation[name])
                if (
                    source is None
                    or source["trial"]["ordinal"] != ordinal
                    or source["trial"]["phase"] != phase
                ):
                    raise ValueError("trial feedback cannot consume another study or confirmation")
            for source in [
                scored,
                *[
                    members_by_id[evaluation[k]]
                    for k in ("retention_baseline_scoring", "retention_candidate_scoring")
                ],
            ]:
                request = json.loads(verify_refs(root, [refs[source["policy"]]])[0])
                if request["operation"] != "score":
                    raise ValueError("original development scorer required")
                for path in request["artifacts"]["score_truth"]:
                    if (
                        not isinstance(path, str)
                        or load(confined(root, path))["kind"] != "development_truth"
                    ):
                        raise ValueError("confirmation or assessment cannot feed the sampler")
            if campaign["purpose"] == "research":
                expected_roster = [
                    dict(task=r["id"], group=r["group"], stratum=r["stratum"]) for r in p["panel"]
                ]
                if sorted(evaluation["roster"], key=lambda r: r["task"]) != sorted(
                    expected_roster, key=lambda r: r["task"]
                ):
                    raise ValueError("the exact frozen 128-task panel is required")
        for plan in plans:
            meta = plan["trial"]
            if campaign["purpose"] == "research" and plan["stage"] != "tuning":
                raise ValueError("separate finite tuning stage required")
            for variant in variants(plan):
                if meta["phase"] not in {"training", "confirmation"} or plan["kind"] != "training":
                    continue
                if "trial_choice" not in variant:
                    raise ValueError("explicit frozen trial alternatives required")
                request = json.loads(verify_refs(root, [refs[variant["policy"]]])[0])
                paths = request["artifacts"]["training"]
                if (
                    request["operation"] != "train"
                    or len(paths) != 1
                    or not isinstance(paths[0], str)
                ):
                    raise ValueError("one original training-only variant request required")
                envelope = load(confined(root, paths[0]))
                if (seal(envelope)!=envelope or envelope.get('version')!='discovery-channel/v1'
                        or envelope.get('kind')!='training_request' or envelope.get('role')!='learner'):
                    raise ValueError('original training-only variant channel required')
                data = envelope["data"]
                package = json.loads(verify_refs(root, [data["artifact"]], max_bytes=256 << 20)[0])
                from .progressive_learner import compatibility,validate_input
                from oeis_learn.generalization.progressive.projections import _validate_channel
                from oeis_learn.generalization.progressive.trial_scopes import materialize
                _validate_channel('training_request',data)
                if compatibility(package)!=data['compatibility']:
                    raise ValueError('original variant base compatibility differs')
                # The retained package is the immutable data base. Calibration
                # and installed execution both expand the selected finite law;
                # freeze must validate that same input, not demand one copy of
                # every training reservoir per possible hyperparameter choice.
                package=materialize(package,variant)
                validate_input(package,dict(purpose=campaign['purpose'],track=campaign['track'],payload=variant),allow_pending=True)
                params = variant["trial_choices"][variant["trial_choice"]]["parameters"]
                actual = {
                    k: package["update_policy"][k]
                    for k in (
                        "lr_multiplier",
                        "schedule",
                        "effective_batch",
                        "weight_decay",
                        "dropout",
                    )
                }
                if params.get("shape", "reference") != package["profile"]["shape"] or any(
                    actual[k] != params[k] for k in actual
                ):
                    raise ValueError(
                        "proposal settings differ from original frozen learner package"
                    )
                if meta["phase"] == "confirmation":
                    if (
                        variant["initializer"]["mode"] != "random"
                        or variant["initializer"]["artifact"] is not None
                    ):
                        raise ValueError("confirmation must initialize fresh optimizer and weights")
                    if campaign["purpose"] == "research" and (
                        variant["token_target"] != 25000000
                        or package["update_policy"]["schedule_horizon_tokens"] != 400000000
                        or (
                            params["schedule"] == "warmup_cosine"
                            and package["update_policy"]["warmup_tokens"] != 1250000
                        )
                    ):
                        raise ValueError("frozen fresh confirmation horizon differs")


def project_variant(root, store, task, operation, artifacts, derived, output):
    """Host-only expansion of a frozen finite law; never accepts runtime code."""
    from oeis_learn.generalization.progressive.projections import describe
    from oeis_learn.generalization.progressive.trial_scopes import materialize
    from oeis_learn.generalization.progressive.models import resolve_profile
    from .progressive_oeis import _Derived

    plan = task["payload"]
    if "trial_choice" not in plan or operation not in {"train", "generate"}:
        return artifacts, derived
    result = {k: list(v) for k, v in artifacts.items()}
    refs = list(derived.artifacts) if derived is not None else []
    for channel, items in result.items():
        for i, item in enumerate(items):
            envelope = json.loads(verify_refs(root, [item.ref])[0])
            data = envelope["data"]
            if operation == "train" and item.kind == "training_request":
                package = json.loads(verify_refs(root, [data["artifact"]], max_bytes=256 << 20)[0])
                expanded = materialize(package, plan)
                path = output / "training-package.json"
                atomic_write(path, expanded)
                replacement = dict(
                    data,
                    artifact=reference(root, path),
                    compatibility=plan["initializer"]["compatibility"],
                )
            elif operation == "generate" and item.kind == "recipe":
                base = json.loads(verify_refs(root, [data["artifact"]])[0])
                params = plan["trial_choices"][plan["trial_choice"]]["parameters"]
                recipe = resolve_profile(
                    base.get("base_recipe", base),
                    shape=params.get("shape", "reference"),
                    seed=plan["seed"],
                    dropout=params["dropout"],
                )
                path = output / "model-profile.json"
                atomic_write(path, recipe)
                replacement = dict(
                    data,
                    artifact=reference(root, path),
                    compatibility=plan["initializer"]["compatibility"],
                )
            else:
                continue
            path = output / (channel + "-" + str(i) + ".json")
            atomic_write(path, seal(dict(envelope, data=replacement)))
            descriptor = describe(root, reference(root, path)["path"])
            items[i] = descriptor
            refs.append(descriptor.ref)
    return result, _Derived(
        store.campaign["identity"], task["identity"], operation, plan["policy"], tuple(refs)
    )


def prepare_inputs(root, store, task, request, output, budget):
    """Charge measured synchronous input/proposal preparation before role startup."""
    import time
    from .progressive_outputs import resolve

    remaining = budget.task_remaining(task)
    debit = dict.fromkeys(remaining, 0)
    debit["elapsed_us"] = remaining["elapsed_us"]
    if debit["elapsed_us"] <= task["payload"]["startup_us"] + task["payload"]["finalization_us"]:
        raise TimeoutError("task input/finalization allowance exhausted")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    start = time.monotonic_ns() // 1000
    charge = budget.reserve(
        task["payload"]["id"], "host-input", store.ordinal, debit, boot=boot, monotonic_us=start
    )
    try:
        artifacts, derived = resolve(root, store, task, request, output / "slots")
        meta = task["payload"].get("trial")
        if meta and meta["phase"] == "proposal":
            artifacts, derived = prepare_proposal(root, store, task, artifacts, output / "proposal")
        elif meta and meta["phase"] == "selection":
            artifacts, derived = prepare_selection(root, store, task, output / "selection")
        artifacts, derived = project_variant(
            root, store, task, request["operation"], artifacts, derived, output / "variant"
        )
        before(root, store, task)
        return artifacts, derived
    finally:
        budget.commit(
            charge["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
