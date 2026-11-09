"""Bounded collection from a frozen campaign and its committed owned outputs."""

import json
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
    verify_refs,
)
from oeis_learn.generalization.progressive.report import (
    KINDS,
    MAX_BYTES,
    terminal_request,
    validate_rows,
)
from oeis_learn.pipeline_validation.identity import confined


def identity(label):
    return (
        label
        if isinstance(label, str) and len(label) == 71 and label.startswith("sha256:")
        else digest(label)
    )


def collect(root, folder, output):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.campaign import validate_campaign
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.storage import StorageGuard

    from .progressive_evidence import public_report_input

    root = Path(root)
    folder = Path(folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fallback = terminal_request(root, folder)
    if not (folder / "run.json").is_file():
        return fallback
    invocation = load(folder / "run.json")
    if seal(invocation) != invocation:
        raise ValueError("original run binding differs")
    campaign = validate_campaign(
        load(confined(root, invocation["campaign_path"]), maximum=32 << 20)
    )
    if campaign["identity"] != invocation["campaign"]:
        raise ValueError("original report campaign differs")
    guard = StorageGuard(
        [output],
        max_bytes=min(MAX_BYTES, campaign["payload"]["storage"]["package_bytes"]),
        max_inodes=8192,
    )
    tables = {k: [] for k in KINDS}
    errors = []
    decisions = []
    scope_record = None
    scopes = []
    cohorts = []
    source = None

    def error(exc, producer):
        return {
            "type": type(exc).__name__,
            "code": "COLLECTOR_FAILURE",
            "producer": producer,
            "local_traceback": None,
        }

    def qualification(value, artifact):
        if seal(value) != value:
            raise ValueError("original qualification case differs")
        row = {
            k: value[k]
            for k in ("identity", "source", "scope", "device", "status", "requested", "executed")
        }
        row["receipt"] = artifact
        if not any(r["identity"] == row["identity"] for r in tables["qualifications"]):
            tables["qualifications"].append(row)

    for ref in campaign["payload"]["references"]:
        if ref["kind"] not in {"policy", "groups", "qualification"}:
            continue
        try:
            value = json.loads(verify_refs(root, [ref["artifact"]], max_bytes=32 << 20)[0])
            if not isinstance(value, dict):
                continue
            if value.get("version") == "discovery-cohort/v1" or set(value) - {"primary"} == {
                "claim",
                "bootstrap_seed",
                "comparisons",
                "links",
                "costs",
            }:
                tables["authorities"].append(
                    {
                        "artifact": ref["artifact"],
                        "raw": verify_refs(root, [ref["artifact"]], max_bytes=4 << 20)[0].decode(),
                    }
                )
            if value.get("version") == "progressive-public-scope/v1":
                scopes.append((ref["artifact"], value))
            if value.get("version") == "discovery-cohort/v1" and value["payload"]["role"] in {
                "development",
                "confirmation",
            }:
                cohorts.append(value)
            if value.get("version") == "progressive-hip-case/v1":
                qualification(value, ref["artifact"])
                source = value["source"]
            if value.get("version") == "generalization-qualification/v2":
                base = confined(root, ref["artifact"]["path"]).parent
                artifacts = value["producer"]["artifacts"]
                if len(artifacts) > 4096:
                    raise ValueError("bounded original qualification inventory required")
                for original in artifacts:
                    if not original["path"].endswith(".json") or original["bytes"] > 4 << 20:
                        continue
                    raw = verify_refs(base, [original], max_bytes=4 << 20)[0]
                    case = json.loads(raw)
                    if isinstance(case, dict) and case.get("version") == "progressive-hip-case/v1":
                        qualification(
                            case,
                            dict(
                                original,
                                path=str(confined(base, original["path"]).relative_to(root)),
                            ),
                        )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(error(exc, ref["name"]))
    if len(scopes) > 1:
        raise ValueError("one frozen public comparison scope required per campaign report")
    frozen = scopes[0][1] if scopes else None
    if frozen:
        if seal(frozen) != frozen:
            raise ValueError("frozen public scope seal differs")
        scope_record = scopes[0][0]
        source = frozen["source"]
        selected = set(frozen["qualification_producers"].values())
        tables["qualifications"] = [
            q for q in tables["qualifications"] if q["identity"] in selected
        ]
    rosters = []
    for cohort in cohorts:
        rosters.append(
            [
                {
                    "task": identity(r["id"]),
                    "group": identity(r["groups"][0]),
                    "stratum": r["stratum"],
                }
                for r in cohort["payload"]["rows"]
            ]
        )
    merged = {}
    for roster in rosters:
        for row in roster:
            if row["task"] in merged and merged[row["task"]] != row:
                raise ValueError("ambiguous public task membership")
            merged[row["task"]] = row
    all_roster = list(merged.values())
    candidates = [all_roster, *rosters]
    if frozen:
        candidates = [r for r in candidates if digest(r) == frozen["roster"]]
    tables["roster"] = candidates[0] if candidates else []
    roster = {r["task"]: r for r in tables["roster"]}
    control = root / "run-state/discovery-control" / campaign["identity"][7:]
    store = RunStore(control, campaign) if (control / "campaign.json").is_file() else None
    journal_root = root / "run-state/discovery-accounting" / campaign["identity"][7:]
    journal = (
        BudgetJournal(journal_root, campaign["identity"], campaign["payload"]["caps"])
        if (journal_root / "budget-binding.json").is_file()
        else None
    )
    seeds = sorted({p["seed"] for p in campaign["payload"]["tasks"]})
    seen_outcomes = set()
    seen_results = set()
    task_records = {}
    arm_map = {
        "static": "baseline",
        "refreshed": "candidate",
        "continued_synthetic": "baseline",
        "replay": "candidate",
        "no_library": "baseline",
        "library": "candidate",
        "without_source": "baseline",
        "with_source": "candidate",
    }
    for template in campaign["payload"]["tasks"]:
        plan = store.resolved_plan(template) if store else template
        if plan is None:
            continue
        task = identity(plan["id"])
        arm = arm_map.get(plan["arm"], "candidate")
        from oeis_learn.generalization.progressive.contracts import record

        task_record = record(
            "task",
            dict(plan, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        task_records[task_record["identity"]] = plan
        if journal:
            left = journal.plan_remaining(plan)
            spent = {k: plan["caps"][k] - left[k] for k in left}
            useful = 0
            state = root / "run-state/discovery-learning" / campaign["identity"][7:]
            from oeis_learn.generalization.progressive.contracts import record

            t = record(
                "task",
                dict(plan, campaign=campaign["identity"]),
                created_at=campaign["created_at"],
                purpose=campaign["purpose"],
                track=campaign["track"],
            )
            files = sorted((state / t["identity"][7:]).glob("event-*.json"))
            if files:
                latest = load(files[-1])
                useful = latest["state"]["useful_tokens"]
                # Child ledgers include ancestor exposure; only this task's useful
                # delta belongs against its own frozen cap.
                lineage = state / t["identity"][7:] / "lineage.json"
                if lineage.is_file():
                    useful -= load(lineage)["base"]["useful"]
            if not 0 <= useful <= spent["tokens"]:
                raise ValueError("original useful exposure exceeds charged task allowance")
            if files and latest["state"].get("checkpoint"):
                checkpoint = latest["state"]["checkpoint"]
                tables["observations"].append(
                    {
                        "producer": checkpoint["identity"],
                        "data": {
                            "checkpoint": checkpoint["identity"],
                            "cuts": checkpoint["payload"]["cuts"],
                            "task": task,
                        },
                    }
                )
            tables["accounting"].append(
                {
                    "task": task,
                    "arm": arm,
                    "tokens": spent["tokens"],
                    "useful_tokens": useful,
                    "discarded_tokens": spent["tokens"] - useful,
                    "elapsed_us": spent["elapsed_us"],
                    "token_cap": plan["caps"]["tokens"],
                    "elapsed_cap_us": plan["caps"]["elapsed_us"],
                    "resources": {"spent": spent, "caps": plan["caps"]},
                }
            )
        parent = plan["initializer"].get("parent_task")
        if parent is None and "parent_slot" in plan["initializer"]:
            parents = [
                p
                for p in campaign["payload"]["tasks"]
                if p["slot"] == plan["initializer"]["parent_slot"]
                and p["seed"] == plan["seed"]
                and p["arm"] == plan["arm"]
            ]
            if len(parents) == 1:
                parent = parents[0]["id"]
        if parent:
            tables["ancestry"].append(
                {"child": task, "parent": identity(parent), "relation": "continuation"}
            )
        cut = store.state.get("outputs", {}).get(plan["id"]) if store else None
        if not cut:
            continue
        try:
            result = json.loads(verify_refs(root, [cut["result"]], max_bytes=32 << 20)[0])
            base = confined(root, cut["result"]["path"]).parent
            if seal(result) != result or result["operation"] != cut["operation"]:
                raise ValueError("owned result binding differs")
            seen_results.add(result["identity"])
            if result.get("producer_error") and not fallback.get("producer_error"):
                fallback["producer_error"] = public_error(root, base, result["producer_error"])
            public_keys = {
                "charged_tokens",
                "useful_tokens",
                "discarded_tokens",
                "device",
                "diagnostic_only",
                "task_status",
                "actual_elapsed_us",
            }
            data = {
                k: v
                for k, v in result.get("metrics", {}).items()
                if k in public_keys and type(v) in (str, int, float, bool, type(None))
            }
            tables["observations"].append({"producer": result["identity"], "data": data})
            if "timing" in result.get("metrics", {}):
                timing = json.loads(
                    verify_refs(base, [result["metrics"]["timing"]], max_bytes=4 << 20)[0]
                )
                tables["observations"].append({"producer": result["identity"], "data": timing})
            if "scores" in result.get("metrics", {}):
                scores = json.loads(
                    verify_refs(base, [result["metrics"]["scores"]], max_bytes=32 << 20)[0]
                )
                for raw_id, value in scores["outcomes"].items():
                    member = identity(raw_id)
                    if member not in roster:
                        continue
                    key = (member, plan["seed"], arm)
                    if key in seen_outcomes:
                        errors.append(
                            error(
                                ValueError(
                                    "multiple comparison outputs require their own frozen scope"
                                ),
                                "outcomes",
                            )
                        )
                        continue
                    seen_outcomes.add(key)
                    tables["outcomes"].append(
                        dict(
                            roster[member],
                            seed=plan["seed"],
                            arm=arm,
                            outcome=value["equal_budget"]["outcome"],
                        )
                    )
            if "decision_bundle" in result.get("metrics", {}):
                bundle = json.loads(
                    verify_refs(base, [result["metrics"]["decision_bundle"]], max_bytes=32 << 20)[0]
                )
                tables["comparisons"].append(
                    {
                        "id": bundle["decision"]["identity"],
                        "rows": bundle["rows"],
                        "roster": bundle["roster"],
                        "seeds": bundle["seeds"],
                        "design": bundle["design"],
                        "verdict": bundle["decision"]["payload"]["evidence_verdict"],
                        "action": bundle["decision"]["payload"]["next_action"],
                    }
                )
                if (
                    frozen
                    and digest(bundle["roster"]) == frozen["roster"]
                    and bundle["design"] == frozen["design"]
                ):
                    decisions.append(
                        {
                            "verdict": bundle["decision"]["payload"]["evidence_verdict"],
                            "action": bundle["decision"]["payload"]["next_action"],
                        }
                    )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(error(exc, plan["id"]))
    # Retain bounded numeric phase observations, without copying model weights,
    # corpora, private continuations or an unrestricted directory tree.
    design = (
        frozen["design"]
        if frozen
        else {
            "claim": "capability",
            "bootstrap_seed": 0,
            "comparisons": 1,
            "links": [],
            "costs": None,
        }
    )
    scope = (
        frozen["scope"]
        if frozen
        else {
            "required": list(KINDS),
            "qualification_scopes": [],
            "tasks": [identity(p["id"]) for p in campaign["payload"]["tasks"]],
        }
    )
    if frozen:
        seeds = frozen["seeds"]
    # Follow only the owned invocation chain and explicit storage inventory.
    # Failed/partial roles are not committed scientific outcomes, but their
    # original errors and measured resource observations must still survive.
    roots = []
    cursor = folder
    seen = set()
    for _ in range(4096):
        key = str(cursor)
        if key in seen:
            raise ValueError("cyclic run invocation ancestry")
        seen.add(key)
        roots.append(cursor)
        manifest = load(cursor / "run.json")
        if seal(manifest) != manifest or manifest["campaign"] != campaign["identity"]:
            raise ValueError("invocation scope differs")
        previous = manifest.get("previous")
        if not previous:
            break
        cursor = confined(root, previous)
    else:
        raise ValueError("bounded invocation ancestry required")
    first = None
    for invocation_root in reversed(roots):
        files = [invocation_root / "result.json"]
        owners = {}
        for path in sorted((journal_root / "storage-inventory").glob("*.json")):
            inventory = load(path)
            if seal(inventory) != inventory or inventory["campaign"] != campaign["identity"]:
                raise ValueError("owned output inventory differs")
            owned = confined(root, inventory["output"], exists=False)
            if owned.is_relative_to(invocation_root):
                files.append(owned / "role/result.json")
                owners[owned / "role/result.json"] = inventory["task"]
                if owned.name == "host":
                    files.append(owned.parent / "result.json")
        for path in files:
            if not path.is_file():
                continue
            value = load(path, maximum=32 << 20)
            if seal(value) != value:
                raise ValueError("original terminal result differs")
            if value["identity"] not in seen_results:
                seen_results.add(value["identity"])
                metrics = value.get("metrics", {})
                public = {
                    k: v
                    for k, v in metrics.items()
                    if k
                    in {
                        "charged_tokens",
                        "useful_tokens",
                        "discarded_tokens",
                        "device",
                        "diagnostic_only",
                        "task_status",
                        "actual_elapsed_us",
                        "completed_tasks",
                        "primary_scaling_eligible",
                        "autonomous_discoveries",
                    }
                }
                tables["observations"].append({"producer": value["identity"], "data": public})
                for name in ("cycle", "source_results"):
                    if metrics.get(name):
                        original = json.loads(
                            verify_refs(root, [metrics[name]], max_bytes=4 << 20)[0]
                        )
                        if seal(original) != original:
                            raise ValueError("original host cycle summary differs")
                        # These summaries contain public outcomes, identities and
                        # cost, never learner examples or hidden continuations.
                        tables["observations"].append(
                            {"producer": original["identity"], "data": original}
                        )
                if (
                    value.get("mechanical_status") == "COMPLETE"
                    and metrics.get("scores")
                    and path in owners
                ):
                    plan = task_records.get(owners[path])
                    if plan is None:
                        raise ValueError("owned score task is outside frozen campaign")
                    scores = json.loads(
                        verify_refs(path.parent, [metrics["scores"]], max_bytes=32 << 20)[0]
                    )
                    arm = arm_map.get(plan["arm"], "candidate")
                    for raw_id, outcome in scores.get("outcomes", {}).items():
                        member = identity(raw_id)
                        key = (member, plan["seed"], arm)
                        if member not in roster:
                            continue
                        if key in seen_outcomes:
                            raise ValueError("ambiguous owned score within public scope")
                        seen_outcomes.add(key)
                        tables["outcomes"].append(
                            dict(
                                roster[member],
                                seed=plan["seed"],
                                arm=arm,
                                outcome=outcome["equal_budget"]["outcome"],
                            )
                        )
            if value.get("producer_error"):
                original_error = public_error(root, path.parent, value["producer_error"])
                previous_trace = first.get("local_traceback") if first else None
                if first is None or (
                    first["code"] == original_error["code"]
                    and previous_trace
                    and not confined(root, previous_trace["path"], exists=False).is_file()
                ):
                    first = original_error
            for e in value.get("collector_errors", []):
                errors.append(
                    public_error(root, path.parent, e)
                    if {"type", "code", "producer", "local_traceback"} <= set(e)
                    else error(ValueError(e.get("message", "collector failed")), "collector")
                )
            progress = sorted((path.parent / "progress").glob("*.json"))
            if progress:
                value_progress = load(progress[-1], maximum=4 << 20)
                tables["observations"].append(
                    {"producer": value["identity"], "data": value_progress}
                )
    if first is not None:
        fallback["producer_error"] = first
    refs = {}
    with guard.activate():
        for kind, rows in tables.items():
            validate_rows(kind, rows)
            path = output / (kind + ".json")
            atomic_write(path, rows)
            refs[kind] = [reference(root, path)]
        value = public_report_input(
            output,
            campaign=campaign["identity"],
            source=source or fallback["source"],
            seeds=seeds,
            design=design,
            scope=scope,
            tables=refs,
            decisions=decisions,
            producer_error=fallback.get("producer_error"),
            collector_errors=[*fallback.get("collector_errors", []), *errors],
            campaign_record=campaign,
            scope_record=scope_record,
            terminal=fallback["terminal"],
        )
    return value


def public_error(root, base, value):
    result = {k: value.get(k) for k in ("type", "code", "producer", "local_traceback")}
    if result["local_traceback"]:
        original = result["local_traceback"]
        path = confined(base, original["path"], exists=False)
        result["local_traceback"] = dict(original, path=str(path.relative_to(root)))
    return result
