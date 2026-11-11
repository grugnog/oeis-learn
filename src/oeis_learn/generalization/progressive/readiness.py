"""Reconcile capability evidence without promoting a scientific or hardware claim."""

from collections import Counter, defaultdict

from oeis_learn.pipeline_validation.identity import confined, seal

from .cohorts import STRATA, census
from .contracts import load, validate, verify_refs


def build(root, cohort, exposure, coverage, evaluations):
    available = census(cohort, exposure=exposure)
    rows = cohort["payload"]["rows"]
    roster = {r["id"]: r for r in rows}
    deficits = [
        s for s, v in available["strata"].items() if v["task_deficit"] or v["group_deficit"]
    ]
    if set(available["strata"]) != set(STRATA):
        deficits.append("all_four_transfer_strata")
    audits = {}
    for audit in coverage:
        a = validate("coverage", audit)["payload"]
        if a["task"] not in roster or a["task"] in audits:
            raise ValueError("coverage task membership differs")
        audits[a["task"]] = a
    if set(audits) != set(roster):
        deficits.append("coverage_inventory")
    models = set()
    model_scopes = {}
    evaluated = defaultdict(set)
    outcomes = defaultdict(dict)
    costs = {}
    for evaluation in evaluations:
        if set(evaluation) != {"candidates", "scores"}:
            raise ValueError("closed evaluation references required")
        blobs = verify_refs(root, [evaluation["candidates"], evaluation["scores"]])
        import json

        candidate, score = (json.loads(b) for b in blobs)
        if (
            seal(candidate) != candidate
            or seal(score) != score
            or candidate.get("version") != "discovery-candidates/v1"
            or score.get("version") != "discovery-scoring/v1"
            or score["candidates"] != evaluation["candidates"]["sha256"]
            or not set(score["outcomes"])
            or not set(score["outcomes"]) <= set(roster)
            or set(candidate["records"]) != set(score["outcomes"])
        ):
            raise ValueError("score/candidate/cohort binding differs")
        model = candidate["model_control"]
        if model not in {"retained_D", "random", "earlier_checkpoint"}:
            raise ValueError("registered checkpoint controls required")
        # Bounded generation tasks may partition one frozen panel. Preserve
        # original per-part producers and require an exact, disjoint union at
        # the same model/checkpoint/recipe/method scope; never merge best scores.
        scope={k:candidate.get(k) for k in ('model','checkpoint','recipe','methods')}
        if model in model_scopes and model_scopes[model]!=scope:
            raise ValueError('partitioned capability model scope differs')
        if evaluated[model] & set(candidate['records']):
            raise ValueError('duplicate partitioned capability member')
        model_scopes[model]=scope
        evaluated[model].update(candidate['records'])
        models.add(model)
        if model not in costs:
            costs[model]=dict(total_elapsed_us=0,per_task={},producers=[])
        costs[model]['total_elapsed_us']+=candidate['elapsed_us']
        costs[model]['per_task'].update({k:v['costs'] for k,v in candidate['records'].items()})
        costs[model]['producers'].append(evaluation)
        required = {
            "raw",
            "rescue",
            "equal_budget",
            "shuffled",
            "absent",
            "template",
            "polynomial",
            "linear_recurrence",
        }
        for task, methods in score["outcomes"].items():
            if set(methods) != required:
                raise ValueError("missing baseline methods")
            for method, value in methods.items():
                if (
                    type(value.get("success")) is not bool
                    or type(value.get("complete")) is not bool
                    or value.get("outcome")
                    != (
                        "success"
                        if value["success"]
                        else "failure"
                        if value["complete"]
                        else "unresolved"
                    )
                ):
                    raise ValueError("contradictory task outcome")
                cell = outcomes[roster[task]["stratum"]].setdefault(
                    model + ":" + method, Counter(tasks=0, successes=0, complete=0, unresolved=0)
                )
                cell.update(
                    tasks=1,
                    successes=int(value["success"]),
                    complete=int(value["complete"]),
                    unresolved=int(value["outcome"] == "unresolved"),
                )
    if "retained_D" not in models:
        deficits.append("retained_D_evaluation")
    if not models.intersection({"random", "earlier_checkpoint"}):
        deficits.append("initialization_control")
    for model in sorted(models):
        if evaluated[model]!=set(roster):
            deficits.append('capability_member_inventory:'+model)
    strata = {}
    for s in STRATA:
        by_method = {}
        for method, count in outcomes[s].items():
            by_method[method] = dict(
                count,
                # Unexecuted partitions stay in the original denominator.
                reported_tasks=count['tasks'],
                expected_tasks=sum(r['stratum']==s for r in rows),
                yield_fraction=count["successes"] / sum(r['stratum']==s for r in rows),
                completion=count["complete"] / sum(r['stratum']==s for r in rows),
            )
        strata[s] = {
            "availability": available["strata"].get(s, {"tasks": 0, "groups": 0}),
            "methods": by_method,
            "representation": dict(
                Counter(
                    audits[r["id"]]["classification"]
                    for r in rows
                    if r["stratum"] == s and r["id"] in audits
                )
            ),
        }
    return seal(
        {
            "version": "discovery-readiness/v1",
            "status": "INCOMPLETE" if deficits else "COMPLETE",
            "scientific_status": "UNMEASURED",
            "qualification_status": "PENDING",
            "cohort": cohort["identity"],
            "historical_exposure": exposure["identity"],
            "exposure_events": len(exposure["payload"]["events"]),
            "strata": strata,
            "costs": costs,
            "missing": sorted(deficits),
            "interpretation": "finite capability and representation audit; runtime limits and unresolved witnesses are not inexpressibility",
        }
    )


def from_request(root, request):
    if (
        not isinstance(request, dict)
        or set(request) != {"version", "identity", "cohort", "exposure", "coverage", "evaluations"}
        or request["version"] != "capability-readiness-request/v1"
        or seal(request) != request
    ):
        raise ValueError("sealed capability readiness request required")
    refs = [request["cohort"], request["exposure"], *request["coverage"]]
    verify_refs(root, refs)
    cohort = load(confined(root, request["cohort"]["path"]))
    exposure = load(confined(root, request["exposure"]["path"]))
    coverage = [load(confined(root, r["path"])) for r in request["coverage"]]
    return build(root, cohort, exposure, coverage, request["evaluations"])
