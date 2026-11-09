"""Derive a decision table from committed generator/scorer envelopes only."""

from oeis_learn.generalization.progressive.contracts import seal


def decision_from_outputs(values):
    templates = [
        v["data"]
        for v in values
        if isinstance(v["data"], dict)
        and v["data"].get("version") == "progressive-decision-template/v1"
    ]
    if len(templates) != 1:
        raise ValueError("one frozen decision template required")
    template = templates[0]
    if set(template) != {
        "version",
        "source_slot",
        "target_slot",
        "roster",
        "seeds",
        "design",
        "policy",
        "remaining",
        "kwargs",
        "producers",
    }:
        raise ValueError("closed producer-bound decision template required")
    owned = {}
    for value in values:
        data = value["data"]
        if not isinstance(data, dict) or data.get("version") != "discovery-owned-output/v1":
            continue
        key = data["task"], data["operation"]
        if seal(data) != data or key in owned:
            raise ValueError("unique committed owned output required")
        owned[key] = data
    roster = {r["task"]: r for r in template["roster"]}
    rows = []
    cells = set()
    for producer in template["producers"]:
        if set(producer) != {"arm", "seed", "generation_task", "scoring_task"}:
            raise ValueError("closed paired evaluation producer required")
        cell = producer["arm"], producer["seed"]
        if (
            cell in cells
            or cell[0] not in {"baseline", "candidate"}
            or cell[1] not in template["seeds"]
        ):
            raise ValueError("unique frozen paired cell required")
        cells.add(cell)
        generated = owned[producer["generation_task"], "generate"]
        scored = owned[producer["scoring_task"], "score"]
        g, s = generated["record"], scored["record"]
        if (
            seal(g) != g
            or seal(s) != s
            or s.get("candidates") != generated["artifact"]["sha256"]
            or set(g["records"]) != set(roster)
            or s.get("denominator") != len(roster)
        ):
            raise ValueError("bound original decision roster/checkpoint differs")
        for task, r in roster.items():
            value = s["outcomes"].get(task, {}).get("equal_budget", {})
            success = value.get("success") is True
            complete = value.get("complete") is True
            if success and not complete:
                raise ValueError("contradictory original scored outcome")
            rows.append(
                dict(
                    r,
                    arm=cell[0],
                    seed=cell[1],
                    outcome="success" if success else "failure" if complete else "unresolved",
                    complete=complete,
                )
            )
    if cells != {(a, s) for a in ("baseline", "candidate") for s in template["seeds"]}:
        raise ValueError("complete paired producer roster required")
    data = dict(template)
    data.pop("producers")
    data["version"] = "progressive-decision-input/v1"
    data["rows"] = rows
    return [
        seal(
            dict(version="discovery-channel/v1", role="verifier", kind="public_evidence", data=data)
        )
    ]
