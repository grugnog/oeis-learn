"""Development eligibility derived from a complete bound outcome roster."""

from collections import defaultdict
from fractions import Fraction
from .contracts import seal


def objective(evidence, *, elapsed_us, root=None):
    if (
        type(evidence) is not dict
        or set(evidence) - {"producer_refs","cost_scope"}
        != {
            "version",
            "identity",
            "role",
            "checkpoint",
            "qualification",
            "scope",
            "roster",
            "outcomes",
            "retention",
            "generator",
            "scorer",
            "costs",
        }
        or seal(evidence) != evidence
        or evidence["version"] != "progressive-trial-evidence/v1"
        or evidence["role"] != "development"
        or evidence["scope"] not in {"DIAGNOSTIC_ONLY", "NATIVE_VERIFIED"}
    ):
        raise ValueError("bound development execution evidence required")
    if evidence['scope']=='NATIVE_VERIFIED' and evidence.get('cost_scope')!='supervised task intervals; startup and cleanup enclosed in phase costs':
        raise ValueError('original enclosed startup/cleanup cost scope required')
    if evidence["scope"] == "NATIVE_VERIFIED" or 'producer_refs' in evidence:
        verify_producers(root, evidence)
    import re

    for k in ("checkpoint", "qualification", "generator", "scorer"):
        if type(evidence[k]) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", evidence[k]):
            raise ValueError("original trial producer identities required")
    costs = evidence["costs"]
    if (
        type(costs) is not dict
        or set(costs) != {"startup", "training", "inference", "search", "verification"}
        or any(type(v) is not int or v < 0 for v in costs.values())
        or sum(costs.values()) != elapsed_us
    ):
        raise ValueError("complete nonoverlapping trial costs required")
    roster = evidence["roster"]
    rows = evidence["outcomes"]
    if (
        not isinstance(roster, list)
        or not 1 <= len(roster) <= 4096
        or not isinstance(rows, list)
        or len(rows) > 4096
    ):
        raise ValueError("bounded frozen development roster required")
    members = {}
    for r in roster:
        if (
            set(r) != {"task", "group", "stratum"}
            or r["task"] in members
            or r["stratum"] not in {"composition", "unseen_structure", "parameter", "oeis"}
            or any(type(v) is not str or not 1 <= len(v) <= 1024 for v in r.values())
        ):
            raise ValueError("closed unique development membership required")
        members[r["task"]] = r
    outcomes = {}
    for r in rows:
        if (
            set(r) != {"task", "group", "stratum", "outcome"}
            or r["task"] not in members
            or {k: r[k] for k in members[r["task"]]} != members[r["task"]]
            or r["task"] in outcomes
            or r["outcome"] not in {"success", "failure", "unresolved"}
        ):
            raise ValueError("original development outcomes differ from roster")
        outcomes[r["task"]] = r["outcome"]
    cells = defaultdict(list)
    for task, r in members.items():
        cells[r["stratum"], r["group"]].append(outcomes.get(task, "unresolved"))
    scores = []
    for stratum in ("composition", "unseen_structure"):
        families = [v for (s, g), v in cells.items() if s == stratum]
        if len(families) < 16:
            raise ValueError("independent development support deficit")
        values = [v for group in families for v in group]
        if Fraction(values.count("unresolved"), len(values)) > Fraction(5, 100):
            raise ValueError("development completion below 95 percent")
        scores.append(sum(Fraction(v.count("success"), len(v)) for v in families) / len(families))
    retention = evidence["retention"]
    if not isinstance(retention, list) or not 1 <= len(retention) <= 4096:
        raise ValueError("retention evidence required")
    seen = set()
    loss = 0
    for r in retention:
        if (
            set(r) != {"task", "baseline", "candidate"}
            or r["task"] in seen
            or r["baseline"] not in {"success", "failure", "unresolved"}
            or r["candidate"] not in {"success", "failure", "unresolved"}
        ):
            raise ValueError("complete paired retention required")
        seen.add(r["task"])
        loss += (r["baseline"] != "failure") - (r["candidate"] == "success")
    if Fraction(loss, len(retention)) > Fraction(2, 100):
        raise ValueError("retention regression")
    return float(sum(scores) / 2)


def eligible(row, *, root=None):
    try:
        value = objective(row.get("evidence"), elapsed_us=row["elapsed_us"], root=root)
        if (
            row.get("completion", 1) < 0.95
            or row.get("retention_loss", 0) > 0.02
            or row.get("executable", True) is not True
        ):
            return None
        return value if row.get("value") == value else None
    except (ValueError, KeyError, TypeError):
        return None


def verify_producers(root, evidence):
    import json
    from .contracts import verify_refs

    refs = evidence.get("producer_refs")
    names = {
        "generation",
        "scoring",
        "checkpoint",
        "qualification",
        "retention_baseline_generation",
        "retention_baseline_scoring",
        "retention_candidate_generation",
        "retention_candidate_scoring",
    }
    if root is None or type(refs) is not dict or set(refs) != names:
        raise ValueError("original research producer closure required")
    docs = {}
    for k, ref in refs.items():
        blob = verify_refs(root, [ref], max_bytes=(64 << 30) if k == "checkpoint" else (32 << 20))[
            0
        ]
        if k != "checkpoint":
            docs[k] = json.loads(blob)
            if seal(docs[k]) != docs[k]:
                raise ValueError("original producer seal differs")
    q = docs["qualification"]
    if (
        q.get("identity") != evidence["qualification"]
        or q.get("status") not in {"PASS", "COMPLETE"}
        or (q.get('device')!='actual_HIP' if evidence['scope']=='NATIVE_VERIFIED' else q.get('device') not in {'actual_HIP','cpu'})
    ):
        raise ValueError("compatible actual research qualification required")
    for prefix in ("", "retention_baseline_", "retention_candidate_"):
        generated = docs[prefix + "generation"]
        scored = docs[prefix + "scoring"]
        if (
            generated.get("version") != "discovery-candidates/v1"
            or scored.get("version") != "discovery-scoring/v1"
            or scored.get("candidates") != refs[prefix + "generation"]["sha256"]
        ):
            raise ValueError("original generator/scorer binding differs")
        if (
            prefix != "retention_baseline_"
            and generated.get("checkpoint") != refs["checkpoint"]["sha256"]
        ):
            raise ValueError("original selected trial checkpoint differs")
    if (
        docs["generation"]["identity"] != evidence["generator"]
        or docs["scoring"]["identity"] != evidence["scorer"]
    ):
        raise ValueError("trial producer identity differs")

    def outcome(scores, task):
        r = scores.get("outcomes", {}).get(task, {}).get("equal_budget")
        if r is None:
            return "unresolved"
        if type(r.get("success")) is not bool or type(r.get("complete")) is not bool:
            raise ValueError("original completion missing")
        return "success" if r["success"] else "failure" if r["complete"] else "unresolved"

    roster = {r["task"] for r in evidence["roster"]}
    if set(docs["generation"]["records"]) != roster or docs["scoring"].get("denominator") != len(
        roster
    ):
        raise ValueError("filtered original trial roster")
    if any(outcome(docs["scoring"], r["task"]) != r["outcome"] for r in evidence["outcomes"]):
        raise ValueError("trial outcomes differ from original scorer")
    retention = {r["task"] for r in evidence["retention"]}
    for arm in ("baseline", "candidate"):
        if set(docs["retention_" + arm + "_generation"]["records"]) != retention:
            raise ValueError("retention producer roster differs")
        if any(
            outcome(docs["retention_" + arm + "_scoring"], r["task"]) != r[arm]
            for r in evidence["retention"]
        ):
            raise ValueError("retention outcomes differ from original scorer")
