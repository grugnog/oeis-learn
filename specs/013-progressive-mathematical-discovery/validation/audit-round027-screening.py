"""Read-only sensitivity analysis of round027's first six 1M screening looks."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from oeis_learn.decoder.bigint_codec import EOS_ID
from oeis_learn.generalization.progressive.contracts import seal, verify_refs
from oeis_learn.pipeline_validation.pilot_clocks import DRAIN, clock_record

CAMPAIGN = "sha256:b35444dcbdc68693917a61d05ac3dc11efbdebee568a074b80f2596191035304"
TRIALS = range(6)
STRATA = ("composition", "unseen_structure")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(value):
    require(seal(value) == value, "saved record identity mismatch")
    return value


def saved(path):
    return checked(json.loads(path.read_bytes()))


def study_results(root, campaign):
    proposal = next(p for p in campaign["payload"]["tasks"] if p["id"] == "trial-00-propose")
    study = proposal["trial"]["study"]
    folder = root / "run-state/discovery-studies" / CAMPAIGN[7:] / study[7:]
    binding = saved(folder / "binding.json")
    require(binding["study"] == study, "study binding differs")
    tip = binding["identity"]
    states = {}
    for index, path in enumerate(sorted(folder.glob("event-*.json"))):
        event = saved(path)
        state = event["state"]
        ordinal = state["proposal"]["ordinal"]
        require(path.name == f"event-{index:04d}.json" and event["previous"] == tip
                and state["proposal"]["study"] == study and event["ordinal"] == str(ordinal),
                "trial journal chain differs")
        states[ordinal] = state
        tip = event["identity"]
    return states, tip


def artifact(root, ref):
    return checked(json.loads(verify_refs(root, [ref])[0]))


def bound_ref(root, outputs, name, operation, metric):
    cut = outputs[name]
    result = artifact(root, cut["result"])
    require(cut["operation"] == result["operation"] == operation
            and result["mechanical_status"] == "COMPLETE", "unexpected producer: " + name)
    ref = result["metrics"][metric]
    path = Path(cut["result"]["path"]).parent / ref["path"]
    return dict(ref, path=str(path))


def classify(rec, score, clock):
    """Change only a frozen unknown whose entire remaining uncertainty is capped decoding."""
    details = rec["costs"]["correct"]["details"]
    upstream = rec["arm_evidence"]["equal_budget"]
    require(score["upstream"] == upstream, "generator/scorer arm evidence differs")
    positions = [d["position"] for d in details]
    require(len(set(positions)) == len(positions) and all(i in range(8) for i in positions),
            "invalid attempt positions")
    reasons = Counter()
    full_caps = 0
    if positions != list(range(8)) or clock["started"] != 8 or clock["terminal"] != 8:
        reasons["missing_or_unfinished_attempt"] += 1
    if clock["termination"] != "attempt_roster_terminal":
        reasons["incomplete_portfolio_clock"] += 1
    for detail in details:
        generation, fitting = detail["generation"], detail["fitting"]
        if generation["complete"] is False:
            tokens = detail["tokens"]
            if (generation["reason"] == "token_cap_without_eos" and len(tokens) == 1024
                    and EOS_ID not in tokens and not detail["source"] and detail["complete"] is False
                    and detail["source_sha256"] is None and detail["raw"]["outcome"] == "not_attempted"
                    and fitting["complete"] is False and fitting["outcome"] == "not_attempted"):
                full_caps += 1
            else:
                reasons["other_or_unproven_generation_stop"] += 1
        elif generation["complete"] is not True or fitting["complete"] is not True:
            reasons["incomplete_fitting"] += 1
    selected = rec["selections"]["equal_budget"] is not None
    scoring = score["scoring"]
    if selected:
        if scoring["complete"] is not True or (not score["success"] and scoring["outcome"] != "continuation_mismatch"):
            reasons["incomplete_selected_candidate_score"] += 1
    elif scoring["outcome"] != "no_selected_candidate" or score["candidate_sha256"] is not None:
        reasons["unproven_absent_selection"] += 1
    original = score["outcome"]
    require(original == ("success" if score["success"] else "failure" if score["complete"] else "unresolved"),
            "score outcome differs")
    if original == "success":
        require(selected and scoring["complete"] is True, "unverified success")
    recoded = original == "unresolved" and full_caps > 0 and not reasons
    return ("failure" if recoded else original), full_caps, reasons


def panel(root, refs, outputs, task_prefix, prefix, expected):
    gref, sref = refs[prefix + "generation"], refs[prefix + "scoring"]
    require(gref == bound_ref(root, outputs, task_prefix + "-generate", "generate", "candidates")
            and sref == bound_ref(root, outputs, task_prefix + "-score", "score", "scores"),
            "study inputs differ from committed milestone outputs")
    generated, scored = artifact(root, gref), artifact(root, sref)
    require(generated["version"] == "discovery-candidates/v1" and generated["methods"] == ["correct"]
            and scored["version"] == "discovery-scoring/v1" and scored["candidates"] == gref["sha256"]
            and set(expected) == set(generated["records"]) == set(scored["outcomes"])
            and scored["denominator"] == len(expected), "screening producer pair or roster differs")
    summary = checked(generated["clock"])
    require(summary["policy"] == DRAIN, "unexpected search policy")
    clocks = {}
    for c in summary["clocks"]:
        fields = {k: c[k] for k in ("policy", "prompt", "mode", "snapshot", "start", "scope_deadline",
                                    "finish", "expected", "started", "terminal")}
        require(c == clock_record(**fields) and c["policy"] == DRAIN and c["mode"] == "correct"
                and c["snapshot"] == generated["model"] and c["prompt"] not in clocks,
                "unexpected portfolio clock")
        clocks[c["prompt"]] = c
    require(set(clocks) == set(expected), "portfolio clock roster differs")
    outcomes, counts, blockers = {}, Counter(), Counter()
    for key, old in expected.items():
        score = scored["outcomes"][key]["equal_budget"]
        require(score["outcome"] == old, "study row differs from original score")
        outcome, caps, reasons = classify(generated["records"][key], score, clocks[key])
        outcomes[key] = outcome
        counts["full_cap_attempts"] += caps
        counts["recoded_cases"] += outcome != old
        counts["cases_with_cap_and_other_gap"] += bool(caps and reasons)
        if outcome == "unresolved":
            blockers.update(reasons.keys() if reasons else ["no_proven_cap"])
    return outcomes, dict(counts, outcomes=dict(Counter(outcomes.values())),
                          unresolved_reasons_by_case=dict(blockers),
                          generator=gref["sha256"], scorer=sref["sha256"])


def evaluate(roster, outcomes, baseline, candidate):
    coverage, scores = {}, []
    supported = True
    for stratum in STRATA:
        groups = defaultdict(list)
        for row in roster:
            if row["stratum"] == stratum:
                groups[row["group"]].append(outcomes[row["task"]])
        values = [v for group in groups.values() for v in group]
        require(len(values) == 32 and len(groups) >= 16, "unexpected development panel")
        unknown = values.count("unresolved")
        coverage[stratum] = dict(unresolved=unknown, total=len(values))
        supported &= Fraction(unknown, len(values)) <= Fraction(5, 100)
        scores.append(sum(Fraction(g.count("success"), len(g)) for g in groups.values()) / len(groups))
    require(len(baseline) == 128 and baseline.keys() == candidate.keys(), "retention panel differs")
    loss = Fraction(sum(v != "failure" for v in baseline.values())
                    - sum(v == "success" for v in candidate.values()), len(baseline))
    return dict(development=coverage, retention_loss_pp=round(float(100 * loss), 4),
                verified_yield=float(sum(scores) / len(scores)), passes_1m_checks=bool(supported and loss <= Fraction(2, 100)))


def audit(root):
    session = root / "run-state/discovery/013-research-round-027"
    campaign = saved(session / "prepare-research/frozen/campaign.json")
    require(campaign["identity"] == CAMPAIGN, "unexpected campaign")
    states, tip = study_results(root, campaign)
    control = root / "run-state/discovery-control" / CAMPAIGN[7:]
    transition = saved(max(control.glob("transition-[0-9][0-9][0-9][0-9].json")))
    outputs = transition["state"]["outputs"]
    rows = []
    for i in TRIALS:
        result = states[i]["result"]
        require(result["status"] == "INFEASIBLE" and result["role"] == "development", "trial status differs")
        e = checked(result["evidence"])
        require(e["scope"] == "NATIVE_VERIFIED", "unexpected screening evidence")
        old = {r["task"]: r["outcome"] for r in e["outcomes"]}
        require(len(old) == len(e["roster"]) == len(e["outcomes"]) == 128 and
                {(r["task"], r["group"], r["stratum"]) for r in e["roster"]} ==
                {(r["task"], r["group"], r["stratum"]) for r in e["outcomes"]}, "development roster differs")
        b, c = ({r["task"]: r[arm] for r in e["retention"]} for arm in ("baseline", "candidate"))
        trial = f"trial-{i:02d}"
        entries = [(trial + "-1000000-development", "", old), (trial + "-retention-baseline", "retention_baseline_", b),
                   (trial + "-1000000-retention", "retention_candidate_", c)]
        views, panels = [], {}
        for name, prefix, expected in entries:
            view, info = panel(root, e["producer_refs"], outputs, name, prefix, expected)
            views.append(view)
            panels[prefix.rstrip("_") or "development"] = info
        frozen = evaluate(e["roster"], old, b, c)
        require(not frozen["passes_1m_checks"], "could not reproduce frozen gate failure")
        rows.append(dict(trial=i, settings=states[i]["proposal"]["parameters"], frozen_status=result["status"],
                         frozen=frozen, exploratory=evaluate(e["roster"], *views), panels=panels))
    passing = [r for r in rows if r["exploratory"]["passes_1m_checks"]]
    ordered = sorted(passing, key=lambda r: (-r["exploratory"]["verified_yield"], r["trial"]))
    return dict(scope="Exploratory 1M checkpoint sensitivity; equal yields are ties. No 4M or confirmation selection.",
                rule="A proven 1024-token no-EOS draw is a spent attempt. Only otherwise resolved cases change.",
                campaign=CAMPAIGN, study_tip=tip, transition=transition["identity"], trials=rows,
                passing_trials_by_1m_yield=[r["trial"] for r in ordered])


if __name__ == "__main__":
    print(json.dumps(audit(ROOT), indent=2, allow_nan=False))
