"""Inspect the original trial-05 portfolio without drawing or fitting another program."""

from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
from oeis_learn.generalization.selector_audit import candidate_pool, check_original
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.reference import evaluate

from .program_search_routing import feasible


def replay_source(source, prompt, truth):
    """Only a completed, indexed disagreement is a negative label."""
    if truth is None or prompt is None:
        return "unknown"
    if (truth["id"] != prompt["id"] or len(truth["indices"]) != 100 or len(truth["values"]) != 100
            or truth["indices"][:20] != prompt["indices"] or truth["values"][:20] != prompt["values"]
            or truth["indices"] != [str(i) for i in range(int(prompt["indices"][0]), int(prompt["indices"][0]) + 100)]):
        raise ValueError("scoring truth differs from original indexed prefix")
    result = evaluate(source, truth["indices"])
    if result.outcome != "ok" or result.coverage != 100:
        return "unknown"
    if list(result.values[:20]) != prompt["values"]:
        return "prefix_mismatch"
    return "solve" if list(result.values) == truth["values"] else "miss"


def selected_source(envelope, score, prompt):
    if envelope is None:
        if score["candidate_sha256"] is not None or score["success"]:
            raise ValueError("scored an absent selection")
        return None
    chosen = validate_selection_envelope(envelope)
    if (source_identity(chosen["source"]) != chosen["source_sha256"]
            or chosen["source_sha256"] != score["candidate_sha256"]
            or chosen["verified"] is not True or chosen["visible_count"] != 20):
        raise ValueError("selected candidate hash or admission differs")
    if prompt is not None:
        visible = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
        if envelope["prompt_sha256"] != visible.sha256:
            raise ValueError("chosen program belongs to another prompt")
    return chosen


def inspect_bank(rec, score, companion, clock, prompt, truth, witness):
    portfolio = rec["costs"]["correct"]
    envelope = rec["selections"]["equal_budget"]
    chosen = selected_source(envelope, score, prompt)
    if portfolio["equal_budget"] != envelope or score["upstream"] != rec["arm_evidence"]["equal_budget"]:
        raise ValueError("portfolio and committed equal-budget choice differ")
    details = portfolio["details"]
    if [d["position"] for d in details] != list(range(len(details))) or len(details) != clock["started"]:
        raise ValueError("original attempt order differs")
    unfinished, late = [], []
    for d in details:
        if d["generation"]["complete"]:
            if source_identity(d["source"]) != d["source_sha256"]:
                raise ValueError("original generated source hash differs")
            if not d["fitting"]["complete"]:
                unfinished.append(d["position"])
            timing = d["fitting"].get("clock", {})
            if timing.get("completed", 0) >= timing.get("deadline", float("inf")) or (
                    "late" in str(d["fitting"].get("reason"))):
                late.append(d["position"])
    from oeis_learn.decoder.bigint_codec import EOS_ID
    spent = sum(d["generation"].get("reason") == "token_cap_without_eos"
                and len(d.get("tokens", [])) == 1024 and EOS_ID not in d["tokens"] and not d["source"]
                and d["source_sha256"] is None and d["fitting"]["outcome"] == "not_attempted"
                for d in details if d["generation"]["complete"] is False)
    finished = sum(d["generation"]["complete"] is True and d["fitting"]["complete"] is True for d in details)
    # A spent no-EOS draw can be closed by the pinned companion rule. A selected
    # success alone cannot settle the outstanding draw/fit dispositions.
    closed = (clock["termination"] == "attempt_roster_terminal" and len(details) == 8
              and (finished + spent == 8 or (companion == "failure" and not late
                   and finished + spent + len(unfinished) == 8 and all(
                       details[i]["fitting"].get("reason") == "control_range" for i in unfinished))))
    exact = False
    if witness is not None:
        if (truth is None or witness["task"] != truth["id"] or witness["indices"] != truth["indices"]
                or witness["values"] != truth["values"]):
            raise ValueError("independent witness belongs to another case")
        exact = bool(witness["source"] and replay_source(witness["source"], prompt, truth) == "solve")
    gap, candidates = None, None
    if "archive" not in portfolio:
        gap = "missing_archive"
    else:
        try:
            candidates = candidate_pool(portfolio, reconstruct=True)
            admissions = {r["source_sha256"] for d in details for r in d["fitting"].get("admission", [])
                          if r["outcome"] == "verified_solution"}
            if admissions != {c["source_sha256"] for c in candidates}:
                gap = "incomplete_admission_index"
        except ValueError as exc:
            if str(exc) != "missing eligible candidate source; subset replay forbidden":
                raise
            gap = "missing_discarded_source"
    verdicts = {}
    fault = False
    if candidates is not None and gap is None:
        selected_id = chosen["source_sha256"] if chosen else None
        if chosen is None or chosen["selection"] == "visible-only/v2":
            check_original(candidates, selected_id)
            if chosen and (chosen["provenance"]["snapshot"] != clock["snapshot"]
                           or details[chosen["attempt_ordinal"]]["fitting"]["job_id"] != chosen["job_id"]):
                raise ValueError("sealed selection attempt or model differs")
        else:
            retained = portfolio["archive"]["retained"]
            costs = {r["id"]: r["cost"] for r in retained}
            min_bytes = min(len(c["source"].encode()) for c in candidates)
            ties = [c for c in candidates if len(c["source"].encode()) == min_bytes]
            if len(ties) > 1 and any(c["source_sha256"] not in costs for c in ties):
                gap = "missing_original_tie_cost"
            else:
                expected = min(ties, key=lambda c: (costs.get(c["source_sha256"], 0), c["source_sha256"]))
                if expected["source_sha256"] != selected_id:
                    raise ValueError("original cost/length selection not reproduced")
        for candidate in candidates:
            key = candidate["source_sha256"]
            verdict = replay_source(candidate["source"], prompt, truth)
            if verdict == "prefix_mismatch":
                # Repeat a deterministic contradiction on the original input.
                fault |= replay_source(candidate["source"], prompt, truth) == verdict
                verdict = "unknown"
            verdicts[key] = verdict
        if chosen is not None:
            v = verdicts[selected_id]
            if (score["success"] and v == "miss") or (score["scoring"]["outcome"] == "continuation_mismatch" and v == "solve"):
                fault |= replay_source(chosen["source"], prompt, truth) == v
    selected = ("success" if score["success"] else "failure" if score["scoring"]["outcome"]
                == "continuation_mismatch" and score["scoring"]["complete"] else companion)
    bank = None if gap else list(verdicts.values())
    return dict(feasible=feasible(selected=selected, bank=bank, closed=closed, exact_witness=exact),
                fault=fault, unfinished_fit=bool(unfinished), unfinished_ordinals=unfinished, late_ordinals=late,
                bank_gap=gap, eligible_verdicts=verdicts, selected=chosen["source_sha256"] if chosen else None,
                search_closed=closed, exact_witness=exact, band_witness=False, replay=None)
