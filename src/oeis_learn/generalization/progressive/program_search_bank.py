"""Inspect the original trial-05 portfolio without drawing or fitting another program."""

import math

from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
from oeis_learn.generalization.selector_audit import candidate_sources, check_original
from oeis_learn.generalization.selector_fit_replay import original_attempt
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


def checked_bank(portfolio, chosen, fits=()):
    """Rebuild only original admissions, including a missing source recovered at its recorded attempt."""
    if "archive" not in portfolio:
        return [], set(), "missing_archive"
    candidates, missing = candidate_sources(portfolio, reconstruct=True)
    eligible = {r["source_sha256"] for r in candidates} | missing
    admissions = {d["position"]: {r["source_sha256"] for r in d["fitting"].get("admission", [])
                                  if r["outcome"] == "verified_solution"} for d in portfolio["details"]}
    if eligible != set().union(*admissions.values()):
        return candidates, missing, "incomplete_admission_index"
    for fit in fits:
        for source in fit["admitted"]:
            sha = source_identity(source)
            if sha in missing and sha in admissions[fit["position"]]:
                candidates.append(dict(source_sha256=sha, source=source))
                missing.remove(sha)
    if missing:
        return candidates, missing, "missing_discarded_source"
    selected = chosen["source_sha256"] if chosen else None
    if chosen is None or chosen["selection"] == "visible-only/v2":
        check_original(candidates, selected)
    else:
        costs = {r["id"]: r["cost"] for r in portfolio["archive"]["retained"]}
        shortest = min(len(c["source"].encode()) for c in candidates)
        ties = [c for c in candidates if len(c["source"].encode()) == shortest]
        if len(ties) > 1 and any(c["source_sha256"] not in costs for c in ties):
            return candidates, missing, "missing_original_tie_cost"
        if min(ties, key=lambda c: (costs.get(c["source_sha256"], 0), c["source_sha256"]))["source_sha256"] != selected:
            raise ValueError("original cost/length selection not reproduced")
    return candidates, missing, None


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
    if witness is not None and truth is not None:
        if (witness["task"] != truth["id"] or witness["indices"] != truth["indices"]
                or witness["values"] != truth["values"]):
            raise ValueError("independent witness belongs to another case")
        exact = bool(witness["source"] and replay_source(witness["source"], prompt, truth) == "solve")
    candidates, missing, gap = checked_bank(portfolio, chosen)
    if chosen and chosen["selection"] == "visible-only/v2" and (
            chosen["provenance"]["snapshot"] != clock["snapshot"]
            or details[chosen["attempt_ordinal"]]["fitting"]["job_id"] != chosen["job_id"]):
        raise ValueError("sealed selection attempt or model differs")
    verdicts = {}
    fault = False
    if gap in (None, "missing_original_tie_cost"):
        selected_id = chosen["source_sha256"] if chosen else None
        for candidate in candidates:
            key = candidate["source_sha256"]
            verdict = replay_source(candidate["source"], prompt, truth)
            if verdict == "prefix_mismatch":
                # Repeat a deterministic contradiction on the original input.
                fault |= replay_source(candidate["source"], prompt, truth) == verdict
                verdict = "unknown"
            verdicts[key] = verdict
        if selected_id in verdicts:
            v = verdicts[selected_id]
            if (score["success"] and v == "miss") or (score["scoring"]["outcome"] == "continuation_mismatch" and v == "solve"):
                fault |= replay_source(chosen["source"], prompt, truth) == v
    selected = ("success" if score["success"] else "failure" if score["scoring"]["outcome"]
                == "continuation_mismatch" and score["scoring"]["complete"] else companion)
    bank = None if gap else list(verdicts.values())
    mask = feasible(selected=selected, bank=bank, closed=closed, exact_witness=exact)
    replay = None
    # Only accounted-for native work on an original decoded source can be replayed.
    # A fresh decoder draw cannot fill in a lost source or change its original choice.
    replay_inputs = []
    if (mask == "NSP" and (unfinished or missing) and gap in (None, "missing_discarded_source")
            and clock["termination"] == "attempt_roster_terminal" and len(details) == 8
            and finished + spent + len(unfinished) == 8 and prompt is not None and truth is not None):
        original = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
        ordinals = set(unfinished)
        for sha in missing:
            ordinals.add(next(d["position"] for d in details if any(
                a["outcome"] == "verified_solution" and a["source_sha256"] == sha
                for a in d["fitting"].get("admission", []))))
        for i in sorted(ordinals):
            draw = details[i]
            timing = draw["fitting"].get("clock", {})
            if "admitted" not in timing or "deadline" not in timing or "seed" not in draw:
                break
            packet = original_attempt(prompt["id"], draw, i, draw["fitting"]["outcome"], original)
            replay_inputs.append({k: packet[k] for k in ("position", "seed", "source_sha256", "prompt_sha256",
                                                         "original_resource", "original_clock", "service_seconds")})
        else:
            # A newly fitted program did not belong to the original eligible bank.
            # Only recovering an originally admitted identity can prove a selection miss.
            resolves = "N" + ("S" if missing and selected == "failure" else "") + ("P" if exact else "")
            replay = dict(ordinals=sorted(ordinals), cost_us=sum(math.ceil(p["service_seconds"] * 1_000_000)
                          for p in replay_inputs), resolves=resolves)
    return dict(feasible=mask,
                fault=fault, unfinished_fit=bool(unfinished), unfinished_ordinals=unfinished, late_ordinals=late,
                bank_gap=gap, eligible_verdicts=verdicts, selected=chosen["source_sha256"] if chosen else None,
                search_closed=closed, exact_witness=exact, band_witness=False, replay=replay,
                replay_inputs=replay_inputs, missing_sources=sorted(missing))
