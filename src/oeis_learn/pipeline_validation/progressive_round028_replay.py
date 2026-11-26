"""One paid, visible-only native replay of an inconclusive diagnostic prefix."""

import math
import time

from oeis_learn.generalization.progressive.contracts import digest, load, read_bound_record, seal
from oeis_learn.generalization.progressive.program_search_bank import feasible, replay_source
from oeis_learn.generalization.progressive.program_search_routing import project, route
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require
from .progressive_round028_panels import NativeRoles


class DiagnosticRole(NativeRoles):
    fixed_attempt = True
    machine_bound_attempts = True


def replay_attempts(data, client):
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.synthesis.admission import VisiblePrompt, source_identity

    prompt = data["prompt"]
    require(
        set(prompt) == {"id", "indices", "values"} and len(prompt["values"]) == 20,
        "visible replay authority required",
    )
    visible = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
    results = []
    for row in data["attempts"]:
        require(
            row["case"] == prompt["id"]
            and row["prompt_sha256"] == visible.sha256
            and source_identity(row["source"]) == row["source_sha256"]
            and 0 < row["service_seconds"] <= 4,
            "original replay inputs differ",
        )
        if row["original_resource"] != client.limits.identity or row[
            "original_runtime"
        ] != client.runtime_identity("bigint-body/v1"):
            results.append(
                dict(
                    position=row["position"],
                    complete=False,
                    admitted=[],
                    elapsed_us=0,
                    reason="changed_native_profile",
                )
            )
            continue
        start = time.monotonic()
        result = client.fit(
            row["source"],
            visible,
            policy=fitting_policy(row["source"]),
            seed=row["seed"],
            deadline=start + row["service_seconds"],
        )
        results.append(
            dict(
                position=row["position"],
                complete=result["complete"],
                admitted=[r["source"] for r in result.get("admitted", [])],
                elapsed_us=math.ceil((time.monotonic() - start) * 1e6),
            )
        )
    return {"attempts": results}


def replay_decision(votes, support, budget, run_case):
    initial = project(votes, training_support=support, replay_budget_us=None)
    requested = route(initial)
    affordable = dict(initial, replay_budget_us=budget)
    decision = route(affordable)
    result = dict(original=requested, reservation_us=budget, elapsed_us=0, original_service_us=0)
    if decision["action"] != "REPLAY":
        result.update(
            projection=affordable,
            decision=decision,
            deficit_us=max(0, requested["replay_cost_us"] - budget),
        )
        return result
    selected = sorted((r for r in votes if r["replay"]), key=lambda r: r["task"])[
        : decision["replay_cases"]
    ]
    updates = {}
    for row in selected:
        update, elapsed = run_case(row)
        require(
            update["feasible"] in {"N", "S", "P", "NS", "NP", "SP", "NSP"}
            and set(update["feasible"]) <= set(row["feasible"]),
            "unwarranted replay vote",
        )
        updates[row["task"]] = dict(row, **update, replay=None)
        result["elapsed_us"] += elapsed
        result["original_service_us"] += row["replay"]["cost_us"]
    final = project(
        [updates.get(r["task"], r) for r in votes],
        training_support=support,
        replay_budget_us=budget,
        replay_used=True,
    )
    return dict(result, projection=final, decision=route(final), deficit_us=0)


def replayed_vote(row, portfolio, selected, fits, prompt, truth):
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
    from oeis_learn.generalization.progressive.program_search_bank import checked_bank
    from oeis_learn.synthesis.admission import source_identity

    envelope = portfolio["equal_budget"]
    chosen = validate_selection_envelope(envelope) if envelope else None
    candidates, missing, gap = checked_bank(portfolio, chosen, fits)
    verdicts = {c["source_sha256"]: replay_source(c["source"], prompt, truth) for c in candidates}
    fault = "prefix_mismatch" in verdicts.values()
    new_solution = any(source_identity(body) not in verdicts and replay_source(body, prompt, truth) == "solve"
                       for fit in fits for body in fit["admitted"])
    mask = feasible(selected=selected, bank=None if gap or missing or fault else list(verdicts.values()),
                    closed=all(f["complete"] for f in fits) and not new_solution, exact_witness=row["exact_witness"])
    return dict(feasible=mask, fault=row["fault"] or fault)


def diagnostic_replay(args, source, root, folder, revision, store, host_factory, report):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.selector_fit_replay import original_attempt
    from oeis_learn.synthesis.admission import VisiblePrompt
    from .progressive_round028 import reopen
    from .progressive_round028_timing import original_channels

    area = confined(folder, "private/diagnostic-replay", exists=False)
    reservation = area / "registration.json"
    seconds = getattr(args, "replay_budget_seconds", None)
    saved = load(confined(area, reservation)) if reservation.is_file() else None
    if seconds is None and saved is None:
        return report
    if seconds is not None and (type(seconds) is not int or seconds < 0):
        raise ValueError("nonnegative replay reservation required")
    budget = seconds * 1_000_000 if seconds is not None else saved["budget_us"]
    reg = seal(
        dict(
            candidate=revision,
            source=digest(collect_snapshot(source)),
            audit=report["receipt"]["report"],
            budget_us=budget,
        )
    )
    retain(reservation, reg)
    cut = reopen(root, folder, store.campaign, store.state["audit"])
    votes = [
        dict(c["diagnosis"], task=c["task"], stratum=c["stratum"])
        for c in cut["cases"]
        if c["stratum"] in ("composition", "unseen_structure")
    ]
    cases = {c["task"]: c for c in cut["cases"]}
    native = DiagnosticRole(args, source, root, revision, area, reg, host_factory)
    records = None

    def run_case(row):
        nonlocal records
        if records is None:
            ref = cut["producers"]["trial-05-1M"]["generation"]
            records = read_bound_record(root, ref, version="discovery-candidates/v1")["records"]
        case = cases[row["task"]]
        prompts, truths = original_channels(root, [case])
        visible = VisiblePrompt(tuple(prompts[0]["indices"]), tuple(prompts[0]["values"]))
        details = records[row["task"]]["costs"]["correct"]["details"]
        attempts = []
        for expected in row["replay_inputs"]:
            pos = expected["position"]
            draw = details[pos]
            packet = original_attempt(row["task"], draw, pos, draw["fitting"]["outcome"], visible)
            require(
                all(packet[k] == v for k, v in expected.items()), "changed original fit allowance"
            )
            packet["original_runtime"] = ((draw.get("native") or {}).get("evidence") or {}).get(
                "runtime_sha256"
            )
            attempts.append(packet)
        block = digest([reg["identity"], row["task"]])[7:]
        result = native(
            "fit_replay",
            seal(dict(block=block, prompt=prompts[0], attempts=attempts)),
            seconds=120 + math.ceil(row["replay"]["cost_us"] / 1e6),
        )
        fits = result["attempts"]
        require([v["position"] for v in fits] == row["replay"]["ordinals"], "replay order changed")
        observation = case["observations"]["trial-05-1M"]
        selected = ("failure" if observation.get("scoring", {}).get("outcome") == "continuation_mismatch"
                    and observation["scoring"]["complete"] else observation["companion"])
        return replayed_vote(row, records[row["task"]]["costs"]["correct"], selected,
                             fits, prompts[0], truths[0]), native.last_wall_us

    result = replay_decision(votes, cut["support"]["eligible_rows"] > 0, budget, run_case)
    receipt = seal(dict(registration=reg["identity"], **result))
    retain(area / "receipt.json", receipt)
    return dict(
        report, decision=result["decision"], diagnostic_replay=receipt, mechanical_status="COMPLETE"
    )
