"""Registered round028 decision on 64 distinct exposed primary cases."""

from collections import Counter

VALUES = {"N": 0, "S": -1, "P": 1}
MASKS = {"N", "S", "P", "NS", "NP", "SP", "NSP"}


def feasible(*, selected, bank, closed, exact_witness):
    """Bank is None if any eligible identity is missing; entries are independent verdicts."""
    if selected not in {"success", "failure", "unresolved", "unavailable"}:
        raise ValueError("unknown selected outcome")
    if bank is not None and set(bank) - {"solve", "miss", "unknown"}:
        raise ValueError("unknown candidate disposition")
    if selected == "success":
        return "N"
    if bank is None:
        return "NSP"
    if "solve" in bank:
        return "S" if selected == "failure" else "NS"
    if "unknown" in bank or not closed:
        return "NSP"
    return "P" if exact_witness else "NP"


def bounds(histogram):
    lower = upper = 0
    for mask, count in histogram.items():
        if mask not in MASKS or type(count) is not int or count < 0:
            raise ValueError("invalid feasible-set histogram")
        lower += min(VALUES[v] for v in mask) * count
        upper += max(VALUES[v] for v in mask) * count
    return lower, upper


def project(cases, *, training_support, replay_budget_us=0, replay_used=False):
    cases = sorted(cases, key=lambda case: case["task"])
    if (len({r["task"] for r in cases}) != 64
            or Counter(r["stratum"] for r in cases) != {"composition": 32, "unseen_structure": 32}):
        raise ValueError("one vote per registered primary case required")
    histogram = Counter(r["feasible"] for r in cases)
    fit = Counter(r["feasible"] for r in cases if r["unfinished_fit"])
    replays = []
    for row in cases:
        offer = row["replay"]
        if offer is not None:
            ordinals = offer["ordinals"]
            if not ordinals or ordinals != sorted(set(ordinals)) or any(
                    type(v) is not int or not 0 <= v < 8 for v in ordinals):
                raise ValueError("replay must bind original attempt order")
            replays.append(dict(mask=row["feasible"], resolves=offer["resolves"], cost_us=offer["cost_us"]))
    return dict(histogram=dict(histogram), unfinished_fit=dict(fit), faults=sum(r["fault"] for r in cases),
                band_witnesses=sum(r["band_witness"] for r in cases), training_support=training_support,
                replay_budget_us=replay_budget_us, replay_used=replay_used, replay_options=replays)


def route(p):
    if set(p) != {"histogram", "unfinished_fit", "faults", "band_witnesses", "training_support",
                  "replay_budget_us", "replay_used", "replay_options"}:
        raise ValueError("closed public routing projection required")
    hist = p["histogram"]
    lo, hi = bounds(hist)
    bounds(p["unfinished_fit"])
    if (sum(hist.values()) != 64 or any(n > hist.get(mask, 0) for mask, n in p["unfinished_fit"].items())
            or any(type(p[k]) is not int or not 0 <= p[k] <= 64 for k in ("faults", "band_witnesses"))
            or type(p["training_support"]) is not bool or type(p["replay_used"]) is not bool
            or p["replay_budget_us"] is not None
            and (type(p["replay_budget_us"]) is not int or p["replay_budget_us"] < 0)):
        raise ValueError("invalid diagnostic coverage, funding or availability")
    result = dict(denominator=64, lower=lo, upper=hi, proved_selection=hist.get("S", 0),
                  proved_proposal=hist.get("P", 0), replay_cases=0, replay_cost_us=0)
    # Resolve the most favorable possible outcome for each branch separately.
    # This searches prefixes linearly, including all still-unknown votes after each prefix.
    best_lower, best_upper = lo, hi
    can_prove_s = bool(result["proved_selection"])
    used = Counter()
    cost = 0
    prefix = None
    for i, offer in enumerate(p["replay_options"], 1):
        if set(offer) != {"mask", "resolves", "cost_us"}:
            raise ValueError("invalid replay fields")
        mask, resolves, amount = offer["mask"], offer["resolves"], offer["cost_us"]
        if (mask not in MASKS or len(mask) < 2 or resolves not in MASKS or not set(resolves) <= set(mask)
                or type(amount) is not int or amount <= 0):
            raise ValueError("unbound replay option or cost")
        used[mask] += 1
        if used[mask] > hist.get(mask, 0):
            raise ValueError("duplicate replay vote")
        cost += amount
        best_lower += max(map(VALUES.get, resolves)) - min(map(VALUES.get, mask))
        best_upper += min(map(VALUES.get, resolves)) - max(map(VALUES.get, mask))
        can_prove_s |= "S" in resolves
        if prefix is None and (p["replay_budget_us"] is None or cost <= p["replay_budget_us"]) and (
                (p["training_support"] and best_lower > 0) or (can_prove_s and best_upper <= 0)):
            prefix = i, cost
    if p["faults"]:
        action = "ISOLATE_FAULT"
    elif lo > 0 and result["proved_proposal"] and p["training_support"]:
        action = "MATCHED_GENERATOR"
    elif result["proved_selection"] and hi <= 0:
        action = "CPU_SELECTOR"
    elif prefix is not None and not p["replay_used"]:
        result["replay_cases"], result["replay_cost_us"] = prefix
        if p["replay_budget_us"] is None:
            # A missing workstation reservation cannot decide whether to spend or retain.
            return dict(result, action=None, blocker="REPLAY_RESERVATION_UNMEASURED")
        action = "REPLAY"
    else:
        action = "RETAIN_REFERENCE"
    return dict(result, action=action)
