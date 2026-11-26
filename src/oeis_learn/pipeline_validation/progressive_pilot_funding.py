"""Read the pilot's durable role costs and completion evidence without opening a worker."""
from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require
from .progressive_pilot_recovery import assigned_request


def role_costs(area, block, role):
    return [load(confined(area, f)) for f in sorted((area / "costs").glob(block + "*-" + role + ".json"))
            if f.name.startswith((block + "-" + role + ".", block + "-resume-", block + "-scoreretry-"))]


def finish_role(area, registration, block, role):
    costs = role_costs(area, block, role)
    require(costs and all(c == seal(c) for c in costs), "completed pilot role lacks costs")
    retain(area / "completed" / (block + "-" + role + ".json"),
           seal(dict(registration=registration, block=block, role=role, costs=[c["identity"] for c in costs])))


def sharing_receipt(cuts, looks, executions):
    # The request digest includes the ordered private roster. Only its identity is released.
    requests = {arm: dict(arm=arm, target=look["target"], registration=look["registration"], seed=look["seed"],
                          scope=digest({k: v for k, v in look.items() if k not in ("arm", "identity")}))
                for arm, look in looks.items()}
    return seal(dict(cuts=cuts, requests=requests, executions=executions))


def funding_state(area, registration, prices, limit, *, stopped_after=None):
    from .progressive_pilot_capacity import paid_allocations

    completed = set()
    charges, opened, keys, paid = [], [], set(), set()
    for path in sorted((area / "reservations").glob("*.json")):
        reservation = load(confined(area, path))
        block, role, ordinal = reservation["key"].split("/")
        base = "-".join(block.split("-")[:2])
        require(reservation == seal(reservation) and (base, role) in prices and ordinal == "0"
                and path.name == f"{block}-{role}.json", "invalid pilot role reservation")
        keys.add(reservation["key"])
        cost = area / "costs" / path.name
        if cost.is_file():
            c = load(confined(area, cost))
            require(c == seal(c) and (c["block"], c["role"]) == (block, role)
                    and c["charged_us"] == (reservation["us"] if c["wall_us"] is None else c["wall_us"])
                    and 0 <= c["lease_us"] <= c["charged_us"], "invalid settled pilot charge")
            charges.append([block, role, c["charged_us"]])
            paid.add(reservation["key"])
            continue
        state, amount = "unstarted", reservation["estimate_us"]
        claim = area / "assignments" / path.name
        if claim.is_file():
            role_dir = confined(area, "roles/" + assigned_request(area, claim)[7:], exists=False)
            attempts = list(role_dir.glob("attempt-*"))
            if attempts:
                state, amount = "open", reservation["us"]
                if len(attempts) == 1 and (attempts[0] / "end.json").is_file():
                    end = load(confined(area, attempts[0] / "end.json"))
                    require(end == seal(end) and end["request"][7:] == role_dir.name, "changed pilot end")
                    if not end.get("supervisor_wall_unknown"):
                        state, amount = "ended", end["elapsed_us"]
        opened.append([block, role, state, amount])
    require(len(charges) == len(list((area / "costs").glob("*.json"))), "pilot charge lacks a reservation")
    if (area / "accounting/budget-binding.json").is_file():
        journal = BudgetJournal(area / "accounting", registration, dict(dict.fromkeys(KEYS, 0), elapsed_us=limit))
        snapshot = journal.snapshot()
        require(set(snapshot["committed"]) <= paid and set(snapshot["pending"]) <= keys,
                "pilot budget journal differs from role reservations")
    if keys:
        rosters = {}
        if (area / "freeze.json").is_file():
            freeze = load(confined(area, "freeze.json"))
            require(freeze == seal(freeze) and freeze["registration"] == registration, "pilot funding freeze differs")
            rosters = {f"{arm}-{target}": {r["task"] for r in look["members"]}
                       for target, looks in freeze["graph"].items() for arm, look in looks.items()}
        _, finished = paid_allocations(area, {}, rosters)
        completed.update(finished & set(prices))
    for path in sorted((area / "completed").glob("*.json")):
        c = load(confined(area, path))
        block, role = c["block"], c["role"]
        require(c == seal(c) and c["registration"] == registration and (block, role) in prices
                and c["costs"] == [r["identity"] for r in role_costs(area, block, role)],
                "completed pilot role charges differ")
        completed.add((block, role))
    proof = area / "shared-zero.json"
    return seal(dict(registration=registration, charges=charges, opened=opened,
                     completed=sorted(map(list, completed)), stopped_after=stopped_after,
                     shared_zero=load(confined(area, proof)) if proof.is_file() else None))
