"""Development ranking and outcome-blind fresh-confirmation reservations."""

from oeis_learn.pipeline_validation.identity import seal

from .trials import integer, scalar
from .trial_evidence import eligible as eligible_value
from .tuning import ranking_uncertainty


def rank(rows, *, target=4000000, root=None):
    integer(target, 1)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate candidate")
    eligible = []
    for r in rows:
        integer(r["tokens"])
        integer(r["elapsed_us"])
        if r["role"] != "development":
            raise ValueError("development ranking only")
        if r["value"] is not None:
            scalar(r["value"])
        if (
            r["status"] == "COMPLETE"
            and r["tokens"] >= target
            and eligible_value(r, root=root) is not None
        ):
            eligible.append(r)
    ordered = sorted(eligible, key=lambda r: (-r["value"], r["elapsed_us"], r["id"]))
    frontier = [
        r["id"]
        for r in ordered
        if not any(
            o["value"] >= r["value"]
            and o["elapsed_us"] <= r["elapsed_us"]
            and (o["value"] > r["value"] or o["elapsed_us"] < r["elapsed_us"])
            for o in ordered
        )
    ]
    return {
        "ordered": [r["id"] for r in ordered],
        "frontier": frontier,
        "uncertainty": ranking_uncertainty(eligible),
        "equal_wall_order": [
            r["id"]
            for r in sorted(
                eligible, key=lambda r: (-r["value"] / max(1, r["elapsed_us"]), r["id"])
            )
        ],
    }


def confirmation(rows, *, baseline, panels, seeds, target=4000000, root=None):
    ranking = rank(rows, target=target, root=root)
    if baseline not in ranking["ordered"]:
        raise ValueError("feasible baseline required")
    selected = [baseline] + [i for i in ranking["ordered"] if i != baseline][:3]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("two paired confirmation seeds required")
    for seed in seeds:
        integer(seed)
    if len(panels) != len(selected) or len({p["identity"] for p in panels}) != len(panels):
        raise ValueError("reserved independent panels required")
    groups = set()
    for panel in panels:
        if (
            set(panel) != {"identity", "role", "exposed", "groups"}
            or panel["role"] != "confirmation"
            or panel["exposed"] is not False
        ):
            raise ValueError("unused confirmation-only panel required")
        if not panel["groups"] or groups.intersection(panel["groups"]):
            raise ValueError("reused confirmation groups")
        groups.update(panel["groups"])
    return seal(
        {
            "version": "progressive-finalists/v1",
            "ranking": ranking,
            "selected": selected,
            "panels": panels,
            "seeds": seeds,
            "initialization": "fresh_random",
            "optimizer": "fresh",
            "tokens": 25000000,
            "seconds": 43200,
            "schedule_horizon_tokens": 400000000,
            "warmup_tokens": 1250000,
            "final_multiplier": 0.1,
            "sampler_feedback": False,
        }
    )
