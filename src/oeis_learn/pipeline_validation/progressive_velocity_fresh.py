"""Prepare the independent seed-one fallback after a paid ordinary exposure rejection."""

from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.generator import draw, draw_exclusion, validate_policy
from oeis_learn.generalization.progressive.models import resolve_profile, validate_profile
from oeis_learn.generalization.progressive.training_rows import (
    matched_stream,
    pair_seeds,
    validate_row,
)
from oeis_learn.generalization.progressive.velocity_trials import validate_start
from .progressive_preparation import verify_draw


def frame(overlap, old_stream, block):
    authority = digest(["round029-disjoint-seed-one/v1", overlap["receipt"]])
    seeds = pair_seeds(authority, 0)
    if any(seeds[k] == old_stream["seeds"][k] for k in seeds):
        raise ValueError("fresh initialization and training stream must differ from the pilot")
    law = validate_policy(dict(old_stream["law"], seed=seeds["training_families"]))
    count = min(block, law["max_draws"])
    return seal(dict(authority=authority, law=law, draws=count, ordinary_end=(count + 1) // 2))


def census(plan, candidate, eligible):
    """Screen both ordinal ranges and all custodian bridges in one component graph."""
    rows = []
    for ordinal in range(plan["draws"]):
        row, _ = verify_draw(draw(plan["law"], ordinal))
        if row is not None and not draw_exclusion(row, plan["law"]):
            rows.append(row)
    cases = [candidate(row) for row in rows]
    accepted, counts = eligible(cases)
    ordinary = [
        r for r in rows if r["draw_identity"] in accepted and r["ordinal"] < plan["ordinary_end"]
    ]
    prospective = [r for r in cases if r["id"] in accepted and r["ordinal"] >= plan["ordinary_end"]]
    # Familiar slots keep their original target. Familiar candidates cannot supply
    # a new composition for a non-familiar slot, even if their structure is absent
    # from this small ordinary frame.
    broader = [r["ordinal"] for r in prospective if r["stratum"] != "parameter"]
    return (
        ordinary,
        broader,
        dict(
            counts,
            reference_rejected=plan["draws"] - len(rows),
            ordinary_eligible=len(ordinary),
            prospective_eligible=len(prospective),
            prospective_familiar=len(prospective) - len(broader),
            prospective_native=len(broader),
        ),
    )


def check_counts(counts, plan):
    keys = (
        "candidates",
        "eligible",
        "excluded",
        "reference_rejected",
        "ordinary_eligible",
        "prospective_eligible",
        "prospective_native",
        "prospective_familiar",
    )
    if (
        any(type(counts.get(k)) is not int or counts[k] < 0 for k in keys)
        or counts["candidates"] + counts["reference_rejected"] != plan["draws"]
        or counts["eligible"] + counts["excluded"] != counts["candidates"]
        or counts["ordinary_eligible"] + counts["prospective_eligible"] != counts["eligible"]
        or counts["prospective_native"] + counts["prospective_familiar"]
        != counts["prospective_eligible"]
        or counts["ordinary_eligible"] > plan["ordinary_end"]
        or counts["prospective_eligible"] > plan["draws"] - plan["ordinary_end"]
    ):
        raise ValueError("fresh exposure counts differ from the frozen draw frame")


def profile_for(original, plan):
    validate_profile(original)
    return resolve_profile(
        original["base_recipe"],
        shape=original["shape"],
        dropout=original["model"]["dropout"],
        seed=pair_seeds(plan["authority"], 0)["initialization"],
    )


def validate_inputs(data):
    plan = data["frame"]
    law = validate_policy(plan["law"])
    rows = data["ordinary_rows"]
    size = len(rows)
    ordinary = [r["ordinal"] for r in rows]
    broader = data["eligible_ordinals"]
    seeds = pair_seeds(plan["authority"], 0)
    if (
        plan != seal(plan)
        or not 0 < plan["draws"] <= law["max_draws"]
        or plan["ordinary_end"] != (plan["draws"] + 1) // 2
        or not size
        or ordinary != sorted(set(ordinary))
        or broader != sorted(set(broader))
        or any(type(i) is not int or not 0 <= i < plan["ordinary_end"] for i in ordinary)
        or any(type(i) is not int or not plan["ordinary_end"] <= i < plan["draws"] for i in broader)
        or data["pair_seed"] != seeds["training_families"]
        or law["seed"] != data["pair_seed"]
        or validate_profile(data["profile"])["model"]["seed"] != seeds["initialization"]
    ):
        raise ValueError(
            "fresh reference inputs differ from the independent seed or frozen draw ranges"
        )
    for row in rows:
        validate_row(row, native_verified=False)
        item = draw(law, row["ordinal"])
        if row["native_verified"] is not False or any(row[k] != item[k] for k in item):
            raise ValueError("fresh ordinary reference row differs from its registered draw")


def native_rows(data, client):
    rows = []
    for reference in data["ordinary_rows"]:
        row, reason = verify_draw(draw(data["frame"]["law"], reference["ordinal"]), client=client)
        if row is None or seal(dict(row, native_verified=False)) != reference:
            raise ValueError(
                f"fresh ordinary native witness differs at ordinal {reference['ordinal']}: {reason}"
            )
        rows.append(row)
    return rows


def ordinary_stream(data, rows):
    if len(rows) != len(data["ordinary_rows"]) or any(
        not row["native_verified"] or seal(dict(row, native_verified=False)) != ref
        for ref, row in zip(data["ordinary_rows"], rows, strict=True)
    ):
        raise ValueError("fresh native witnesses do not match the eligible reference rows")
    plan = data["frame"]
    return matched_stream([rows], {"probes": {}}, plan["authority"], 0, law=plan["law"])


def reopen(data, package, broad):
    """Bind the native witnesses and random start to the exact custodian request."""
    rows = package["ordinary_rows"]
    stream = ordinary_stream(data, rows)
    start = validate_start(package["start"])
    from oeis_learn.generalization.progressive.training_rows import generator_common_policy
    from .progressive_training import resolve_update_policy

    policy = resolve_update_policy(
        data["profile"],
        objective="ordinary",
        effective_batch=32,
        microbatch_size=data["microbatch"],
        weight_decay=0,
        schedule_horizon_tokens=4_000_000,
    )
    if (
        set(package) != {"identity", "request", "stream", "ordinary_rows", "start"}
        or package != seal(package)
        or package["request"] != data["identity"]
        or package["stream"] != stream
        or start["ordinary"] != stream["identity"]
        or start["seed_authority"] != stream["registration"]
        or start["pair"] != 0
        or start["broad"] != (broad["identity"] if broad["available"] else None)
        or start["initial_weights"] == data["previous_weights"]
        or start["codec"] != data["profile"]["codec"]
        or start["common_policy"] != generator_common_policy(data["profile"], policy)
    ):
        raise ValueError(
            "fresh shared start differs from its native witnesses or common update policy"
        )
    return dict(
        data,
        stream=stream,
        ordinary_rows=rows,
        start=start,
        initial_weights=start["initial_weights"],
    )
