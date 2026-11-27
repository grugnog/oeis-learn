"""Frozen finite OEIS yield gate; discovery count never confers learning gain."""

from collections import Counter
from fractions import Fraction

from .admission import validate_member
from .archive import Archive
from .contracts import digest, seal


def policy(*, min_groups=32, min_structures=16):
    return seal(
        dict(
            version="progressive-oeis-yield-policy/v1",
            maximum_tasks=512,
            attempts_per_task=8,
            portfolio_us=60_000_000,
            maximum_active_us=43_200_000_000,
            min_groups=min_groups,
            min_structures=min_structures,
            minimum_rate="0.05",
        )
    )


def validate_policy(value):
    if set(value) != set(policy()) or seal(value) != value:
        raise ValueError("closed frozen yield policy required")
    if any(
        value[k] != policy()[k]
        for k in policy()
        if k not in {"identity", "min_groups", "min_structures"}
    ):
        raise ValueError("fixed finite OEIS search limits required")
    for k, maximum in [("min_groups", 512), ("min_structures", 4096)]:
        if type(value[k]) is not int or not 1 <= value[k] <= maximum:
            raise ValueError("positive frozen yield threshold required")
    return value


def audit(
    pool,
    admissions,
    attempts,
    *,
    costs,
    selected=None,
    purpose="acceptance",
    scope="DIAGNOSTIC_ONLY",
    created_at="2026-11-08T00:00:00Z",
):
    p = validate_policy(selected or policy())
    if purpose not in {"acceptance", "research"} or scope not in {
        "DIAGNOSTIC_ONLY",
        "CHECKPOINT_NATIVE",
    }:
        raise ValueError("explicit yield scope required")
    if purpose == "research" and (scope != "CHECKPOINT_NATIVE" or p != policy()):
        raise ValueError("diagnostic yield thresholds cannot authorize research")
    if not isinstance(pool, list) or not 1 <= len(pool) <= p["maximum_tasks"]:
        raise ValueError("finite outcome-blind training pool required")
    members = {m["id"]: validate_member(m) for m in pool}
    if len(members) != len(pool):
        raise ValueError("duplicate source member")
    if set(costs) != {"search_us", "admission_us", "package_us"} or any(
        type(v) is not int or v < 0 for v in costs.values()
    ):
        raise ValueError("measured nonnegative active cost required")
    total = sum(costs.values())
    active = costs["search_us"] + costs["admission_us"]
    if not active or total > p["maximum_active_us"]:
        raise ValueError("yield task active reservation exhausted or unmeasured")
    if len(attempts) > len(pool) * p["attempts_per_task"]:
        raise ValueError("bounded attempt ledger required")
    seen, by_task, rejections, accepted_attempts = set(), Counter(), Counter(), set()
    for item in attempts:
        if (
            set(item) != {"task", "ordinal", "candidate", "outcome", "reason", "elapsed_us"}
            or item["task"] not in members
            or type(item["ordinal"]) is not int
            or not 0 <= item["ordinal"] < p["attempts_per_task"]
            or item["outcome"] not in {"accepted", "rejected", "unresolved"}
            or not isinstance(item["reason"], str)
            or not 1 <= len(item["reason"]) <= 256
            or type(item["elapsed_us"]) is not int
            or item["elapsed_us"] < 0
        ):
            raise ValueError("closed original attempt record required")
        key = (item["task"], item["ordinal"])
        if key in seen:
            raise ValueError("duplicate attempt ordinal")
        seen.add(key)
        by_task[item["task"]] += item["elapsed_us"]
        if item["outcome"] == "accepted":
            accepted_attempts.add((item["task"], item["candidate"]))
        else:
            rejections[item["outcome"] + ":" + item["reason"]] += 1
    if any(n > p["portfolio_us"] for n in by_task.values()) or sum(by_task.values()) > active:
        raise ValueError("original portfolio/search cost exceeded or omitted")
    if not isinstance(admissions, list) or len(admissions) > 4096:
        raise ValueError("bounded committed admission inventory required")
    valid, invalid, supplied = {}, [], set()
    for value in admissions:
        q = value.get("payload", {}) if isinstance(value, dict) else {}
        supplied.add((q.get("task"), q.get("source_hash")))
        try:
            a = Archive.validate_entry(value)
            q = a["payload"]
            m = members[q["task"]]
            if (
                a["purpose"] != purpose
                or q["native_verified"] is not True
                or q["member"] != digest(m)
                or q["group"] != m["group"]
                or q["source_snapshot"] != m["source_snapshot"]
                or q["indices"] != m["indices"]
                or q["values_hash"] != digest(m["values"])
                or q["visible_values"] != m["values"][:20]
                or (q["task"], q["source_hash"]) not in accepted_attempts
            ):
                raise ValueError("admission differs from original training member/attempt")
            valid[a["identity"]] = a
        except (KeyError, ValueError, TypeError) as exc:
            invalid.append(
                dict(
                    identity=value.get("identity") if isinstance(value, dict) else None,
                    reason=str(exc)[:256],
                )
            )
    bound = {(a["payload"]["task"], a["payload"]["source_hash"]) for a in valid.values()}
    missing = accepted_attempts - bound - supplied
    invalid.extend(
        dict(identity=source, reason="accepted attempt lacks committed admission")
        for _, source in sorted(missing)
    )
    groups = {a["payload"]["group"] for a in valid.values()}
    structures = {a["payload"]["structure"] for a in valid.values()}
    rate = Fraction(len(groups) * 60_000_000, active)
    deficits = dict(
        groups=max(0, p["min_groups"] - len(groups)),
        structures=max(0, p["min_structures"] - len(structures)),
        rate=max(0.0, float(Fraction(p["minimum_rate"]) - rate)),
    )
    eligible = not invalid and not any(deficits.values())
    return seal(
        dict(
            version="discovery-oeis-yield/v1",
            created_at=created_at,
            parents=[],
            purpose=purpose,
            track="oeis_derived",
            payload=dict(
                policy=p["identity"],
                pool=digest(pool),
                attempts=digest(attempts),
                admissions=sorted(valid),
                accepted_groups=len(groups),
                structures=len(structures),
                invalid_admissions=len(invalid),
                invalid=invalid,
                rejections=dict(sorted(rejections.items())),
                unattempted=sorted(set(members) - set(by_task)),
                attempted=len(seen),
                costs=costs,
                total_us=total,
                groups_per_active_minute=float(rate),
                deficits=deficits,
                eligible=eligible,
                next_action="STOP_AFFECTED"
                if invalid
                else "ADVANCE_REGISTERED"
                if eligible
                else "DEFER_BRANCH",
                scope=scope,
                autonomous_discoveries=len(groups) if scope == "CHECKPOINT_NATIVE" else 0,
            ),
        )
    )
