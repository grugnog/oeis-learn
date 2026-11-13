import pytest

from oeis_learn.generalization.progressive.contracts import digest
from tests.contract.test_progressive_admission import member
from tests.unit.test_progressive_archive import admitted


def fixture(count=32):
    pool, targets, attempts = [], [], []
    for i in range(count):
        m = member() | dict(id=f"t{i}", group=f"g{i}")
        a = admitted(m["id"], "source-" + str(i))
        a["payload"].update(
            member=digest(m),
            group=m["group"],
            source_snapshot=m["source_snapshot"],
            values_hash=digest(m["values"]),
        )
        from oeis_learn.pipeline_validation.identity import seal

        a = seal(a)
        pool.append(m)
        targets.append(a)
        attempts.append(
            dict(
                task=m["id"],
                ordinal=0,
                candidate=a["payload"]["source_hash"],
                outcome="accepted",
                reason="finite_verified",
                elapsed_us=1_000_000,
            )
        )
    return pool, targets, attempts


def test_yield_counts_groups_structures_cost_and_every_rejection():
    from oeis_learn.generalization.progressive.yield_audit import audit

    p, a, t = fixture()
    # All targets share a canonical structure despite different numeric sources.
    result = audit(
        p, a, t, costs=dict(search_us=32_000_000, admission_us=1_000_000, package_us=1_000)
    )
    assert result["payload"]["accepted_groups"] == 32
    assert result["payload"]["structures"] == 1
    assert result["payload"]["deficits"]["structures"] == 15
    assert result["payload"]["next_action"] == "DEFER_BRANCH"
    assert result["payload"]["total_us"] == 33_001_000
    assert result["payload"]["autonomous_discoveries"] == 0


def test_repeated_groups_and_invalid_admission_cannot_pass():
    from oeis_learn.generalization.progressive.yield_audit import audit

    p, a, t = fixture()
    a[0]["payload"]["group"] = "forged"
    from oeis_learn.pipeline_validation.identity import seal

    a[0] = seal(a[0])
    result = audit(p, a, t, costs=dict(search_us=32_000_000, admission_us=1, package_us=1))
    assert result["payload"]["invalid_admissions"] == 1
    assert result["payload"]["next_action"] == "STOP_AFFECTED"


@pytest.mark.parametrize("change", ["duplicate", "attempts", "time", "pool"])
def test_finite_audit_limits_cannot_be_waived(change):
    from oeis_learn.generalization.progressive.yield_audit import audit

    p, a, t = fixture(1)
    costs = dict(search_us=1_000_000, admission_us=1, package_us=1)
    if change == "duplicate":
        t.append(dict(t[0]))
    if change == "attempts":
        t[0]["ordinal"] = 8
    if change == "time":
        costs["search_us"] = 43_200_000_001
    if change == "pool":
        p = [dict(p[0], id=str(i)) for i in range(513)]
    with pytest.raises(ValueError):
        audit(p, a, t, costs=costs)


def test_positive_diagnostic_gate_does_not_claim_autonomous_discovery():
    from oeis_learn.generalization.progressive.yield_audit import audit, policy

    p, a, t = fixture(1)
    result = audit(
        p,
        a,
        t,
        costs=dict(search_us=1_000_000, admission_us=1, package_us=1),
        selected=policy(min_groups=1, min_structures=1),
    )
    assert result["payload"]["eligible"]
    assert result["payload"]["autonomous_discoveries"] == 0
    with pytest.raises(ValueError, match="diagnostic"):
        audit(
            p,
            a,
            t,
            costs=dict(search_us=1_000_000, admission_us=1, package_us=1),
            selected=policy(min_groups=1, min_structures=1),
            purpose="research",
        )


def test_low_rate_and_rejections_remain_visible():
    from oeis_learn.generalization.progressive.yield_audit import audit, policy

    p, a, t = fixture(1)
    t.append(
        dict(
            task="t0",
            ordinal=1,
            candidate=None,
            outcome="unresolved",
            reason="timeout",
            elapsed_us=1,
        )
    )
    result = audit(
        p,
        a,
        t,
        costs=dict(search_us=2_000_000_000, admission_us=1, package_us=1),
        selected=policy(min_groups=1, min_structures=1),
    )
    assert result["payload"]["deficits"]["rate"] > 0
    assert result["payload"]["rejections"] == {"unresolved:timeout": 1}
    assert not result["payload"]["eligible"]
