"""Operator-facing child pilot contracts, independent of a physical GPU."""

import copy

import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.program_search_pilot import (
    allocation,
    pilot_registration,
)
from oeis_learn.pipeline_validation.progressive_round028_pilot import (
    outcome_channels,
    research_rosters,
)


def parent():
    return dict(
        status="VERIFIED",
        evidence_verdict="INCOMPLETE",
        next_action="RETAIN_BASELINE",
        report=digest("old"),
        audit=digest("audit"),
        timing=digest("timing"),
        workstation={"status": "VERIFIED", "identity": digest("receipt")},
        comparison={"fresh_scored_cases": 0, "assessment": "SEALED"},
        funding={
            "pilot_us": 37296952335,
            "eight_case_batch_pilot_us": 40167927767,
            "repeat_if_earned_us": 40308089268,
        },
    )


def test_larger_allocation_uses_measured_pilot_alone_and_keeps_original_decision():
    p = parent()
    saved = copy.deepcopy(p)
    assert allocation(p)["headroom_us"] == 2303047665
    assert allocation(p)["eight_case_batch_headroom_us"] == -567927767
    assert allocation(p)["repeat_funded"] is False
    a = pilot_registration(p, "1" * 40, digest("source"))
    assert a["predecessor"] == p["report"] and a == seal(a) and p == saved
    assert pilot_registration(p, "2" * 40, digest("other"))["identity"] != a["identity"]
    bad = copy.deepcopy(p)
    bad["funding"]["pilot_us"] = 11 * 3600 * 1000000 + 1
    with pytest.raises(ValueError, match="exceeds"):
        pilot_registration(bad, "1" * 40, digest("source"))


def test_training_scoring_and_visible_inputs_have_disjoint_authority():
    from oeis_learn.generalization.progressive.graph import generator_look
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts

    panels = generator_design_and_cuts(0)[0]
    source = {
        name: [dict(id=row["task"], group=row["group"], stratum=row["stratum"]) for row in rows]
        for name, rows in panels.items()
    }
    source["assessment"] = [{"id": "never-open"}]
    assert research_rosters(source) == panels
    look = generator_look(
        panels, registration=digest("child"), pair=0, arm="ordinary", target=0, split="development"
    )
    witnesses = {
        r["task"]: {
            "id": r["task"],
            "indices": list(map(str, range(100))),
            "values": list(map(str, range(100))),
            "source": "PRIVATE",
        }
        for r in look["members"]
    }
    visible, truth = outcome_channels(look, witnesses)
    assert all(set(r) == {"id", "indices", "values"} and len(r["values"]) == 20 for r in visible)
    assert all(set(r) == {"id", "indices", "values"} and len(r["values"]) == 100 for r in truth)
    assert "never-open" not in {r["id"] for r in truth}
