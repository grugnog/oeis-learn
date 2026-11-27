"""Prospective grant binding and logical identity across physical retries."""

import pytest

from oeis_learn.generalization.progressive.budgets import KEYS
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.velocity_records import (
    grant_terms,
    operation_key,
    validate_grant,
    validate_result,
)


def grant(scope="qualification", **changed):
    caps = dict.fromkeys(KEYS, 0)
    caps["elapsed_us"] = 120
    fields = dict(
        version="round029-grant/v1",
        study_id="016-research-round-029",
        registration_or_parent_ref=digest("parent"),
        scope=scope,
        state_root_id=digest("physical"),
        allowed_operations=["qualify"],
        total_us=120,
        holdback_us=20,
        resources=caps,
    )
    fields.update(changed)
    authorization = seal(dict(version="round029-authorization/v1", terms=digest(fields)))
    return seal(dict(fields, authorizing_record_ref=authorization["identity"])), authorization


@pytest.mark.parametrize("scope", ["qualification", "cpu", "exploration", "confirmation"])
def test_independently_bound_scope(scope):
    actions = {
        "qualification": "qualify",
        "cpu": "cpu",
        "exploration": "run",
        "confirmation": "run",
    }
    g, authorization = grant(scope, allowed_operations=[actions[scope]])
    assert (
        validate_grant(
            g,
            authorization,
            root=digest("physical"),
            parent=digest("parent"),
            action=actions[scope],
        )
        == g
    )
    assert digest(grant_terms(g)) == authorization["terms"]
    for root, parent in [
        (digest("foreign"), digest("parent")),
        (digest("physical"), digest("other")),
    ]:
        with pytest.raises(ValueError):
            validate_grant(g, authorization, root=root, parent=parent, action=actions[scope])


@pytest.mark.parametrize(
    "changes",
    [
        {"holdback_us": 121},
        {"holdback_us": -1},
        {"total_us": 120.0},
        {"total_us": True},
        {"scope": "research"},
        {"allowed_operations": ["assess"]},
    ],
)
def test_invalid_grants(changes):
    g, a = grant(**changes)
    with pytest.raises(ValueError):
        validate_grant(
            g, a, root=g["state_root_id"], parent=g["registration_or_parent_ref"], action="qualify"
        )


def test_grant_substitution_and_logical_coordinates():
    g, a = grant()
    altered = seal(dict(g, holdback_us=0))
    with pytest.raises(ValueError):
        validate_grant(
            altered,
            a,
            root=g["state_root_id"],
            parent=g["registration_or_parent_ref"],
            action="qualify",
        )
    coord = dict(
        study=digest("study"),
        stage="evaluate",
        method="00",
        seed=1,
        look=4000000,
        panel="development",
        profile="common",
        input_mode="true",
        role="search",
        ordinal=0,
    )
    assert operation_key(coord, 0)["logical"] == operation_key(coord, 1)["logical"]
    assert operation_key(coord, 0)["physical"] != operation_key(coord, 1)["physical"]
    assert (
        operation_key(coord, 0)["logical"]
        != operation_key(dict(coord, profile="full"), 0)["logical"]
    )
    result = dict(
        version="round029-result/v1",
        action="preflight",
        session_id="016-research-round-029",
        mechanical_status="INSPECTED",
        scientific_verdict=None,
        last_committed_ref=None,
        remaining_authority=None,
        next_action="qualify",
        reason="No measurements yet",
    )
    assert validate_result(result) == result
    with pytest.raises(ValueError):
        validate_result(dict(result, reason=""))


def test_artifact_checks_owner_format_and_bytes(tmp_path):
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.generalization.progressive.velocity_records import open_stored

    path = tmp_path / "recipe.json"
    path.write_text("{}")
    record = dict(owner="custodian", format="recipe", artifact=reference(tmp_path, path))
    assert open_stored(tmp_path, record, owner="custodian", format="recipe") == path
    for owner, fmt in (("search", "recipe"), ("custodian", "weights")):
        with pytest.raises(ValueError, match="owner or format"):
            open_stored(tmp_path, record, owner=owner, format=fmt)
    path.write_text("[]")
    with pytest.raises(ValueError, match="identity"):
        open_stored(tmp_path, record, owner="custodian", format="recipe")
