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
        action="run",
        session_id="016-research-round-029",
        mechanical_status="BLOCKED",
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


@pytest.mark.parametrize("scope,action", [("qualification", "qualify"), ("cpu", "cpu"), ("exploration", "run")])
def test_confirmation_branch_needs_a_separately_bound_child(scope, action):
    parent, child, root = digest("parent"), digest("frozen child"), digest("physical")
    g, a = grant(scope, allowed_operations=[action, "confirmation"])
    with pytest.raises(ValueError, match="phase"):
        validate_grant(g, a, root=root, parent=parent, action=action)
    g, a = grant("confirmation", registration_or_parent_ref=child, allowed_operations=["run", "confirmation"])
    assert validate_grant(g, a, root=root, parent=child, action="run") == g
    with pytest.raises(ValueError):
        validate_grant(g, a, root=root, parent=parent, action="run")


def test_public_action_documents_match_the_installed_schema(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from oeis_learn.generalization.progressive.contracts import schema

    document = schema("round029")
    registry = Registry().with_resources((d["$id"], Resource.from_contents(d)) for d in (document, schema("common")))
    wire = Draft202012Validator(document, registry=registry)
    cmd = [sys.executable, "-S", "-B", str(Path(__file__).resolve().parents[2] / "scripts/foundation_run.py"),
           "--engine", "podman", "discovery", "round029"]
    for action, workspace, code, state in [
        ("preflight", tmp_path, 0, "INSPECTED"), ("status", tmp_path, 0, "INSPECTED"),
        ("verify", tmp_path, 2, "BLOCKED"), ("qualify", tmp_path, 1, "FAILED"),
        ("status", tmp_path / ("x" * 1200), 1, "FAILED")]:
        proc = subprocess.run([*cmd, "--workspace", str(workspace), "--action", action], capture_output=True)
        result = json.loads(proc.stdout)
        assert (proc.returncode, result["mechanical_status"]) == (code, state)
        wire.validate(result)
        assert validate_result(result) == result
        if action == "verify":
            assert result["verification_scope"] == "private_workspace"
        if state == "FAILED":
            assert result["first_error"]["reason"] == result["reason"]
        if workspace != tmp_path:
            assert str(workspace.name) in result["first_error"]["reason"]
        if action == "preflight":
            with pytest.raises(ValueError):
                validate_result(dict(result, scientific_verdict="pending score interpreted as a gain"))
            with pytest.raises(ValueError):
                validate_result({k: v for k, v in result.items() if k != "inventory"})
            with pytest.raises(ValueError):
                validate_result(dict(result, receipt_path="wrong_action"))


def test_launcher_binds_confirmation_grant_to_an_accepted_child(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id
    from oeis_learn.pipeline_validation import progressive_round028, progressive_round029

    monkeypatch.setattr(progressive_round029, "candidate_revision", lambda *a: "1" * 40)
    monkeypatch.setattr(progressive_round028, "git", lambda root, op, *args:
                        velocity_policy()["donor_revision"] if op == "rev-parse" else None)
    for i, accepted in enumerate([
        seal(dict(scope="exploration")),
        seal(dict(scope="confirmation", parent_ref=digest("screen"))),
        seal(dict(scope="confirmation", parent_ref=digest("screen"), panel_ref=digest("untouched"))),
    ]):
        root = tmp_path / str(i)
        area = root / "run-state/discovery" / SESSION
        area.mkdir(parents=True)
        atomic_write(area / "accepted.json", accepted)
        g, approval = grant("confirmation", registration_or_parent_ref=accepted["identity"],
                            state_root_id=physical_root_id(root), allowed_operations=["run", "confirmation"])
        auth = root / "run-state/discovery-authorizations" / (approval["identity"][7:] + ".json")
        atomic_write(auth, approval)
        atomic_write(root / "grant.json", g)
        args = SimpleNamespace(action="run", workspace=str(root), archive=None, qualification_archive=None,
                               authority=root / "grant.json")
        result, code = progressive_round029.execute(args, tmp_path)
        if i < 2:
            assert code == 1 and result["mechanical_status"] == "FAILED"
            assert not (area / "run.lock").exists()
        else:
            assert code == 2 and result["next_action"] == "qualify"
            assert (area / "run.lock").exists()
