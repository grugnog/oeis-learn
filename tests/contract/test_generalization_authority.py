"""Phase A shape tests. Schema validity is never provenance or acceptance."""
import copy
import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/generalization"


def validator(name):
    resources = files("oeis_learn.generalization").joinpath("schemas")
    documents = [json.loads(p.read_text()) for p in resources.iterdir() if p.name.endswith(".json")]
    registry = Registry().with_resources((d["$id"], Resource.from_contents(d)) for d in documents)
    document = next(d for d in documents if d["$id"].endswith(f"/{name}.schema.json"))
    Draft202012Validator.check_schema(document)
    return Draft202012Validator(document, registry=registry)


@pytest.mark.parametrize("name", [
    "preparation", "study", "cohort", "refresh", "checkpoint", "replay", "qualification", "evidence",
])
def test_authority_shapes_and_unknown_fields(name):
    record = json.loads((FIXTURES / "authority.json").read_text())[name]
    v = validator(name)
    v.validate(record)
    for key, value in [("version", "unsupported/v99"), ("hidden_values", ["99"])]:
        changed = copy.deepcopy(record)
        changed[key] = value
        assert list(v.iter_errors(changed))
    for key in record:
        changed = copy.deepcopy(record)
        del changed[key]
        assert list(v.iter_errors(changed)), key


def test_nested_hidden_fields_and_malformed_authority_rejected():
    records = json.loads((FIXTURES / "authority.json").read_text())
    for name, path in [("study", "selection"), ("preparation", "budgets"),
                       ("qualification", "hardware"), ("replay", "verification")]:
        record = copy.deepcopy(records[name])
        record[path]["hidden_values"] = ["99"]
        assert list(validator(name).iter_errors(record))
    for value in ["", "sha256:xyz", "0" * 64]:
        record = copy.deepcopy(records["study"])
        record["identity"] = value
        assert list(validator("study").iter_errors(record))
    for value in [True, -1, 2**32]:
        record = copy.deepcopy(records["study"])
        record["seeds"] = [value]
        assert list(validator("study").iter_errors(record))


def test_weights_only_is_not_full_state_and_shapes_do_not_verify_hashes():
    records = json.loads((FIXTURES / "authority.json").read_text())
    record = copy.deepcopy(records["checkpoint"])
    record["kind"] = "weights_only"
    assert list(validator("checkpoint").iter_errors(record))
    record = copy.deepcopy(records["qualification"])
    record["hardware"]["hip_available"] = False
    assert list(validator("qualification").iter_errors(record))
    # A well-shaped but fabricated hash passes shape checks: Phase B must verify bytes/identity.
    record = copy.deepcopy(records["study"])
    record["identity"] = "sha256:" + "f" * 64
    validator("study").validate(record)


def test_all_schema_resources_are_valid_and_shapes_reject_unsafe_references():
    documents = files("oeis_learn.generalization").joinpath("schemas")
    for path in documents.iterdir():
        if path.name.endswith(".json"):
            Draft202012Validator.check_schema(json.loads(path.read_text()))
    records = json.loads((FIXTURES / "authority.json").read_text())
    for bad_path in ["/outside", "../escape", "safe/../../escape", "safe\\escape", "safe\n"]:
        record = copy.deepcopy(records["study"])
        record["source_closure"]["path"] = bad_path
        assert list(validator("study").iter_errors(record))
    for field in ["arms", "seeds"]:
        record = copy.deepcopy(records["study"])
        record[field] *= 2
        assert list(validator("study").iter_errors(record))


def test_fixture_inventory_and_independent_mechanical_expectations():
    import hashlib
    from fractions import Fraction
    from oeis_learn.synthesis.reference import evaluate

    inventory = json.loads((FIXTURES / "inventory.json").read_text())
    for name, entry in inventory["files"].items():
        raw = (FIXTURES / name).read_bytes()
        assert entry == {"bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
    cases = json.loads((FIXTURES / "mechanical-cases.json").read_text())
    for case in cases:
        actual = evaluate(case["source"], case["indices"])
        assert actual.outcome == "ok", (case["id"], actual)
        assert list(actual.values) == case["expected"]
    expected = json.loads((FIXTURES / "policy-expectations.json").read_text())
    assert Fraction(expected["metrics"]["row_lower"]) == Fraction(2, 4)
    assert Fraction(expected["metrics"]["macro_lower"]) == (Fraction(1)+Fraction(1, 3))/2
    assert expected["replay"]["target_values"][:20] == expected["replay"]["candidate_values"][:20]
    assert expected["replay"]["target_values"][20:] != expected["replay"]["candidate_values"][20:]
