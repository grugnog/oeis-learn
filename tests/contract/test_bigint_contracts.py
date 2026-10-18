"""Phase 1 shape/rejection tests, not Phase 2 arithmetic qualification."""
import copy
import json
from pathlib import Path

import pytest

from oeis_learn.synthesis.contracts import (
    ContractError, RECORD_KINDS, handle_request, validate_record, validate_request,
)

ROOT = Path(__file__).resolve().parents[2]


def request():
    return json.loads((ROOT / "tests/fixtures/bigint/request.json").read_text())


def test_stub_never_admits():
    for operation in ("fit", "execute", "replay_unsat"):
        value = request()
        value["operation"] = operation
        result = handle_request(value)
        assert result["outcome"] == "unsupported"
        assert result["reason"] == "phase_2_not_implemented"
        assert result["admitted"] is False
        assert result["request_id"] == value["request_id"]
        assert result["worker_generation"] == value["worker_generation"]
        assert result["assignments"] == []


@pytest.mark.parametrize("field,value", [
    ("hidden_values", ["99"]), ("language_profile", "unknown"),
    ("resource_profile", "unknown"), ("solver_profile", "unknown"),
    ("protocol", "synthesis-worker/v2"), ("operation", "admit"),
    ("worker_generation", True), ("contract_sha256", "bad"),
    ("seed", -1), ("visible_values", [4]), ("visible_values", ["-0"]),
    ("visible_values", ["01"]), ("visible_values", ["+1"]),
    ("visible_values", ["1e4"]), ("visible_values", ["1\n"]),
    ("visible_values", ["1" * 1236]), ("visible_indices", ["0", "0"]),
    ("visible_indices", ["0", "1"]), ("candidates", []),
])
def test_bad_requests_fail_closed(field, value):
    payload = request()
    payload[field] = value
    with pytest.raises(ContractError):
        validate_request(payload)


def test_nested_unknown_fields_duplicate_candidates_and_counts_rejected():
    value = request()
    value["candidates"][0]["hidden_values"] = []
    with pytest.raises(ContractError):
        validate_request(value)
    value = request()
    value["candidates"] *= 2
    with pytest.raises(ContractError):
        validate_request(value)
    value = request()
    value["candidates"] = [dict(value["candidates"][0], candidate_id=str(i)) for i in range(33)]
    with pytest.raises(ContractError):
        validate_request(value)
    value = request()
    value["remaining_budget"]["refunded_seconds"] = 1
    with pytest.raises(ContractError):
        validate_request(value)


def test_every_record_stub_is_versioned_strict_and_disabled():
    examples = json.loads((ROOT / "tests/fixtures/bigint/record-stubs.json").read_text())
    assert set(examples) == set(RECORD_KINDS)
    for kind, value in examples.items():
        validate_record(kind, value)
        for field, bad in (("hidden_values", []), ("version", "unknown"), ("enabled", True)):
            changed = copy.deepcopy(value)
            changed[field] = bad
            with pytest.raises(ContractError):
                validate_record(kind, changed)
    with pytest.raises(ContractError):
        validate_record("Unknown", {})
