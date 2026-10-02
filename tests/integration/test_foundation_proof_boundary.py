"""Finite matches and legacy certificates must never become a proof claim."""

import copy
import uuid
import pytest
from oeis_learn.sandbox.pipeline import Runtime, verify, digest_bytes
from oeis_learn.experiments.models import validate_artifact


def candidate(evidence, terms, scope="full"):
    return verify(
        evidence,
        terms,
        scope,
        run_id=str(uuid.uuid4()),
        attempt_index=0,
        prompt_sha256=digest_bytes(b"visible"),
    )


def test_finite_match_not_a_proof_and_imported_proven_rejected():
    result = candidate(Runtime().evaluate("i256.zero", range(100)), ["0"] * 100)
    assert result["outcome"] == "full_horizon_match"
    assert result["proof_status"] == "not_claimed"
    for status in ("PROVEN", "proven", "shifted_identity"):
        forged = {**result, "proof_status": status}
        with pytest.raises(ValueError):
            validate_artifact(forged)


def test_shifted_identity_false_proof_cannot_override_executed_values():
    # a(n)=n, b(n)=n+1 is not a(n)=b(n), regardless of a legacy proof label.
    source = "local.get $n i64.extend_i32_s i64.const 0 i64.const 0 i64.const 0"
    result = candidate(Runtime().evaluate(source, range(100)), [str(n + 1) for n in range(100)])
    assert result["outcome"] == "wrong_values" and result["verified_terms"] == 0
    assert result["proof_status"] == "not_claimed"


def test_missing_evidence_or_truth_cannot_produce_match():
    good = Runtime().evaluate("i256.zero", range(20))
    for mutate in [
        lambda x: x.identities.clear(),
        lambda x: setattr(x, "reference_sha256", None),
        lambda x: setattr(x, "fuel", None),
        lambda x: setattr(x, "elapsed_ns", 2_000_000_001),
    ]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError):
            candidate(bad, ["0"] * 20, "prefix")
    with pytest.raises(ValueError):
        candidate(good, ["0"] * 19, "prefix")
    bad = copy.deepcopy(good)
    bad.reference_outputs.pop()
    assert candidate(bad, ["0"] * 20, "prefix")["outcome"] == "incomplete_output"
    bad = copy.deepcopy(good)
    bad.reference_outputs[0] = "1"
    assert candidate(bad, ["0"] * 20, "prefix")["outcome"] == "runtime_failure"


def test_forged_resource_and_reference_records_rejected():
    good = Runtime().evaluate("i256.zero", range(20))
    mutations = [
        lambda x: x.fuel_terms.__setitem__(0, 1_000_001),
        lambda x: x.reference_terms[0].__setitem__("steps", 1_000_001),
        lambda x: x.reference_terms[0].__setitem__("value", 9),
        lambda x: setattr(x, "reference_sha256", digest_bytes(b"forged")),
        lambda x: x.identities.__setitem__("runtime_sha256", digest_bytes(b"stale")),
    ]
    for mutate in mutations:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError):
            candidate(bad, ["0"] * 20, "prefix")


@pytest.mark.parametrize("label", ["prefix_match", "full_horizon_match", "PROVEN"])
def test_raw_evidence_cannot_preclaim_verification(label):
    forged = Runtime().evaluate("i256.zero", range(20))
    forged.outcome = label
    forged.identities.clear()
    with pytest.raises(ValueError, match="raw execution evidence"):
        candidate(forged, ["0"] * 20, "prefix")
