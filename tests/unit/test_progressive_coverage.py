from dataclasses import replace
import json
import pytest
from oeis_learn.generalization.progressive.coverage import audit
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.contracts import ResourceProfile
from tests.helpers.progressive_fixture import manifest


def witness(tmp_path, example):
    path = tmp_path / "witness.json"
    path.write_text(json.dumps({k: example[k] for k in ("source", "indices", "values")}))
    return reference(tmp_path, "witness.json")


def test_no_witness_or_neural_failure_is_not_inexpressibility(tmp_path):
    r = audit(
        tmp_path, "x", None, missing_operations=["division"], created_at="2026-11-08T00:00:00Z"
    )
    assert r["payload"]["classification"] == "unresolved"
    assert r["payload"]["missing_operations"] == ["division"]


def test_reference_checks_independent_fixture_and_missing_native_is_unresolved(tmp_path):
    for example in manifest()["programs"]:
        path = tmp_path / example["id"]
        path.mkdir()
        ref = witness(path, example)
        result = audit(path, example["id"], ref, created_at="2026-11-08T00:00:00Z")
        assert result["payload"]["classification"] == "unresolved"
        expected = (
            "witness_counterexample"
            if example["id"] == "prefix-overfit"
            else "native_evidence_unavailable"
        )
        assert result["payload"]["reason"] == expected


def test_signed_overflow_and_grammar_have_independent_expectations():
    signed = manifest()["programs"][1]
    assert evaluate(signed["source"], signed["indices"]).values == tuple(signed["values"])
    limits = replace(ResourceProfile(), input_bits=4, literal_bits=4, intermediate_bits=4)
    assert (
        evaluate("(program (return (mul $index $index)))", ["7"], limits).reason == "integer_limit"
    )
    assert evaluate("(program (return (div $index (int 0))))", ["1"]).reason == "invalid_source"
    from oeis_learn.decoder.bigint_codec import encode_body, validate_body, CodecError

    valid = encode_body("(program (return (int -2)))")
    with pytest.raises(CodecError):
        validate_body(valid[:-1])


def test_native_witness_acceptance_uses_packaged_service(tmp_path, fitting_client):
    ref = witness(tmp_path, manifest()["programs"][0])
    r = audit(tmp_path, "square", ref, client=fitting_client, created_at="2026-11-08T00:00:00Z")
    assert r["payload"]["classification"] == "supported"
    r = audit(
        tmp_path,
        "square-budget",
        ref,
        client=fitting_client,
        token_cap=1,
        created_at="2026-11-08T00:00:00Z",
    )
    assert r["payload"]["classification"] == "over_budget"


def test_retired_fitting_profile_is_explicitly_unsupported():
    from oeis_learn.synthesis.fitting import fit_program

    r = fit_program("(program (return (int 0)))", [0] * 20, profile="retired-scalar/v1")
    assert r.outcome == "unsupported"
    assert r.response["admitted"] == []
