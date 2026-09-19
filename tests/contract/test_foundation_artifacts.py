"""Contract tests for the 007 foundation external trust-boundary artifact schema.

Covers every required/unknown field, profile mismatch, hash/path rule, wrong
horizon, PROVEN label, floating integer, incomplete match and partial
checkpoint in ``artifacts.schema.json`` and ``data-model.md``. Also verifies
that the schema examples are explicitly synthetic fixtures and are never
eligible run evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from jsonschema import ValidationError

H0 = "sha256:" + "0" * 64


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def artifacts_schema(contracts_007_dir: Path) -> Dict[str, Any]:
    return _load_json(contracts_007_dir / "artifacts.schema.json")


@pytest.fixture(scope="session")
def schema_examples(contracts_007_dir: Path) -> list[Dict[str, Any]]:
    return _load_json(contracts_007_dir / "schema-examples.json")


@pytest.fixture
def valid_visible_prompt(schema_examples: list[Dict[str, Any]]) -> Dict[str, Any]:
    return copy.deepcopy(schema_examples[0])


@pytest.fixture
def valid_candidate_result(schema_examples: list[Dict[str, Any]]) -> Dict[str, Any]:
    return copy.deepcopy(schema_examples[1])


@pytest.fixture
def valid_checkpoint_manifest(schema_examples: list[Dict[str, Any]]) -> Dict[str, Any]:
    return copy.deepcopy(schema_examples[2])


def _assert_invalid(instance: Dict[str, Any], validate_contract) -> None:
    with pytest.raises(ValidationError):
        validate_contract(instance, "artifacts")


# ---------------------------------------------------------------------------
# Schema examples are fixtures, never eligible run evidence
# ---------------------------------------------------------------------------


def test_schema_examples_validate_against_artifacts_schema(schema_examples, validate_contract):
    for example in schema_examples:
        validate_contract(example, "artifacts")


def test_schema_examples_are_explicitly_synthetic_fixtures(artifacts_schema, schema_examples):
    description = artifacts_schema.get("description", "")
    assert "synthetic" in description.lower()
    assert "not executed-run evidence" in description.lower()
    # Every example must carry the exact foundation/v1 version; none may claim
    # a real digest (the all-zero fixture digests are placeholders, not evidence).
    for example in schema_examples:
        assert example["schema_version"] == "foundation/v1"
        assert example["kind"] in {"visible_prompt", "candidate_result", "checkpoint_manifest"}


def test_schema_examples_do_not_present_fixture_digests_as_evidence(artifacts_schema, schema_examples):
    # Fixture digests are all-zero placeholders; they must never be treated as
    # real identity evidence. The schema description records this explicitly.
    for example in schema_examples:
        for key, value in example.items():
            if isinstance(value, str) and value.startswith("sha256:"):
                assert value == H0, f"{example['kind']}.{key} uses a non-fixture digest"
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    if isinstance(v2, str) and v2.startswith("sha256:"):
                        assert v2 == H0, f"{example['kind']}.{key}.{k2} uses a non-fixture digest"


# ---------------------------------------------------------------------------
# Required and unknown fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,required",
    [
        ("valid_visible_prompt", ["kind", "schema_version", "request_nonce", "language_profile", "observed_terms"]),
        (
            "valid_candidate_result",
            [
                "kind",
                "schema_version",
                "purpose",
                "run_id",
                "attempt_index",
                "stage",
                "language_profile",
                "resource_profile",
                "source_sha256",
                "checkpoint_sha256",
                "prompt_sha256",
                "outcome",
                "reason",
                "outputs",
                "verified_terms",
                "first_failure_index",
                "selected",
                "reference_evidence_sha256",
                "expected_terms_sha256",
                "usage",
                "proof_status",
            ],
        ),
        (
            "valid_checkpoint_manifest",
            [
                "kind",
                "schema_version",
                "run_id",
                "generation",
                "completed_update",
                "blob_path",
                "blob_sha256",
                "contract_sha256",
                "pool_sha256",
                "codec_sha256",
                "runtime_sha256",
                "effective_config_sha256",
                "ledger_sequence",
                "charged_budget_ns",
                "next_sample_ids",
                "payload_state_keys",
                "checkpoint_state",
                "previous_manifest_sha256",
            ],
        ),
    ],
)
def test_every_required_field_is_required(fixture, required, request, validate_contract):
    instance = request.getfixturevalue(fixture)
    for field in required:
        d = copy.deepcopy(instance)
        del d[field]
        _assert_invalid(d, validate_contract)
        # re-added field makes it valid again (shape only)
        d[field] = instance[field]
        validate_contract(d, "artifacts")


@pytest.mark.parametrize("fixture", ["valid_visible_prompt", "valid_candidate_result", "valid_checkpoint_manifest"])
def test_unknown_top_level_field_rejected(fixture, request, validate_contract):
    instance = request.getfixturevalue(fixture)
    d = copy.deepcopy(instance)
    d["extra_field"] = "anything"
    _assert_invalid(d, validate_contract)


@pytest.mark.parametrize("fixture", ["valid_visible_prompt", "valid_candidate_result", "valid_checkpoint_manifest"])
def test_unknown_schema_version_rejected(fixture, request, validate_contract):
    instance = request.getfixturevalue(fixture)
    d = copy.deepcopy(instance)
    d["schema_version"] = "foundation/v2"
    _assert_invalid(d, validate_contract)


def test_unknown_nested_field_rejected_in_metric(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["usage"]["fuel"]["bogus"] = 1
    _assert_invalid(d, validate_contract)


def test_unknown_nested_field_rejected_in_usage(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["usage"]["extra"] = 1
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# Profile mismatch and digest rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_chain",
    [
        ("valid_visible_prompt", "language_profile"),
        ("valid_candidate_result", "language_profile"),
        ("valid_candidate_result", "resource_profile"),
        ("valid_candidate_result", "source_sha256"),
        ("valid_candidate_result", "prompt_sha256"),
        ("valid_checkpoint_manifest", "blob_sha256"),
        ("valid_checkpoint_manifest", "contract_sha256"),
        ("valid_checkpoint_manifest", "pool_sha256"),
        ("valid_checkpoint_manifest", "codec_sha256"),
        ("valid_checkpoint_manifest", "runtime_sha256"),
        ("valid_checkpoint_manifest", "effective_config_sha256"),
    ],
)
def test_digest_must_be_sha256_64_lowercase_hex(path_chain, request, validate_contract):
    fixture, field = path_chain
    instance = request.getfixturevalue(fixture)
    for bad in [
        "sha256:" + "0" * 63,  # too short
        "sha256:" + "0" * 65,  # too long
        "sha256:" + "0" * 63 + "g",  # non-hex
        "sha256:" + "0" * 32 + "A" * 32,  # uppercase
        "sha256:" + "z" * 64,  # non-hex
        "md5:" + "0" * 64,  # wrong scheme
        "0" * 64,  # missing scheme
        "",  # empty
    ]:
        d = copy.deepcopy(instance)
        d[field] = bad
        _assert_invalid(d, validate_contract)


def test_checkpoint_sha256_accepts_null_but_rejects_bad_digest(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["checkpoint_sha256"] = None
    validate_contract(d, "artifacts")
    d["checkpoint_sha256"] = "sha256:" + "0" * 63
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# Path rule for checkpoint blob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.pt",
        "sub/../escape.pt",
        "/absolute/path.pt",
        "a/b/c.pt",
        "checkpoint-000003.txt",
        "checkpoint-000003",
        "checkpoint-000003.pt.bak",
        "checkpoint-000003.pT",
        "",
    ],
)
def test_blob_path_must_be_a_safe_relative_pt_filename(valid_checkpoint_manifest, bad_path, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["blob_path"] = bad_path
    _assert_invalid(d, validate_contract)


def test_blob_path_accepts_plain_relative_filename(valid_checkpoint_manifest, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["blob_path"] = "checkpoint-000003.pt"
    validate_contract(d, "artifacts")


# ---------------------------------------------------------------------------
# IntegerText rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["1.5", "1e3", "1E3", "0.0", "01", "-0", "+1", " 1", "1 ", "1_000", "true", "false", "abc", "", "1.0e0"])
def test_integer_text_rejects_floats_notation_bools_and_noncanonical(bad, valid_visible_prompt, validate_contract):
    d = copy.deepcopy(valid_visible_prompt)
    d["observed_terms"][0] = bad
    _assert_invalid(d, validate_contract)


def test_integer_text_accepts_zero_positive_and_negative_canonical(valid_visible_prompt, validate_contract):
    d = copy.deepcopy(valid_visible_prompt)
    # replace the first five entries with canonical edge spellings; keep 20 total
    d["observed_terms"][:5] = ["0", "-1", "-9", "-999999999999999999999", "7"]
    validate_contract(d, "artifacts")


def test_integer_text_rejects_boolean_value_not_string(valid_visible_prompt, validate_contract):
    d = copy.deepcopy(valid_visible_prompt)
    d["observed_terms"][0] = True
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# Horizon / match consistency
# ---------------------------------------------------------------------------


def test_visible_prompt_requires_exactly_20_terms(valid_visible_prompt, validate_contract):
    d = copy.deepcopy(valid_visible_prompt)
    d["observed_terms"] = d["observed_terms"][:19]
    _assert_invalid(d, validate_contract)
    d2 = copy.deepcopy(valid_visible_prompt)
    d2["observed_terms"] = d2["observed_terms"] + ["21"]
    _assert_invalid(d2, validate_contract)


def test_prefix_match_requires_exactly_20_outputs(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "prefix_match"
    d["outputs"] = [str(i) for i in range(100)]
    d["verified_terms"] = 20
    d["stage"] = "prefix"
    d["reason"] = None
    d["first_failure_index"] = None
    d["reference_evidence_sha256"] = H0
    d["expected_terms_sha256"] = H0
    _assert_invalid(d, validate_contract)


def test_full_horizon_match_requires_exactly_100_outputs(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "full_horizon_match"
    d["outputs"] = [str(i) for i in range(20)]
    d["verified_terms"] = 100
    d["stage"] = "full"
    d["reason"] = None
    d["first_failure_index"] = None
    d["reference_evidence_sha256"] = H0
    d["expected_terms_sha256"] = H0
    _assert_invalid(d, validate_contract)


def test_full_horizon_match_requires_verified_terms_100(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "full_horizon_match"
    d["verified_terms"] = 99
    d["stage"] = "full"
    d["reason"] = None
    d["first_failure_index"] = None
    d["reference_evidence_sha256"] = H0
    d["expected_terms_sha256"] = H0
    _assert_invalid(d, validate_contract)


def test_prefix_match_requires_stage_prefix(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "prefix_match"
    d["outputs"] = [str(i) for i in range(20)]
    d["verified_terms"] = 20
    d["stage"] = "full"  # wrong
    d["reason"] = None
    d["first_failure_index"] = None
    d["reference_evidence_sha256"] = H0
    d["expected_terms_sha256"] = H0
    _assert_invalid(d, validate_contract)


def test_match_outcomes_require_evidence_digests(valid_candidate_result, validate_contract):
    for outcome in ("prefix_match", "full_horizon_match"):
        d = copy.deepcopy(valid_candidate_result)
        d["outcome"] = outcome
        d["outputs"] = [str(i) for i in range(20 if outcome == "prefix_match" else 100)]
        d["verified_terms"] = 20 if outcome == "prefix_match" else 100
        d["stage"] = "prefix" if outcome == "prefix_match" else "full"
        d["reason"] = None
        d["first_failure_index"] = None
        d["reference_evidence_sha256"] = None
        d["expected_terms_sha256"] = None
        _assert_invalid(d, validate_contract)


def test_non_match_outcome_requires_reason(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "wrong_values"
    d["reason"] = None
    _assert_invalid(d, validate_contract)
    d["reason"] = "first divergence at index 3"
    validate_contract(d, "artifacts")


# ---------------------------------------------------------------------------
# Incomplete match / partial output
# ---------------------------------------------------------------------------


def test_incomplete_output_outcome_is_allowed_only_with_reason(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "incomplete_output"
    d["reason"] = "only 60 terms produced within deadline"
    validate_contract(d, "artifacts")


def test_outputs_may_be_partial_for_non_match_outcomes(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outcome"] = "execution_limit"
    d["reason"] = "fuel exhausted at index 40"
    d["outputs"] = [str(i) for i in range(40)]
    d["verified_terms"] = 0
    validate_contract(d, "artifacts")


def test_outputs_cannot_exceed_100(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["outputs"] = [str(i) for i in range(101)]
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# PROVEN / proof boundary
# ---------------------------------------------------------------------------


def test_proof_status_must_be_exactly_not_claimed(valid_candidate_result, validate_contract):
    for bad in ("PROVEN", "proven", "claimed", "not_proven", "UNKNOWN"):
        d = copy.deepcopy(valid_candidate_result)
        d["proof_status"] = bad
        _assert_invalid(d, validate_contract)


def test_proof_status_is_required(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    del d["proof_status"]
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# Partial checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_state_must_be_complete(valid_checkpoint_manifest, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["checkpoint_state"] = "partial"
    _assert_invalid(d, validate_contract)


def test_payload_state_keys_require_all_eleven_unique(valid_checkpoint_manifest, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["payload_state_keys"] = ["model", "optimizer"]
    _assert_invalid(d, validate_contract)
    d2 = copy.deepcopy(valid_checkpoint_manifest)
    d2["payload_state_keys"] = d2["payload_state_keys"] + ["model"]  # duplicate
    _assert_invalid(d2, validate_contract)
    d3 = copy.deepcopy(valid_checkpoint_manifest)
    d3["payload_state_keys"] = d3["payload_state_keys"] + ["bogus"]
    _assert_invalid(d3, validate_contract)


def test_completed_update_and_generation_are_nonnegative(valid_checkpoint_manifest, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["completed_update"] = -1
    _assert_invalid(d, validate_contract)
    d2 = copy.deepcopy(valid_checkpoint_manifest)
    d2["generation"] = -1
    _assert_invalid(d2, validate_contract)


def test_charged_budget_ns_is_nonnegative_integer(valid_checkpoint_manifest, validate_contract):
    d = copy.deepcopy(valid_checkpoint_manifest)
    d["charged_budget_ns"] = -5
    _assert_invalid(d, validate_contract)
    d2 = copy.deepcopy(valid_checkpoint_manifest)
    d2["charged_budget_ns"] = True
    _assert_invalid(d2, validate_contract)


# ---------------------------------------------------------------------------
# Metric semantics (measured / disabled / unavailable)
# ---------------------------------------------------------------------------


def test_measured_metric_requires_integer_value_and_source(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["usage"]["fuel"]["state"] = "measured"
    d["usage"]["fuel"]["value"] = None
    _assert_invalid(d, validate_contract)
    d["usage"]["fuel"]["value"] = 5
    d["usage"]["fuel"]["source"] = None
    _assert_invalid(d, validate_contract)
    d["usage"]["fuel"]["source"] = "wasmtime instrumentation"
    d["usage"]["fuel"]["reason"] = None  # measured requires null reason
    validate_contract(d, "artifacts")


def test_disabled_or_unavailable_metric_requires_null_value_and_reason(valid_candidate_result, validate_contract):
    for state in ("disabled", "unavailable"):
        d = copy.deepcopy(valid_candidate_result)
        d["usage"]["fuel"]["state"] = state
        d["usage"]["fuel"]["value"] = 5
        _assert_invalid(d, validate_contract)
        d["usage"]["fuel"]["value"] = None
        d["usage"]["fuel"]["reason"] = None
        _assert_invalid(d, validate_contract)
        d["usage"]["fuel"]["reason"] = "not measured in fixture"
        validate_contract(d, "artifacts")


def test_metric_value_must_be_nonnegative_integer_or_null(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["usage"]["elapsed_ns"]["state"] = "measured"
    d["usage"]["elapsed_ns"]["value"] = -1
    d["usage"]["elapsed_ns"]["source"] = "clock"
    _assert_invalid(d, validate_contract)


# ---------------------------------------------------------------------------
# Model purpose requires checkpoint identity
# ---------------------------------------------------------------------------


def test_model_purpose_requires_checkpoint_digest(valid_candidate_result, validate_contract):
    d = copy.deepcopy(valid_candidate_result)
    d["purpose"] = "model"
    d["checkpoint_sha256"] = None
    _assert_invalid(d, validate_contract)
    d["checkpoint_sha256"] = H0
    validate_contract(d, "artifacts")
