"""Contract tests for frozen benchmark manifest schema conformance."""

from __future__ import annotations

import copy
import pytest
from jsonschema import ValidationError


@pytest.fixture
def valid_manifest_data() -> dict:
    h = "sha256:" + "0" * 64
    return {
        "schema_version": "1.0",
        "cohort_id": "test_cohort_v1",
        "manifest_sha256": h,
        "source": {
            "name": "OEIS stripped/names",
            "revision": "2026-09-04",
            "retrieved_at": "2026-09-04T12:00:00Z",
            "content_sha256": h,
            "license_notice": "CC BY-NC 4.0",
        },
        "observed_horizon": 20,
        "unseen_horizon": 100,
        "targets": [
            {
                "oeis_id": "A000045",
                "name": "Fibonacci numbers",
                "offset": 0,
                "family": "LINEAR_RECURRENCE_2",
                "curriculum_stage": 2,
                "observed_terms": [str(i) for i in range(20)],
                "unseen_terms": [str(i + 20) for i in range(100)],
                "result_profile": "i256x4_v1",
                "terms_sha256": h,
                "formula_definition_id": "def_A000045_v1",
                "term_fingerprint": h,
                "program_fingerprints": [h],
                "tags": ["core", "fibonacci"],
            }
        ],
        "exclusions": [
            {
                "oeis_id": "A999999",
                "reason_code": "INSUFFICIENT_TERMS",
                "message": "Only 45 terms available",
            }
        ],
    }


def test_valid_manifest_conforms(valid_manifest_data, validate_contract):
    validate_contract(valid_manifest_data, "benchmark-manifest")


def test_rejects_wrong_observed_horizon(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["observed_horizon"] = 15
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")


def test_rejects_wrong_unseen_horizon(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["unseen_horizon"] = 50
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")


def test_rejects_insufficient_observed_terms(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["targets"][0]["observed_terms"] = [str(i) for i in range(19)]
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")


def test_rejects_insufficient_unseen_terms(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["targets"][0]["unseen_terms"] = [str(i) for i in range(99)]
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")


def test_rejects_invalid_result_profile(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["targets"][0]["result_profile"] = "i32_scalar"
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")


def test_rejects_invalid_oeis_id(valid_manifest_data, validate_contract):
    d = copy.deepcopy(valid_manifest_data)
    d["targets"][0]["oeis_id"] = "45"
    with pytest.raises(ValidationError):
        validate_contract(d, "benchmark-manifest")
