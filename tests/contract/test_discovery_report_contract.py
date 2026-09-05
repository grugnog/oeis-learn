"""Contract tests for discovery report and symbolic definition registry schemas."""

from __future__ import annotations

import copy
import pytest
from jsonschema import ValidationError


@pytest.fixture
def valid_discovery_report_data() -> dict:
    h = "sha256:" + "0" * 64
    return {
        "schema_version": "1.0",
        "report_id": "rep_disc_001",
        "run_id": "run_008_test",
        "created_at": "2026-09-04T12:00:00Z",
        "checkpoint_sha256": h,
        "benchmark_manifest_sha256": h,
        "protocol": {
            "protocol_id": h,
            "search_indices": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "validation_indices": list(range(20)),
            "unseen_indices": list(range(20, 120)),
            "distance_threshold": 0.8,
            "max_candidates": 50,
            "precision_digits": 500,
            "seed": 42,
            "definition_registry_sha256": h,
        },
        "claims": [],
        "candidate_dispositions": [],
        "summary": {
            "latent_candidates": 0,
            "unique_claims": 0,
            "duplicate_candidates": 0,
            "numerical_conjectures": 0,
            "symbolically_proven": 0,
            "rejected": 0,
            "insufficient_evidence": 0,
        },
        "errors": [],
    }


def test_discovery_report_conforms(valid_discovery_report_data, validate_contract):
    validate_contract(valid_discovery_report_data, "discovery-report")


def test_symbolic_definitions_conforms(validate_contract):
    h = "sha256:" + "0" * 64
    defs_data = {
        "schema_version": "1.0",
        "registry_id": "symbolic_definitions_v1",
        "registry_sha256": h,
        "created_at": "2026-09-04T12:00:00Z",
        "definitions": [
            {
                "definition_id": "def_A000290_v1",
                "sequence_ref": {
                    "oeis_id": "A000290",
                    "index_scale": 1,
                    "index_shift": 0,
                },
                "kind": "CLOSED_FORM",
                "source": {
                    "reference": "OEIS A000290",
                    "revision": "2026-09-04",
                    "content_sha256": h,
                },
                "expression": "n**2",
                "domain": {
                    "integer_only": True,
                    "lower_bound": 0,
                    "upper_bound": None,
                },
                "assumptions": ["n >= 0"],
                "parser_version": "sympy-1.12",
            }
        ],
    }
    validate_contract(defs_data, "symbolic-definitions")
