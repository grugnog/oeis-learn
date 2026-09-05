"""Contract tests for paired experiment manifest schema conformance."""

from __future__ import annotations

import copy
import json
import pytest
from jsonschema import ValidationError


@pytest.fixture
def valid_inference_manifest_data() -> dict:
    h = "sha256:" + "0" * 64
    with open("configs/experiments/trustworthy_inference_v1.json") as f:
        return json.load(f)


@pytest.fixture
def valid_curriculum_manifest_data() -> dict:
    with open("configs/experiments/trustworthy_curriculum_v1.json") as f:
        return json.load(f)


def test_inference_manifest_conforms(valid_inference_manifest_data, validate_contract):
    validate_contract(valid_inference_manifest_data, "experiment-manifest")


def test_curriculum_manifest_conforms(valid_curriculum_manifest_data, validate_contract):
    validate_contract(valid_curriculum_manifest_data, "experiment-manifest")


def test_rejects_fewer_than_three_seeds(valid_inference_manifest_data, validate_contract):
    d = copy.deepcopy(valid_inference_manifest_data)
    d["seeds"] = [42, 137]  # Less than 3
    with pytest.raises(ValidationError):
        validate_contract(d, "experiment-manifest")


def test_rejects_trial_hours_exceeding_four(valid_inference_manifest_data, validate_contract):
    d = copy.deepcopy(valid_inference_manifest_data)
    d["max_trial_hours"] = 5.0
    with pytest.raises(ValidationError):
        validate_contract(d, "experiment-manifest")


def test_training_ablation_requires_exact_decision_schedule(valid_curriculum_manifest_data, validate_contract):
    d = copy.deepcopy(valid_curriculum_manifest_data)
    d["decision_schedule"] = [0, 50, 100]  # Must be [0, 100, 200, 300, 400, 500]
    with pytest.raises(ValidationError):
        validate_contract(d, "experiment-manifest")
