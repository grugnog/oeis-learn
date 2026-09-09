"""Contract tests for warmstart transfer configuration schema validation."""

from __future__ import annotations

import pytest
from jsonschema import Draft7Validator, ValidationError


def test_warmstart_transfer_config_valid(load_schema):
    schema = load_schema("warmstart-transfer-config.schema.json")
    validator = Draft7Validator(schema)

    valid_config = {
        "base_checkpoint_path": "checkpoints/model_epoch_050.v2.pt",
        "strategy": "STRATEGY_C_PROGRESSIVE_SFT_BRIDGE",
        "vocabulary_expansion": {
            "projection_method": "CONVEX_HULL_SEMANTIC_PROJECTION",
            "logit_cap_threshold": 30.0,
            "norm_calibration": True,
        },
        "embedding_warmup_phase": {
            "steps": 1500,
            "learning_rate": 1e-4,
            "frozen_modules": ["encoder", "decoder.layers"],
        },
        "joint_sft_bridge_phase": {
            "steps": 5000,
            "encoder_lr": 1e-5,
            "decoder_lr": 5e-5,
            "heads_lr": 1e-4,
            "dataset_size": 20000,
        },
        "transition_gate": {
            "max_advantage_collapse_rate": 0.15,
            "min_grammar_validity": 0.985,
            "min_candidate_pass_rate": 0.75,
        },
        "rl_reanchoring_phase": {
            "initial_beta_kl": 0.00,
            "target_beta_kl": 0.04,
            "bandit_gamma_floor": 0.25,
            "seed_edb_with_canaries": True,
        },
    }

    validator.validate(valid_config)


def test_warmstart_transfer_config_invalid_acr(load_schema):
    schema = load_schema("warmstart-transfer-config.schema.json")
    validator = Draft7Validator(schema)

    invalid_config = {
        "base_checkpoint_path": "checkpoints/model_epoch_050.v2.pt",
        "strategy": "STRATEGY_C_PROGRESSIVE_SFT_BRIDGE",
        "vocabulary_expansion": {
            "projection_method": "CONVEX_HULL_SEMANTIC_PROJECTION",
            "logit_cap_threshold": 30.0,
            "norm_calibration": True,
        },
        "embedding_warmup_phase": {
            "steps": 1500,
            "learning_rate": 1e-4,
            "frozen_modules": [],
        },
        "joint_sft_bridge_phase": {
            "steps": 5000,
            "encoder_lr": 1e-5,
            "decoder_lr": 5e-5,
            "heads_lr": 1e-4,
            "dataset_size": 20000,
        },
        "transition_gate": {
            "max_advantage_collapse_rate": 0.35,  # Exceeds 0.15 maximum
            "min_grammar_validity": 0.985,
            "min_candidate_pass_rate": 0.75,
        },
        "rl_reanchoring_phase": {
            "initial_beta_kl": 0.00,
            "target_beta_kl": 0.04,
            "bandit_gamma_floor": 0.25,
            "seed_edb_with_canaries": True,
        },
    }

    with pytest.raises(ValidationError):
        validator.validate(invalid_config)
