"""Contract test validating SYMPLE configuration schema in configs/train_tier1.yaml."""

from __future__ import annotations

import json
import os
import pytest
import yaml


def test_symple_configuration_schema_contract():
    schema_path = "specs/004-decoupled-grounding-and-symple-engine/contracts/symple-config.schema.json"
    assert os.path.exists(schema_path), f"Schema file missing: {schema_path}"

    with open(schema_path, "r") as f:
        schema = json.load(f)

    assert schema["title"] == "SYMPLEConfigurationSchema"
    assert "symple" in schema["properties"]
    assert "solver" in schema["properties"]
    assert "parsimony" in schema["properties"]

    # Verify train_tier1.yaml conforms to the contract
    config_path = "configs/train_tier1.yaml"
    assert os.path.exists(config_path), f"Config file missing: {config_path}"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "symple" in cfg
    symple_cfg = cfg["symple"]
    assert symple_cfg["active_prompts"] == 2
    assert symple_cfg["min_group_size"] >= 8
    assert symple_cfg["max_group_size"] <= 64
    assert symple_cfg["exp3_gamma"] == 0.15
    assert symple_cfg["exp3_alpha"] == 0.05
    assert symple_cfg["beta_sft_replay"] == 0.50

    assert "solver" in cfg
    assert cfg["solver"]["enable_diophantine"] is True
    assert cfg["solver"]["enable_smt"] is True
    assert cfg["solver"]["smt_timeout_ms"] == 250

    assert "parsimony" in cfg
    assert cfg["parsimony"]["enable_dce"] is True
    assert cfg["parsimony"]["hard_waste_threshold"] == 0.30
