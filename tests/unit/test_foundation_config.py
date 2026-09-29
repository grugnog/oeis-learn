"""Regression tests for strict foundation config loading (T005)."""

import copy
import io

import pytest
import yaml

from oeis_learn.experiments.config import ConfigError, load_config

SMOKE_PATH = "configs/foundation/wat_smoke.yaml"


def _load() -> dict:
    return yaml.safe_load(open(SMOKE_PATH, encoding="utf-8"))


def test_valid_smoke_loads():
    cfg = load_config(SMOKE_PATH)
    assert cfg.kind == "smoke"
    assert cfg.data["learning"]["track"] == "strict_generic"
    assert cfg.data["training"]["batch_size"] == 4
    assert cfg.data["training"]["updates"] == 3
    assert cfg.data["training"]["scheduler"] == "disabled"
    assert cfg.data["training"]["scaler"] == "disabled"
    assert cfg.data["model"]["dropout"] == 10
    assert cfg.data["model"]["film"] is True
    assert cfg.data["model"]["summary_tokens"] is False


def _dump(tmp_path: pytest.TempPathFactory, d: dict) -> str:
    p = str(tmp_path / "bad.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return p


def test_unknown_top_level_key_rejected(tmp_path):
    d = _load()
    d["bogus"] = 1
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_unknown_nested_key_rejected(tmp_path):
    d = _load()
    d["model"]["bogus"] = 1
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_unknown_disabled_group_rejected(tmp_path):
    d = _load()
    d["disabled"]["quantization"] = "disabled"
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_mismatched_lock_rejected(tmp_path):
    d = _load()
    d["locks"]["codec"] = "not-the-frozen-codec"
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_incomplete_identity_rejected(tmp_path):
    d = _load()
    del d["locks"]["resource"]
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_undeclared_disabled_feature_rejected(tmp_path):
    d = _load()
    d["disabled"]["solvers"] = "unavailable"  # smoke requires explicit 'disabled'
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_wrong_schema_version_rejected(tmp_path):
    d = _load()
    d["schema_version"] = "foundation/v0"
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, d))


def test_effective_model_profile_is_explicit():
    cfg = load_config(SMOKE_PATH)
    profile = cfg.effective_model_profile()
    # Every constructor setting must be persisted; no implicit default.
    assert profile["d_model"] == 256
    assert profile["encoder_layers"] == 4
    assert profile["decoder_layers"] == 4
    assert profile["heads"] == 4
    assert profile["feed_forward"] == 1024
    assert profile["dropout"] == 10
    assert profile["film"] is True
    assert profile["summary_tokens"] is False
    assert profile["dtype"] == "fp32"
    assert profile["position"] == "sinusoidal"
    assert len(profile["modulus"]) == 100
    assert profile["primes"] == [2, 3, 5, 7, 11, 13]
