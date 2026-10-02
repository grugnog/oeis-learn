"""Regression tests for strict foundation config loading (T005)."""

import copy

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
    assert profile["dropout"] == 0.1
    assert profile["film"] is True
    assert profile["summary_tokens"] is False
    assert profile["dtype"] == "fp32"
    assert profile["position"] == "sinusoidal"
    assert len(profile["modulus"]) == 100
    assert profile["primes"] == [2, 3, 5, 7, 11, 13]


@pytest.mark.parametrize(
    "section", [None, "learning", "model", "training", "disabled", "limits", "locks"]
)
def test_every_required_config_key_is_required(section, tmp_path):
    original = _load()
    mapping = original if section is None else original[section]
    for key in mapping:
        value = copy.deepcopy(original)
        del (value if section is None else value[section])[key]
        with pytest.raises(ConfigError, match="missing"):
            load_config(_dump(tmp_path, value))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "d_model", -2),
        ("model", "heads", 0),
        ("model", "encoder_layers", True),
        ("model", "codec_output_vocab", 0),
        ("model", "primes", [1]),
        ("model", "modulus", [0]),
        ("limits", "max_body_tokens", 2048),
        ("training", "learning_rate_exp", -3),
        ("model", "dropout", 10.0),
        ("locks", "codec_sha256", "sha256:" + "f" * 64),
    ],
)
def test_invalid_semantics_rejected(section, key, value, tmp_path):
    data = _load()
    data[section][key] = value
    with pytest.raises(ConfigError):
        load_config(_dump(tmp_path, data))


def test_duplicate_yaml_keys_rejected(tmp_path):
    p = tmp_path / "duplicate.yaml"
    p.write_text("schema_version: foundation/v1\nschema_version: foundation/v1\n")
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


def test_config_copies_and_persisted_effective_values(tmp_path):
    import json

    from oeis_learn.experiments.artifacts import compute_canonical_digest, compute_file_hash

    cfg = load_config(SMOKE_PATH)
    cfg.to_dict()["model"]["primes"].append(97)
    assert cfg.data["model"]["primes"] == [2, 3, 5, 7, 11, 13]
    path = cfg.persist_effective(tmp_path).as_path()
    record = json.loads(path.read_text())
    assert record["model"]["dropout"] == 0.1
    assert record["training"]["learning_rate"] == pytest.approx(0.0003)
    assert record["training"]["weight_decay"] == 0.01
    assert compute_file_hash(path) == compute_canonical_digest(record)


def test_frozen_profile_and_packaged_schema_match_contract(tmp_path):
    import json
    from pathlib import Path

    from oeis_learn.experiments.models import _artifact_schema
    from oeis_learn.experiments.profiles import validate_profile_file

    validate_profile_file("configs/foundation/wat_profile.yaml")
    data = yaml.safe_load(Path("configs/foundation/wat_profile.yaml").read_text())
    data["language"]["operators"].append("call")
    p = tmp_path / "bad-profile.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        validate_profile_file(p)
    assert _artifact_schema() == json.loads(
        Path("specs/007-experiment-foundation/contracts/artifacts.schema.json").read_text()
    )
