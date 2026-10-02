"""Strict foundation/v1 configuration loading (007).

Config files are YAML. Loading is strict: unknown keys are rejected at every
level, disabled features (solvers/optimizers/scaffolds/RL/proving) must be
declared explicitly as unavailable/disabled, profile locks must match the
frozen profile identities, and every effective model constructor setting must
be persisted before work begins so that no implicit default can change on
resume.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

from oeis_learn.experiments.profiles import (
    CODEC_PROFILE_ID,
    EVALUATION_PROFILE_ID,
    LANGUAGE_PROFILE_ID,
    RESOURCE_PROFILE_ID,
    profile_digests,
)

# The only schema_version accepted in foundation configs.
CONFIG_SCHEMA_VERSION = "foundation/v1"

# Feature groups that are disabled/unavailable in the foundation/v1 smoke.
DISABLED_FEATURE_GROUPS = ("solvers", "optimizers", "scaffolds", "rl", "proving")


class ConfigError(ValueError):
    """Strict configuration failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# ---------------------------------------------------------------------------
# Strict schema tree. Values are (kind, ...) where kind is one of:
#   "str", "int", "bool", "enum", "list", "map", "nullable-*"
# ---------------------------------------------------------------------------

_LEARNING_ENUM = ("strict_generic",)
_OBJECTIVE_ENUM = ("sft",)
_INIT_ENUM = ("random",)
_POLICY_ENUM = ("prefix_rebased_zero",)
_STATE_ENUM = ("disabled", "unavailable", "measured")


def _expect(schema: dict[str, Any], path: str, data: Any) -> None:
    kind = schema[0]
    if kind == "map":
        allowed = schema[1]
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
        missing = set(allowed) - data.keys()
        _require(not missing, f"{path}: missing required keys {sorted(missing)}")
        for key, value in data.items():
            if key not in allowed:
                raise ConfigError(f"{path}: unknown key {key!r}")
            _expect(allowed[key], f"{path}.{key}", value)
        return
    if kind == "list":
        item_schema = schema[1]
        if not isinstance(data, list):
            raise ConfigError(f"{path}: expected a list, got {type(data).__name__}")
        for i, item in enumerate(data):
            _expect(item_schema, f"{path}[{i}]", item)
        return
    if kind == "enum":
        choices = schema[1]
        if data not in choices:
            raise ConfigError(f"{path}: {data!r} not in {choices!r}")
        return
    if kind == "int":
        _require(
            isinstance(data, int) and not isinstance(data, bool), f"{path}: expected an integer"
        )
        return
    if kind == "str":
        _require(isinstance(data, str) and data, f"{path}: expected a non-empty string")
        return
    if kind == "bool":
        _require(isinstance(data, bool), f"{path}: expected a boolean")
        return
    raise ConfigError(f"{path}: invalid schema kind {kind!r}")


_TOP_SCHEMA: dict[str, Any] = {
    "schema_version": ("str",),
    "kind": ("enum", ("smoke",)),
    "learning": (
        "map",
        {
            "track": ("enum", _LEARNING_ENUM),
            "objective": ("enum", _OBJECTIVE_ENUM),
            "initialization": ("enum", _INIT_ENUM),
            "horizon": ("list", ("int",)),
            "policy": ("enum", _POLICY_ENUM),
        },
    ),
    "model": (
        "map",
        {
            "d_model": ("int",),
            "encoder_layers": ("int",),
            "decoder_layers": ("int",),
            "heads": ("int",),
            "feed_forward": ("int",),
            "dropout": ("int",),
            "film": ("bool",),
            "summary_tokens": ("bool",),
            "dtype": ("enum", ("fp32",)),
            "modulus": ("list", ("int",)),
            "primes": ("list", ("int",)),
            "codec_output_vocab": ("int",),
            "position": ("enum", ("sinusoidal",)),
            "encoder_max_seq_len": ("int",),
            "decoder_max_seq_len": ("int",),
            "max_valuation": ("int",),
            "chunk_size": ("int",),
            "logit_cap_threshold": ("int",),
        },
    ),
    "training": (
        "map",
        {
            "batch_size": ("int",),
            "updates": ("int",),
            "optimizer": ("enum", ("adamw",)),
            "learning_rate": ("int",),  # stored as fixed-point integer (3e-4 -> 3 * 10**-4)
            "learning_rate_exp": ("int",),
            "weight_decay": ("int",),
            "weight_decay_exp": ("int",),
            "grad_norm_cap": ("int",),
            "scheduler": ("enum", _STATE_ENUM),
            "scaler": ("enum", _STATE_ENUM),
        },
    ),
    "disabled": (
        "map",
        {
            "solvers": ("enum", _STATE_ENUM),
            "optimizers": ("enum", _STATE_ENUM),
            "scaffolds": ("enum", _STATE_ENUM),
            "rl": ("enum", _STATE_ENUM),
            "proving": ("enum", _STATE_ENUM),
        },
    ),
    "limits": (
        "map",
        {
            "max_body_tokens": ("int",),
            "max_source_bytes": ("int",),
        },
    ),
    "locks": (
        "map",
        {
            "language": ("str",),
            "codec": ("str",),
            "resource": ("str",),
            "evaluation": ("str",),
            "language_sha256": ("str",),
            "codec_sha256": ("str",),
            "resource_sha256": ("str",),
            "evaluation_sha256": ("str",),
        },
    ),
}


def _validate_semantics(data: dict[str, Any], source: str) -> None:
    kind = data["kind"]
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{source}: schema_version must be {CONFIG_SCHEMA_VERSION!r}")

    if kind == "smoke":
        _require(
            data["learning"]["track"] == "strict_generic", "smoke.track must be strict_generic"
        )
        _require(data["learning"]["objective"] == "sft", "smoke.objective must be sft")
        _require(
            data["learning"]["initialization"] == "random", "smoke.initialization must be random"
        )
        _require(data["learning"]["horizon"] == [20, 100], "smoke.horizon must be [20, 100]")
        _require(
            data["learning"]["policy"] == "prefix_rebased_zero",
            "smoke.policy must be prefix_rebased_zero",
        )

        model = data["model"]
        _require(model["dropout"] == 10, "smoke.model.dropout must be 0.1 (fixed-point 10)")
        _require(model["film"] is True, "smoke.model.film must be enabled")
        _require(model["summary_tokens"] is False, "smoke.model.summary_tokens must be disabled")
        _require(model["dtype"] == "fp32", "smoke.model.dtype must be fp32")
        _require(model["position"] == "sinusoidal", "smoke.model.position must be sinusoidal")

        training = data["training"]
        _require(training["batch_size"] == 4, "smoke.training.batch_size must be 4")
        _require(training["updates"] == 3, "smoke.training.updates must be 3")
        _require(training["scheduler"] == "disabled", "smoke.training.scheduler must be disabled")
        _require(training["scaler"] == "disabled", "smoke.training.scaler must be disabled")

    from oeis_learn.decoder.program_codec import FOUNDATION_VOCAB_SIZE

    model = data["model"]
    for key in (
        "d_model",
        "heads",
        "encoder_layers",
        "decoder_layers",
        "feed_forward",
        "encoder_max_seq_len",
        "decoder_max_seq_len",
        "max_valuation",
        "chunk_size",
        "logit_cap_threshold",
    ):
        _require(model[key] > 0, f"model.{key} must be positive")
    _require(
        model["d_model"] % model["heads"] == 0 and model["d_model"] % 2 == 0,
        "d_model must be even and divisible by heads",
    )
    _require(
        model["encoder_max_seq_len"] >= 20 and model["decoder_max_seq_len"] >= 1024,
        "position capacity is shorter than the frozen horizon/context",
    )
    _require(
        model["codec_output_vocab"] == FOUNDATION_VOCAB_SIZE,
        "model vocabulary must be explicitly codec-sized",
    )
    _require(
        model["modulus"] == list(range(2, 102)), "smoke requires the explicit 100-modulus inventory"
    )
    _require(
        model["primes"] == [2, 3, 5, 7, 11, 13],
        "smoke requires the explicit p-adic prime inventory",
    )
    expected_training = {
        "learning_rate": 3,
        "learning_rate_exp": -4,
        "weight_decay": 1,
        "weight_decay_exp": -2,
        "grad_norm_cap": 1,
    }
    for key, value in expected_training.items():
        _require(data["training"][key] == value, f"smoke.training.{key} must be {value}")
    _require(
        data["limits"] == {"max_body_tokens": 1024, "max_source_bytes": 65536},
        "limits differ from frozen codec",
    )
    for key, digest in profile_digests().items():
        _require(data["locks"][key + "_sha256"] == digest, f"locks.{key}_sha256 content mismatch")

    for group in DISABLED_FEATURE_GROUPS:
        state = data["disabled"][group]
        _require(
            state == "disabled",
            f"smoke.disabled.{group} must be 'disabled' in the foundation/v1 smoke",
        )

    locks = data["locks"]
    _require(
        locks.get("language") == LANGUAGE_PROFILE_ID,
        "locks.language must match the frozen language profile (missing or mismatched identity)",
    )
    _require(
        locks.get("codec") == CODEC_PROFILE_ID,
        "locks.codec must match the frozen codec profile (missing or mismatched identity)",
    )
    _require(
        locks.get("resource") == RESOURCE_PROFILE_ID,
        "locks.resource must match the frozen resource profile (missing or mismatched identity)",
    )
    _require(
        locks.get("evaluation") == EVALUATION_PROFILE_ID,
        "locks.evaluation must match the frozen evaluation profile (missing or mismatched identity)",
    )


@dataclass
class FoundationConfig:
    """A strictly loaded foundation configuration with all effective settings."""

    _data: dict[str, Any]
    source: str

    @property
    def data(self) -> dict[str, Any]:
        return deepcopy(self._data)

    @property
    def kind(self) -> str:
        return self.data["kind"]

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def effective_model_profile(self) -> dict[str, Any]:
        """Persisted effective model constructor settings (no implicit defaults)."""
        model = deepcopy(self._data["model"])
        model["dropout"] /= 100
        return model

    def model_constructor_kwargs(self) -> dict[str, Any]:
        """Exact arguments for the existing encoder and decoder constructors."""
        from oeis_learn.decoder.program_codec import PAD_ID, codec_digest

        m = self.effective_model_profile()
        common = dict(
            d_model=m["d_model"], n_heads=m["heads"], d_ff=m["feed_forward"], dropout=m["dropout"]
        )
        return {
            "encoder": dict(
                common,
                n_encoder_layers=m["encoder_layers"],
                max_seq_len=m["encoder_max_seq_len"],
                primes=m["primes"],
                max_valuation=m["max_valuation"],
                moduli_count=len(m["modulus"]),
                base_moduli=m["modulus"],
                use_film=m["film"],
                enable_summary_tokens=m["summary_tokens"],
            ),
            "decoder": dict(
                common,
                vocab_size=m["codec_output_vocab"],
                n_decoder_layers=m["decoder_layers"],
                max_seq_len=m["decoder_max_seq_len"],
                pad_idx=PAD_ID,
                chunk_size=m["chunk_size"],
                logit_cap_threshold=float(m["logit_cap_threshold"]),
                codec_sha256=codec_digest(),
            ),
        }

    def effective_training_profile(self) -> dict[str, Any]:
        training = deepcopy(self._data["training"])
        for key in ("learning_rate", "weight_decay"):
            training[key] *= 10 ** training.pop(key + "_exp")
        training["grad_norm_cap"] = float(training["grad_norm_cap"])
        return training

    def persist_effective(self, root: Path):
        from oeis_learn.experiments.artifacts import ArtifactPath, atomic_write, canonical_bytes

        payload = self.to_dict()
        payload["model"] = self.effective_model_profile()
        payload["constructors"] = self.model_constructor_kwargs()
        payload["training"] = self.effective_training_profile()
        path = ArtifactPath(root, "effective-config.json")
        atomic_write(path, canonical_bytes(payload))
        return path

    def disabled_features(self) -> dict[str, str]:
        return dict(self.data["disabled"])

    def locks(self) -> dict[str, str]:
        return dict(self.data["locks"])


def load_config(path: str | PathLike) -> FoundationConfig:
    """Load and strictly validate a foundation config file.

    Rejects unknown keys at every level, mismatched profile locks, incomplete
    identities and undeclared disabled features. Returns a FoundationConfig
    carrying every effective setting.
    """
    import os

    p = os.fspath(path) if hasattr(path, "__fspath__") else str(path)
    with open(p, encoding="utf-8") as f:
        try:
            raw = yaml.load(f, Loader=_UniqueLoader)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: config must be a YAML mapping")

    _expect(("map", _TOP_SCHEMA), "config", raw)
    _validate_semantics(raw, p)
    return FoundationConfig(raw, p)


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently retaining the last value."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigError("config keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
