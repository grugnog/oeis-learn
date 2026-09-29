"""Strict foundation/v1 configuration loading (007).

Config files are YAML. Loading is strict: unknown keys are rejected at every
level, disabled features (solvers/optimizers/scaffolds/RL/proving) must be
declared explicitly as unavailable/disabled, profile locks must match the
frozen profile identities, and every effective model constructor setting must
be persisted before work begins so that no implicit default can change on
resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from oeis_learn.experiments.profiles import (
    CODEC_PROFILE_ID,
    EVALUATION_PROFILE_ID,
    LANGUAGE_PROFILE_ID,
    RESOURCE_PROFILE_ID,
)
from oeis_learn.experiments.models import SCHEMA_VERSION as _MODEL_VERSION

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


def _expect(schema: Dict[str, Any], path: str, data: Any) -> None:
    kind = schema[0]
    if kind == "map":
        allowed = schema[1]
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
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
        _require(isinstance(data, int) and not isinstance(data, bool), f"{path}: expected an integer")
        return
    if kind == "str":
        _require(isinstance(data, str) and data, f"{path}: expected a non-empty string")
        return
    if kind == "bool":
        _require(isinstance(data, bool), f"{path}: expected a boolean")
        return
    raise ConfigError(f"{path}: invalid schema kind {kind!r}")


_TOP_SCHEMA: Dict[str, Any] = {
    "schema_version": ("str",),
    "kind": ("enum", ("smoke", "profile", "sampler")),
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
        },
    ),
}


def _validate_semantics(data: Dict[str, Any], source: str) -> None:
    kind = data["kind"]
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{source}: schema_version must be {CONFIG_SCHEMA_VERSION!r}")

    if kind == "smoke":
        _require(data["learning"]["track"] == "strict_generic", "smoke.track must be strict_generic")
        _require(data["learning"]["objective"] == "sft", "smoke.objective must be sft")
        _require(data["learning"]["initialization"] == "random", "smoke.initialization must be random")
        _require(data["learning"]["horizon"] == [20, 100], "smoke.horizon must be [20, 100]")
        _require(data["learning"]["policy"] == "prefix_rebased_zero", "smoke.policy must be prefix_rebased_zero")

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

    for group in DISABLED_FEATURE_GROUPS:
        state = data["disabled"][group]
        _require(state == "disabled", f"smoke.disabled.{group} must be 'disabled' in the foundation/v1 smoke")

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

    data: Dict[str, Any]
    source: str

    @property
    def kind(self) -> str:
        return self.data["kind"]

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def effective_model_profile(self) -> Dict[str, Any]:
        """Persisted effective model constructor settings (no implicit defaults)."""
        return dict(self.data["model"])

    def disabled_features(self) -> Dict[str, str]:
        return dict(self.data["disabled"])

    def locks(self) -> Dict[str, str]:
        return dict(self.data["locks"])


def load_config(path: Union[str, "PathLike"]) -> FoundationConfig:
    """Load and strictly validate a foundation config file.

    Rejects unknown keys at every level, mismatched profile locks, incomplete
    identities and undeclared disabled features. Returns a FoundationConfig
    carrying every effective setting.
    """
    import os

    p = os.fspath(path) if hasattr(path, "__fspath__") else str(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: config must be a YAML mapping")

    _expect(("map", _TOP_SCHEMA), "config", raw)
    _validate_semantics(raw, p)
    return FoundationConfig(raw, p)
