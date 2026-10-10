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


@dataclass(frozen=True)
class QualifiedServices:
    """Explicit capability dispatch after G7 and profile validation."""

    contract_sha256: str
    g7_sha256: str
    _profiles_json: bytes

    @property
    def capabilities(self):
        from oeis_learn.experiments.artifacts import load_json

        return tuple(sorted(load_json(self._profiles_json)))

    def _settings(self, name):
        from oeis_learn.experiments.artifacts import load_json

        profiles = load_json(self._profiles_json)
        if name not in profiles:
            raise ConfigError(f"{name}: capability disabled/pending")
        return profiles[name]["settings"]

    def ground(self, source, visible_terms):
        from oeis_learn.decoder.grounding import ground

        settings = self._settings("solver")
        return ground(
            source,
            visible_terms,
            bound=settings["coefficient_bound"],
            timeout_ms=settings["timeout_ms"],
        )

    def optimize(self, source):
        from oeis_learn.sandbox.optimizer import optimize_wat_program

        return optimize_wat_program(
            source,
            hard_waste_threshold=self._settings("optimizer")["waste_threshold_percent"] / 100,
        )

    def prover(self):
        from oeis_learn.discovery.symbolic_prover import SymbolicProver

        return SymbolicProver(timeout_ms=self._settings("formula_identity")["timeout_ms"])

    def runner(self, result_dir):
        from oeis_learn.sandbox.runner import FoundationRunner

        settings = self._settings("native")
        return FoundationRunner(
            result_dir,
            backend="rust",
            workers=settings["workers"],
            cache_bytes=settings["cache_bytes"],
        )


def activate_qualified_contract(contract, *, g7_report, evidence_root, resolve):
    """Resolve content-addressed inputs before activating any repaired service."""
    from oeis_learn.experiments.qualified_models import validate_record
    from oeis_learn.experiments.artifacts import compute_canonical_digest
    from oeis_learn.evaluation.repair_gate import require_g7

    validate_record(contract)
    if contract["kind"] != "qualified_contract":
        raise ConfigError("qualified contract required")
    if any(
        contract["modes"][m]["enabled"]
        for m in ("learning", "replay", "search", "program_proof", "extended_runtime")
    ):
        raise ConfigError("G8/G9 capabilities are pending their own qualification")

    def resolve_checked(digest):
        value = resolve(digest)
        # Content-addressed records omit their own identity when hashed.
        excluded = (
            "identity_sha256" if isinstance(value, dict) and "identity_sha256" in value else None
        )
        if compute_canonical_digest(value, excluded) != digest:
            raise ConfigError("referenced content identity mismatch")
        return value

    base = resolve_checked(contract["base_contract_sha256"])
    if base.get("kind") == "smoke":
        _expect(("map", _TOP_SCHEMA), "qualified.base", base)
        _validate_semantics(base, "qualified.base")
    else:
        required = {
            "schema_version",
            "kind",
            "purpose",
            "run_id",
            "profiles",
            "track",
            "initialization",
            "objective",
            "constitution_version",
            "observed_terms",
            "total_terms",
            "index_policy",
            "enabled_subsystems",
            "run_provenance",
            "contract_id",
        }
        _require(set(base) == required, "complete foundation base contract required")
        _require(
            base["contract_id"] == compute_canonical_digest(base, "contract_id"),
            "base contract identity mismatch",
        )
        _require(
            base["schema_version"] == "foundation/v1"
            and base["kind"] == "experiment_contract"
            and base["track"] == "strict_generic"
            and base["initialization"] == "random"
            and base["objective"] == "sft"
            and base["constitution_version"] == "2.0.0",
            "incompatible base contract",
        )
        _require(
            base["profiles"] == profile_digests()
            and base["observed_terms"] == 20
            and base["total_terms"] == 100
            and base["index_policy"] == "prefix_rebased_zero"
            and base["enabled_subsystems"] == ["sft"],
            "incompatible base profiles/horizon",
        )
    for item in contract["parent_chain"]:
        parent = resolve_checked(item["contract_sha256"])
        validate_record(parent)
        if (
            parent["track"] != "strict_generic"
            or parent["initialization"] != item["initialization"]
        ):
            raise ConfigError("parent provenance mismatch")
        expected_parent = (
            parent["parent_chain"][0]["contract_sha256"] if parent["parent_chain"] else None
        )
        if expected_parent != item["parent_sha256"]:
            raise ConfigError("parent chain identity mismatch")
    for digest in contract["profiles"].values():
        resolve_checked(digest)
    gate = require_g7(g7_report, evidence_root=evidence_root)
    if contract["gates"]["G7"] != gate:
        raise ConfigError("qualified G7 identity mismatch")
    profile_schemas = {
        "solver": {
            "coefficient_bound": ("int",),
            "timeout_ms": ("int",),
            "max_parameters": ("int",),
        },
        "optimizer": {"waste_threshold_percent": ("int",)},
        "native": {"workers": ("int",), "cache_bytes": ("int",)},
        "formula_identity": {"timeout_ms": ("int",)},
    }
    active = {}
    for name in profile_schemas:
        mode = contract["modes"][name]
        if not mode["enabled"]:
            continue
        profile = resolve_checked(mode["profile_sha256"])
        _expect(
            (
                "map",
                {
                    "version": ("enum", ("qualified-service/v1",)),
                    "service": ("enum", (name,)),
                    "settings": ("map", profile_schemas[name]),
                },
            ),
            name,
            profile,
        )
        settings = profile["settings"]
        if name == "solver":
            _require(
                0 <= settings["coefficient_bound"] <= 1000
                and 1 <= settings["timeout_ms"] <= 2000
                and settings["max_parameters"] == 8,
                "solver limits mismatch",
            )
        elif name == "optimizer":
            _require(
                0 <= settings["waste_threshold_percent"] <= 100, "optimizer threshold mismatch"
            )
        elif name == "native":
            _require(
                1 <= settings["workers"] <= 8 and 0 <= settings["cache_bytes"] <= 1 << 30,
                "native limits mismatch",
            )
        else:
            _require(1 <= settings["timeout_ms"] <= 2000, "formula worker limits mismatch")
        active[name] = deepcopy(profile)
    from oeis_learn.experiments.artifacts import canonical_bytes

    return QualifiedServices(contract["identity_sha256"], gate, canonical_bytes(active))


def load_qualified_config(path, **activation):
    with open(path, encoding="utf-8") as handle:
        contract = yaml.load(handle, Loader=_UniqueLoader)
    return activate_qualified_contract(contract, **activation)
