"""Foundation/v1 data model: strict semantic validation beyond JSON shape.

The JSON schema in ``contracts/artifacts.schema.json`` enforces shape only.
This module implements the semantic rules from ``data-model.md`` that the
schema cannot express: digest spelling, exact integer parsing against the
record's numerical profile, path containment, enum/nullability relationships,
and acyclic reference rules.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Union

from jsonschema import Draft202012Validator

from oeis_learn.experiments.profiles import I256_MAX, I256_MIN

SCHEMA_VERSION = "foundation/v1"

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
INTEGER_TEXT_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
BLOB_PATH_RE = re.compile(r"^[A-Za-z0-9_-]+[.]pt$")

VALID_KINDS = ("visible_prompt", "candidate_result", "checkpoint_manifest")


class FoundationValidationError(ValueError):
    """Semantic validation failure for a foundation/v1 artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FoundationValidationError(message)


def parse_integer_text(value: str, lo: int, hi: int, context: str) -> int:
    """Exact integer parsing that enforces the record's numerical profile.

    Floats, scientific notation, booleans, ``-0``, leading-zero spellings and
    values outside ``[lo, hi]`` are rejected; ``value`` is returned as a
    Python int only when it is a canonical integer spelling within range.
    """
    if not isinstance(value, str) or not INTEGER_TEXT_RE.fullmatch(value):
        raise FoundationValidationError(
            f"{context}: expected canonical integer text (0 or -?[1-9][0-9]*), got {value!r}"
        )
    parsed = int(value, 10)
    if not (lo <= parsed <= hi):
        raise FoundationValidationError(f"{context}: integer {value} outside [{lo}, {hi}]")
    return parsed


def check_digest(value: Any, context: str, *, nullable: bool = False) -> Optional[str]:
    if value is None:
        if nullable:
            return None
        raise FoundationValidationError(f"{context}: digest must not be null")
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise FoundationValidationError(
            f"{context}: expected sha256: + 64 lowercase hex, got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    state: str
    value: Optional[int]
    unit: str
    source: Optional[str]
    reason: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any], context: str) -> "Metric":
        return cls(
            state=d["state"],
            value=d["value"],
            unit=d["unit"],
            source=d["source"],
            reason=d["reason"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "reason": self.reason,
        }


def _check_metric(metric: Dict[str, Any], context: str) -> None:
    state = metric["state"]
    _require(state in {"measured", "disabled", "unavailable"}, f"{context}.state invalid")
    value = metric["value"]
    if state == "measured":
        _require(isinstance(value, int) and value >= 0, f"{context}.value must be a nonnegative integer when measured")
        _require(isinstance(metric["source"], str) and metric["source"], f"{context}.source required when measured")
        _require(metric["reason"] is None, f"{context}.reason must be null when measured")
    else:
        _require(value is None, f"{context}.value must be null when {state}")
        _require(isinstance(metric["reason"], str) and metric["reason"], f"{context}.reason required when {state}")


# ---------------------------------------------------------------------------
# Artifact kinds
# ---------------------------------------------------------------------------


@dataclass
class VisiblePrompt:
    kind: str
    schema_version: str
    request_nonce: str
    language_profile: str
    observed_terms: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisiblePrompt":
        return cls(
            kind=d["kind"],
            schema_version=d["schema_version"],
            request_nonce=d["request_nonce"],
            language_profile=d["language_profile"],
            observed_terms=d["observed_terms"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "request_nonce": self.request_nonce,
            "language_profile": self.language_profile,
            "observed_terms": self.observed_terms,
        }


@dataclass
class CandidateResult:
    kind: str
    schema_version: str
    purpose: str
    run_id: str
    attempt_index: int
    stage: str
    language_profile: str
    resource_profile: str
    source_sha256: str
    checkpoint_sha256: Optional[str]
    prompt_sha256: str
    outcome: str
    reason: Optional[str]
    outputs: List[str]
    verified_terms: int
    first_failure_index: Optional[int]
    selected: bool
    reference_evidence_sha256: Optional[str]
    expected_terms_sha256: Optional[str]
    usage: Dict[str, Metric]
    proof_status: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateResult":
        return cls(
            kind=d["kind"],
            schema_version=d["schema_version"],
            purpose=d["purpose"],
            run_id=d["run_id"],
            attempt_index=d["attempt_index"],
            stage=d["stage"],
            language_profile=d["language_profile"],
            resource_profile=d["resource_profile"],
            source_sha256=d["source_sha256"],
            checkpoint_sha256=d["checkpoint_sha256"],
            prompt_sha256=d["prompt_sha256"],
            outcome=d["outcome"],
            reason=d["reason"],
            outputs=d["outputs"],
            verified_terms=d["verified_terms"],
            first_failure_index=d["first_failure_index"],
            selected=d["selected"],
            reference_evidence_sha256=d["reference_evidence_sha256"],
            expected_terms_sha256=d["expected_terms_sha256"],
            usage={k: Metric.from_dict(v, k) for k, v in d["usage"].items()},
            proof_status=d["proof_status"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "run_id": self.run_id,
            "attempt_index": self.attempt_index,
            "stage": self.stage,
            "language_profile": self.language_profile,
            "resource_profile": self.resource_profile,
            "source_sha256": self.source_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_sha256": self.prompt_sha256,
            "outcome": self.outcome,
            "reason": self.reason,
            "outputs": self.outputs,
            "verified_terms": self.verified_terms,
            "first_failure_index": self.first_failure_index,
            "selected": self.selected,
            "reference_evidence_sha256": self.reference_evidence_sha256,
            "expected_terms_sha256": self.expected_terms_sha256,
            "usage": {k: v.to_dict() for k, v in self.usage.items()},
            "proof_status": self.proof_status,
        }


@dataclass
class CheckpointManifest:
    kind: str
    schema_version: str
    run_id: str
    generation: int
    completed_update: int
    blob_path: str
    blob_sha256: str
    contract_sha256: str
    pool_sha256: str
    codec_sha256: str
    runtime_sha256: str
    effective_config_sha256: str
    ledger_sequence: int
    charged_budget_ns: int
    next_sample_ids: List[str]
    payload_state_keys: List[str]
    checkpoint_state: str
    previous_manifest_sha256: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointManifest":
        return cls(
            kind=d["kind"],
            schema_version=d["schema_version"],
            run_id=d["run_id"],
            generation=d["generation"],
            completed_update=d["completed_update"],
            blob_path=d["blob_path"],
            blob_sha256=d["blob_sha256"],
            contract_sha256=d["contract_sha256"],
            pool_sha256=d["pool_sha256"],
            codec_sha256=d["codec_sha256"],
            runtime_sha256=d["runtime_sha256"],
            effective_config_sha256=d["effective_config_sha256"],
            ledger_sequence=d["ledger_sequence"],
            charged_budget_ns=d["charged_budget_ns"],
            next_sample_ids=d["next_sample_ids"],
            payload_state_keys=d["payload_state_keys"],
            checkpoint_state=d["checkpoint_state"],
            previous_manifest_sha256=d["previous_manifest_sha256"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generation": self.generation,
            "completed_update": self.completed_update,
            "blob_path": self.blob_path,
            "blob_sha256": self.blob_sha256,
            "contract_sha256": self.contract_sha256,
            "pool_sha256": self.pool_sha256,
            "codec_sha256": self.codec_sha256,
            "runtime_sha256": self.runtime_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            "ledger_sequence": self.ledger_sequence,
            "charged_budget_ns": self.charged_budget_ns,
            "next_sample_ids": self.next_sample_ids,
            "payload_state_keys": self.payload_state_keys,
            "checkpoint_state": self.checkpoint_state,
            "previous_manifest_sha256": self.previous_manifest_sha256,
        }


def _check_blob_path(path: str) -> None:
    """Safe relative artifact path: no '..', no absolute path, no symlink escape."""
    if not isinstance(path, str) or not BLOB_PATH_RE.fullmatch(path):
        raise FoundationValidationError(f"blob_path: expected safe relative *.pt filename, got {path!r}")
    p = PurePosixPath(path)
    _require(not p.is_absolute(), f"blob_path {path!r} must be relative")
    _require(".." not in p.parts, f"blob_path {path!r} must not contain '..'")


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


def validate_artifact(data: Dict[str, Any]) -> None:
    """Validate a foundation/v1 artifact (JSON shape + semantic rules)."""
    kind = data.get("kind")
    _require(kind in VALID_KINDS, f"unknown artifact kind {kind!r}")
    _require(data.get("schema_version") == SCHEMA_VERSION, "schema_version must be foundation/v1")

    if kind == "visible_prompt":
        _validate_visible_prompt(data)
    elif kind == "candidate_result":
        _validate_candidate_result(data)
    else:  # checkpoint_manifest
        _validate_checkpoint_manifest(data)


def _validate_visible_prompt(data: Dict[str, Any]) -> None:
    check_digest(data["language_profile"], "language_profile")
    _require(len(data["observed_terms"]) == 20, "visible_prompt.observed_terms must have exactly 20 values")
    for i, term in enumerate(data["observed_terms"]):
        parse_integer_text(term, I256_MIN, I256_MAX, f"observed_terms[{i}]")


def _validate_candidate_result(data: Dict[str, Any]) -> None:
    check_digest(data["language_profile"], "language_profile")
    check_digest(data["resource_profile"], "resource_profile")
    check_digest(data["source_sha256"], "source_sha256")
    check_digest(data["checkpoint_sha256"], "checkpoint_sha256", nullable=True)
    check_digest(data["prompt_sha256"], "prompt_sha256")
    check_digest(data["reference_evidence_sha256"], "reference_evidence_sha256", nullable=True)
    check_digest(data["expected_terms_sha256"], "expected_terms_sha256", nullable=True)

    _require(data["proof_status"] == "not_claimed", "proof_status must be exactly 'not_claimed'")
    for i, out in enumerate(data["outputs"]):
        parse_integer_text(out, I256_MIN, I256_MAX, f"outputs[{i}]")

    for key in ("fuel", "reference_steps", "elapsed_ns", "peak_rss_bytes"):
        _check_metric(data["usage"][key], f"usage.{key}")

    outcome = data["outcome"]
    if outcome == "prefix_match":
        _require(len(data["outputs"]) == 20, "prefix_match requires exactly 20 outputs")
        _require(data["verified_terms"] == 20, "prefix_match requires verified_terms == 20")
        _require(data["stage"] == "prefix", "prefix_match requires stage == 'prefix'")
        _require(data["reason"] is None, "prefix_match requires reason null")
        _require(data["first_failure_index"] is None, "prefix_match requires first_failure_index null")
        check_digest(data["reference_evidence_sha256"], "reference_evidence_sha256")
        check_digest(data["expected_terms_sha256"], "expected_terms_sha256")
    elif outcome == "full_horizon_match":
        _require(len(data["outputs"]) == 100, "full_horizon_match requires exactly 100 outputs")
        _require(data["verified_terms"] == 100, "full_horizon_match requires verified_terms == 100")
        _require(data["stage"] == "full", "full_horizon_match requires stage == 'full'")
        _require(data["reason"] is None, "full_horizon_match requires reason null")
        _require(data["first_failure_index"] is None, "full_horizon_match requires first_failure_index null")
        check_digest(data["reference_evidence_sha256"], "reference_evidence_sha256")
        check_digest(data["expected_terms_sha256"], "expected_terms_sha256")
    else:
        _require(isinstance(data["reason"], str) and data["reason"], f"{outcome} requires a non-empty reason")

    if data["purpose"] == "model":
        check_digest(data["checkpoint_sha256"], "checkpoint_sha256", nullable=False)
    else:
        check_digest(data["checkpoint_sha256"], "checkpoint_sha256", nullable=True)


def _validate_checkpoint_manifest(data: Dict[str, Any]) -> None:
    _check_blob_path(data["blob_path"])
    check_digest(data["blob_sha256"], "blob_sha256")
    check_digest(data["contract_sha256"], "contract_sha256")
    check_digest(data["pool_sha256"], "pool_sha256")
    check_digest(data["codec_sha256"], "codec_sha256")
    check_digest(data["runtime_sha256"], "runtime_sha256")
    check_digest(data["effective_config_sha256"], "effective_config_sha256")
    check_digest(data["previous_manifest_sha256"], "previous_manifest_sha256", nullable=True)
    for sid in data["next_sample_ids"]:
        check_digest(sid, "next_sample_ids[]")
    _require(data["checkpoint_state"] == "complete", "checkpoint_state must be 'complete'")
    _require(
        data["payload_state_keys"] == [
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "python_rng",
            "numpy_rng",
            "torch_cpu_rng",
            "torch_gpu_rng",
            "named_rng",
            "data_order",
            "counters",
        ],
        "payload_state_keys must be the exact ordered eleven-key set",
    )


# ---------------------------------------------------------------------------
# JSON shape validation against the 007 schema
# ---------------------------------------------------------------------------


def build_schema_validator(schema: Dict[str, Any]) -> Draft202012Validator:
    """Return a validator for the supplied artifacts schema."""
    return Draft202012Validator(schema)


def validate_shape(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate JSON shape only (unknown fields, required fields, enums)."""
    errors = list(build_schema_validator(schema).iter_errors(data))
    if errors:
        raise FoundationValidationError(
            "schema validation failed: " + "; ".join(e.message for e in errors)
        )
