"""Immutable evaluation protocol definition, canonical hashing, and candidate seed derivation."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any, Dict


def canonical_json_dumps(obj: Any) -> str:
    """Serializes a Python object to canonical JSON string with sorted keys and no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonical_json_hash(obj: Any) -> str:
    """Computes SHA-256 digest over canonical JSON representation, prefixed with 'sha256:'."""
    raw = canonical_json_dumps(obj).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def derive_candidate_seed(
    base_seed: int,
    protocol_id: str,
    sequence_id: str,
    candidate_index: int,
) -> int:
    """Derives a deterministic signed 64-bit seed for a specific candidate rollout."""
    payload = f"{base_seed}:{protocol_id}:{sequence_id}:{candidate_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Interpret first 8 bytes as signed 64-bit integer
    (val,) = struct.unpack(">q", digest[:8])
    return val


@dataclass(frozen=True)
class EvaluationProtocol:
    """Immutable protocol governing a reproducible synthesis evaluation."""

    schema_version: str
    protocol_id: str
    checkpoint_sha256: str
    benchmark_manifest_sha256: str
    observed_horizon: int
    unseen_horizon: int
    candidate_budget: int
    base_seed: int
    temperature: float
    top_p: float
    max_tokens: int
    constant_resolution: bool
    solver_timeout_ms: int
    max_placeholders: int
    fuel_per_invocation: int
    memory_limit_mib: int
    mdl_ratio_max: float
    native_evaluator_required: bool
    code_revision: str
    environment_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if self.observed_horizon != 20:
            raise ValueError(f"observed_horizon must be 20, got {self.observed_horizon}")
        if self.unseen_horizon != 100:
            raise ValueError(f"unseen_horizon must be 100, got {self.unseen_horizon}")
        if self.candidate_budget not in (1, 8, 16):
            raise ValueError(
                f"candidate_budget must be one of [1, 8, 16], got {self.candidate_budget}"
            )
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.mdl_ratio_max > 1.2:
            raise ValueError(f"mdl_ratio_max must be <= 1.2, got {self.mdl_ratio_max}")
        if not self.native_evaluator_required:
            raise ValueError("native_evaluator_required must be true for qualified evaluation")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationProtocol:
        clean = dict(data)
        # Compute protocol_id if not provided or to ensure canonical hash match
        fields_for_id = {k: v for k, v in clean.items() if k != "protocol_id"}
        computed_id = canonical_json_hash(fields_for_id)
        clean["protocol_id"] = clean.get("protocol_id", computed_id) or computed_id
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
