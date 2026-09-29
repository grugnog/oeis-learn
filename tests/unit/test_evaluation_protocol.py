"""Unit tests for immutable evaluation protocol normalization, hashing, and candidate seed derivation."""

from __future__ import annotations

import json
import pytest
from oeis_learn.evaluation.protocol import (
    EvaluationProtocol,
    canonical_json_dumps,
    canonical_json_hash,
    derive_candidate_seed,
)


def sample_protocol_dict() -> dict:
    return {
        "schema_version": "1.0",
        "checkpoint_sha256": "sha256:" + "a" * 64,
        "benchmark_manifest_sha256": "sha256:" + "b" * 64,
        "observed_horizon": 20,
        "unseen_horizon": 100,
        "candidate_budget": 8,
        "base_seed": 42,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 128,
        "constant_resolution": True,
        "solver_timeout_ms": 250,
        "max_placeholders": 4,
        "fuel_per_invocation": 10000,
        "memory_limit_mib": 16,
        "mdl_ratio_max": 1.2,
        "native_evaluator_required": True,
        "code_revision": "test_rev",
        "environment_fingerprint": "sha256:" + "c" * 64,
    }


def test_canonical_json_serialization_order():
    d1 = {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}}
    d2 = {"a": 1, "nested": {"y": 20, "z": 10}, "b": 2}
    assert canonical_json_dumps(d1) == canonical_json_dumps(d2)
    assert canonical_json_dumps(d1) == '{"a":1,"b":2,"nested":{"y":20,"z":10}}'


def test_canonical_json_hash():
    d = {"a": 1, "b": 2}
    h = canonical_json_hash(d)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_protocol_id_computation_and_immutability():
    data = sample_protocol_dict()
    proto = EvaluationProtocol.from_dict(data)
    assert proto.protocol_id.startswith("sha256:")
    # Protocol must be frozen/immutable
    with pytest.raises(Exception):
        proto.candidate_budget = 16  # type: ignore


def test_protocol_version_rejection():
    data = sample_protocol_dict()
    data["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        EvaluationProtocol.from_dict(data)


def test_protocol_horizon_enforcement():
    data = sample_protocol_dict()
    data["observed_horizon"] = 10
    with pytest.raises(ValueError, match="observed_horizon must be 20"):
        EvaluationProtocol.from_dict(data)

    data = sample_protocol_dict()
    data["unseen_horizon"] = 50
    with pytest.raises(ValueError, match="unseen_horizon must be 100"):
        EvaluationProtocol.from_dict(data)


def test_protocol_budget_enum_enforcement():
    data = sample_protocol_dict()
    data["candidate_budget"] = 4
    with pytest.raises(ValueError, match="candidate_budget must be one of"):
        EvaluationProtocol.from_dict(data)


def test_candidate_seed_derivation_determinism():
    base_seed = 42
    protocol_id = "sha256:" + "0" * 64
    seq_id = "A000045"
    s1 = derive_candidate_seed(base_seed, protocol_id, seq_id, 0)
    s2 = derive_candidate_seed(base_seed, protocol_id, seq_id, 0)
    assert s1 == s2
    assert isinstance(s1, int)
    # Check within 64-bit signed integer range
    assert -9223372036854775808 <= s1 <= 9223372036854775807


def test_candidate_seed_stability_across_budgets():
    base_seed = 12345
    protocol_id = "sha256:" + "f" * 64
    seq_id = "A000290"

    seeds_b1 = [derive_candidate_seed(base_seed, protocol_id, seq_id, i) for i in range(1)]
    seeds_b8 = [derive_candidate_seed(base_seed, protocol_id, seq_id, i) for i in range(8)]
    seeds_b16 = [derive_candidate_seed(base_seed, protocol_id, seq_id, i) for i in range(16)]

    assert seeds_b1[0] == seeds_b8[0] == seeds_b16[0]
    assert seeds_b8 == seeds_b16[:8]
    # Seeds for different indices must be different
    assert len(set(seeds_b16)) == 16
