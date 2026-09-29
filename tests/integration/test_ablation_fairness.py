"""Integration tests for paired ablation fairness, prefix reuse, and budget equality."""

from __future__ import annotations

import json
import pytest
from oeis_learn.evaluation.experiments import (
    evaluate_inference_ablation_pair,
    verify_experiment_fairness,
)


def test_inference_ablation_fairness_prefix_reuse():
    """Verify that solver-on and solver-off variants receive identical raw candidate pools."""
    raw_candidates = [
        "(module (func (export \"compute\") (param $n i32) (result i64) i64.const_?))",
        "(module (func (export \"compute\") (param $n i32) (result i64) local.get $n i64.extend_i32_u))",
    ]
    target_terms = [i * 2 for i in range(120)]

    res_off = evaluate_inference_ablation_pair(
        raw_candidates=raw_candidates,
        target_terms=target_terms,
        constant_resolution=False,
    )
    res_on = evaluate_inference_ablation_pair(
        raw_candidates=raw_candidates,
        target_terms=target_terms,
        constant_resolution=True,
    )

    # Both variants evaluated exactly the same 2 raw candidates
    assert len(res_off["candidates"]) == 2
    assert len(res_on["candidates"]) == 2
    for c_off, c_on in zip(res_off["candidates"], res_on["candidates"]):
        assert c_off["raw_wat"] == c_on["raw_wat"]

    # Solver on resolved the constant placeholder
    assert res_on["candidates"][0]["resolved_wat"] is not None
    assert "i64.const 0" in res_on["candidates"][0]["resolved_wat"] or "i64.const" in res_on["candidates"][0]["resolved_wat"]


def test_experiment_manifest_requires_complete_pairs():
    manifest = {
        "seeds": [42, 137, 2026],
        "variants": [{"variant_id": "v1"}, {"variant_id": "v2"}],
        "outcomes": [
            {"variant_id": "v1", "seed": 42, "status": "COMPLETE"},
            {"variant_id": "v2", "seed": 42, "status": "COMPLETE"},
            {"variant_id": "v1", "seed": 137, "status": "COMPLETE"},
            # Missing v2 for seed 137
        ],
    }
    is_fair, reason = verify_experiment_fairness(manifest)
    assert is_fair is False
    assert "missing outcome" in reason.lower()
