"""Integration tests for automated canary preflight qualification across the 6 landmark sequences."""

from __future__ import annotations

import json
from pathlib import Path
from jsonschema import Draft7Validator
from oeis_learn.cli.evaluate_canaries import run_canary_evaluation


def test_reference_kernel_diagnostics_all_six_pass():
    schema_path = (
        Path(__file__).resolve().parent.parent.parent
        / "specs"
        / "007-experiment-foundation"
        / "contracts"
        / "reference-kernel-diagnostic.schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft7Validator(schema)

    report = run_canary_evaluation(
        checkpoint_path="runs/011_multilimb_256bit_production/checkpoints/sft_bridge_ready.pt",
        result_profile="i256x4_v1",
        fuel_budget=100000,
    )

    validator.validate(report)

    assert report["all_canaries_passed"] is True
    assert report["qualified"] is False and report["checkpoint_evaluated"] is None
    assert len(report["canary_results"]) == 6

    for r in report["canary_results"]:
        assert r["verdict"] == "EXTRAPOLATING_SUCCESS"
        assert r["observed_matched"] is True
        assert r["unseen_matched"] is True
        assert r["overflow_prevented"] is (True if r["scalar_64bit_overflow_term"] < 120 else None)
        assert r["fuel_consumed"] <= report["fuel_budget"]


def test_reference_kernel_fuel_failure_does_not_claim_overflow_evidence():
    report = run_canary_evaluation(fuel_budget=1)
    assert not report["all_canaries_passed"] and not report["qualified"]
    for result in report["canary_results"]:
        assert result["verdict"] == "FUEL_TRAP"
        assert result["execution_status"] == "OUT_OF_FUEL"
        assert result["overflow_prevented"] is None
