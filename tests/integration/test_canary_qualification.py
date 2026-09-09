"""Integration tests for automated canary preflight qualification across the 6 landmark sequences."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from jsonschema import Draft7Validator
from oeis_learn.cli.evaluate_canaries import run_canary_evaluation


def test_canary_qualification_all_six_pass():
    schema_path = (
        Path(__file__).resolve().parent.parent.parent
        / "specs"
        / "006-multilimb-curriculum-scaling"
        / "contracts"
        / "canary-benchmark.schema.json"
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft7Validator(schema)

    report = run_canary_evaluation(
        checkpoint_path="runs/011_multilimb_256bit_production/checkpoints/sft_bridge_ready.pt",
        result_profile="i256x4_v1",
        fuel_budget=20000,
    )

    validator.validate(report)

    assert report["all_canaries_passed"] is True
    assert len(report["canary_results"]) == 6

    for r in report["canary_results"]:
        assert r["verdict"] == "EXTRAPOLATING_SUCCESS"
        assert r["observed_matched"] is True
        assert r["unseen_matched"] is True
        assert r["overflow_prevented"] is True
        assert r["fuel_consumed"] <= 20000
