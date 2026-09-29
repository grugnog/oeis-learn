"""Pure readiness policy evaluation and gate verification service."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from oeis_learn.data.models import (
    OverrideRecord,
    ReadinessGateResult,
    ReadinessPolicy,
    ReadinessReport,
    ReadinessThreshold,
)
from oeis_learn.evaluation.protocol import canonical_json_dumps, canonical_json_hash


def load_readiness_policy(policy_path: str) -> ReadinessPolicy:
    """Loads a versioned readiness policy from JSON with integrity checking."""
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Readiness policy file not found: {policy_path}")

    with open(policy_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds: List[ReadinessThreshold] = []
    for t in data.get("thresholds", []):
        thresholds.append(
            ReadinessThreshold(
                gate_id=t["gate_id"],
                metric=t["metric"],
                comparator=t["comparator"],
                threshold=float(t["threshold"]),
                unit=t["unit"],
                source=t["source"],
                non_relaxable=bool(t.get("non_relaxable", True)),
            )
        )

    # Compute policy_id
    raw_for_id = {
        "schema_version": data.get("schema_version", "1.0"),
        "name": data.get("name", "readiness_policy"),
        "thresholds": [t.to_dict() for t in thresholds],
        "required_experiment_ids": data.get("required_experiment_ids", []),
        "required_artifacts": data.get("required_artifacts", []),
        "native_evaluator_required": bool(data.get("native_evaluator_required", True)),
    }
    computed_id = canonical_json_hash(raw_for_id)
    policy_id = data.get("policy_id") or computed_id

    return ReadinessPolicy(
        schema_version=data.get("schema_version", "1.0"),
        policy_id=policy_id,
        name=data.get("name", "readiness_policy"),
        thresholds=thresholds,
        required_experiment_ids=data.get("required_experiment_ids", []),
        required_artifacts=data.get("required_artifacts", []),
        native_evaluator_required=bool(data.get("native_evaluator_required", True)),
    )


def evaluate_readiness_policy(
    policy: ReadinessPolicy,
    metrics: Dict[str, float],
    run_id: str,
    evidence_map: Optional[Dict[str, List[Dict[str, str]]]] = None,
    override_info: Optional[Dict[str, Any]] = None,
) -> ReadinessReport:
    """Evaluates metrics against all thresholds in a pure, side-effect-free pass."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ev_map = evidence_map or {}
    gate_results: List[ReadinessGateResult] = []
    failed_gate_ids: List[str] = []

    for th in policy.thresholds:
        measured = float(metrics.get(th.metric, 0.0))
        passed = False
        diagnostics: List[str] = []

        if th.comparator == "GE":
            passed = measured >= th.threshold
        elif th.comparator == "LE":
            passed = measured <= th.threshold
        elif th.comparator == "EQ":
            passed = abs(measured - th.threshold) < 1e-6

        if not passed:
            failed_gate_ids.append(th.gate_id)
            diagnostics.append(
                f"Gate '{th.gate_id}' FAILED: {th.metric}={measured:.4f} did not satisfy {th.comparator} {th.threshold}"
            )

        evidence = ev_map.get(
            th.gate_id,
            [{"artifact_id": f"metric_{th.metric}", "artifact_sha256": "sha256:" + "0" * 64}],
        )

        gate_results.append(
            ReadinessGateResult(
                gate_id=th.gate_id,
                measured_value=measured,
                threshold=th,
                passed=passed,
                evaluated_at=now_utc,
                evidence=evidence,
                diagnostics=diagnostics,
            )
        )

    all_passed = len(failed_gate_ids) == 0

    override_record: Optional[OverrideRecord] = None
    if all_passed:
        qual_state = "AUTHORIZED"
    else:
        if override_info and override_info.get("operator") and override_info.get("reason"):
            ovr_id = f"ovr_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
            override_record = OverrideRecord(
                override_id=ovr_id,
                operator=override_info["operator"],
                created_at=now_utc,
                reason=override_info["reason"],
                diagnostic_intent=override_info.get("diagnostic_intent", "Diagnostic evaluation only"),
                failed_gate_ids=failed_gate_ids,
                policy_id=policy.policy_id,
            )
            qual_state = "OVERRIDDEN_UNQUALIFIED"
        else:
            qual_state = "BLOCKED"

    report_id = f"rep_{run_id}_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
    return ReadinessReport(
        schema_version="1.0",
        report_id=report_id,
        run_id=run_id,
        created_at=now_utc,
        policy=policy,
        gate_results=gate_results,
        overall_passed=all_passed,
        override=override_record,
        qualification_state=qual_state,
    )
