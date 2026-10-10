"""Pure readiness policy evaluation and gate verification service."""

from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any, Dict, List, Optional
from oeis_learn.data.models import (
    OverrideRecord,
    ReadinessGateResult,
    ReadinessPolicy,
    ReadinessReport,
    ReadinessThreshold,
)
from oeis_learn.evaluation.protocol import canonical_json_hash


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
        raw = metrics.get(th.metric)
        try:
            measured = float(raw) if raw is not None else None
        except (TypeError, ValueError, OverflowError):
            measured = None
        if measured is not None and not math.isfinite(measured):
            measured = None
        passed = False
        diagnostics: List[str] = []
        if measured is None:
            diagnostics.append(f"Unavailable finite observation: {th.metric}")
        elif th.comparator == "GE":
            passed = measured >= th.threshold
        elif th.comparator == "LE":
            passed = measured <= th.threshold
        elif th.comparator == "EQ":
            passed = abs(measured - th.threshold) < 1e-6

        if not passed:
            failed_gate_ids.append(th.gate_id)
            diagnostics.append(
                f"Gate '{th.gate_id}' FAILED: {th.metric}={measured} did not satisfy {th.comparator} {th.threshold}"
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
        qual_state = (
            "BLOCKED"  # Threshold-only legacy diagnostics cannot authorize model synthesis.
        )
    else:
        if override_info and override_info.get("operator") and override_info.get("reason"):
            ovr_id = f"ovr_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
            override_record = OverrideRecord(
                override_id=ovr_id,
                operator=override_info["operator"],
                created_at=now_utc,
                reason=override_info["reason"],
                diagnostic_intent=override_info.get(
                    "diagnostic_intent", "Diagnostic evaluation only"
                ),
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


def foundation_readiness(checkpoint, *, software_complete):
    """Software evidence cannot stand in for the later hardware/run gates."""
    from oeis_learn.experiments.profiles import profile_digests

    if (
        checkpoint.contract["profiles"] != profile_digests()
        or checkpoint.contract["constitution_version"] != "2.0.0"
        or not checkpoint.manifest["blob_sha256"]
    ):
        raise ValueError("genuine checkpoint and adopted profile evidence required")
    return {
        "qualified": False,
        "qualification": "diagnostic"
        if checkpoint.contract["purpose"] == "diagnostic"
        else "pending_run_gates",
        "software_complete": bool(software_complete),
        "checkpoint_sha256": checkpoint.manifest["blob_sha256"],
        "missing_gates": [
            "G0: actual-device feasibility and adopted-contract provenance for this run",
            "G1: bound exact-runtime conformance evidence",
            "G2: bound evaluation/isolation evidence",
            "G3: bound generic-admission evidence",
            "G4: bound complete-resume evidence",
            "G5: real HIP updates and runtime containment",
            "G6: bound finite-result/proof-boundary evidence",
        ],
        "full_007_additional_gates": ["G7", "G8", "G9", "G10"],
        "lifecycle_is_qualification": False,
    }


def require_architecture_measurement_gates(g7_report, *, evidence_root=None):
    """Architecture/learning measurements require G7 independently of smoke."""
    from oeis_learn.evaluation.repair_gate import require_g7

    require_g7(g7_report, evidence_root=evidence_root)


def require_learning_readiness(report, *, evidence_root, diagnostic=False):
    """G8 CPU diagnostics cannot authorize hardware-qualified learning."""
    from oeis_learn.evaluation.learning_gate import require_g8

    return require_g8(report, evidence_root=evidence_root, diagnostic=diagnostic)


def require_discovery_readiness(report, *, evidence_root):
    """Unavailable native evidence is pending; failed mandatory cases remain failures."""
    from oeis_learn.evaluation.discovery_gate import require_g9

    return require_g9(report, evidence_root=evidence_root)


def release_readiness(
    *,
    checkpoint=None,
    baseline_bundle=None,
    g7=None,
    g8=None,
    g9=None,
    wheel_junit=None,
    measurement_root=None,
    release_source=None,
    phase10_junit=None,
    role_junit=None,
    dispositions=None,
):
    """Replay supplied evidence; missing run/hardware gates remain explicit.

    Each extension is a (report, evidence_root) tuple. A status string or schema
    fixture cannot stand in for its validator. Baseline promotion still requires
    the phase-6 coordinated device/containment path; this aggregator cannot invent it.
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from oeis_learn.experiments.artifacts import load_json
    from oeis_learn.experiments.qualified_models import validate_record

    baseline = (
        foundation_readiness(checkpoint, software_complete=False)
        if checkpoint is not None
        else {"qualified": False, "missing_gates": ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]}
    )
    gates = {
        f"G{i}": dict(status="pending", reason="bound run/device/containment evidence pending")
        for i in range(7)
    }
    if baseline_bundle is not None:
        try:
            from oeis_learn.evaluation.baseline_gate import require_baseline

            if checkpoint is None:
                raise ValueError("baseline evidence requires the actual checkpoint")
            bundle, root = baseline_bundle
            gates.update(require_baseline(bundle, checkpoint, evidence_root=root))
            baseline["qualified"] = True
        except (ValueError, OSError, KeyError, TypeError) as exc:
            gates["G0"] = dict(status="FAIL", reason=str(exc))
    for name, value, validator in (
        ("G7", g7, require_architecture_measurement_gates),
        ("G8", g8, require_learning_readiness),
        ("G9", g9, require_discovery_readiness),
    ):
        if value is None:
            gates[name] = dict(status="pending", reason="missing current executable evidence")
            continue
        try:
            report, root = value
            if name == "G8":
                validator(report, evidence_root=root, diagnostic=True)
                if report["hardware_status"] == "PENDING" and report["hardware"] is None:
                    gates[name] = dict(
                        status="pending",
                        software_status="PASS",
                        reason="replayed CPU evidence; qualified hardware still pending",
                        report_sha256=report["report_sha256"],
                    )
                    continue
            validator(report, evidence_root=root)
            gates[name] = dict(status="PASS", report_sha256=report["report_sha256"])
        except (ValueError, OSError, TypeError, KeyError) as exc:
            gates[name] = dict(status="FAIL", reason=str(exc))
    software = {}
    if wheel_junit:
        try:
            cases = ET.parse(wheel_junit).findall(".//testcase")
            expected = {
                "test_packaged_resources_and_actual_execution",
                "test_public_measure_dry_run_and_actual",
                "test_census_public_invalid_input",
                "test_installed_native_has_no_fallback",
                "test_installed_train_resume_search",
                "test_installed_discover_and_fresh_certificate",
            }
            if (
                len(cases) != len(expected)
                or {c.attrib.get("name") for c in cases} != expected
                or any(
                    any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))
                    for c in cases
                )
            ):
                raise ValueError("installed-wheel/native cases missing, failed or skipped")
            software["installed_wheel_native"] = dict(status="PASS", cases=len(cases))
        except (ValueError, OSError, ET.ParseError) as exc:
            software["installed_wheel_native"] = dict(status="FAIL", reason=str(exc))
    else:
        software["installed_wheel_native"] = dict(
            status="pending", reason="no clean-wheel evidence supplied"
        )
    if measurement_root:
        try:
            root = Path(measurement_root)
            from oeis_learn.evaluation.experiment_runner import load_measurement

            report = load_measurement(root)
            for record in report["decisions"]:
                validate_record(record)
            software["measurements"] = dict(
                status="PASS" if report["qualified"] else "pending", reason=report["stop_reason"]
            )
        except (ValueError, OSError, TypeError, KeyError) as exc:
            software["measurements"] = dict(status="FAIL", reason=str(exc))
    else:
        software["measurements"] = dict(
            status="pending", reason="qualified engineering/pilot evidence unavailable"
        )
    if release_source is not None:
        try:
            from oeis_learn.evaluation.release_source import audit_release

            cohort, archive, snapshot = release_source
            audited = audit_release(cohort, archive=archive, snapshot=snapshot)
            software["release_source"] = dict(
                status=audited["status"], report_sha256=audited["report_sha256"]
            )
        except (ValueError, OSError, TypeError, KeyError) as exc:
            software["release_source"] = dict(status="FAIL", reason=str(exc))
    else:
        software["release_source"] = dict(
            status="pending",
            reason="actual indexed release source/structural archive audit missing",
        )
    if phase10_junit:
        try:
            cases = ET.parse(phase10_junit).findall(".//testcase")
            if role_junit:
                cases.extend(ET.parse(role_junit).findall(".//testcase"))
            modules = {c.attrib.get("classname", "").split(".")[-1] for c in cases}
            required = {
                "test_foundation_source_policy",
                "test_foundation_legacy_wrappers",
                "test_foundation_measurement",
                "test_foundation_phase10_integration",
                "test_foundation_roles",
                "test_foundation_overflow_diagnostics",
            }
            if not required <= modules or any(
                any(c.find(t) is not None for t in ("failure", "error", "skipped")) for c in cases
            ):
                raise ValueError(
                    "mandatory source/wrapper/telemetry/trace/coordinator integration incomplete"
                )
            software["integration"] = dict(status="PASS", cases=len(cases))
        except (ValueError, OSError, ET.ParseError) as exc:
            software["integration"] = dict(status="FAIL", reason=str(exc))
    else:
        software["integration"] = dict(
            status="pending", reason="current executable phase10 integration evidence missing"
        )
    if dispositions:
        try:
            record = load_json(Path(dispositions).read_bytes())
            from oeis_learn.evaluation.review_dispositions import require_dispositions

            software["dispositions"] = dict(status="PASS", inputs=require_dispositions(record))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            software["dispositions"] = dict(status="FAIL", reason=str(exc))
    else:
        software["dispositions"] = dict(
            status="pending", reason="current original-input dispositions missing"
        )
    status = (
        "FAIL"
        if any(c["status"] == "FAIL" for c in software.values())
        else "PASS"
        if all(c["status"] == "PASS" for c in software.values())
        else "pending"
    )
    gates["G10"] = dict(
        status=status,
        checks=software,
        missing=[k for k, v in software.items() if v["status"] != "PASS"],
    )
    full = all(value["status"] == "PASS" for value in gates.values())
    return dict(
        baseline_ready=baseline["qualified"],
        full_release_qualified=full,
        state="full_release_qualified"
        if full
        else "baseline_ready"
        if baseline["qualified"]
        else "baseline_pending",
        gates=gates,
        missing_gates=[key for key, value in gates.items() if value["status"] != "PASS"],
        full_007_complete=full,
        lifecycle_is_qualification=False,
    )
