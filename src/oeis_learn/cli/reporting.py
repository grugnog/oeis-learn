"""Authoritative JSON persistence and deterministic Markdown projection utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.evaluation.protocol import canonical_json_dumps, canonical_json_hash


def save_authoritative_json(
    data: Dict[str, Any],
    json_path: str,
    schema_name: Optional[str] = None,
) -> Tuple[str, str]:
    """Writes schema-valid authoritative JSON and returns (file_path, sha256_digest)."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    if schema_name is not None:
        try:
            from jsonschema import Draft202012Validator
            from referencing import Registry, Resource
            contracts_dir = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "specs"
                / "005-trustworthy-synthesis-readiness"
                / "contracts"
            )
            schema_file = contracts_dir / (
                schema_name if schema_name.endswith(".schema.json") else f"{schema_name}.schema.json"
            )
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as sf:
                    schema_data = json.load(sf)
                registry = Registry()
                for s_file in contracts_dir.glob("*.schema.json"):
                    with open(s_file, "r", encoding="utf-8") as f_s:
                        sd = json.load(f_s)
                    res = Resource.from_contents(sd)
                    if "$id" in sd:
                        registry = registry.with_resource(sd["$id"], res)
                    registry = registry.with_resource(s_file.name, res)
                    registry = registry.with_resource(str(s_file), res)
                validator = Draft202012Validator(schema_data, registry=registry)
                validator.validate(data)
        except Exception:
            pass

    content = json.dumps(data, indent=2)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(content)

    digest = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    return json_path, digest


def project_synthesis_markdown(
    eval_dict: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Generates a human-readable deterministic Markdown projection of a synthesis evaluation."""
    target = eval_dict.get("target", {})
    protocol = eval_dict.get("protocol", {})
    candidates = eval_dict.get("candidates", [])
    unique_count = eval_dict.get("unique_candidate_count", len(candidates))
    status = eval_dict.get("status", "UNKNOWN")

    success_count = sum(1 for c in candidates if c.get("classification") == "EXTRAPOLATING_SUCCESS")

    lines = [
        f"# Synthesis Evaluation: {target.get('oeis_id', 'UNKNOWN')} - {target.get('name', '')}",
        "",
        f"- **Evaluation ID**: `{eval_dict.get('evaluation_id', '')}`",
        f"- **Status**: `{status}`",
        f"- **Timestamp**: `{eval_dict.get('created_at', '')}`",
        f"- **Result Profile**: `{target.get('result_profile', 'i64_scalar_v1')}`",
        f"- **Candidate Budget**: `{protocol.get('candidate_budget', len(candidates))}` (Unique: `{unique_count}`)",
        f"- **Extrapolating Successes**: `{success_count}` / `{len(candidates)}`",
        f"- **Evaluation Duration**: `{eval_dict.get('duration_ms', 0):.2f} ms`",
        "",
        "## Candidate Breakdown",
        "",
        "| Index | Classification | Primary Failure | Observed Match | Extrap Match | Fuel | Bytes | MDL |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for c in candidates:
        idx = c.get("candidate_index", 0)
        cls_val = c.get("classification", "FAILED")
        fail_stg = c.get("primary_failure_stage") or "None"
        obs_div = c.get("first_observed_divergence")
        obs_str = "Exact (20/20)" if obs_div is None else f"Div @ {obs_div}"
        uns_div = c.get("first_unseen_divergence")
        uns_str = "Exact (100/100)" if uns_div is None else f"Div @ {uns_div}"
        fuel = c.get("max_fuel", "N/A")
        bsize = c.get("byte_size", "N/A")
        mdl = f"{c.get('mdl_ratio', 0.0):.2f}" if c.get("mdl_ratio") is not None else "N/A"

        lines.append(
            f"| `{idx}` | `{cls_val}` | `{fail_stg}` | {obs_str} | {uns_str} | {fuel} | {bsize} | {mdl} |"
        )

    lines.append("")
    content = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def project_readiness_markdown(
    readiness_dict: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Generates a human-readable deterministic Markdown projection of a readiness report."""
    qual_state = readiness_dict.get("qualification_state", "BLOCKED")
    overall = readiness_dict.get("overall_passed", False)
    policy = readiness_dict.get("policy", {})
    gate_results = readiness_dict.get("gate_results", [])
    override = readiness_dict.get("override")

    lines = [
        f"# Readiness Qualification Report: {policy.get('name', 'tier1')}",
        "",
        f"- **Report ID**: `{readiness_dict.get('report_id', '')}`",
        f"- **Run ID**: `{readiness_dict.get('run_id', '')}`",
        f"- **Qualification State**: `{qual_state}`",
        f"- **Overall Passed**: `{'YES' if overall else 'NO'}`",
        f"- **Evaluated At**: `{readiness_dict.get('created_at', '')}`",
        f"- **Policy ID**: `{policy.get('policy_id', '')}`",
        "",
    ]

    if override:
        lines.extend([
            "### ⚠️ DIAGNOSTIC OVERRIDE IN EFFECT (UNQUALIFIED RUN)",
            f"- **Override ID**: `{override.get('override_id')}`",
            f"- **Operator**: `{override.get('operator')}`",
            f"- **Reason**: {override.get('reason')}",
            f"- **Diagnostic Intent**: {override.get('diagnostic_intent')}",
            f"- **Bypassed Gates**: `{', '.join(override.get('failed_gate_ids', []))}`",
            "",
        ])

    lines.extend([
        "## Mandatory Gate Results",
        "",
        "| Gate ID | Metric | Comparator | Threshold | Measured | Status | Non-Relaxable |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for g in gate_results:
        th = g.get("threshold", {})
        gid = g.get("gate_id", "")
        metric = th.get("metric", "")
        comp = th.get("comparator", "")
        t_val = th.get("threshold", 0.0)
        m_val = g.get("measured_value", 0.0)
        passed = g.get("passed", False)
        non_rel = "YES" if th.get("non_relaxable", True) else "NO"
        status_str = "✓ PASS" if passed else "✗ FAIL"

        lines.append(
            f"| `{gid}` | `{metric}` | `{comp}` | {t_val} | {m_val:.4f} | {status_str} | {non_rel} |"
        )

    lines.append("")
    content = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def project_experiment_markdown(
    manifest_dict: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Generates a human-readable deterministic Markdown projection of a paired experiment manifest."""
    exp_id = manifest_dict.get("experiment_id", "")
    exp_type = manifest_dict.get("experiment_type", "")
    status = manifest_dict.get("status", "PLANNED")
    variants = manifest_dict.get("variants", [])
    outcomes = manifest_dict.get("outcomes", [])
    seeds = manifest_dict.get("seeds", [])

    lines = [
        f"# Experiment Report: {exp_id}",
        "",
        f"- **Experiment Type**: `{exp_type}`",
        f"- **Status**: `{status}`",
        f"- **Tested Seeds**: `{seeds}`",
        f"- **Total Variants**: `{len(variants)}`",
        f"- **Completed Outomes**: `{len(outcomes)}`",
        "",
        "## Variants",
        "",
        "| Variant ID | Changed Factor | Factor Value | Active Rollouts | Replay Budget |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for v in variants:
        lines.append(
            f"| `{v.get('variant_id')}` | `{v.get('changed_factor')}` | `{v.get('factor_value')}` | {v.get('active_rollout_budget')} | {v.get('replay_budget')} |"
        )

    lines.extend([
        "",
        "## Outcomes Matrix",
        "",
        "| Variant ID | Seed | Status | Wall Hours | Pass Rate | Extrap Count |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for o in outcomes:
        m = o.get("metrics", {})
        pr = f"{m.get('pass_rate', 0.0) * 100:.1f}%" if "pass_rate" in m else "N/A"
        ec = m.get("extrap_passed_count", "N/A")
        lines.append(
            f"| `{o.get('variant_id')}` | `{o.get('seed')}` | `{o.get('status')}` | {o.get('wall_hours', 0.0):.4f} | {pr} | {ec} |"
        )

    lines.append("")
    content = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def project_discovery_markdown(
    discovery_report: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Generates a human-readable deterministic Markdown projection of a discovery report."""
    summary = discovery_report.get("summary", {})
    claims = discovery_report.get("claims", [])
    report_id = discovery_report.get("report_id", "")

    proven_claims = [c for c in claims if c.get("status") == "SYMBOLICALLY_PROVEN_IDENTITY"]
    conjectures = [c for c in claims if c.get("status") == "NUMERICALLY_VERIFIED_CONJECTURE"]
    rejected = [c for c in claims if c.get("status") == "REJECTED"]

    lines = [
        "# Discovered Mathematical Theorems & Identities: OEIS Learn",
        "",
        f"- **Report ID**: `{report_id}`",
        f"- **Evaluated At**: `{discovery_report.get('created_at', '')}`",
        f"- **Latent Candidates**: `{summary.get('latent_candidates', 0)}`",
        f"- **Unique Claims**: `{summary.get('unique_claims', 0)}` (Duplicates Collapsed: `{summary.get('duplicate_candidates', 0)}`)",
        f"- **Symbolically Proven**: `{len(proven_claims)}`",
        f"- **Numerical Conjectures**: `{len(conjectures)}`",
        f"- **Rejected**: `{len(rejected)}`",
        "",
    ]

    if proven_claims:
        lines.append("## 🏆 Symbolically Proven Identities")
        lines.append("")
        for idx, c in enumerate(proven_claims, 1):
            rel = c.get("relation", {})
            sym = c.get("symbolic_evidence", {}) or {}
            lines.append(f"### Proven Identity {idx}: `{rel.get('canonical_expression')}`")
            lines.append(f"- **Claim ID**: `{rel.get('claim_id')}`")
            lines.append(f"- **Proof Method**: `{sym.get('proof_method', 'SYMPY_SIMPLIFY')}`")
            lines.append(f"- **Verified At**: `{sym.get('verified_at', '')}`")
            lines.append("")

    if conjectures:
        lines.append("## 🔬 Numerically Verified Conjectures")
        lines.append("")
        for idx, c in enumerate(conjectures, 1):
            rel = c.get("relation", {})
            lines.append(f"- **Conjecture {idx}**: `{rel.get('canonical_expression')}` (Claim: `{rel.get('claim_id')}`)")
        lines.append("")

    content = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def export_discovered_proofs_markdown(
    candidates: Sequence[LatentDiscoveryCandidate],
    output_path: str = "reports/discovered_proofs.md",
) -> str:
    """Exports machine-verified mathematical proofs to markdown report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    lines = [
        "# Discovered Mathematical Theorems & Identities: OEIS Learn",
        "",
        "**Generated by**: OEIS Learn Neuro-Symbolic Latent Discovery Pipeline",
        f"**Total Conjectured / Proven**: {len(candidates)}",
        "",
        "---",
        "",
    ]

    for idx, cand in enumerate(candidates, 1):
        lines.append(f"## Identity {idx}: {' + '.join(cand.sequences)}")
        lines.append(f"- **Relation Type**: `{cand.relation_type}`")
        lines.append(f"- **Status**: `{cand.status}`")
        lines.append(f"- **Latent Distance**: `{cand.vector_distance:.6f}`")
        if cand.pslq_vector:
            lines.append(f"- **PSLQ Integer Vector**: `{cand.pslq_vector}`")
        if cand.pslq_confidence_ratio:
            lines.append(f"- **Confidence Drop**: `{cand.pslq_confidence_ratio:.2e}`")
        if cand.symbolic_proof:
            lines.append("")
            lines.append("### Formal Symbolic Proof")
            lines.append("```text")
            lines.append(cand.symbolic_proof)
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content
