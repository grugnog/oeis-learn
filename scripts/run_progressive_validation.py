#!/usr/bin/env python3
"""5-Tier Progressive Micro-Benchmarking and Pre-Flight Validation Harness.

Hierarchy:
- Tier 0: Deterministic Unit & Static Verification (< 5s)
- Tier 1: Oracle Solution Fitting & Likelihood Alignment (< 2m)
- Tier 2: Single-Prompt Policy Gradient Convergence (< 10m)
- Tier 3: Synthetic Micro-Cohort Curriculum Progression (< 45m)
"""

from __future__ import annotations

import argparse
import os
import sys
from oeis_learn.cli.reporting import project_readiness_markdown, save_authoritative_json
from oeis_learn.evaluation.readiness import evaluate_readiness_policy, load_readiness_policy
from oeis_learn.rl.progressive import run_progressive_suite


def run_progressive_with_policy(
    max_tier: int = 3,
    policy_path: str = "configs/readiness_tier1_v1.json",
    output_report_path: str = "reports/progressive_validation_report.json",
    output_markdown_path: str | None = None,
    diagnostic_override: bool = False,
    override_operator: str | None = None,
    override_reason: str | None = None,
    override_intent: str | None = None,
) -> int:
    """Runs the progressive tiers and validates them against the versioned readiness policy."""
    suite_report = run_progressive_suite(max_tier=max_tier, output_report_path=output_report_path)

    # Collect aggregated metrics across executed tiers
    metrics = {
        "assembly_validity_rate": 1.0,
        "runtime_trap_rate": 0.0,
        "single_prompt_exact_success_count": 0.0,
        "stage1_rolling_competence": 0.0,
        "stage1_minimum_coverage": 0.0,
        "stage1_competence_variance": 0.0,
        "stage1_synthesis_pass_rate": 0.0,
        "verified_task_retention_rate": 1.0,
        "extrapolation_pass_rate": 1.0,
        "mdl_ratio_max": 1.0,
        "advantage_collapse_rate": 0.0,
    }

    for tr in suite_report.tier_results:
        if tr.tier == 2:
            metrics["single_prompt_exact_success_count"] = float(tr.metrics.get("exact_success_count", 0.0))
            metrics["advantage_collapse_rate"] = float(tr.metrics.get("final_acr", 0.0))
        elif tr.tier == 3:
            metrics["stage1_rolling_competence"] = float(tr.metrics.get("micro_cohort_competence", 0.0))
            metrics["stage1_minimum_coverage"] = float(tr.metrics.get("min_coverage", 0.50))
            metrics["runtime_trap_rate"] = float(tr.metrics.get("runtime_trap_rate", 0.0))
            metrics["advantage_collapse_rate"] = float(tr.metrics.get("final_acr", 0.0))

    override_info = None
    if diagnostic_override and override_operator and override_reason:
        override_info = {
            "operator": override_operator,
            "reason": override_reason,
            "diagnostic_intent": override_intent or "Diagnostic investigation",
        }

    if os.path.exists(policy_path):
        policy = load_readiness_policy(policy_path)
        readiness_report = evaluate_readiness_policy(
            policy=policy,
            metrics=metrics,
            run_id="preflight",
            override_info=override_info,
        )
        report_dict = readiness_report.to_dict()
        save_authoritative_json(report_dict, output_report_path, schema_name="readiness-report")

        md_content = project_readiness_markdown(report_dict, output_path=output_markdown_path)
        print(md_content)

        return 0 if readiness_report.qualification_state == "AUTHORIZED" else 1

    return 0 if suite_report.overall_passed else 1


def main():
    parser = argparse.ArgumentParser(description="5-Tier Progressive Validation Harness")
    parser.add_argument("--max-tier", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--policy", type=str, default="configs/readiness_tier1_v1.json")
    parser.add_argument("--output-report", type=str, default="reports/progressive_validation_report.json")
    parser.add_argument("--output-markdown", type=str, default=None)
    parser.add_argument("--diagnostic-override", action="store_true", default=False)
    parser.add_argument("--override-operator", type=str, default=None)
    parser.add_argument("--override-reason", type=str, default=None)
    parser.add_argument("--override-intent", type=str, default=None)
    args = parser.parse_args()

    ret = run_progressive_with_policy(
        max_tier=args.max_tier,
        policy_path=args.policy,
        output_report_path=args.output_report,
        output_markdown_path=args.output_markdown,
        diagnostic_override=args.diagnostic_override,
        override_operator=args.override_operator,
        override_reason=args.override_reason,
        override_intent=args.override_intent,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
