"""Portable discovery command surface. Model and proposer imports stay installed."""

COMMANDS = {
    "init": ("reference-campaign", "output"),
    "qualify": ("blueprint", "output"),
    "prepare": ("blueprint", "qualification", "output"),
    "freeze": ("blueprint", "prepared", "qualification", "output"),
    "acceptance": ("campaign", "output"),
    "run": ("campaign", "acceptance", "output"),
    "status": ("run",),
    "pause": ("run",),
    "resume": ("run", "output"),
    "report": ("run", "output"),
    "verify-bundle": ("archive", "output"),
    "assess": ("campaign", "selection", "acceptance", "output"),
}

# Explicit inventory avoids the unrelated legacy rl/progressive harness whose
# tests also begin with test_progressive_. New cases remain in source closure.
CPU_TESTS = (
    "tests/unit/test_oeis_corpus.py",
    "tests/unit/test_progressive_research_inputs.py",
    "tests/unit/test_progressive_research_census.py",
    "tests/unit/test_progressive_qualification_cleanup.py",
    "tests/unit/test_progressive_workstation.py",
    "tests/integration/test_progressive_component_flow.py",
    "tests/integration/test_progressive_reporting.py",
    "tests/integration/test_progressive_short_cycle.py",
    "tests/integration/test_progressive_acceptance_profiles.py",
    "tests/integration/test_progressive_calibration.py",
    "tests/integration/test_progressive_handoff.py",
    "tests/contract/test_progressive_diagnostic_blueprints.py",
    "tests/integration/test_progressive_acceptance.py",
    "tests/contract/test_progressive_assets.py",
    "tests/contract/test_progressive_authority.py",
    "tests/contract/test_progressive_cli.py",
    "tests/contract/test_progressive_evaluation.py",
    "tests/contract/test_progressive_packaging.py",
    "tests/contract/test_progressive_roles.py",
    "tests/integration/test_progressive_preparation.py",
    "tests/integration/test_progressive_lifecycle.py",
    "tests/integration/test_progressive_oeis_preparation.py",
    "tests/integration/test_progressive_evaluation.py",
    "tests/integration/test_progressive_stream.py",
    "tests/integration/test_progressive_runtime.py",
    "tests/unit/test_progressive_baselines.py",
    "tests/unit/test_progressive_budgets.py",
    "tests/unit/test_progressive_recovery_transactions.py",
    "tests/unit/test_progressive_ci.py",
    "tests/unit/test_progressive_cohorts.py",
    "tests/unit/test_progressive_coverage.py",
    "tests/unit/test_progressive_decisions.py",
    "tests/unit/test_progressive_exposure.py",
    "tests/unit/test_progressive_failures.py",
    "tests/unit/test_progressive_freeze.py",
    "tests/unit/test_progressive_metrics.py",
    "tests/unit/test_progressive_targets.py",
    "tests/unit/test_progressive_scaling.py",
    "tests/unit/test_progressive_dag_decisions.py",
    "tests/unit/test_progressive_commands_lifecycle.py",
    "tests/unit/test_progressive_heartbeat.py",
    "tests/unit/test_progressive_continuation.py",
    "tests/unit/test_progressive_archive.py",
    "tests/unit/test_progressive_yield_audit.py",
    "tests/unit/test_progressive_replay.py",
    "tests/unit/test_progressive_components.py",
    "tests/unit/test_progressive_dossiers.py",
    "tests/unit/test_progressive_polynomial_certificate.py",
    "tests/unit/test_progressive_quarantine.py",
    "tests/contract/test_progressive_sources.py",
    "tests/contract/test_progressive_bundle.py",
    "tests/contract/test_progressive_assessment.py",
    "tests/contract/test_progressive_qualification.py",
    "tests/unit/test_progressive_short_profile.py",
    "tests/unit/test_progressive_combinatorial.py",
    "tests/integration/test_progressive_sources.py",
    "tests/unit/test_progressive_claims.py",
    "tests/unit/test_progressive_component_mining.py",
    "tests/unit/test_progressive_component_search.py",
    "tests/integration/test_progressive_oeis.py",
    "tests/integration/test_progressive_components.py",
    "tests/integration/test_progressive_claims.py",
    "tests/contract/test_progressive_admission.py",
    "tests/unit/test_progressive_sensitivity.py",
    "tests/unit/test_progressive_stream.py",
    "tests/unit/test_progressive_curriculum.py",
    "tests/unit/test_progressive_optimizer.py",
    "tests/unit/test_progressive_optimizer_specifications.py",
    "tests/unit/test_progressive_training_policy.py",
    "tests/unit/test_progressive_trials.py",
    "tests/unit/test_progressive_tuning.py",
    "tests/unit/test_progressive_trial_eligibility.py",
    "tests/unit/test_progressive_trial_scopes.py",
    "tests/unit/test_progressive_outputs.py",
    "tests/unit/test_progressive_diversity_counts.py",
    "tests/unit/test_progressive_supply.py",
    "tests/unit/test_progressive_iteration.py",
    "tests/unit/test_progressive_contingencies.py",
    "tests/unit/test_progressive_contingency_specifications.py",
    "tests/integration/test_progressive_tuning.py",
    "tests/integration/test_progressive_recovery.py",
    "tests/integration/test_progressive_recovery_updates.py",
    "tests/unit/test_progressive_trial_specifications.py",
    "tests/unit/test_progressive_models.py",
    "tests/unit/test_discovery_readiness.py",
)


def register(commands):
    group = commands.add_parser("discovery", help="bounded spec 013 discovery operations")
    operations = group.add_subparsers(dest="operation", required=True)
    for name, arguments in COMMANDS.items():
        command = operations.add_parser(name)
        for argument in arguments:
            command.add_argument("--" + argument, required=True)


def main(args, root, host_factory):
    from .progressive_commands import main as execute

    return execute(args, root, host_factory)
