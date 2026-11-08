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
    "tests/contract/test_progressive_assets.py",
    "tests/contract/test_progressive_authority.py",
    "tests/contract/test_progressive_cli.py",
    "tests/contract/test_progressive_evaluation.py",
    "tests/contract/test_progressive_packaging.py",
    "tests/contract/test_progressive_roles.py",
    "tests/integration/test_progressive_preparation.py",
    "tests/integration/test_progressive_evaluation.py",
    "tests/integration/test_progressive_runtime.py",
    "tests/unit/test_progressive_baselines.py",
    "tests/unit/test_progressive_budgets.py",
    "tests/unit/test_progressive_ci.py",
    "tests/unit/test_progressive_cohorts.py",
    "tests/unit/test_progressive_coverage.py",
    "tests/unit/test_progressive_decisions.py",
    "tests/unit/test_progressive_exposure.py",
    "tests/unit/test_progressive_failures.py",
    "tests/unit/test_progressive_freeze.py",
    "tests/unit/test_progressive_metrics.py",
    "tests/unit/test_progressive_targets.py",
    "tests/unit/test_progressive_sensitivity.py",
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
