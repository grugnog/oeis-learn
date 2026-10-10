"""Explicit inventory of existing historical launchers. Never relabel past evidence."""

from pathlib import Path
from oeis_learn.experiments.artifacts import compute_file_hash

PATHS = (
    "scripts/run_008_learning_run.py",
    "scripts/run_009_recurrence_sprint.py",
    "scripts/run_010_overnight_curriculum.py",
    "scripts/extract_oeis_data.py",
    "scripts/analyze_oeis_corpus_precision.py",
    "scripts/run_progressive_validation.py",
    "scripts/run_qualification_smoke.py",
    "scripts/run_trustworthy_ablations.py",
    "src/oeis_learn/cli/train_run011.py",
    "src/oeis_learn/cli/train_warmstart.py",
    "src/oeis_learn/cli/evaluate_canaries.py",
)


def wrapper_inventory(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    return [
        dict(
            path=name,
            sha256=compute_file_hash(root / name),
            status="legacy_unverified",
            replacement="oeis-learn foundation",
            disposition="shared solver, sampler and runtime consumers; historical named-teacher data is diagnostic",
        )
        for name in PATHS
        if (root / name).is_file()
    ]


def epoch_diagnostics(metrics):
    """Aggregate observed group variance and solver calls; missing data stays null."""
    import math

    variances = [row.get("reward_variance") for row in metrics]
    observed = [
        value
        for value in variances
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
    ]
    durations = [duration for row in metrics for duration in row.get("solver_durations_ms", [])]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError("actual finite solver durations required")
    return dict(
        advantage_collapse_rate=sum(v == 0 for v in observed) / len(observed)
        if observed and len(observed) == len(metrics)
        else None,
        variance_groups_observed=len(observed),
        modular_filter_rejection_rate=None,
        diophantine_solve_duration_ms=sum(durations) / len(durations) if durations else None,
        solver_calls_observed=len(durations),
        unavailable=dict(
            modular_filter_rejection_rate="filter events not exposed by legacy wrapper",
            **(
                {"diophantine_solve_duration_ms": "no observed solver calls"}
                if not durations
                else {}
            ),
            **(
                {"advantage_collapse_rate": "incomplete group variance observations"}
                if not observed or len(observed) != len(metrics)
                else {}
            ),
        ),
    )


def complete_legacy_run(context):
    """Lifecycle completion never grants release qualification to a legacy launcher."""
    context.set_qualification_state("legacy_unverified")
    context.set_status("COMPLETED")
    return dict(
        status="SUCCESS",
        run_id=context.metadata.run_id,
        qualified=False,
        purpose="legacy_unverified",
    )
