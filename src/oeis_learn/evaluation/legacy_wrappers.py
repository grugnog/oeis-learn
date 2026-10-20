"""Explicit inventory of existing historical launchers. Never relabel past evidence."""

from pathlib import Path
from oeis_learn.experiments.artifacts import compute_file_hash

PATHS = (
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


def load_legacy_configuration(path):
    """One parser for existing launchers; historical contents are preserved verbatim."""
    import yaml
    from oeis_learn.experiments.config import _UniqueLoader

    value = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
    if not isinstance(value, dict) or not value:
        raise ValueError("nonempty historical configuration required")
    return value


def require_legacy_parent(path):
    if path and not Path(path).is_file():
        raise FileNotFoundError(f"Explicit historical parent checkpoint missing: {path}")


def latest_legacy_checkpoint(directory):
    """Resolve historical epochs numerically, including runs beyond epoch 999."""
    import re

    candidates = []
    for path in Path(directory).glob("model_epoch_*.v2.pt"):
        match = re.fullmatch(r"model_epoch_(\d+)\.v2\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), str(path), path))
    if not candidates:
        return None
    epoch, _, path = max(candidates)
    return path, epoch


def legacy_readiness_metrics(trainer, scheduler, epoch_metrics, pass_rates):
    observed = epoch_diagnostics(epoch_metrics)
    competence, coverage = scheduler.compute_stage_competence(1)
    record = trainer.telemetry.latest_record
    return dict(
        runtime_trap_rate=float(record.compiler_trap_rate) if record else None,
        stage1_rolling_competence=float(competence),
        stage1_minimum_coverage=float(coverage),
        stage1_competence_variance=float(scheduler.compute_epoch_variance()),
        stage1_synthesis_pass_rate=float(pass_rates[-1]) if pass_rates else None,
        advantage_collapse_rate=observed["advantage_collapse_rate"],
        diophantine_solve_duration_ms=observed["diophantine_solve_duration_ms"],
        modular_filter_rejection_rate=observed["modular_filter_rejection_rate"],
    )
