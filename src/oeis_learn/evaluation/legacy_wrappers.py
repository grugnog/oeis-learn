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
