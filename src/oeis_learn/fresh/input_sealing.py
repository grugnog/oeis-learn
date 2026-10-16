"""Explicitly seal a near-complete extended diagnostic without further sampling."""

import uuid
from pathlib import Path

from .common import digest, file_hash, lock, read
from .input_export import collect, export


def completion_record(config, training, development, plan_hash, budget):
    requested = config["archive"]["training_prompts"]
    minimum = (95 * requested + 99) // 100
    if (
        config["run"].get("profile") != "fresh-extended-12h"
        or requested != 2048
        or not minimum <= training < requested
        or development < config["archive"]["development_structures"]
        or budget["status"] != "paused"
        or budget["charged_budget_ns"] < config["archive"]["seconds"] * 10**9
    ):
        raise ValueError(
            "partial seal requires extended inputs with >=95% of 2048 distinct prompts"
        )
    return {
        "mode": "accepted_shortfall",
        "requested_training_prompts": requested,
        "actual_training_prompts": training,
        "minimum_training_prompts": minimum,
        "admitted_development_structures": development,
        "exported_development_structures": config["archive"]["development_structures"],
        "plan_sha256": plan_hash,
        "budget": budget,
        "qualified": False,
    }


def seal(output):
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.evaluation.foundation_cohort import build_cohort
    from oeis_learn.evaluation.release_source import read_indexed_source
    from oeis_learn.experiments.learning_profiles import load_holdouts
    from oeis_learn.tracking.budget_ledger import BudgetLedger

    from .inputs import check_inputs

    root = Path(output).resolve()
    with lock(root / "prepare.lock"):
        if (root / "prepared.json").exists():
            return check_inputs(root)
        plan = read(root / "archive-plan.json")
        config = plan["config"]
        if plan["version"] != "fresh-admission-plan/v1":
            raise ValueError("unknown admission plan")
        if file_hash(root / "indexed-source/manifest.json") != plan["source_manifest"]:
            raise ValueError("frozen source manifest changed")
        report = read(root / "archive-preparation-report.json")
        if report.get("stop_reason") != "lifetime_budget":
            raise ValueError("partial seal requires lifetime-budget exhaustion")
        if not any((root / "archive-budget/budget-events").glob("*.jsonl")):
            raise ValueError("original preparation ledger required")
        # Reopen the original identity/cap. This never issues another allowance.
        ledger = BudgetLedger(
            root / "archive-budget",
            str(uuid.UUID(digest(plan)[7:39])),
            config["archive"]["seconds"] * 10**9,
        )
        if ledger.remaining_ns() > 0 or ledger.snapshot()["status"] != "paused":
            raise ValueError("partial seal requires an exhausted, paused original ledger")
        retained_budget = report["budget"]
        if (
            retained_budget["charged_budget_ns"] != ledger.snapshot()["charged_budget_ns"]
            or retained_budget["run_id"] != ledger.snapshot()["run_id"]
        ):
            raise ValueError("preparation report differs from original exhausted budget")
        holdouts = load_holdouts(read(root / "holdouts.json"))
        if holdouts.identity() != plan["holdouts"]:
            raise ValueError("frozen holdouts changed")
        # Archive construction replays packet integrity, provenance and eligibility.
        archive = ProgramArchive(root / "archive", holdouts=holdouts)
        training, development = {}, {}
        collect(archive, archive.freeze()["entries"], training, development)
        if "training_mixture" in plan:
            from .bootstrap_views import MIXTURE, training_mixture

            if plan["training_mixture"] != MIXTURE:
                raise ValueError("unknown frozen training mixture")
            training = training_mixture(training, config["archive"]["training_prompts"])
        if (len(training), len(development)) != (
            report["training_prompts"],
            report["development_structures"],
        ):
            raise ValueError("admitted counts differ from retained preparation report")
        completion = completion_record(
            config,
            len(training),
            len(development),
            file_hash(root / "archive-plan.json"),
            retained_budget,
        )
        rows, _ = read_indexed_source(root / "indexed-source/manifest.json")
        built = build_cohort(rows, affine=True, **config["cohort"])
        imported = read(root / "indexed-source/import-report.json")
        return export(
            root, config, imported, built, archive, training, development, completion=completion
        )
