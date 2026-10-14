"""Sealing/export boundaries with a fixture archive and a real durable budget ledger."""

import shutil
import uuid
from unittest.mock import patch

import pytest

from oeis_learn.data.structural_splits import StructuralHoldouts, split_for_structure
from oeis_learn.experiments.learning_profiles import holdout_profile
from oeis_learn.fresh.common import digest, file_hash, inventory, read, write
from oeis_learn.fresh.input_sealing import completion_record, seal
from oeis_learn.tracking.budget_ledger import BudgetLedger
from tests.unit.test_extended_fresh import expanded_environment
from tests.unit.test_foundation_local import local
from tests.unit.test_fresh_cli import cli, config
from tests.unit.test_fresh_workflow import FakeClock, host


def seal_fixture(tmp_path, *, count=2008, expired=True):
    root = tmp_path / "inputs"
    cfg = config()
    cfg["run"].update(profile="fresh-extended-12h", training_seconds=36000)
    cfg["archive"].update(training_prompts=2048, development_structures=128, seconds=7200)
    holdouts = StructuralHoldouts(seed=7)
    write(root / "holdouts.json", holdout_profile(holdouts))
    write(root / "indexed-source/manifest.json", {"fixture": True})
    write(root / "indexed-source/import-report.json", {"fixture": True})
    write(root / "cohort.yaml", b"fixture\n")
    plan = {
        "version": "fresh-admission-plan/v1",
        "config": cfg,
        "source_manifest": file_hash(root / "indexed-source/manifest.json"),
        "holdouts": holdouts.identity(),
    }
    write(root / "archive-plan.json", plan)
    clock = FakeClock()
    ledger = BudgetLedger(
        root / "archive-budget", str(uuid.UUID(digest(plan)[7:39])), 7200 * 10**9, clock=clock
    )
    ledger.transition("bootstrap")
    clock.n += (7200 if expired else 100) * 10**9
    ledger.transition("paused")
    budget = ledger.snapshot()
    write(
        root / "archive-preparation-report.json",
        {
            "training_prompts": count,
            "development_structures": 2017,
            "stop_reason": "lifetime_budget",
            "budget": budget,
        },
    )
    dev = []
    n = 0
    while len(dev) < 2017:
        key = digest(["development", n])
        if split_for_structure(key, holdouts.seed) == "development":
            dev.append(key)
        n += 1
    packets = {}
    for i in range(count):
        key = digest(["training", i])
        packets[key] = {
            "tokens": [1],
            "entry": {
                "conditioning_values": [str(i)] * 20,
                "ir_sha256": digest(["structure", i]),
                "training_eligible": True,
                "context_tokens": 1,
            },
        }
    for i, key in enumerate(dev):
        packets[key] = {
            "tokens": [1],
            "entry": {
                "conditioning_values": [str(-1 - i)] * 20,
                "ir_sha256": key,
                "training_eligible": False,
                "context_tokens": 1,
            },
        }

    class FixtureArchive:
        def __init__(self, path, *, holdouts):
            self.holdouts = holdouts

        def freeze(self):
            return {"entries": sorted(packets)}

        def get(self, key):
            return packets[key]

    def release(source, cohort, output):
        (output / "cohort").mkdir(parents=True)

    return root, cfg, budget, clock, FixtureArchive, release


def fixture_patches(clock, archive, release):
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch(
            "oeis_learn.tracking.budget_ledger.BudgetLedger",
            lambda *args: BudgetLedger(*args, clock=clock),
        )
    )
    stack.enter_context(patch("oeis_learn.discovery.program_archive.ProgramArchive", archive))
    stack.enter_context(
        patch("oeis_learn.evaluation.release_source.read_indexed_source", return_value=([], {}))
    )
    stack.enter_context(
        patch(
            "oeis_learn.evaluation.foundation_cohort.build_cohort",
            return_value={"groups": [], "records": {}},
        )
    )
    stack.enter_context(patch("oeis_learn.evaluation.release_source.prepare_release", release))
    stack.enter_context(
        patch("oeis_learn.evaluation.release_source.audit_release", return_value={"fixture": True})
    )
    stack.enter_context(
        patch(
            "oeis_learn.sandbox.runner.FoundationRunner",
            side_effect=AssertionError("no new candidate execution"),
        )
    )
    return stack


def test_seal_records_actual_count_caps_development_and_retries_without_refund(tmp_path):
    root, cfg, budget, clock, archive, release = seal_fixture(tmp_path)
    plan_hash = file_hash(root / "archive-plan.json")
    report_hash = file_hash(root / "archive-preparation-report.json")
    with fixture_patches(clock, archive, release):
        result = seal(root)
        assert len(read(root / "examples.json")["training"]) == 2008
        assert len(read(root / "examples.json")["development"]) == 128
        assert result["config"] == cfg and not result["qualified"]
        completion = result["admission_completion"]
        assert completion["actual_training_prompts"] == 2008
        assert completion["requested_training_prompts"] == 2048
        assert completion["admitted_development_structures"] == 2017
        assert completion["budget"] == budget
        assert file_hash(root / "archive-plan.json") == plan_hash
        assert file_hash(root / "archive-preparation-report.json") == report_hash
        assert seal(root) == result


def test_interrupted_export_retries_same_immutable_completion(tmp_path):
    root, cfg, budget, clock, archive, release = seal_fixture(tmp_path)

    def interrupted(path, value, **kwargs):
        if path.name == "prepared.json":
            raise RuntimeError("interrupted export")
        return write(path, value, **kwargs)

    with fixture_patches(clock, archive, release):
        with patch("oeis_learn.fresh.input_export.write", interrupted):
            with pytest.raises(RuntimeError, match="interrupted export"):
                seal(root)
        completion_hash = file_hash(root / "admission-completion.json")
        result = seal(root)
        assert file_hash(root / "admission-completion.json") == completion_hash
        assert result["admission_completion"]["budget"] == budget


@pytest.mark.parametrize(
    "invalid", ["few", "remaining", "missing_ledger", "report", "holdouts", "source"]
)
def test_seal_rejects_unready_or_changed_evidence(tmp_path, invalid):
    root, cfg, budget, clock, archive, release = seal_fixture(
        tmp_path, count=492 if invalid == "few" else 2008, expired=invalid != "remaining"
    )
    if invalid == "missing_ledger":
        shutil.rmtree(root / "archive-budget")
    if invalid == "report":
        report = read(root / "archive-preparation-report.json")
        report["training_prompts"] += 1
        write(root / "archive-preparation-report.json", report)
    if invalid == "holdouts":
        write(root / "holdouts.json", holdout_profile(StructuralHoldouts(seed=8)))
    if invalid == "source":
        write(root / "indexed-source/manifest.json", {"changed": True})
    with fixture_patches(clock, archive, release), pytest.raises(ValueError):
        seal(root)
    assert not (root / "prepared.json").exists()
    if invalid == "missing_ledger":
        assert not (root / "archive-budget").exists()


@pytest.mark.parametrize(
    "count,accepted", [(1945, False), (1946, True), (2008, True), (2048, False)]
)
def test_partial_threshold_is_closed_and_recorded(count, accepted):
    cfg = config()
    cfg["archive"].update(training_prompts=2048, development_structures=128, seconds=7200)
    cfg["run"]["profile"] = "fresh-extended-12h"
    budget = {"status": "paused", "charged_budget_ns": 7200 * 10**9}
    if accepted:
        assert (
            completion_record(cfg, count, 2017, "plan", budget)["minimum_training_prompts"] == 1946
        )
    else:
        with pytest.raises(ValueError):
            completion_record(cfg, count, 2017, "plan", budget)


def test_cli_requires_explicit_acceptance_and_host_routes_offline_cpu(tmp_path):
    for entrypoint in (cli, local):
        with pytest.raises(SystemExit):
            entrypoint.main(["seal-inputs", "--inputs", "inputs"])
    with (
        patch.object(local, "ROOT", tmp_path),
        patch.object(local, "Host") as factory,
    ):
        assert local.main(["seal-inputs", "--inputs", "inputs", "--accept-shortfall"]) == 0
        factory.return_value.cpu.assert_called_once_with(
            "seal-inputs", "--inputs", "inputs", "--accept-shortfall"
        )


@pytest.mark.parametrize("accepted", [False, True])
def test_extended_host_requires_bound_partial_completion(tmp_path, accepted):
    workspace, policy = expanded_environment(tmp_path)
    root = workspace / "inputs"
    examples = read(root / "examples.json")
    examples["training"] = examples["training"][:2008]
    write(root / "examples.json", examples)
    manifest = read(root / "prepared.json")
    if accepted:
        manifest["config"]["archive"] = {
            "training_prompts": 2048,
            "development_structures": 1,
            "seconds": 7200,
        }
        record = completion_record(
            manifest["config"],
            2008,
            128,
            "plan",
            {"status": "paused", "charged_budget_ns": 7200 * 10**9},
        )
        write(root / "admission-completion.json", record)
        manifest["admission_completion"] = record
    manifest["files"] = inventory(root)
    manifest["files"].pop("prepared.json", None)
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k != "identity"})
    write(root / "prepared.json", manifest)

    class ReachedTraining(Exception):
        pass

    class Coordinator:
        def __init__(self, *args):
            pass

        def run(self, *args):
            raise ReachedTraining

        def close(self):
            pass

    with (
        patch.object(host, "clock", FakeClock()),
        patch.object(host, "process_identity", lambda pid: {"pid": pid}),
    ):
        with pytest.raises(ReachedTraining if accepted else ValueError):
            host.execute(
                policy,
                "image",
                ["run", "inputs", "runs/long", "fresh-extended-12h"],
                coordinator_factory=Coordinator,
            )
    if accepted:
        assert read(workspace / "runs/long/admission-completion.json") == record
