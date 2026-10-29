"""Durable wiring for qualified contracts, debits, checkpoints and final exposure."""

from dataclasses import asdict
from pathlib import Path
import uuid
from .common import read, write, file_hash, confined, lock
from oeis_learn.synthesis.contracts import identity
from oeis_learn.synthesis.state import (
    ExperimentContract,
    BudgetLedger,
    validate_checkpoint,
    FinalPlan,
    FinalStudy,
    analysis_identity,
)


class DurableRun:
    def __init__(self, root, record, ledger):
        self.root = Path(root)
        self.record = record
        self.contract = ExperimentContract(**record["contract"])
        self.ledger = ledger

    @classmethod
    def create(
        cls, root, recipe, manifest, *, image, device, foundation_sha256=None, hardware=None
    ):
        root = Path(root)
        if (root / "run.json").exists():
            raise ValueError("existing run requires exact-contract resume")
        runtime = recipe["execution"]["artifacts"]
        contract = ExperimentContract(
            data=manifest["identity"],
            split=manifest["split_sha256"],
            model=identity(recipe["model"]),
            inputs=identity(recipe["inputs"]),
            decoder=identity(
                {
                    k: recipe["model"][k]
                    for k in ("token_scale", "position_scale", "dropout", "attention")
                }
            ),
            solver=identity(recipe["fitting"]),
            runtime=identity(dict(image=image, artifacts=runtime, hardware=hardware)),
            resources=identity(recipe["execution"]["resources"]),
            policy=recipe["identity"],
            track=recipe["track"]["kind"],
            foundation=foundation_sha256,
            seed=recipe["model"]["seed"],
            budget=recipe["budgets"]["training_seconds"],
        )
        record = dict(
            version="bigint-run/v1",
            contract=asdict(contract),
            contract_sha256=contract.sha256,
            recipe=recipe,
            image=image,
            device=device,
            hardware=hardware,
            inputs_identity=manifest["identity"],
            split_sha256=manifest["split_sha256"],
            final_access="sealed",
        )
        write(root / "run.json", record, immutable=True)
        ledger = BudgetLedger(contract.budget)
        run = cls(root, record, ledger)
        run.persist("prepared")
        return run

    @classmethod
    def resume(cls, root, recipe, *, image):
        root = Path(root)
        record = read(root / "run.json")
        status = read(root / "status.json")
        if record["image"] != image:
            raise ValueError("resume image mismatch")
        if record["recipe"] != recipe:
            raise ValueError("resume requires identical recipe/contract")
        if status["status"] == "completed":
            raise ValueError("finalized runs cannot resume training")
        ledger = BudgetLedger.restore(read(root / "budget.json"))
        run = cls(root, record, ledger)
        if run.contract.sha256 != record["contract_sha256"] or ledger.budget != run.contract.budget:
            raise ValueError("changed run identity or cumulative budget")
        # Missing completion remains charged. Reconcile active debits conservatively.
        for key, row in list(ledger.records.items()):
            if row["status"] == "active":
                ledger.finish(key, "interrupted")
        run.persist("interrupted")
        return run

    def persist(self, status):
        write(self.root / "budget.json", self.ledger.snapshot())
        write(
            self.root / "status.json",
            dict(
                version="bigint-status/v1",
                status=status,
                contract_sha256=self.contract.sha256,
                remaining_seconds=self.ledger.remaining,
                final_access="sealed",
            ),
        )

    def start(self, key, debit):
        self.ledger.start(key, debit)
        self.persist("active")

    def finish(self, key, status):
        self.ledger.finish(key, status)
        self.persist("active" if status == "completed" else status)

    def finalize(self, status, summary):
        write(
            self.root / "summary.json",
            dict(summary, contract_sha256=self.contract.sha256, final_access="sealed"),
        )
        self.persist(status)


class CheckpointStore:
    STATES = ("weights", "optimizer", "rng", "data_order", "selection", "curriculum", "replay")

    def __init__(self, root, contract, *, policies=None):
        self.root = Path(root)
        self.contract = contract
        self.policies = policies

    def _boundary(self, selection):
        if self.policies is None:
            return
        from oeis_learn.pipeline_validation.identity import seal
        boundary = selection.get('pipeline_boundary')
        if (not isinstance(boundary, dict) or boundary.get('version') != 'pipeline-boundary/v1'
            or seal(boundary) != boundary or boundary.get('policies') != self.policies
            or boundary.get('active') != 0 or boundary.get('reserved') != 0):
            raise ValueError('changed policy or unaccounted candidate work at checkpoint boundary')

    def save(self, states, ledger, *, state="active"):
        import io
        import torch

        if set(states) != set(self.STATES):
            raise ValueError("complete checkpoint required")
        self._boundary(states['selection'])
        generation = "state-" + uuid.uuid4().hex
        folder = self.root / generation
        files = {}
        for key in self.STATES:
            stream = io.BytesIO()
            torch.save(states[key], stream)
            write(folder / (key + ".pt"), stream.getvalue())
            files[key] = dict(
                path=generation + "/" + key + ".pt", sha256=file_hash(folder / (key + ".pt"))
            )
        binding = dict(
            contract=self.contract.sha256,
            ledger=ledger.snapshot(),
            state=state,
            **{key: files[key]["sha256"] for key in self.STATES},
        )
        validate_checkpoint(binding, self.contract)
        manifest = dict(version="bigint-checkpoint/v1", binding=binding, files=files)
        write(folder / "checkpoint.json", manifest, immutable=True)
        pointer = dict(
            path=generation + "/checkpoint.json", sha256=file_hash(folder / "checkpoint.json")
        )
        write(self.root / "latest.json", pointer)
        return pointer

    def load(self, pointer):
        import torch

        path = confined(self.root, pointer["path"])
        if file_hash(path) != pointer["sha256"]:
            raise ValueError("checkpoint artifact identity mismatch")
        value = read(path)
        ledger = validate_checkpoint(value["binding"], self.contract)
        states = {}
        for key in self.STATES:
            ref = value["files"][key]
            file = confined(self.root, ref["path"])
            if file_hash(file) != ref["sha256"] or ref["sha256"] != value["binding"][key]:
                raise ValueError("checkpoint state artifact mismatch")
            states[key] = torch.load(file, map_location="cpu", weights_only=False)
        self._boundary(states['selection'])
        return dict(states=states, ledger=ledger)

    def prune(self, selected=None):
        import shutil

        retained = {read(self.root / "latest.json")["path"].split("/")[0]}
        if selected:
            retained.add(selected["path"].split("/")[0])
        for directory in self.root.glob("state-*"):
            if directory.name not in retained:
                shutil.rmtree(directory)


def child_analysis(root, parent, checkpoint, analysis):
    record = dict(
        version="bigint-analysis/v1",
        identity=analysis_identity(parent, checkpoint, analysis),
        parent=parent,
        checkpoint=checkpoint,
        analysis=analysis,
    )
    write(Path(root) / "analysis.json", record, immutable=True)
    return record


def final_plan(value):
    value = dict(value)
    if value.pop("version", None) != "final-evaluation/v1":
        raise ValueError("final-evaluation/v1 plan required")
    decisions = value.pop("decision_manifest", None)
    if not isinstance(decisions, dict) or set(decisions) != set(FinalPlan.DECISIONS):
        raise ValueError("completed decision manifest required")
    for name, sha in value["decisions"]:
        record = decisions[name]
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"completed", "not_planned"}
            or identity(record) != sha
        ):
            raise ValueError("incomplete or changed decision record")
    value["decisions"] = tuple(tuple(x) for x in value["decisions"])
    value["arms"] = tuple(tuple(x) for x in value["arms"])
    value["controls"] = tuple(value["controls"])
    return FinalPlan(**value)


def expose_final(root, plan_record, *, cohort, checkpoint, recipe, portfolio, runtime, seed, debit):
    """Host persists shared exposure/debit before mounting any final-prompt input."""
    root = Path(root)
    plan = final_plan(plan_record)
    if plan.cohort != cohort:
        raise ValueError("final cohort mismatch")
    matches = [
        arm for arm in plan.arms if arm[1:6] == (seed, checkpoint, recipe, portfolio, runtime)
    ]
    if len(matches) != 1:
        raise ValueError("unregistered/ambiguous final arm or changed checkpoint")
    with lock(root / "study.lock"):
        write(root / "plan.json", plan_record, immutable=True)
        study = (
            FinalStudy.restore(read(root / "state.json"), plan)
            if (root / "state.json").exists()
            else FinalStudy(cohort)
        )
        study.authorize(plan)
        # Reconcile an interrupted owner, preserving all reservations and exposure.
        for arm in list(study.active):
            study.finish(arm, "interrupted")
        study.start(matches[0][0], checkpoint, debit, operation="evaluate_final")
        write(root / "state.json", study.snapshot())
    return matches[0][0], study.snapshot()


def finish_final(root, plan_record, arm, status):
    root = Path(root)
    plan = final_plan(plan_record)
    with lock(root / "study.lock"):
        study = FinalStudy.restore(read(root / "state.json"), plan)
        study.finish(arm, status)
        write(root / "state.json", study.snapshot())
    return study.snapshot()
