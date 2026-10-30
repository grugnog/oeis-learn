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


def generalization_checkpoint_policy(history, *, pins):
    """Prospective pure prune decision. Historical CheckpointStore remains unchanged."""
    from oeis_learn.generalization.checkpoints import prune_plan
    return prune_plan(history, pins)


class GeneralizationCheckpointStore:
    MAX_BYTES = 32 << 30

    """Prospective full-state generations; latest moves only after verified publication.

    Private state is trusted run-owned data. Hash verification happens before
    deserialization; callers must never use this to load untrusted shared archives.
    """
    def __init__(self, root, contract, cap):
        self.root=Path(root); self.contract=contract; self.cap=cap

    @staticmethod
    def _encode(value):
        import io,torch
        stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()

    @staticmethod
    def _decode(value):
        import io,torch
        return torch.load(io.BytesIO(value),map_location='cpu',weights_only=False)

    def save(self,states,ledger,*,encode=None):
        from oeis_learn.generalization.state import snapshot,STATE_FIELDS
        from oeis_learn.generalization.contracts import verify_artifact
        from oeis_learn.pipeline_validation.identity import seal
        if set(states)!=STATE_FIELDS:raise ValueError('complete prospective state required')
        encode=encode or self._encode
        generation='generalization-'+uuid.uuid4().hex
        folder=self.root/generation;folder.mkdir(parents=True,exist_ok=False)
        def blob(name,value):
            data=encode(value)
            if not isinstance(data,bytes):raise ValueError('state serializer must return bytes')
            import shutil
            used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if used+len(data)>self.MAX_BYTES or len(data)>shutil.disk_usage(self.root).free:
                raise OSError('checkpoint storage reservation exhausted; existing pins retained')
            path=folder/(name+'.pt');write(path,data,immutable=True)
            ref=dict(path=str(path.relative_to(self.root)),sha256=file_hash(path),bytes=len(data))
            verify_artifact(self.root,ref);return ref
        bound={key:blob(key,states[key]) for key in STATE_FIELDS-{'rng','data'}}
        bound['rng']={key:blob('rng-'+key,value) for key,value in states['rng'].items()}
        bound['data']=states['data']
        manifest=snapshot(bound,contract=self.contract,ledger=ledger)
        path=folder/'state.json';write(path,manifest,immutable=True)
        ref=dict(path=str(path.relative_to(self.root)),sha256=file_hash(path),bytes=path.stat().st_size)
        verify_artifact(self.root,ref)
        write(self.root/'generalization-latest.json',ref)
        return ref

    def load(self,ref,*,decode=None):
        import json
        from oeis_learn.generalization.state import restore
        from oeis_learn.generalization.contracts import verify_artifact
        decode=decode or self._decode
        record=json.loads(verify_artifact(self.root,ref))
        result=restore(record,contract=self.contract,cap=self.cap,root=self.root)
        states=result['states']
        for key in states.keys()-{'rng','data'}:states[key]=decode(verify_artifact(self.root,states[key]))
        states['rng']={key:decode(verify_artifact(self.root,value)) for key,value in states['rng'].items()}
        return dict(states=states,ledger=result['ledger'])

    def describe(self,ref,*,initialization,parent=None,pins):
        """Verify a generation and expose its exact weights without unpickling.

        Token exposure describes committed parameters, excluding a reserved but
        unfinished update. The original ledger remains in the full-state file.
        """
        import json
        from oeis_learn.generalization.state import restore
        from oeis_learn.generalization.contracts import verify_artifact,shape
        from oeis_learn.pipeline_validation.identity import seal
        record=json.loads(verify_artifact(self.root,ref))
        restored=restore(record,contract=self.contract,cap=self.cap,root=self.root)
        ledger=restored['ledger']
        descriptor=seal(dict(version='generalization-checkpoint/v1',kind='full_state',
            weights=record['states']['weights'],state=ref,initialization=initialization,
            parent=parent,tokens=ledger.consumed-(ledger.pending[1] if ledger.debited else 0),
            updates=ledger.updates,pins=list(pins),compatibility=self.contract))
        shape('checkpoint',descriptor)
        return descriptor

    def retain(self,history,pins):
        """Verify every retained generation before atomically publishing the pin set.

        Only generations explicitly present in this run's history may be removed;
        incomplete writes and unrelated files are never guessed to be disposable.
        """
        import json,shutil
        from oeis_learn.generalization.checkpoints import prune_plan
        from oeis_learn.generalization.contracts import verify_artifact,artifact_refs
        remove=prune_plan(history,pins)
        for key in pins:
            record=json.loads(verify_artifact(self.root,history[key]))
            for ref in artifact_refs(record):verify_artifact(self.root,ref)
        latest=read(self.root/'generalization-latest.json')
        if latest not in [history[k] for k in pins]:raise ValueError('latest generation must remain pinned')
        write(self.root/'generalization-pins.json',{key:history[key] for key in pins})
        keep={Path(history[k]['path']).parent for k in pins}
        for key in remove:
            relative=Path(history[key]['path']).parent
            if relative in keep:continue
            if len(relative.parts)!=1 or not relative.name.startswith('generalization-'):
                raise ValueError('unowned checkpoint generation')
            # verify_artifact also rejects symlinks before any recursive removal.
            verify_artifact(self.root,history[key])
            shutil.rmtree(self.root/relative)
