"""One frozen OEIS cycle over existing isolated generation/admission/train/score roles.

The host only handles sealed records and explicit projections. CPU subprocesses
are acceptance-only; research uses the existing container supervisor and HIP.
"""

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
    validate,
    record,
    verify_refs,
)
from oeis_learn.generalization.progressive.campaign import validate_campaign, validate_task
from oeis_learn.pipeline_validation.identity import confined


@dataclass(frozen=True)
class _Derived:
    campaign: str
    task: str
    operation: str
    policy: str
    artifacts: tuple


def check_derived(value, *, campaign, task, operation):
    if not isinstance(value, _Derived) or (
        value.campaign != campaign["identity"]
        or value.task != task["identity"]
        or value.operation != operation
        or value.policy != task["payload"]["policy"]
    ):
        raise ValueError("only the frozen host cycle may route its committed outputs")
    return list(value.artifacts)


def _read(root, ref, maximum=256 << 20):
    return json.loads(verify_refs(root, [ref], max_bytes=maximum)[0])


def validate_cycle(root, campaign, plan):
    if campaign.get("track") != "oeis_derived":
        raise ValueError("separate OEIS-derived study required")
    if (
        set(plan)
        != {
            "version",
            "identity",
            "cycle",
            "pool",
            "training",
            "panels",
            "yield_policy",
            "seconds",
            "forced",
            "design",
        }
        or plan["version"] != "progressive-oeis-cycle/v1"
        or seal(plan) != plan
        or type(plan["cycle"]) is not int
        or plan["cycle"] != 0
    ):
        raise ValueError("exactly one frozen OEIS cycle required; a further cycle is a new study")
    from oeis_learn.generalization.progressive.admission import validate_member
    from oeis_learn.generalization.progressive.yield_audit import validate_policy

    validate_policy(plan["yield_policy"])
    pool = _read(root, plan["pool"])
    if not isinstance(pool, list) or not 1 <= len(pool) <= 512:
        raise ValueError("finite outcome-blind pool required")
    if len({m["id"] for m in pool}) != len(pool):
        raise ValueError("duplicate pool member")
    for m in pool:
        validate_member(m)
    if set(plan["seconds"]) != {"generate", "admit-training", "train", "score"} or any(
        type(s) is not int or not 1 <= s <= (86400 if op == "train" else 21600)
        for op, s in plan["seconds"].items()
    ):
        raise ValueError("finite role duration required")
    if plan["forced"] is not None and campaign["purpose"] != "acceptance":
        raise ValueError("forced candidate fixtures are DIAGNOSTIC_ONLY")
    if plan["forced"] is not None and (
        set(_read(root, plan["forced"])) != {"pool", "evaluation"}
        or set(_read(root, plan["forced"])["pool"]) != {m["id"] for m in pool}
    ):
        raise ValueError("diagnostic fixture must bind the original visible roster")
    from oeis_learn.generalization.progressive.cohorts import validate_cohort

    panels = []
    if set(plan["panels"]) != {"transfer", "retention", "exposure"}:
        raise ValueError("separate transfer and retention panels required")
    exposure = _read(root, plan["panels"]["exposure"])
    for kind in ("transfer", "retention"):
        c = _read(root, plan["panels"][kind])
        validate_cohort(c, exposure=exposure, other_cohorts=panels)
        if (
            c["payload"]["role"] != "confirmation"
            or c["track"] != campaign["track"]
            or c["purpose"] != campaign["purpose"]
        ):
            raise ValueError("only frozen independent confirmation panels may promote OEIS replay")
        if any(
            r["id"] in {m["id"] for m in pool} or set(r["groups"]) & {m["group"] for m in pool}
            for r in c["payload"]["rows"]
        ):
            raise ValueError("training feedback is not transfer evidence")
        panels.append(c)
    from oeis_learn.generalization.progressive.cohorts import validate_frozen_panels

    histories = {exposure["identity"]: exposure}
    for item in campaign["payload"]["references"]:
        if item["kind"] == "groups":
            original = _read(root, item["artifact"])
            if original.get("version") == "discovery-exposure/v1":
                histories[original["identity"]] = original
    validate_frozen_panels(root, panels, histories)
    train_windows = {digest(m["values"][i : i + 20]) for m in pool for i in range(81)}
    for panel in panels:
        for row in panel["payload"]["rows"]:
            truth = _read(root, row["truth"])
            if any(digest(truth["values"][i : i + 20]) in train_windows for i in range(81)):
                raise ValueError("training feedback overlaps a transfer/retention window")
    if set(plan["design"]) != {"bootstrap_seed", "comparisons", "maximum_total_us"} or (
        type(plan["design"]["maximum_total_us"]) is not int
        or plan["design"]["maximum_total_us"] <= 0
        or type(plan["design"]["comparisons"]) is not int
        or plan["design"]["comparisons"] < 2
    ):
        raise ValueError("both starting/control comparisons and total cost must be frozen")
    plans = {p["id"]: p for p in campaign["payload"]["tasks"]}
    pairs = {}
    selected_slots = set()
    if not isinstance(plan["training"], list) or len(plan["training"]) != 4:
        raise ValueError("two paired seeds and replay/synthetic arms required")
    ids = set()
    for entry in plan["training"]:
        if (
            set(entry) != {"task", "channel", "model"}
            or entry["task"] not in plans
            or entry["task"] in ids
        ):
            raise ValueError("original distinct task slots required")
        ids.add(entry["task"])
        p = plans[entry["task"]]
        if p["arm"] not in {"replay", "continued_synthetic"} or p["stage"] != "oeis":
            raise ValueError("declared OEIS replay/control task required")
        from .progressive_learner import compatibility, validate_input
        from oeis_learn.generalization.progressive.projections import _validate_channel

        channel = load(confined(root, entry["channel"]), maximum=256 << 20)
        allowed = [
            r["artifact"] for r in campaign["payload"]["references"] if r["role"] == "learner"
        ]
        if (
            reference(root, entry["channel"]) not in allowed
            or channel.get("role") != "learner"
            or seal(channel) != channel
        ):
            raise ValueError("original learner projection required")
        _validate_channel("training_request", channel["data"])
        package = _read(root, channel["data"]["artifact"])
        validate_input(
            package,
            dict(purpose=campaign["purpose"], track=campaign["track"], payload=p),
            allow_pending=True,
        )
        if compatibility(package) != p["initializer"]["compatibility"]:
            raise ValueError("training initializer differs from original frozen package")
        models = [load(confined(root, path)) for path in entry["model"]]
        if len(models) != 2 or {v["kind"] for v in models} != {"checkpoint", "recipe"}:
            raise ValueError("one original model and checkpoint required")
        generator_refs = [
            r["artifact"] for r in campaign["payload"]["references"] if r["role"] == "generator"
        ]
        if any(reference(root, path) not in generator_refs for path in entry["model"]):
            raise ValueError("model channels were not frozen")
        checkpoint = next(v["data"] for v in models if v["kind"] == "checkpoint")
        recipe = next(v["data"] for v in models if v["kind"] == "recipe")
        if _read(root, recipe["artifact"]) != package["profile"]:
            raise ValueError("generation must use the actual learner model profile")
        if p["initializer"]["mode"] == "random":
            if checkpoint.get("control") != "random":
                raise ValueError("random initializer requires its identified random model control")
        elif p["initializer"]["mode"] == "weights_only":
            if checkpoint["artifact"] != p["initializer"]["artifact"]:
                raise ValueError("starting checkpoint differs from paired training initializer")
        else:
            raise ValueError(
                "cycle initializer needs a resolved random/weights-only child or external full-state support"
            )
        selected_slots.add(p["slot"])
        pairs.setdefault(p["seed"], {})[p["arm"]] = p
    if len(selected_slots) != 1:
        raise ValueError("cycle tasks must share one eligible frozen stage slot")
    if len(pairs) != 2 or any(set(v) != {"replay", "continued_synthetic"} for v in pairs.values()):
        raise ValueError("two paired seeds required")
    for pair in pairs.values():
        a, b = pair.values()
        if a["initializer"] != b["initializer"] or a["token_target"] != b["token_target"]:
            raise ValueError(
                "matched declared initialization and charged-token milestones required"
            )
    return pool, panels


class Cycle:
    def __init__(
        self,
        root,
        output,
        campaign,
        manager,
        request,
        journal,
        *,
        host=None,
        runtime=None,
        pythonpath=None,
    ):
        self.root, self.output = Path(root), Path(output)
        self.host_root = self.output / "host"
        self.campaign, self.manager = (
            validate_campaign(campaign),
            validate_task(manager, campaign=campaign),
        )
        self.journal, self.host, self.runtime, self.pythonpath = journal, host, runtime, pythonpath
        expected = self.root / "run-state/discovery-accounting" / campaign["identity"][7:]
        if (
            journal.root.resolve() != expected.resolve()
            or journal.binding["campaign"] != campaign["identity"]
            or journal.binding["caps"] != campaign["payload"]["caps"]
        ):
            raise ValueError("original canonical campaign accounting required")
        if host is None and campaign["purpose"] != "acceptance":
            raise ValueError("research requires isolated container/HIP execution")
        refs = [
            r["artifact"]
            for r in campaign["payload"]["references"]
            if r["kind"] == "policy" and r["artifact"]["sha256"] == manager["payload"]["policy"]
        ]
        if (
            len(refs) != 1
            or _read(root, refs[0]) != request
            or request["operation"] != "oeis-cycle"
            or set(request["artifacts"]) != {"cycle"}
            or len(request["artifacts"]["cycle"]) != 1
        ):
            raise ValueError("original frozen cycle task request required")
        self.request = request
        self.plan = load(confined(root, request["artifacts"]["cycle"][0]), maximum=256 << 20)
        if not any(
            r["artifact"] == reference(root, request["artifacts"]["cycle"][0])
            for r in campaign["payload"]["references"]
        ):
            raise ValueError("cycle policy was not frozen")
        self.pool, self.panels = validate_cycle(root, campaign, self.plan)
        self.tasks = {
            p["id"]: validate_task(
                record(
                    "task",
                    dict(p, campaign=campaign["identity"]),
                    created_at=campaign["created_at"],
                    purpose=campaign["purpose"],
                    track=campaign["track"],
                ),
                campaign=campaign,
            )
            for p in campaign["payload"]["tasks"]
        }
        if manager["payload"]["id"] != self.plan["training"][0]["task"] or any(
            self.tasks[e["task"]]["payload"]["policy"] != manager["payload"]["policy"]
            for e in self.plan["training"]
        ):
            raise ValueError("one frozen owner for every cycle slot required")
        self.ordinal, self.elapsed, self.receipts = 0, 0, []
        self.started = time.monotonic_ns() // 1000
        marker = journal.root / "oeis-cycle.json"
        if marker.exists():
            raise ValueError(
                "cycle already attempted; recovery must use its original cuts, never a second cycle"
            )
        atomic_write(
            marker,
            seal(
                dict(
                    campaign=campaign["identity"],
                    policy=self.plan["identity"],
                    output=str(self.output.relative_to(self.root)),
                )
            ),
        )
        self.overhead_us = min(60_000_000, journal.task_remaining(manager)["elapsed_us"] // 10)
        cost = dict.fromkeys(journal.binding["caps"], 0)
        cost["elapsed_us"] = self.overhead_us
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        charge = journal.reserve(
            manager["payload"]["id"], "oeis-host", 0, cost, boot=boot, monotonic_us=self.started
        )
        journal.commit(charge["key"])  # conservative non-refundable host work reserve

    def channel(self, kind, role, data):
        value = seal(dict(version="discovery-channel/v1", kind=kind, role=role, data=data))
        path = self.host_root / "routed" / (value["identity"][7:] + ".json")
        if not path.exists():
            atomic_write(path, value)
        return str(path.relative_to(self.root))

    def phase(self, operation, artifacts, task=None):
        from .progressive_roles import REGISTRY, invoke
        from oeis_learn.generalization.progressive.projections import describe

        task = task or self.manager
        artifacts = {k: [describe(self.root, p) for p in paths] for k, paths in artifacts.items()}
        refs = tuple(a.ref for values in artifacts.values() for a in values)
        authority = _Derived(
            self.campaign["identity"], task["identity"], operation, task["payload"]["policy"], refs
        )
        target = self.output / f"phase-{self.ordinal:05d}-{operation}"
        self.ordinal += 1
        remaining = self.journal.task_remaining(task)
        seconds = min(
            self.plan["seconds"][operation],
            (remaining["elapsed_us"] - task["payload"]["finalization_us"]) // 1_000_000,
        )
        if seconds <= task["payload"]["startup_us"] / 1_000_000:
            raise TimeoutError("cycle enclosing allowance exhausted")
        if time.monotonic_ns() // 1000 - self.started - self.elapsed > self.overhead_us:
            raise TimeoutError("frozen cycle host work reserve exhausted")
        began = time.monotonic_ns() // 1000
        if self.host is None:
            result = self.cpu_phase(operation, artifacts, target, task, seconds)
        else:
            from oeis_learn.generalization.progressive.budgets import (
                role_work_requirements,
                WORK_ROLES,
            )

            values = [
                load(confined(self.root, a.ref["path"]), maximum=256 << 20)
                for v in artifacts.values()
                for a in v
            ]
            costs = (
                role_work_requirements(operation, values)
                if operation in WORK_ROLES
                else dict.fromkeys(remaining, 0)
            )
            costs["elapsed_us"] = seconds * 1_000_000
            result = invoke(
                self.host,
                self.request["image"],
                self.root,
                operation,
                artifacts,
                output=str(target.relative_to(self.root)),
                campaign=self.campaign,
                task=task,
                journal=self.journal,
                ordinal=100000 + self.ordinal,
                reservation=costs,
                seconds=seconds,
                runtime=self.runtime,
                derived=authority,
            )
        duration = time.monotonic_ns() // 1000 - began
        self.elapsed += duration
        if hasattr(self, "host_guard"):
            from oeis_learn.generalization.progressive.storage import usage

            own_outputs = [
                self.root / load(p)["output"]
                for p in (self.journal.root / "storage-inventory").glob("*.json")
                if load(p)["task"] == self.manager["identity"]
                and load(p)["output"] != str(self.host_root.relative_to(self.root))
            ]
            self.host_guard.prior_bytes, self.host_guard.prior_inodes = usage(own_outputs)
        self.receipts.append(
            dict(
                operation=operation,
                task=task["identity"],
                result=reference(self.root, target / "role/result.json"),
                elapsed_us=duration,
            )
        )
        if (
            operation == "admit-training"
            and result["mechanical_status"] != "COMPLETE"
            and result.get("producer_error", {}).get("code") != "ADMISSION_REJECTED"
        ):
            raise RuntimeError(
                "admission integrity/producer failure: " + str(result.get("producer_error"))
            )
        if result["mechanical_status"] != "COMPLETE" and operation != "admit-training":
            raise RuntimeError(
                "cycle producer failed: " + str(result.get("producer_error") or result["reason"])
            )
        return result, target, duration

    def cpu_phase(self, operation, artifacts, target, task, seconds):
        """Acceptance-only process adapter to the same installed dispatcher."""
        from oeis_learn.generalization.progressive.budgets import role_work_requirements, WORK_ROLES
        from oeis_learn.generalization.progressive.projections import project
        from .progressive_roles import REGISTRY
        from oeis_learn.generalization.progressive.storage import (
            register_invocation,
            owned_guard,
            usage,
        )

        values = [
            load(confined(self.root, a.ref["path"]), maximum=256 << 20)
            for v in artifacts.values()
            for a in v
        ]
        costs = (
            role_work_requirements(operation, values)
            if operation in WORK_ROLES
            else dict.fromkeys(self.journal.binding["caps"], 0)
        )
        costs["elapsed_us"] = seconds * 1_000_000
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        start = time.monotonic_ns() // 1000
        charge = self.journal.reserve(
            task["payload"]["id"],
            operation,
            100000 + self.ordinal,
            costs,
            boot=boot,
            monotonic_us=start,
        )
        state = (
            self.root
            / "run-state/discovery-learning"
            / self.campaign["identity"][7:]
            / task["identity"][7:]
        )
        proof = register_invocation(self.root, self.campaign, task, self.journal.root, target)
        guard = owned_guard(
            dict(campaign=self.campaign, task=task, storage=proof), output=target, state_root=state
        )
        with guard.activate():
            project(
                self.root,
                target / "inputs",
                operation,
                artifacts,
                registry=REGISTRY,
                authorized=[a.ref for v in artifacts.values() for a in v],
            )
        used, inodes = usage([target])
        proof["prior_bytes"] += used + (1 << 20)
        proof["prior_inodes"] += inodes + 64
        if operation == "train":
            authority = seal(
                dict(
                    version="progressive-supervisor/v1",
                    campaign=self.campaign,
                    task=task,
                    clock=dict(
                        reservation_key=charge["key"],
                        boot=boot,
                        start_us=start,
                        end_us=start + costs["elapsed_us"],
                    ),
                    invocation=f"oeis-{self.ordinal}",
                    storage=proof,
                )
            )
        else:
            with self.journal.lock():
                event = next(
                    e
                    for e in self.journal._read()[0]
                    if e["operation"] == charge["key"] and e["event"] == "RESERVE"
                )
            authority = seal(
                dict(
                    version="progressive-role-budget/v1",
                    campaign=self.campaign,
                    task=task,
                    operation=operation,
                    inputs=digest(sorted(values, key=lambda v: v["identity"])),
                    reservation=event,
                    storage=proof,
                )
            )
        atomic_write(target / "authority.json", authority)
        code = """import sys
from pathlib import Path
from oeis_learn.pipeline_validation.progressive_role import execute
from oeis_learn.pipeline_validation.progressive_evidence import failure
p=Path(sys.argv[1])
try:
 r=execute(sys.argv[2],inputs=p/'inputs',output=p/'role',authority=p/'authority.json',accounting=Path(sys.argv[3]),state_root=Path(sys.argv[4]),control=p/'control')
except Exception as exc:
 r=failure(p/'role',sys.argv[2],exc,producer='installed-'+sys.argv[2])
raise SystemExit(0 if r['mechanical_status']=='COMPLETE' else 2 if r['mechanical_status']=='PARTIAL' else 1)
"""
        env = dict(
            os.environ,
            OEIS_DISCOVERY_PURPOSE="acceptance",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        if self.pythonpath:
            env["PYTHONPATH"] = str(self.pythonpath)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(target),
                operation,
                str(self.journal.root),
                str(state),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=seconds,
            check=False,
        )
        self.journal.commit(
            charge["key"],
            boot=boot,
            monotonic_us=time.monotonic_ns() // 1000,
            termination_proven=True,
        )
        atomic_write(
            target / "process.json",
            seal(
                dict(
                    returncode=process.returncode,
                    stdout=process.stdout[-4096:],
                    stderr=process.stderr[-4096:],
                )
            ),
        )
        if not (target / "role/result.json").exists():
            raise RuntimeError(
                "installed cycle role produced no terminal record: " + process.stderr[-4096:]
            )
        return load(target / "role/result.json")

    def evaluate(self, task, model, *, label):
        outcomes = []
        for kind, panel in zip(("transfer", "retention"), self.panels):
            rows = [
                r for r in panel["payload"]["rows"] if kind == "retention" or r["stratum"] == "oeis"
            ]
            if not rows:
                raise ValueError("frozen OEIS/retention support missing")
            visible = [
                self.channel("visible_prompt", "generator", _read(self.root, r["visible"]))
                for r in rows
            ]
            generated, target, _ = self.phase(
                "generate",
                dict(
                    visible=visible,
                    model=model
                    + [
                        self.channel(
                            "oeis_generation",
                            "generator",
                            dict(
                                methods=["correct"],
                                forced={
                                    r["id"]: _read(self.root, self.plan["forced"])["evaluation"][
                                        r["id"]
                                    ]
                                    for r in rows
                                }
                                if self.plan["forced"]
                                else None,
                            ),
                        )
                    ],
                ),
                task,
            )
            candidates = load(target / "role/generation/candidates.json", maximum=512 << 20)
            scorer_inputs = dict(
                candidates=[self.channel("candidate", "scorer", candidates)],
                score_truth=[
                    self.channel("confirmation_truth", "scorer", _read(self.root, r["truth"]))
                    for r in rows
                ],
            )
            _, scored, _ = self.phase("score", scorer_inputs, task)
            score = load(scored / "role/scoring/scores.json", maximum=512 << 20)
            for row in rows:
                outcome = score["outcomes"][row["id"]]["equal_budget"]
                outcomes.append(
                    dict(
                        seed=task["payload"]["seed"],
                        arm=label,
                        stratum="retention" if kind == "retention" else "oeis",
                        group=row["groups"][0],
                        task=row["id"],
                        outcome=outcome["outcome"],
                    )
                )
        return outcomes

    def run(self):
        from oeis_learn.generalization.progressive.archive import Archive
        from oeis_learn.generalization.progressive.replay import freeze_replay, RULE
        from oeis_learn.generalization.progressive.yield_audit import audit
        from .progressive_evidence import publish
        from .progressive_learner import compatibility

        attempts, admissions = [], []
        search_us = admission_us = 0
        initial = self.plan["training"][0]
        forced = _read(self.root, self.plan["forced"])["pool"] if self.plan["forced"] else None
        for offset in range(0, len(self.pool), 128):
            pool = self.pool[offset : offset + 128]
            visible = [
                self.channel(
                    "visible_prompt",
                    "generator",
                    dict(id=m["id"], indices=m["indices"][:20], values=m["values"][:20]),
                )
                for m in pool
            ]
            mode = self.channel(
                "oeis_generation",
                "generator",
                dict(
                    methods=["correct"],
                    forced={m["id"]: forced[m["id"]] for m in pool} if forced is not None else None,
                ),
            )
            _, generated, elapsed = self.phase(
                "generate", dict(visible=visible, model=initial["model"] + [mode])
            )
            search_us += elapsed
            candidates = load(generated / "role/generation/candidates.json", maximum=512 << 20)
            for m in pool:
                chosen = candidates["records"][m["id"]]["selections"]["equal_budget"]
                accepted = None
                reason = "no_visible_candidate"
                source = None
                if chosen:
                    source = chosen["selected"]["source"]
                    candidate = self.channel(
                        "candidate", "admission", dict(task=m["id"], source=source)
                    )
                    truth = self.channel("training_continuation", "admission", m)
                    result, produced, elapsed = self.phase(
                        "admit-training", dict(candidates=[candidate], admission_truth=[truth])
                    )
                    admission_us += elapsed
                    if result["mechanical_status"] == "COMPLETE":
                        accepted = load(produced / "role/admission.json")
                        admissions.append(accepted)
                        reason = "finite_verified"
                    else:
                        reason = result.get("producer_error", {}).get("message") or result["reason"]
                # All eight original portfolio slots remain visible. Only the
                # frozen visible-only choice is submitted to continuation admission.
                for ordinal in range(8):
                    selected = ordinal == 7
                    attempts.append(
                        dict(
                            task=m["id"],
                            ordinal=ordinal,
                            candidate=digest(source) if selected and source else None,
                            outcome="accepted"
                            if selected and accepted
                            else "rejected"
                            if selected
                            else "unresolved",
                            reason=reason[:256]
                            if selected
                            else "not_admitted_by_visible_selection",
                            elapsed_us=0,
                        )
                    )
        package_start = time.monotonic_ns() // 1000
        atomic_write(
            self.host_root / "search-ledger.json",
            seal(dict(attempts=attempts, receipts=self.receipts)),
        )
        package_us = max(1, time.monotonic_ns() // 1000 - package_start)
        report = audit(
            self.pool,
            admissions,
            attempts,
            costs=dict(
                search_us=max(1, search_us), admission_us=admission_us, package_us=package_us
            ),
            selected=self.plan["yield_policy"],
            purpose=self.campaign["purpose"],
            scope="DIAGNOSTIC_ONLY" if forced is not None else "CHECKPOINT_NATIVE",
        )
        atomic_write(self.host_root / "yield-audit.json", report)
        if not report["payload"]["eligible"]:
            return publish(
                self.output,
                operation="oeis-cycle",
                mechanical_status="COMPLETE",
                scientific_status="INVALID"
                if report["payload"]["invalid_admissions"]
                else "INCOMPLETE",
                next_action=report["payload"]["next_action"],
                reason="OEIS_YIELD_DEFICIT",
                campaign=self.campaign["identity"],
                metrics=dict(
                    yield_audit=reference(self.root, self.host_root / "yield-audit.json"),
                    task_status="COMPLETE",
                    completed_tasks=[],
                    autonomous_discoveries=report["payload"]["autonomous_discoveries"],
                ),
            )
        archive = Archive(
            self.host_root / "archive",
            max_bytes=min(self.manager["payload"]["caps"]["archive_bytes"], 512 << 20),
        )
        cut = archive.publish(admissions)
        atomic_write(self.host_root / "archive.json", cut)
        all_rows = {"starting": [], "continued_synthetic": [], "replay": []}
        completed = []
        for entry in self.plan["training"]:
            task = self.tasks[entry["task"]]
            original = load(confined(self.root, entry["channel"]))
            base = _read(self.root, original["data"]["artifact"])
            if base.get("oeis_rules") != RULE:
                raise ValueError("archive update rule was not frozen in the training input")
            replay = freeze_replay(archive, cut, arm=task["payload"]["arm"])
            package = seal(dict(base, oeis_replay=replay))
            if compatibility(package) != task["payload"]["initializer"]["compatibility"]:
                raise ValueError("replay changed the frozen learning compatibility")
            path = self.host_root / "packages" / (task["identity"][7:] + ".json")
            atomic_write(path, package)
            channel = self.channel(
                "training_request",
                "learner",
                dict(original["data"], artifact=reference(self.root, path)),
            )
            trained, _, _ = self.phase("train", dict(training=[channel]), task)
            # Baseline generation/scoring spends this same task's allowances.
            # Initialize learning first; evaluate the original frozen initializer
            # afterwards so inference cannot make a new learner look reopened.
            if task["payload"]["arm"] == "replay":
                all_rows["starting"] += self.evaluate(task, entry["model"], label="baseline")
            state = (
                self.root
                / "run-state/discovery-learning"
                / self.campaign["identity"][7:]
                / task["identity"][7:]
            )
            from oeis_learn.generalization.progressive.recovery import Recovery

            with Recovery(self.journal, task, compatibility(package), state_root=state) as owner:
                restored = owner.restore()
                expected_archive = replay["archive"]["identity"] if replay["archive"] else None
                if restored["states"]["replay"]["archive"] != expected_archive:
                    raise ValueError("checkpoint/archive cut differs")
                full = _read(
                    state / "checkpoints",
                    owner.state["checkpoint"]["payload"]["full_state"],
                    64 << 30,
                )
                weights = full["states"]["weights"]
                verify_refs(state / "checkpoints", [weights], max_bytes=64 << 30)
                checkpoint = self.channel(
                    "checkpoint",
                    "generator",
                    dict(
                        artifact=reference(self.root, state / "checkpoints" / weights["path"]),
                        compatibility=compatibility(package),
                        profile="bigint-body/v1",
                    ),
                )
            recipe = [p for p in entry["model"] if load(confined(self.root, p))["kind"] == "recipe"]
            if len(recipe) != 1:
                raise ValueError("one frozen research model recipe required")
            all_rows[task["payload"]["arm"]] += self.evaluate(
                task,
                [checkpoint, *recipe],
                label="candidate" if task["payload"]["arm"] == "replay" else "baseline",
            )
            completed.append(task["payload"]["id"])
        comparisons = {}
        from oeis_learn.generalization.progressive.metrics import compare
        from oeis_learn.generalization.progressive.decisions import classify

        candidate = all_rows["replay"]
        roster = [
            {k: r[k] for k in ("task", "group", "stratum")}
            for r in candidate
            if r["seed"] == min(r["seed"] for r in candidate)
        ]
        seeds = sorted({r["seed"] for r in candidate})
        for label in ("starting", "continued_synthetic"):
            metrics = compare(
                all_rows[label] + candidate,
                seed=self.plan["design"]["bootstrap_seed"],
                comparisons=self.plan["design"]["comparisons"],
                roster=roster,
                seeds=seeds,
                primary=("oeis",),
            )
            verdict = "INCOMPLETE"
            if metrics["support_status"] == "SUFFICIENT" and all(
                v is not None for v in metrics["retention_losses"].values()
            ):
                verdict = classify(
                    mean_gain=metrics["mean_gain"],
                    seed_effects=list(metrics["seed_effects"].values()),
                    completion_by_arm_seed_stratum=[
                        metrics["cells"][a][s]["oeis"]["completion"]
                        for a in ("baseline", "candidate")
                        for s in seeds
                    ],
                    retention_losses=list(metrics["retention_losses"].values()),
                    minimum_groups=metrics["minimum_groups"],
                    lower=metrics["lower"],
                    upper=metrics["upper"],
                    integrity=True,
                    claim="capability",
                )
            comparisons[label] = dict(metrics=metrics, verdict=verdict)
        total_us = time.monotonic_ns() // 1000 - self.started
        cost_ok = total_us <= self.plan["design"]["maximum_total_us"]
        passed = cost_ok and all(v["verdict"] == "CONFIRMED_GAIN" for v in comparisons.values())
        final = seal(
            dict(
                version="progressive-oeis-cycle-result/v1",
                plan=self.plan["identity"],
                archive=cut["identity"],
                comparisons=comparisons,
                discovery_us=search_us + admission_us,
                cycle_active_us=total_us,
                cost_within_frozen_limit=cost_ok,
                equal_total_compute=False,
                completed_tasks=completed,
                receipts=self.receipts,
                scope="DIAGNOSTIC_ONLY" if forced is not None else "CHECKPOINT_NATIVE",
                autonomous_discoveries=report["payload"]["autonomous_discoveries"],
            )
        )
        atomic_write(self.host_root / "cycle.json", final)
        return publish(
            self.output,
            operation="oeis-cycle",
            mechanical_status="COMPLETE",
            scientific_status="CONFIRMED_GAIN"
            if passed
            else "INCOMPLETE"
            if any(v["verdict"] == "INCOMPLETE" for v in comparisons.values())
            else "INCONCLUSIVE",
            next_action="RETAIN_BASELINE",
            reason="ONE_CYCLE_COMPLETE_NEW_STUDY_REQUIRED",
            campaign=self.campaign["identity"],
            metrics=dict(
                cycle=reference(self.root, self.host_root / "cycle.json"),
                completed_tasks=completed,
                task_status="COMPLETE",
                autonomous_discoveries=final["autonomous_discoveries"],
            ),
        )


def run_cycle(
    root, output, campaign, manager, request, journal, *, host=None, runtime=None, pythonpath=None
):
    cycle = Cycle(
        root,
        output,
        campaign,
        manager,
        request,
        journal,
        host=host,
        runtime=runtime,
        pythonpath=pythonpath,
    )
    from oeis_learn.generalization.progressive.storage import register_invocation, owned_guard

    proof = register_invocation(Path(root), campaign, manager, journal.root, cycle.host_root)
    state = (
        Path(root)
        / "run-state/discovery-learning"
        / campaign["identity"][7:]
        / manager["identity"][7:]
    )
    guard = owned_guard(
        dict(campaign=campaign, task=manager, storage=proof),
        output=cycle.host_root,
        state_root=state,
    )
    guard.roots.append(cycle.output / "result.json")
    cycle.host_guard = guard
    with guard.activate():
        return cycle.run()
