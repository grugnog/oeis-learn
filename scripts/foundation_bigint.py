"""Standard-library host orchestration for qualified bigint roles."""

import json
import math
import shutil
import time
import uuid

from oeis_learn.fresh.common import read, write, digest, confined, lock
from oeis_learn.fresh.bigint_state import DurableRun, expose_final, finish_final, child_analysis


class Allowance:
    """An already durably debited role reservation; no refunds on interruption."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = time.monotonic()
        self.previous = {}

    def remaining(self):
        return max(0.0, self.seconds - (time.monotonic() - self.started))

    def record(self, status):
        self.previous = dict(status=status, remaining=self.remaining())


def _copy(destination, values, files=None):
    from foundation_host import copy_view

    return copy_view(destination, values, files)


def resolve_recipe(host, args, job):
    from foundation_local import relative

    output = job / "resolved.json"
    argv = ["bigint-resolve", "--config", relative(args.config), "--output", relative(output)]
    if getattr(args, "seed", None) is not None:
        argv += ["--seed", str(args.seed)]
    host.cpu(*argv, timeout=300)
    return read(output)


def validate_options(args, recipe):
    if getattr(args, "encoder", None) is not None:
        raise ValueError(
            "--encoder is a retained-profile option; use inputs.profile/mask in the bigint recipe"
        )
    if getattr(args, "seed", None) is not None and args.seed != recipe["model"]["seed"]:
        raise ValueError("conflicting seed; existing run initialization is immutable")
    mapping = {
        "max_updates": ("learning", "max_updates"),
        "budget_seconds": ("budgets", "training_seconds"),
        "validation_every_updates": ("learning", "validation_every"),
        "decoder_token_scale": ("model", "token_scale"),
        "decoder_position_scale": ("model", "position_scale"),
        "dropout": ("model", "dropout"),
    }
    explicit = getattr(args, "explicit_options", set())
    for option, (section, key) in mapping.items():
        flag = "--" + option.replace("_", "-")
        value = getattr(args, option, None)
        if flag in explicit and value != recipe[section][key]:
            raise ValueError("conflicting explicit config/" + option)
    if "--view" in explicit and args.view != "diverse":
        raise ValueError("bigint learnability requires diverse32")
    if "--diagnostic-seconds" in explicit:
        raise ValueError(
            "retained diagnostic_seconds is unsupported; use the registered recipe/cost preflight"
        )


def _foundation(recipe, workspace):
    path = recipe["track"]["foundation"]
    if path is None:
        return None, None, None
    root = confined(workspace, path)
    record = read(root / "run.json")
    selected = read(root / "checkpoint-identity.json")["selected"]
    lineage = read(root / "lineage.json")
    files = read(root / "ancestry.json")
    from oeis_learn.fresh.provenance import validate_ancestry

    validate_ancestry(lineage, files)
    if read(root / "status.json")["status"] != "completed":
        raise ValueError("shared foundation must be a completed compatible run")
    for section in ("execution", "inputs"):
        if record["recipe"][section] != recipe[section]:
            raise ValueError("foundation numerical/input profile mismatch")
    for key in (
        "d_model",
        "heads",
        "encoder_layers",
        "decoder_layers",
        "token_scale",
        "position_scale",
        "dropout",
        "attention",
    ):
        if record["recipe"]["model"][key] != recipe["model"][key]:
            raise ValueError("foundation model/codec mismatch")
    if recipe["track"]["kind"] == "generic_only" and lineage["track"] != "generic_only":
        raise ValueError("assisted ancestry cannot enter strict recipient")
    return root, record, dict(selected=selected, lineage=lineage, ancestry=files)


def execute(host, args, *, coordinator_factory=None):
    import signal

    previous = signal.getsignal(signal.SIGTERM)

    def cancelled(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cancelled)
    try:
        return _execute(host, args, coordinator_factory=coordinator_factory)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _execute(host, args, *, coordinator_factory=None):
    from foundation_local import ROOT, relative, state_path
    from foundation_host import Coordinator
    from oeis_learn.fresh.bigint_inputs import check
    from oeis_learn.fresh.cost_policy import screen_cost, generation_seconds
    from oeis_learn.fresh.bigint_bundle import pack
    from oeis_learn.fresh.provenance import (
        root_lineage,
        descendant,
        validate_assistance,
        fresh_cohort,
    )

    coordinator_factory = coordinator_factory or Coordinator
    operation = args.command
    if operation == "compare":
        out = ROOT / relative(args.output)
        host.cpu(
            "bigint-compare", "--runs", *[relative(p) for p in args.runs], "--output", str(out)
        )
        return
    destination = ROOT / relative(args.run)
    job = state_path(args.run)
    job.mkdir(parents=True, exist_ok=True)
    token = digest([str(ROOT), relative(args.run), "bigint"])[7:39]
    policy = dict(host.policy, run_token=token)
    saved_record = read(destination / "run.json") if (destination / "run.json").exists() else None
    pinned = operation == "resume" or getattr(args, "split", None) == "final"
    image = host.engine.image(saved_record["image"]) if pinned and saved_record else host.ensure()
    if pinned and args.image and host.engine.image(args.image) != image:
        raise ValueError("resume/final image substitution prohibited")
    host.select_gpu()
    policy["render_node"] = host.policy["render_node"]
    with host.gpu_lock(), lock(job / "controller.lock"):
        host.engine.cleanup(token)
        coordinator = None
        run = None
        active = None
        analysis_ledger = None
        analysis = None
        try:
            if operation in {"train", "learnability"}:
                if not args.config:
                    raise ValueError("bigint training requires --config")
                recipe = resolve_recipe(host, args, job)
                validate_options(args, recipe)
                if operation == "learnability":
                    for name in ("max_updates", "budget_seconds", "view"):
                        value = getattr(args, name, None)
                        if name == "view" and value not in (None, "diverse"):
                            raise ValueError("bigint learnability requires diverse32")
                        expected = (
                            recipe["learning"]["max_updates"]
                            if name == "max_updates"
                            else recipe["budgets"]["training_seconds"]
                        )
                        if name != "view" and value is not None and value != expected:
                            raise ValueError("conflicting explicit config/" + name)
                inputs = ROOT / relative(args.inputs)
                manifest = check(inputs)
                if manifest["config"]["data"] != recipe["data"]:
                    raise ValueError("prepared data recipe mismatch")
                if (
                    recipe["track"]["kind"] == "generic_only" and manifest.get("assisted_count", 0)
                ) or (
                    recipe["track"]["kind"] == "assisted"
                    and (
                        not manifest.get("assisted_count", 0)
                        or manifest["config"]["track"]["assisted_manifest"]
                        != recipe["track"]["assisted_manifest"]
                    )
                ):
                    raise ValueError("prepared assistance track/manifest mismatch")
                foundation, foundation_record, parent = _foundation(recipe, ROOT)
                if parent:
                    for cohort in manifest["cohorts"].values():
                        fresh_cohort(cohort["groups"], parent["lineage"])
                if recipe["track"]["kind"] == "assisted":
                    assistance = read(ROOT / relative(recipe["track"]["assisted_manifest"]))[
                        "manifest"
                    ]
                    if (
                        validate_assistance(assistance)
                        != read(inputs / "assistance.json")["manifest_sha256"]
                    ):
                        raise ValueError("reviewed assistance changed since preparation")
                cost = recipe["learning"]["cost_preflight"]
                if cost is None:
                    raise ValueError(
                        "measured complete-screen cost_preflight required before training; run Phase 5 preflight"
                    )
                if (
                    cost["image"] != image
                    or cost["profile_sha256"]
                    != digest(
                        {
                            k: recipe[k]
                            for k in ("execution", "inputs", "model", "fitting", "search")
                        }
                    )
                    or cost["scoring_data"] is not False
                ):
                    raise ValueError("cost preflight image/profile or non-scoring fixture mismatch")
                forecast = screen_cost(
                    **cost["costs"],
                    updates=recipe["learning"]["max_updates"],
                    cadence=recipe["learning"]["validation_every"],
                    budget=recipe["budgets"]["training_seconds"],
                )
                if not forecast["within_budget"]:
                    raise ValueError(
                        "complete-screen projection exceeds80%; preregister revised recipe"
                    )
                probe_seconds = 30
                probe = coordinator_factory(job / "roles", policy, image, Allowance(probe_seconds))
                try:
                    hardware = probe.run(
                        "bigint-probe",
                        "probe",
                        _copy(job / "probe-input", {"settings.json": dict(device="cuda")}),
                        job / "roles/probe",
                        probe_seconds - 5,
                    )
                finally:
                    probe.close()
                run = DurableRun.create(
                    destination,
                    recipe,
                    manifest,
                    image=image,
                    device="cuda",
                    hardware=hardware,
                    foundation_sha256=parent["selected"]["sha256"] if parent else None,
                )
                write(
                    destination / "input-manifest.json",
                    {
                        k: manifest[k]
                        for k in ("identity", "split_sha256", "counts", "cohorts", "exclusions")
                    },
                    immutable=True,
                )
                write(job / "source.json", dict(inputs=relative(args.inputs)), immutable=True)
                write(destination / "cost-projection.json", forecast, immutable=True)
            else:
                saved = read(destination / "run.json")
                recipe = saved["recipe"]
                inputs = ROOT / read(job / "source.json")["inputs"]
                manifest = check(inputs)
                if (
                    saved["version"] != "bigint-run/v1"
                    or operation == "resume"
                    and saved["image"] != image
                ):
                    raise ValueError("run/image mismatch")
                validate_options(args, recipe)
                foundation, foundation_record, parent = _foundation(recipe, ROOT)
                if args.config:
                    requested = resolve_recipe(host, args, job)
                    if requested != recipe:
                        raise ValueError(
                            "analysis recipe differs from parent; numerical/profile changes require an explicit new experiment"
                        )
                run = (
                    DurableRun.resume(destination, recipe, image=image)
                    if operation == "resume"
                    else DurableRun(
                        destination,
                        saved,
                        __import__(
                            "oeis_learn.synthesis.state", fromlist=["BudgetLedger"]
                        ).BudgetLedger.restore(read(destination / "budget.json")),
                    )
                )
            write(
                job / "active.json",
                __import__("foundation_host").process_identity(__import__("os").getpid()),
            )
            examples = read(inputs / "examples.json")
            view = "concrete" if recipe["learning"]["target_view"] == "literal" else "parameterized"
            learner = examples["views"][view]
            if operation == "learnability":
                wanted = set(examples["tiny_ids"])
                learner = dict(
                    learner, training=[r for r in learner["training"] if r["id"] in wanted]
                )
            role_recipe = recipe

            def settings(seconds, **extra):
                value = dict(
                    version="bigint-role/v1",
                    recipe=role_recipe,
                    contract=run.record["contract"],
                    contract_sha256=run.contract.sha256,
                    device="cuda",
                    seconds=max(0.01, seconds - 5),
                    ledger=run.ledger.snapshot(),
                )
                value.update(extra)
                return value

            def launch(name, role, values, seconds, files=None, *, training=False):
                nonlocal coordinator, active
                seconds = min(
                    float(seconds), run.ledger.remaining if training else analysis_ledger.remaining
                )
                if seconds <= 5:
                    raise TimeoutError("insufficient remaining allowance including cleanup")
                active = "role-" + uuid.uuid4().hex
                if training:
                    run.start(active, seconds)
                else:
                    analysis_ledger.start(active, seconds)
                    write(analysis / "budget.json", analysis_ledger.snapshot())
                values = dict(values)
                values["settings.json"] = dict(
                    values.get("settings.json", settings(seconds)),
                    seconds=max(0.01, seconds - 5),
                    ledger=run.ledger.snapshot(),
                )
                source = _copy(job / "inputs" / active, values, files)
                out = job / "roles" / active
                coordinator = coordinator_factory(job / "roles", policy, image, Allowance(seconds))
                try:
                    result = coordinator.run("bigint-" + name, role, source, out, seconds - 5)
                    if training:
                        run.finish(active, "completed")
                    else:
                        analysis_ledger.finish(active, "completed")
                        write(analysis / "budget.json", analysis_ledger.snapshot())
                    return result, out
                except BaseException:
                    if training:
                        run.finish(active, "interrupted")
                    else:
                        analysis_ledger.finish(active, "interrupted")
                        write(analysis / "budget.json", analysis_ledger.snapshot())
                    raise
                finally:
                    coordinator.close()
                    coordinator = None
                    active = None

            def checkpoint_files():
                refs = {}
                for path in (destination / "checkpoints").rglob("*"):
                    if path.is_file():
                        refs[
                            "checkpoints/" + str(path.relative_to(destination / "checkpoints"))
                        ] = path
                if (destination / "selected.json").exists():
                    refs["selected.json"] = destination / "selected.json"
                return refs

            if operation in {"train", "learnability", "resume"}:
                interval = recipe["learning"]["validation_every"]
                cost = recipe["learning"]["cost_preflight"]["costs"]
                while True:
                    progress = (
                        read(destination / "checkpoint-identity.json")["update"]
                        if (destination / "checkpoint-identity.json").exists()
                        else 0
                    )
                    if progress >= recipe["learning"]["max_updates"]:
                        break
                    count = min(interval, recipe["learning"]["max_updates"] - progress)
                    seconds = (
                        math.ceil(
                            (
                                count * cost["training_p95"]
                                + 256 * cost["teacher_forcing_p95"]
                                + generation_seconds(96, cost)
                                + 96 * (cost["fit_p95"] + cost["reference_p95"])
                                + cost["checkpoint_p95"]
                            )
                            / 0.8
                        )
                        + 5
                    )
                    files = checkpoint_files()
                    if foundation and progress == 0:
                        for path in (foundation / "checkpoints").rglob("*"):
                            if path.is_file():
                                files[
                                    "foundation/checkpoints/"
                                    + str(path.relative_to(foundation / "checkpoints"))
                                ] = path
                    result, out = launch(
                        "train",
                        "learner",
                        {
                            "settings.json": settings(
                                seconds,
                                examples_sha256=digest(learner),
                                update_limit=count,
                                foundation=dict(
                                    record=foundation_record, selected=parent["selected"]
                                )
                                if parent
                                else None,
                            ),
                            "examples.json": learner,
                        },
                        seconds,
                        files,
                        training=True,
                    )
                    for folder in (out / "checkpoints").glob("state-*"):
                        target = destination / "checkpoints" / folder.name
                        if not target.exists():
                            shutil.copytree(folder, target)
                    write(
                        destination / "checkpoints/latest.json",
                        read(out / "checkpoints/latest.json"),
                    )
                    for name in ("selected.json", "checkpoint-identity.json"):
                        shutil.copyfile(out / name, destination / name)
                    if result["update"] <= progress:
                        raise TimeoutError(
                            "training made no update before reserved allowance expired"
                        )
                    if result["early_stop"]:
                        break
                selected = read(destination / "checkpoint-identity.json")["selected"]
                spent = run.contract.budget - run.ledger.remaining
                if parent:
                    assistance = None
                    if recipe["track"]["assisted_manifest"]:
                        assistance = read(ROOT / relative(recipe["track"]["assisted_manifest"]))[
                            "manifest"
                        ]
                        validate_assistance(assistance)
                    lineage = descendant(
                        parent["lineage"],
                        track=recipe["track"]["kind"],
                        cost=spent,
                        manifest=assistance,
                    )
                    ancestry = dict(
                        parent["ancestry"], **{parent["lineage"]["sha256"]: parent["lineage"]}
                    )
                else:
                    lineage = root_lineage(
                        seed=recipe["model"]["seed"],
                        cost=spent,
                        checkpoint_sha256=selected["sha256"],
                        recipe_sha256=recipe["identity"],
                    )
                    ancestry = {}
                write(destination / "lineage.json", lineage)
                write(destination / "ancestry.json", ancestry)
            selected = read(destination / "checkpoint-identity.json")["selected"]
            role_recipe = __import__("copy").deepcopy(recipe)
            if getattr(args, "fitting", None) == "raw":
                role_recipe["fitting"]["disabled"] = True
            if operation == "diagnose":
                masks = (
                    [
                        None,
                        *[
                            dict(recipe["inputs"]["mask"], **{group: False})
                            for group in ("magnitude", "modular", "difference", "exact_bytes")
                        ],
                    ]
                    if args.ablate_inputs == "all"
                    else [None]
                )
            else:
                masks = [None]
            analysis = (
                (ROOT / relative(args.output))
                if getattr(args, "output", None)
                else destination / "analysis"
                if operation in {"evaluate", "diagnose"}
                else destination
            )
            if operation in {"evaluate", "diagnose"}:
                child_analysis(
                    analysis,
                    run.contract.sha256,
                    selected["sha256"],
                    digest(
                        dict(
                            image=image,
                            fitting=args.fitting or "paired",
                            ablations=masks,
                            split=getattr(args, "split", "development"),
                        )
                    ),
                )
            if operation in {"evaluate", "diagnose"}:
                from oeis_learn.synthesis.state import BudgetLedger

                if (analysis / "budget.json").exists():
                    raise ValueError(
                        "analysis output already has debits; use an explicitly new child output"
                    )
                analysis_ledger = BudgetLedger(recipe["budgets"]["training_seconds"])
                write(
                    analysis / "run.json",
                    dict(run.record, parent=run.contract.sha256, image=image, analysis=True),
                    immutable=True,
                )
                if (destination / "lineage.json").exists():
                    write(
                        analysis / "lineage.json",
                        read(destination / "lineage.json"),
                        immutable=True,
                    )
                write(
                    analysis / "checkpoint-identity.json",
                    read(destination / "checkpoint-identity.json"),
                    immutable=True,
                )
                write(analysis / "status.json", dict(status="active", final_access="sealed"))
            summaries = {}
            observations = {}
            final_arm = None
            plan = None
            try:
                final = getattr(args, "split", None) == "final"
                if final:
                    if (
                        getattr(args, "fitting", "paired") == "raw"
                        and not recipe["fitting"]["disabled"]
                    ):
                        raise ValueError("raw final arm requires a separately frozen raw recipe")
                    if operation != "evaluate":
                        raise ValueError("only explicit evaluate may expose final cohort")
                    if not args.final_plan:
                        raise ValueError("explicit matching --final-plan required")
                    plan = read(ROOT / relative(args.final_plan))
                    if plan["denominator"] != manifest["cohorts"]["final"]["count"]:
                        raise ValueError("final plan denominator mismatch")
                    shared = (
                        ROOT
                        / "run-state/final-study"
                        / manifest["cohorts"]["final"]["identity"][7:]
                    )
                    final_arm, exposure = expose_final(
                        shared,
                        plan,
                        cohort=manifest["cohorts"]["final"]["identity"],
                        checkpoint=selected["sha256"],
                        recipe=recipe["identity"],
                        portfolio=digest(recipe["search"]),
                        runtime=run.contract.runtime,
                        seed=recipe["model"]["seed"],
                        debit=min(
                            plan["budget"],
                            recipe["search"]["seconds"] * manifest["cohorts"]["final"]["count"]
                            + recipe["evaluation"]["hidden_seconds"]
                            * manifest["cohorts"]["final"]["count"],
                        ),
                    )
                    write(analysis / "final-study.json", exposure)
                    write(analysis / "final-plan.json", plan, immutable=True)
                    from oeis_learn.synthesis.state import BudgetLedger

                    final_debit = list(exposure["ledger"]["records"].values())[-1]["debit"]
                    analysis_ledger = BudgetLedger(final_debit)
                    groups = {"final": read(inputs / "private/oeis-final-prompts.json")}
                    truths = {"final": read(inputs / "private/oeis-final.json")}
                else:
                    groups = {k: v for k, v in learner.items() if k != "training"}
                    alltruth = read(inputs / "private/synthetic-truth.json")
                    truths = {
                        k: {r["id"]: alltruth[r["id"]] for r in rows} for k, rows in groups.items()
                    }
                    if operation == "learnability":
                        groups = {"tiny": learner["training"]}
                        truths = {"tiny": {r["id"]: alltruth[r["id"]] for r in groups["tiny"]}}
                archive_remaining = recipe["archive"]["max_bytes"]
                for mask_number, mask in enumerate(masks):
                    for split, rows in groups.items():
                        # End-of-screen always includes every128+128 development row.
                        if operation in {"evaluate", "diagnose"}:
                            groups_left = (len(masks) - mask_number) * len(groups) - list(
                                groups
                            ).index(split)
                            portion = analysis_ledger.remaining / max(1, groups_left)
                            seconds = portion * 0.8
                        else:
                            cost = recipe["learning"]["cost_preflight"]["costs"]
                            seconds = (
                                math.ceil(
                                    (
                                        32 * generation_seconds(len(rows), cost)
                                        + len(rows) * 40 * (cost["fit_p95"] + cost["reference_p95"])
                                    )
                                    / 0.8
                                )
                                + 5
                            )
                        result, out = launch(
                            "generate",
                            "generation",
                            {
                                "settings.json": settings(
                                    seconds,
                                    checkpoint=selected,
                                    archive_remaining=archive_remaining,
                                    ablation=mask,
                                    compare_cache=operation == "diagnose",
                                    final_access="exposed" if final else "sealed",
                                ),
                                "prompts.json": [generation_prompt(row) for row in rows],
                            },
                            seconds,
                            checkpoint_files(),
                            training=operation in {"train", "learnability", "resume"},
                        )
                        if result["archive"]["bytes"] > archive_remaining:
                            raise ValueError("aggregate archive cap exceeded")
                        archive_remaining -= result["archive"]["bytes"]
                        files = {name: out / name for name in result["files"]}
                        refs = result["files"]
                        score_seconds = (
                            portion * 0.2
                            if operation in {"evaluate", "diagnose"}
                            else math.ceil(len(rows) * 6 * cost["hidden_scoring_p95"] / 0.8) + 5
                        )
                        scored, score_out = launch(
                            "score",
                            "scorer",
                            {
                                "settings.json": settings(score_seconds, device="cpu"),
                                "truth.json": truths[split],
                                "selections.json": refs,
                            },
                            score_seconds,
                            files,
                            training=operation in {"train", "learnability", "resume"},
                        )
                        scored["generation_metrics"] = result["generation_metrics"]
                        scored["archive"] = dict(
                            result["archive"],
                            role_output=str(out),
                            manifest_sha256=__import__(
                                "oeis_learn.fresh.common", fromlist=["file_hash"]
                            ).file_hash(out / "archive-manifest.json"),
                        )
                        scored["parameter_counts"] = result["counts"]
                        scored["detail_omissions"] = result.get("omitted", [])
                        summaries[f"{mask_number}:{split}"] = scored
                        byid = {r["id"]: r for r in rows}
                        observations[f"{mask_number}:{split}"] = {
                            key: dict(family=byid[key].get("family", key), outcomes=value)
                            for key, value in read(score_out / "outcomes.json").items()
                        }
                        for p in (out / "details").glob("*.json"):
                            target = analysis / "details" / f"{mask_number}-{split}-{p.name}"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(p, target)
                if final:
                    write(
                        analysis / "final-study.json",
                        finish_final(shared, plan, final_arm, "completed"),
                    )
            except BaseException:
                if final_arm:
                    write(
                        analysis / "final-study.json",
                        finish_final(shared, plan, final_arm, "interrupted"),
                    )
                raise
            if operation in {"train", "learnability", "resume"}:
                spent = run.contract.budget - run.ledger.remaining
                if parent:
                    lineage = descendant(
                        parent["lineage"],
                        track=recipe["track"]["kind"],
                        cost=spent,
                        manifest=read(ROOT / relative(recipe["track"]["assisted_manifest"]))[
                            "manifest"
                        ]
                        if recipe["track"]["assisted_manifest"]
                        else None,
                    )
                else:
                    lineage = root_lineage(
                        seed=recipe["model"]["seed"],
                        cost=spent,
                        checkpoint_sha256=selected["sha256"],
                        recipe_sha256=recipe["identity"],
                    )
                write(destination / "lineage.json", lineage)
            summary = dict(
                version="bigint-summary/v1",
                splits=summaries,
                final_access="exposed" if getattr(args, "split", None) == "final" else "sealed",
                automatic_run_authorized=False,
                post_training_ablation=operation == "diagnose",
                retraining_claim=False,
            )
            if operation == "learnability":
                from oeis_learn.fresh.comparison import memorization_gate

                tiny = summaries["0:tiny"]
                rescue = tiny["splits"]["rescue"]
                summary["memorization_gate"] = memorization_gate(
                    rescue["success"],
                    rescue["complete"],
                    termination_reliable=tiny["generation_metrics"]["greedy_terminated"] == 32,
                )
            if (destination / "lineage.json").exists():
                summary["ancestry"] = read(destination / "lineage.json")
            target = analysis if operation in {"evaluate", "diagnose"} else destination
            write(
                target / "comparison.json",
                dict(
                    seed=recipe["model"]["seed"],
                    recipe=recipe["identity"],
                    splits=summaries,
                    track=recipe["track"]["kind"],
                    data=manifest["identity"],
                    observations=observations,
                ),
            )
            write(target / "summary.json", summary)
            if operation in {"evaluate", "diagnose"}:
                write(
                    target / "status.json",
                    dict(status="completed", final_access=summary["final_access"]),
                )
            if operation in {"train", "learnability", "resume"}:
                run.finalize("completed", summary)
            print(
                json.dumps(
                    dict(status="PASS", output=str(target), final_access=summary["final_access"])
                )
            )
        except BaseException:
            if (
                analysis is not None
                and operation in {"evaluate", "diagnose"}
                and (analysis / "run.json").exists()
            ):
                write(
                    analysis / "status.json",
                    dict(
                        status="interrupted",
                        final_access="exposed"
                        if (analysis / "final-study.json").exists()
                        else "sealed",
                    ),
                )
                write(
                    analysis / "summary.json",
                    dict(reason="interrupted; inspect role logs", roles=str(job / "roles")),
                )
                if not (analysis / "comparison.json").exists():
                    write(
                        analysis / "comparison.json",
                        dict(status="unavailable", reason="interrupted"),
                    )
            if run is not None and operation in {"train", "learnability", "resume"}:
                run.finalize(
                    "interrupted",
                    dict(reason="interrupted; inspect role logs", roles=str(job / "roles")),
                )
            raise
        finally:
            if coordinator is not None:
                coordinator.close()
            host.engine.cleanup(token)
            if (
                analysis is not None
                and operation in {"evaluate", "diagnose"}
                and (analysis / "summary.json").exists()
            ):
                pack(
                    analysis,
                    ROOT / "tmp-analysis" / (analysis.name + ".zip"),
                    max_bytes=recipe["reporting"]["bundle_bytes"],
                )
            if run is not None and operation in {"train", "learnability", "resume"}:
                for name in ("checkpoint-identity.json", "comparison.json"):
                    if not (destination / name).exists():
                        write(
                            destination / name,
                            dict(status="unavailable", reason="run interrupted before this stage"),
                        )
                pack(
                    destination,
                    ROOT / "tmp-analysis" / (destination.name + ".zip"),
                    max_bytes=recipe["reporting"]["bundle_bytes"],
                )


def generation_prompt(row):
    from oeis_learn.fresh.bigint_diagnostics import strata

    return dict(
        id=row["id"],
        indices=row["indices"],
        values=row["values"],
        horizon_indices=row["horizon_indices"],
        family=row.get("family", row["id"]),
        strata=strata(row),
    )
