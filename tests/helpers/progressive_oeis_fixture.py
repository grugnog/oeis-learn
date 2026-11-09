"""Hand-selected diagnostic study; no autonomous discovery or HIP authority."""

import copy

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.replay import RULE
from oeis_learn.pipeline_validation.progressive_acceptance import fixture as learning_fixture
from oeis_learn.pipeline_validation.progressive_learner import compatibility
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
from oeis_learn.generalization.progressive.yield_audit import policy
from tests.helpers.progressive_fixture import caps


def fixture(root, *, low=False):
    root.mkdir(parents=True, exist_ok=True)
    _, _, _, _ = learning_fixture(root / "base", interrupt=False)
    from oeis_learn.generalization.progressive.contracts import load

    base = (
        load(root / "base/training-input.json")
        if (root / "base/training-input.json").exists()
        else None
    )
    if base is None:
        channel = load(root / "base/training-channel.json")
        base = load(root / "base" / channel["data"]["artifact"]["path"])
    from oeis_learn.pipeline_validation.progressive_acceptance_limits import storage
    measured = load(root / 'base/acceptance-limits.json')

    def channel(name, kind, role, data):
        p = root / name
        atomic_write(p, seal(dict(version="discovery-channel/v1", kind=kind, role=role, data=data)))
        return name

    task_caps = caps(
        tokens=16384,
        draws=4096,
        rejections=4096,
        attempts=4096,
        native_work=65536,
        elapsed_us=600_000_000,
        disk_bytes=measured['task_bytes'],
        archive_bytes=32 << 20,
    )
    pool = []
    programs = [
        "(program (return (add $index (int 3))))",
        "(program (return (add (mul $index (int 2)) (int 1))))",
        "(program (return (select (lt $index (int 20)) (add $index (int 5)) (int 0))))",
    ]
    for i in range(3):
        pool.append(
            dict(
                id=f"train-{i}",
                group=f"training-group-{i}",
                role="training",
                source_snapshot=digest("snapshot"),
                indices=list(map(str, range(100))),
                values=[
                    str(n + 3 if i == 0 else 2 * n + 1 if i == 1 else n + 5) for n in range(100)
                ],
            )
        )
    atomic_write(root / "pool.json", pool)
    exposure = record(
        "exposure",
        dict(previous=None, detector_limits=["fixture finite groups"], events=[]),
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="oeis_derived",
    )
    atomic_write(root / "exposure.json", exposure)
    panels = {}
    eval_forced = {}
    from oeis_learn.generalization.progressive.cohorts import DEFAULT_PROFILE

    for kind in ("transfer", "retention"):
        rows = []
        for i in range(2):
            id = f"{kind}-{i}"
            xs = list(map(str, range(30, 130)))
            values = [
                str(int(n) ** 2 + 10 + i if kind == "transfer" else -(int(n) ** 2) - 100 - i)
                for n in xs
            ]
            atomic_write(
                root / (id + "-visible.json"), dict(id=id, indices=xs[:20], values=values[:20])
            )
            atomic_write(root / (id + "-truth.json"), dict(id=id, indices=xs, values=values))
            eval_forced[id] = [
                f"(program (return (add (mul $index $index) (int {10 + i}))))"
                if kind == "transfer"
                else f"(program (return (sub (int {-100 - i}) (mul $index $index))))"
            ]
            rows.append(
                dict(
                    id=id,
                    source="oeis" if kind == "transfer" else "synthetic",
                    offset="30",
                    stratum="oeis" if kind == "transfer" else "retention",
                    groups=[id + "-group"],
                    structure=digest(id),
                    behavior=digest(id + "b"),
                    composition=digest(id + "c"),
                    operations=["add"],
                    difficulty=dict(depth=1, nodes=3, work=300),
                    visible=reference(root, id + "-visible.json"),
                    truth=reference(root, id + "-truth.json"),
                )
            )
        cohort = record(
            "cohort",
            dict(
                role="confirmation",
                panel=kind,
                decision="oeis",
                profile=DEFAULT_PROFILE,
                required_per_stratum=1,
                minimum_groups=1,
                group_policy="exclusive",
                detector_limits=["fixture finite groups"],
                exposure=exposure["identity"],
                rows=rows,
                strata=["oeis" if kind == "transfer" else "retention"],
            ),
            created_at=exposure["created_at"],
            purpose="acceptance",
            track="oeis_derived",
        )
        atomic_write(root / (kind + ".json"), cohort)
        panels[kind] = reference(root, kind + ".json")
    panels["exposure"] = reference(root, "exposure.json")
    atomic_write(
        root / "forced.json",
        dict(
            pool={m["id"]: [] if low else [s] for m, s in zip(pool, programs)},
            evaluation=eval_forced,
        ),
    )
    entries = []
    tasks = []
    refs = []
    for seed in (1, 2):
        for arm in ("replay", "continued_synthetic"):
            p = copy.deepcopy(base)
            from oeis_learn.pipeline_validation.progressive_device_acceptance import selected_profile
            p["profile"] = selected_profile(resolve_profile(
                p["profile"]["base_recipe"], shape="smaller_width", seed=seed, dropout=0
            ))
            p["update_policy"] = resolve_update_policy(
                p["profile"],
                schedule_horizon_tokens=400000000,
                schedule="constant",
                microbatch_size=7,
            )
            p["oeis_rules"] = RULE
            p["diagnostic_fault"] = None
            p = seal(p)
            name = f"{arm}-{seed}"
            atomic_write(root / (name + "-input.json"), p)
            train = channel(
                name + "-channel.json",
                "training_request",
                "learner",
                dict(
                    membership="training",
                    artifact=reference(root, name + "-input.json"),
                    compatibility=compatibility(p),
                    profile="bigint-body/v1",
                ),
            )
            atomic_write(root / (name + "-profile.json"), p["profile"])
            atomic_write(
                root / (name + "-dummy.json"), dict(diagnostic="random control has no checkpoint")
            )
            model = [
                channel(
                    name + "-recipe.json",
                    "recipe",
                    "generator",
                    dict(
                        artifact=reference(root, name + "-profile.json"),
                        compatibility=compatibility(p),
                        profile="bigint-body/v1",
                    ),
                ),
                channel(
                    name + "-checkpoint.json",
                    "checkpoint",
                    "generator",
                    dict(
                        artifact=reference(root, name + "-dummy.json"),
                        compatibility=compatibility(p),
                        profile="bigint-body/v1",
                        control="random",
                    ),
                ),
            ]
            entries.append(dict(task=name, channel=train, model=model))
            tasks.append(
                dict(
                    id=name,
                    stage="oeis",
                    slot="oeis",
                    kind="training",
                    arm=arm,
                    seed=seed,
                    initializer=dict(mode="random", artifact=None, compatibility=compatibility(p)),
                    token_target=1,
                    caps=task_caps,
                    startup_us=1_000_000,
                    finalization_us=4_000_000,
                    policy=digest("request"),
                    output=name,
                )
            )
    plan = seal(
        dict(
            version="progressive-oeis-cycle/v1",
            cycle=0,
            pool=reference(root, "pool.json"),
            training=entries,
            panels=panels,
            yield_policy=policy(min_groups=2, min_structures=2),
            seconds=dict(generate=60, **{"admit-training": 30}, train=90, score=30),
            forced=reference(root, "forced.json"),
            design=dict(bootstrap_seed=13, comparisons=2, maximum_total_us=2_400_000_000),
        )
    )
    atomic_write(root / "cycle-plan.json", plan)
    request = dict(
        operation="oeis-cycle",
        artifacts=dict(cycle=["cycle-plan.json"]),
        image="sha256:" + "a" * 64,
        seconds=600,
        reservation=dict.fromkeys(task_caps, 0),
    )
    atomic_write(root / "cycle-request.json", request)
    for task in tasks:
        task["policy"] = reference(root, "cycle-request.json")["sha256"]
    stages = ["capability", "diversity", "tuning", "scaling", "oeis"]
    zero = dict.fromkeys(task_caps, 0)
    nodes = [
        dict(
            id=s,
            stage=s,
            kind="routine",
            dependencies=[] if i == 0 else [stages[i - 1]],
            maximum_visits=1,
            caps={k: v * 4 for k, v in task_caps.items()} if s == "oeis" else zero,
        )
        for i, s in enumerate(stages)
    ]
    edges = [
        dict(
            source=a,
            target=b,
            condition="CONFIRMED_GAIN" if b in {"scaling", "oeis"} else "COMPLETE",
            action="ADVANCE_REGISTERED",
        )
        for a, b in zip(stages, stages[1:])
    ]
    refs = [
        dict(
            name="cycle-request",
            kind="policy",
            role="custodian",
            artifact=reference(root, "cycle-request.json"),
        ),
        dict(
            name="cycle-plan",
            kind="prepared",
            role="custodian",
            artifact=reference(root, "cycle-plan.json"),
        ),
    ]
    for e in entries:
        refs.append(
            dict(
                name=e["task"] + "-training",
                kind="prepared",
                role="learner",
                artifact=reference(root, e["channel"]),
            )
        )
        for i, p in enumerate(e["model"]):
            refs.append(
                dict(
                    name=e["task"] + "-model-" + str(i),
                    kind="prepared",
                    role="generator",
                    artifact=reference(root, p),
                )
            )
    storage_limits = storage(root, measured, tasks=len(tasks))
    campaign = record(
        "campaign",
        dict(
            blueprint=digest("blueprint"),
            runtime=digest("runtime"),
            graph=dict(selected_stages=stages, nodes=nodes, edges=edges, diagnostic_pool=zero),
            tasks=tasks,
            references=refs,
            caps={k: v * 8 for k, v in task_caps.items()} | {'disk_bytes': storage_limits['required_bytes']},
            storage=storage_limits,
            target_policy=digest("targets"),
        ),
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="oeis_derived",
    )
    from oeis_learn.generalization.progressive.campaign import validate_campaign

    validate_campaign(campaign)
    manager = record(
        "task",
        dict(tasks[0], campaign=campaign["identity"]),
        created_at=campaign["created_at"],
        purpose="acceptance",
        track="oeis_derived",
    )
    return campaign, manager, request, plan
