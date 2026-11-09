"""Real CPU/native study fixture with prospective finite scopes; no research claim."""

import copy
from types import SimpleNamespace
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    reference,
    load,
    seal,
    digest,
    record,
)
from oeis_learn.pipeline_validation.progressive_acceptance import fixture
from oeis_learn.pipeline_validation.progressive_learner import compatibility
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
from oeis_learn.generalization.progressive.tuning import BASELINE
from tests.contract.test_progressive_roles import artifact
from tests.helpers.progressive_engine import LocalProcessEngine


def build(
    root,
    monkeypatch,
    *,
    interrupt=False,
    fail=False,
    partial=False,
    capability=False,
    confirmation=False,
    trials=1,
):
    from oeis_learn.pipeline_validation import progressive_roles as roles

    c, old, _, charge = fixture(
        root, interrupt=interrupt, fail=fail, first_cap_updates=1 if partial else 8
    )
    package = load(root / "training-input.json")
    source = "(program (return (add $index (int 1))))"
    truth = [
        dict(
            id=f"{s}-{g}",
            indices=list(map(str, range(100))),
            values=[str(i + 1) for i in range(100)],
        )
        for s in ("composition", "unseen_structure")
        for g in range(16)
    ]
    roster = [dict(task=r["id"], group=r["id"], stratum=r["id"].rsplit("-", 1)[0]) for r in truth]
    q = seal(dict(version="progressive-cpu-study-acceptance/v1", device="cpu", status="COMPLETE"))
    atomic_write(root / "study-q.json", q)
    confirmation_panels = []
    if confirmation:
        from tests.unit.test_progressive_cohorts import fixture as panel_fixture

        panel, _ = panel_fixture(
            root / "fresh-confirmation", counts=(1, 1, 1, 1), panel="fresh-confirmation"
        )
        panel = seal(
            dict(panel, payload=dict(panel["payload"], required_per_stratum=1, minimum_groups=1))
        )
        atomic_write(root / "confirmation-panel.json", panel)
        confirmation_panels = [reference(root, "confirmation-panel.json")]
    spec = seal(
        dict(
            version="progressive-study-policy/v1",
            token_cap=8 * charge,
            target_tokens=(3 if interrupt or partial else 1) * charge,
            elapsed_cap_us=7200000000,
            screening=dict(
                startup_us=1000000,
                evaluation_us=[1000000] * 3,
                training_us=1000000,
                finalization_us=1000000,
            ),
            panel=roster,
            evaluations={
                "dev-score": dict(
                    generation_task="dev-generate",
                    checkpoint_task="train",
                    qualification=reference(root, "study-q.json"),
                    roster=roster,
                    retention_baseline_generation="ret-base-generate",
                    retention_baseline_scoring="ret-base-score",
                    retention_candidate_generation="ret-candidate-generate",
                    retention_candidate_scoring="ret-candidate-score",
                )
            },
            confirmation=dict(panels=confirmation_panels, seeds=[17, 19]),
        )
    )
    if trials > 1:
        if trials != 6 or confirmation or capability:
            raise ValueError("bounded six-trial pruning rehearsal")
        spec["target_tokens"] = 1
        spec["token_cap"] = 16 * charge
        spec["diagnostic_pruning_coordinate"] = 1000000
        first = spec["evaluations"]["dev-score"]
        for ordinal in range(1, trials):
            prefix = f"t{ordinal}-"
            other = dict(first)
            for key in (
                "generation_task",
                "checkpoint_task",
                "retention_baseline_generation",
                "retention_baseline_scoring",
                "retention_candidate_generation",
                "retention_candidate_scoring",
            ):
                other[key] = prefix + other[key]
            spec["evaluations"][prefix + "dev-score"] = other
        spec = seal(spec)
    atomic_write(root / "study-policy.json", spec)
    seed = int(digest([spec["identity"], 0]).split(":")[1][:8], 16)
    profile = resolve_profile(
        package["profile"]["base_recipe"], shape="reference", seed=seed, dropout=0
    )
    package = seal(
        dict(
            package,
            profile=profile,
            update_policy=resolve_update_policy(profile, schedule_horizon_tokens=4000000),
        )
    )
    atomic_write(root / "study-training.json", package)
    comp = compatibility(package)
    train = artifact(
        root,
        "study-channel.json",
        "training_request",
        "learner",
        dict(
            membership="training",
            artifact=reference(root, "study-training.json"),
            compatibility=comp,
            profile="bigint-body/v1",
        ),
    )
    atomic_write(root / "profile.json", profile)
    recipe = artifact(
        root,
        "profile-channel.json",
        "recipe",
        "generator",
        dict(
            artifact=reference(root, "profile.json"), compatibility=comp, profile="bigint-body/v1"
        ),
    )
    # Random control loads a fresh model, while its placeholder artifact is still
    # an original bounded allowed reference rather than a future result.
    (root / "unused-random.bin").write_bytes(b"not-loaded-by-random-control")
    random = artifact(
        root,
        "random-channel.json",
        "checkpoint",
        "generator",
        dict(
            artifact=reference(root, "unused-random.bin"),
            compatibility=comp,
            profile="bigint-body/v1",
            control="random",
        ),
    )
    forced = artifact(
        root,
        "forced-channel.json",
        "oeis_generation",
        "generator",
        dict(methods=["correct"], forced={r["id"]: [source] for r in truth}),
    )
    prompts = [
        artifact(
            root,
            f"prompt-{i}.json",
            "visible_prompt",
            "generator",
            dict(id=r["id"], indices=r["indices"][:20], values=r["values"][:20]),
        )
        for i, r in enumerate(truth)
    ]
    truths = [
        artifact(root, f"truth-{i}.json", "development_truth", "scorer", r)
        for i, r in enumerate(truth)
    ]
    proposal = artifact(
        root,
        "proposal-channel.json",
        "search_space",
        "proposal",
        dict(study=spec["identity"], ordinal=0, history=[], selected_learning=None),
    )
    refs = [
        dict(
            name="study-policy",
            kind="policy",
            role="proposal",
            artifact=reference(root, "study-policy.json"),
        )
    ]
    for i, a in enumerate([train, recipe, random, forced, *prompts, *truths, proposal]):
        refs.append(dict(name=f"input-{i}", kind="prepared", role=a.role, artifact=a.ref))
    plans = []

    def add(name, operation, channels, phase, *, choices=False):
        cap = dict(old[0]["payload"]["caps"])
        cap.update(tokens=0, draws=100000, rejections=100000, attempts=100000, native_work=100000)
        reservation = dict.fromkeys(cap, 0)
        reservation.update(
            elapsed_us=120000000,
            draws=100000,
            rejections=100000,
            attempts=100000,
            native_work=100000,
        )
        if operation == "train":
            cap["tokens"] = (1 if partial else 8) * charge
            reservation = dict.fromkeys(cap, 0) | {"elapsed_us": 120000000}
        request = dict(
            operation=operation,
            artifacts=channels,
            image="sha256:" + "a" * 64,
            seconds=120,
            reservation=reservation,
        )
        path = f"study-request-{name}.json"
        atomic_write(root / path, request)
        ref = reference(root, path)
        refs.append(dict(name=f"request-{name}", role="custodian", kind="policy", artifact=ref))
        plan = copy.deepcopy(old[0]["payload"])
        plan.pop("campaign")
        plan.update(
            id=name,
            output=name,
            slot="first",
            stage="capability",
            kind="training" if operation == "train" else "evaluation",
            seed=seed,
            caps=cap,
            policy=ref["sha256"],
            token_target=(3 if partial or interrupt else 1) * charge if operation == "train" else 0,
            initializer=dict(mode="random", artifact=None, compatibility=comp),
            trial=dict(
                study=spec["identity"],
                ordinal=0,
                phase=phase,
                coordinate=spec["target_tokens"] if phase == "scoring" else 0,
            ),
        )
        if choices:
            plan["trial_choices"] = [
                dict(parameters=BASELINE, compatibility=comp, policy=ref["sha256"])
            ]
        plans.append(plan)

    add("propose", "propose-trial", {"search": ["proposal-channel.json"]}, "proposal")
    add("train", "train", {"training": ["study-channel.json"]}, "training", choices=True)
    weights = dict(task="train", operation="train", output="weights", kind="checkpoint")
    for name, random_control, phase in [
        ("ret-base", True, "retention"),
        ("ret-candidate", False, "retention"),
        ("dev", False, "generation"),
    ]:
        gen = f"{name}-generate"
        add(
            gen,
            "generate",
            {
                "visible": [a.ref["path"] for a in prompts],
                "model": [
                    "profile-channel.json",
                    "forced-channel.json",
                    "random-channel.json" if random_control else weights,
                ],
            },
            phase,
        )
        add(
            f"{name}-score",
            "score",
            {
                "candidates": [
                    dict(task=gen, operation="generate", output="candidates", kind="candidate")
                ],
                "score_truth": [a.ref["path"] for a in truths],
            },
            "scoring" if name == "dev" else "retention",
        )
    if trials > 1:
        import itertools
        from oeis_learn.generalization.progressive.tuning import SPACE

        keys = list(SPACE)
        parameters = [
            dict(zip(keys, values)) for values in itertools.product(*(SPACE[k] for k in keys))
        ]
        old_plans = copy.deepcopy(plans)
        request_refs = {
            r["artifact"]["sha256"]: r["artifact"] for r in refs if r["kind"] == "policy"
        }
        bad = artifact(
            root,
            "last-poor-channel.json",
            "oeis_generation",
            "generator",
            dict(
                methods=["correct"], forced={r["id"]: ["(program (return (int 0)))"] for r in truth}
            ),
        )
        refs.append(dict(name="poor-declaration", kind="prepared", role=bad.role, artifact=bad.ref))
        for ordinal in range(trials):
            trial_seed = int(digest([spec["identity"], ordinal]).split(":")[1][:8], 16)
            selected = plans[: len(old_plans)] if ordinal == 0 else []
            if ordinal:
                prefix = f"t{ordinal}-"
                mapping = {p["id"]: prefix + p["id"] for p in old_plans}

                def rename(v):
                    if isinstance(v, dict):
                        return {
                            k: (
                                mapping.get(child, child)
                                if k == "task" and isinstance(child, str)
                                else rename(child)
                            )
                            for k, child in v.items()
                        }
                    if isinstance(v, list):
                        return [rename(child) for child in v]
                    return v

                for original_plan in old_plans:
                    plan = copy.deepcopy(original_plan)
                    plan.update(id=mapping[plan["id"]], output=mapping[plan["id"]], seed=trial_seed)
                    plan["trial"]["ordinal"] = ordinal
                    request = rename(load(root / request_refs[original_plan["policy"]]["path"]))
                    if ordinal == trials - 1 and original_plan["id"] == "dev-generate":
                        request["artifacts"]["model"] = [
                            bad.ref["path"] if x == "forced-channel.json" else x
                            for x in request["artifacts"]["model"]
                        ]
                    name = "request-" + plan["id"] + ".json"
                    atomic_write(root / name, request)
                    ref = reference(root, name)
                    refs.append(
                        dict(
                            name="policy-" + plan["id"],
                            kind="policy",
                            role="custodian",
                            artifact=ref,
                        )
                    )
                    plan["policy"] = ref["sha256"]
                    selected.append(plan)
                plans.extend(selected)
            for plan in selected:
                plan["seed"] = trial_seed
                plan["trial"]["coordinate"] = 1 if plan["trial"]["phase"] == "scoring" else 0
                if plan["kind"] == "training":
                    plan["token_target"] = 1
                if plan["id"].endswith(("train", "dev-generate", "ret-candidate-generate")):
                    choices = []
                    for params in parameters:
                        prof = resolve_profile(
                            package["profile"]["base_recipe"],
                            seed=trial_seed,
                            dropout=params["dropout"],
                        )
                        updated = resolve_update_policy(
                            prof,
                            schedule_horizon_tokens=4000000,
                            **{
                                k: params[k]
                                for k in (
                                    "lr_multiplier",
                                    "schedule",
                                    "effective_batch",
                                    "weight_decay",
                                )
                            },
                        )
                        variant = seal(dict(package, profile=prof, update_policy=updated))
                        choices.append(
                            dict(
                                parameters=params,
                                compatibility=compatibility(variant),
                                policy=plan["policy"],
                            )
                        )
                    plan["trial_choices"] = choices
    if trials > 1:
        late = copy.deepcopy(next(p for p in plans if p["id"] == "t5-train"))
        late.update(id="t5-late-train", output="t5-late-train", token_target=1000000)
        late["initializer"].update(mode="full_state", artifact=None, parent_task="t5-train")
        plans.append(late)
    if confirmation:
        selection_input = artifact(
            root,
            "selection-input.json",
            "public_evidence",
            "verifier",
            dict(version="progressive-selection-request/v1", study=spec["identity"]),
        )
        refs.append(
            dict(
                name="selection-input",
                kind="prepared",
                role="verifier",
                artifact=selection_input.ref,
            )
        )
        refs.append(
            dict(
                name="confirmation-panel",
                kind="groups",
                role="custodian",
                artifact=confirmation_panels[0],
            )
        )
        add("selection", "publish", {"public": ["selection-input.json"]}, "selection")
        ct = [dict(id="fresh-" + r["id"], indices=r["indices"], values=r["values"]) for r in truth]
        visible = [
            artifact(
                root,
                f"confirm-visible-{i}.json",
                "visible_prompt",
                "generator",
                dict(id=r["id"], indices=r["indices"][:20], values=r["values"][:20]),
            )
            for i, r in enumerate(ct)
        ]
        hidden = [
            artifact(root, f"confirm-truth-{i}.json", "confirmation_truth", "scorer", r)
            for i, r in enumerate(ct)
        ]
        bad = artifact(
            root,
            "confirm-forced.json",
            "oeis_generation",
            "generator",
            dict(methods=["correct"], forced={r["id"]: ["(program (return (int 0)))"] for r in ct}),
        )
        for i, a in enumerate([*visible, *hidden, bad]):
            refs.append(
                dict(name=f"confirm-input-{i}", kind="prepared", role=a.role, artifact=a.ref)
            )
        for fresh_seed in (17, 19):
            fresh_profile = resolve_profile(
                package["profile"]["base_recipe"], shape="reference", seed=fresh_seed, dropout=0
            )
            fresh = seal(
                dict(
                    package,
                    profile=fresh_profile,
                    update_policy=resolve_update_policy(
                        fresh_profile, schedule_horizon_tokens=400000000
                    ),
                    diagnostic_fault=None,
                )
            )
            prefix = "confirm-" + str(fresh_seed)
            atomic_write(root / (prefix + "-package.json"), fresh)
            atomic_write(root / (prefix + "-profile.json"), fresh_profile)
            fresh_comp = compatibility(fresh)
            fc = artifact(
                root,
                prefix + "-channel.json",
                "training_request",
                "learner",
                dict(
                    membership="training",
                    artifact=reference(root, prefix + "-package.json"),
                    compatibility=fresh_comp,
                    profile="bigint-body/v1",
                ),
            )
            fr = artifact(
                root,
                prefix + "-recipe.json",
                "recipe",
                "generator",
                dict(
                    artifact=reference(root, prefix + "-profile.json"),
                    compatibility=fresh_comp,
                    profile="bigint-body/v1",
                ),
            )
            refs.extend(
                [
                    dict(name=prefix + "-input", kind="prepared", role=fc.role, artifact=fc.ref),
                    dict(name=prefix + "-recipe", kind="prepared", role=fr.role, artifact=fr.ref),
                ]
            )
            add(prefix, "train", {"training": [fc.ref["path"]]}, "confirmation", choices=True)
            add(
                prefix + "-generate",
                "generate",
                {
                    "visible": [a.ref["path"] for a in visible],
                    "model": [
                        fr.ref["path"],
                        bad.ref["path"],
                        dict(task=prefix, operation="train", output="weights", kind="checkpoint"),
                    ],
                },
                "confirmation",
                choices=True,
            )
            add(
                prefix + "-score",
                "score",
                {
                    "candidates": [
                        dict(
                            task=prefix + "-generate",
                            operation="generate",
                            output="candidates",
                            kind="candidate",
                        )
                    ],
                    "score_truth": [a.ref["path"] for a in hidden],
                },
                "confirmation",
                choices=True,
            )
            for plan in plans[-3:]:
                plan["seed"] = fresh_seed
                plan["initializer"]["compatibility"] = fresh_comp
                plan["trial_choices"][0]["compatibility"] = fresh_comp
    if capability:
        from tests.unit.test_progressive_cohorts import fixture as cohort_fixture

        cohort, exposure = cohort_fixture(root / "capability-panel")
        template = cohort["payload"]["rows"][0]
        cohort["payload"].update(
            rows=[
                dict(template, id=r["task"], stratum=r["stratum"], groups=[r["group"]])
                for r in roster
            ],
            required_per_stratum=1,
            minimum_groups=1,
        )
        cohort = seal(cohort)
        co = artifact(root, "cap-cohort.json", "cohort", "custodian", cohort)
        ex = artifact(root, "cap-exposure.json", "exposure", "custodian", exposure)
        cp = artifact(root, "cap-public-cohort.json", "public_evidence", "verifier", cohort)
        ep = artifact(root, "cap-public-exposure.json", "public_evidence", "verifier", exposure)
        witnesses = [
            artifact(
                root,
                f"witness-{i}.json",
                "coverage_witness",
                "custodian",
                dict(
                    task=r["id"],
                    source=source,
                    indices=r["indices"],
                    values=r["values"],
                    operator_class=["add"],
                    missing_operations=[],
                    created_at=c["created_at"],
                ),
            )
            for i, r in enumerate(truth)
        ]
        for i, a in enumerate([co, ex, cp, ep, *witnesses]):
            refs.append(dict(name=f"cap-input-{i}", kind="prepared", role=a.role, artifact=a.ref))
        add(
            "cap-census",
            "census",
            {"sources": ["cap-cohort.json", "cap-exposure.json"]},
            "retention",
        )
        plans[-1].pop("trial")
        add(
            "cap-audit",
            "audit-language",
            {"sources": [a.ref["path"] for a in witnesses]},
            "retention",
        )
        plans[-1].pop("trial")
        public = [
            "cap-public-cohort.json",
            "cap-public-exposure.json",
            dict(task="cap-census", operation="census", output="census", kind="public_evidence"),
            dict(
                task="cap-audit",
                operation="audit-language",
                output="coverage",
                kind="public_evidence",
            ),
        ]
        for prefix in ("dev", "ret-base"):
            public += [
                dict(
                    task=prefix + "-generate",
                    operation="generate",
                    output="evaluation",
                    kind="public_evidence",
                ),
                dict(
                    task=prefix + "-score",
                    operation="score",
                    output="scores",
                    kind="public_evidence",
                ),
            ]
        add("cap-readiness", "publish", {"public": public}, "retention")
        plans[-1].pop("trial")
        from tests.unit.test_progressive_decisions import fixture as decision_fixture

        _, _, design, target = decision_fixture()
        seeds = [seed, seed + 1]
        target = seal(target)
        atomic_write(root / "cap-target.json", target)
        refs.append(
            dict(
                name="target-policy",
                kind="policy",
                role="verifier",
                artifact=reference(root, "cap-target.json"),
            )
        )
        c["payload"]["target_policy"] = target["identity"]
        registered = []
        comparison = []
        decision_inputs = []
        for arm, prefix in [("baseline", "ret-candidate"), ("candidate", "dev")]:
            registered.append(
                dict(
                    arm=arm,
                    seed=seed,
                    checkpoint=weights,
                    model=None,
                    candidates=dict(
                        task=prefix + "-generate",
                        operation="generate",
                        output="candidates",
                        kind="candidate",
                    ),
                    scores=dict(
                        task=prefix + "-score",
                        operation="score",
                        output="scores",
                        kind="public_evidence",
                    ),
                )
            )
            comparison.append(
                dict(
                    arm=arm,
                    seed=seed,
                    generation_task=prefix + "-generate",
                    scoring_task=prefix + "-score",
                )
            )
            decision_inputs.extend(
                [
                    dict(
                        task=prefix + "-generate",
                        operation="generate",
                        output="evaluation",
                        kind="public_evidence",
                    ),
                    dict(
                        task=prefix + "-score",
                        operation="score",
                        output="scores",
                        kind="public_evidence",
                    ),
                ]
            )
        gate = seal(
            dict(
                version="progressive-gate-registration/v1",
                source_slot="first",
                target_slot="second",
                panels=[cohort["identity"]],
                seeds=seeds,
                design=digest(design),
                producers=registered,
            )
        )
        atomic_write(root / "cap-gate.json", gate)
        atomic_write(root / "cap-gate-panel.json", cohort)
        refs.extend(
            [
                dict(
                    name="gate-registration",
                    kind="policy",
                    role="verifier",
                    artifact=reference(root, "cap-gate.json"),
                ),
                dict(
                    name="gate-panel",
                    kind="groups",
                    role="custodian",
                    artifact=reference(root, "cap-gate-panel.json"),
                ),
            ]
        )
        template = dict(
            version="progressive-decision-template/v1",
            source_slot="first",
            target_slot="second",
            roster=roster,
            seeds=seeds,
            design=design,
            policy=target,
            remaining={},
            kwargs=dict(target_id="gain", action="RETAIN_BASELINE"),
            producers=comparison,
        )
        dt = artifact(root, "cap-decision-template.json", "public_evidence", "verifier", template)
        refs.append(
            dict(name="decision-template", kind="prepared", role="verifier", artifact=dt.ref)
        )
        add(
            "cap-decision",
            "publish",
            {"public": ["cap-decision-template.json", *decision_inputs]},
            "retention",
        )
        plans[-1].pop("trial")

    cap = {k: sum(p["caps"][k] for p in plans) for k in plans[0]["caps"]}
    c["payload"].update(tasks=plans, references=refs, caps={k: 2 * v for k, v in cap.items()})
    c["payload"]["graph"]["nodes"][0]["caps"] = cap
    c["payload"]["graph"]["nodes"][1]["caps"] = dict.fromkeys(cap, 0)
    c["payload"]["storage"]["checkpoint_bytes"] = 256 << 20
    c["payload"]["storage"]["required_bytes"] = 517 << 20
    c["payload"]["storage"]["required_inodes"] = 16384
    c["payload"]["storage"]["available_inodes"] = 32768
    c = seal(c)
    atomic_write(root / "study-campaign.json", c)
    atomic_write(
        root / "producer/qualification.json", {"covered_capabilities": ["progressive-lifecycle"]}
    )
    # fixture's runtime file is absent; every qualification below is a declared
    # CPU test double and cannot authorize research.
    runtime_path = root / "runtime.json"
    if runtime_path.exists():
        runtime_path.unlink()
    atomic_write(
        runtime_path,
        {"payload": {"qualification": [reference(root, "producer/qualification.json")]}},
    )
    atomic_write(
        root / "accepted/acceptance.json",
        seal(
            dict(
                version="discovery-acceptance/v1",
                scope="progressive-lifecycle",
                qualified=True,
                campaign=c["identity"],
                runtime=c["payload"]["runtime"],
                status="COMPLETE",
            )
        ),
    )
    runtime = SimpleNamespace(
        execution_ready=True,
        identity=c["payload"]["runtime"],
        purpose="acceptance",
        track=c["track"],
        payload={
            "operations": [
                "train",
                "propose-trial",
                "generate",
                "score",
                "census",
                "audit-language",
                "publish",
            ],
            "scope": {"images": ["sha256:" + "a" * 64]},
        },
        producers=(),
    )
    monkeypatch.setattr(roles, "current_runtime", lambda *a, **kw: runtime)
    host = SimpleNamespace(
        engine=LocalProcessEngine(root), policy=dict(memory_bytes=2 << 30, cpus=2)
    )
    return c, spec, host, charge
