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


def build(root, monkeypatch, *, interrupt=False, fail=False, partial=False):
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
    spec = seal(
        dict(
            version="progressive-study-policy/v1",
            token_cap=8 * charge,
            target_tokens=charge,
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
            confirmation=dict(panels=[], seeds=[17, 19]),
        )
    )
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
            token_target=(3 if partial else 1) * charge if operation == "train" else 0,
            initializer=dict(mode="random", artifact=None, compatibility=comp),
            trial=dict(
                study=spec["identity"],
                ordinal=0,
                phase=phase,
                coordinate=charge if phase == "scoring" else 0,
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
    cap = {k: sum(p["caps"][k] for p in plans) for k in plans[0]["caps"]}
    c["payload"].update(tasks=plans, references=refs, caps={k: 2 * v for k, v in cap.items()})
    c["payload"]["graph"]["nodes"][0]["caps"] = cap
    c["payload"]["graph"]["nodes"][1]["caps"] = dict.fromkeys(cap, 0)
    c["payload"]["storage"]["checkpoint_bytes"] = 256 << 20
    c["payload"]["storage"]["required_bytes"] = 517 << 20
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
            "operations": ["train", "propose-trial", "generate", "score"],
            "scope": {"images": ["sha256:" + "a" * 64]},
        },
        producers=(),
    )
    monkeypatch.setattr(roles, "current_runtime", lambda *a, **kw: runtime)
    host = SimpleNamespace(
        engine=LocalProcessEngine(root), policy=dict(memory_bytes=2 << 30, cpus=2)
    )
    return c, spec, host, charge
