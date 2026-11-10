"""Mined expansion -> verified training target -> real CPU update and full restore."""

import copy
import time
import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal
from tests.unit.test_progressive_component_mining import sources


def prepared(root, client=None):
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.component_search import search_input, Proposals
    from oeis_learn.pipeline_validation.progressive_training import distill_component

    store = LibraryStore(root, "library", purpose="acceptance", track="synthetic_strict")
    lib = store.publish(sources(root, client), client=client)
    bundle = search_input(store, lib)
    proposals = Proposals(lib)
    chosen = proposals(4, time.monotonic() + 60)
    member = dict(
        id="training-new",
        group="training-new-group",
        role="training",
        source_snapshot=digest("mechanical fixture"),
        indices=list(map(str, range(100))),
        values=[str(2 * i + 1) for i in range(100)],
    )
    # A concrete component proposal is the selected source; no oracle program is supplied.
    proposal = next(p for p in proposals.report["proposals"] if p["source"] == chosen["source"])
    target = distill_component(bundle, proposal, member, client=client, purpose="acceptance")
    return bundle, target


def test_mined_candidate_updates_weights_and_restores_atomic_library_cut(tmp_path):
    from oeis_learn.generalization.progressive.component_distillation import (
        freeze_distillation,
        ComponentMixer,
    )
    from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate, replay_batch
    from oeis_learn.generalization.progressive.recovery import Recovery
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from tests.helpers.progressive_fixture import task, caps
    from tests.unit.test_progressive_training_policy import profile, policy
    from oeis_learn.decoder.bigint_codec import BOS_ID
    import torch

    bundle, target = prepared(tmp_path)
    package = freeze_distillation(bundle, [target], arm="library")
    mixer = ComponentMixer(package)
    state = mixer.initial()
    rows, charge = replay_batch(package, [target["row"]] * 16, state)
    assert charge["oeis_examples"] > 0
    p = profile()
    cfg = policy(p)
    t = task(caps=caps(tokens=10000, elapsed_us=60000000))
    journal = BudgetJournal(tmp_path / "accounting", t["payload"]["campaign"], t["payload"]["caps"])
    context = {k: {} for k in ("scheduler", "data", "replay", "history", "pins", "selection")}
    context["data"] = {"cursor": 0, "pending_batch": None}
    context["replay"] = state
    context["selection"]["component_input"] = package
    cuts = {
        k: digest(None)
        for k in (
            "stream",
            "allocation",
            "archive",
            "library",
            "cost_journal",
            "selection",
            "milestones",
            "consumed",
        )
    }
    cuts.update(library=bundle["library"]["identity"], archive=mixer.archive)
    tokens = torch.tensor([[BOS_ID, *r["tokens"]] for r in rows])
    with Recovery(journal, t, digest([p, cfg])) as owner:
        worker = RecoverableUpdate(owner, p, cfg, device="cpu")
        worker.initialize(context, cuts)
        before = [v.detach().clone() for v in worker.runner.decoder.parameters()]
        owner.begin(boot="one", monotonic_us=0)
        worker.step(
            [list(map(int, r["values"][:20])) for r in rows],
            tokens,
            entries=[r["draw_identity"] for r in rows],
            update=1,
            context=context,
            cuts=cuts,
            boot="one",
            monotonic_us=1,
            replay=charge,
            replay_package=package,
        )
        assert any(
            not torch.equal(a, b) for a, b in zip(before, worker.runner.decoder.parameters())
        )
        owner.close(boot="one", monotonic_us=100)
    with Recovery(journal, t, digest([p, cfg])) as owner:
        worker = RecoverableUpdate(owner, p, cfg, device="cpu")
        restored = owner.restore()
        assert restored["ledger"].updates == 1
        assert restored["cuts"]["library"] == bundle["library"]["identity"]
        assert restored["states"]["selection"]["component_input"] == package
        assert worker.context["replay"]["oeis_tokens"] == charge["oeis_tokens"]
    control = freeze_distillation(None, [], arm="no_library")
    assert ComponentMixer(control).archive is None
    assert control["library"] is None and control["targets"] == []


def test_training_only_distillation_rejects_exposure_mismatch_and_fake_source(tmp_path):
    from oeis_learn.pipeline_validation.progressive_training import distill_component
    from oeis_learn.generalization.progressive.component_distillation import freeze_distillation

    bundle, target = prepared(tmp_path)
    with pytest.raises(ValueError):
        distill_component(
            bundle,
            target["proposal"],
            dict(target["member"], role="confirmation"),
            purpose="acceptance",
        )
    with pytest.raises(ValueError):
        distill_component(
            bundle,
            dict(target["proposal"], source="(program (return (int 0)))"),
            target["member"],
            purpose="acceptance",
        )
    with pytest.raises(ValueError):
        freeze_distillation(bundle, [target], arm="no_library")
    bad = copy.deepcopy(target)
    bad["row"]["tokens"] = [1]
    bad["row"] = seal(bad["row"])
    bad = seal(bad)
    with pytest.raises(ValueError):
        freeze_distillation(bundle, [bad], arm="library")


def test_native_component_distillation_matches_original_hundred_terms(tmp_path, fitting_client):
    _, target = prepared(tmp_path, fitting_client)
    assert target["row"]["native_verified"]
    assert target["row"]["values"] == [str(2 * i + 1) for i in range(100)]


@pytest.mark.parametrize("arm", ["library", "no_library"])
@pytest.mark.parametrize("seed", [1, 2])
def test_installed_learner_binds_component_cut_and_rejects_changed_resume(tmp_path, arm, seed, fitting_client):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.generalization.progressive.component_distillation import freeze_distillation
    from oeis_learn.generalization.progressive.contracts import (
        load,
        atomic_write,
        reference,
        record,
    )

    c, tasks, channel, _ = fixture(tmp_path, interrupt=True)
    source_root = tmp_path / "mining"
    source_root.mkdir()
    bundle, target = prepared(source_root, fitting_client)
    assert target["row"]["native_verified"]
    package = load(tmp_path / "training-input.json")
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy

    from oeis_learn.pipeline_validation.progressive_device_acceptance import selected_profile
    profile = selected_profile(resolve_profile(
        package["profile"]["base_recipe"], shape="smaller_width", seed=seed, dropout=0.1
    ))
    update_policy = resolve_update_policy(
        profile, schedule_horizon_tokens=400000000, schedule="constant", microbatch_size=7
    )
    package = seal(
        dict(
            package,
            profile=profile,
            update_policy=update_policy,
            component_distillation=freeze_distillation(
                bundle if arm == "library" else None, [target] if arm == "library" else [], arm=arm
            ),
        )
    )
    atomic_write(tmp_path / "component-training-input.json", package)
    channel = seal(
        dict(
            channel,
            data=dict(
                channel["data"],
                artifact=reference(tmp_path, "component-training-input.json"),
                compatibility=compatibility(package),
            ),
        )
    )
    atomic_write(tmp_path / "component-training-channel.json", channel)
    c = copy.deepcopy(c)
    c["payload"]["graph"]["selected_stages"] = [
        "capability",
        "diversity",
        "tuning",
        "scaling",
        "components",
    ]
    for node in c["payload"]["graph"]["nodes"]:
        node["stage"] = "components"
    previous = None
    for stage in ("capability", "diversity", "tuning", "scaling"):
        c["payload"]["graph"]["nodes"].append(
            dict(
                id=stage,
                stage=stage,
                kind="routine",
                dependencies=[previous] if previous else [],
                maximum_visits=1,
                caps=dict.fromkeys(c["payload"]["caps"], 0),
            )
        )
        if previous:
            c["payload"]["graph"]["edges"].append(
                dict(
                    source=previous,
                    target=stage,
                    condition="CONFIRMED_GAIN" if stage == "scaling" else "COMPLETE",
                    action="ADVANCE_REGISTERED",
                )
            )
        previous = stage
    c["payload"]["graph"]["nodes"][0]["dependencies"] = ["scaling"]
    for plan in c["payload"]["tasks"]:
        plan["stage"] = "components"
        plan["arm"] = arm
        plan["seed"] = seed
        plan["initializer"]["compatibility"] = compatibility(package)
    c = seal(c)
    task = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    first, _ = invoke_cpu(
        tmp_path, c, task, channel, ordinal=0, channel_path="component-training-channel.json"
    )
    assert first["mechanical_status"] == "PARTIAL", first
    assert first["metrics"]["discarded_tokens"] > 0
    second, _ = invoke_cpu(
        tmp_path, c, task, channel, ordinal=1, channel_path="component-training-channel.json"
    )
    assert second["mechanical_status"] == "COMPLETE", second
    assert second["metrics"]["useful_tokens"] > 0
    from oeis_learn.generalization.progressive.recovery import Recovery
    from oeis_learn.generalization.progressive.budgets import BudgetJournal

    j = BudgetJournal(tmp_path / "accounting", c["identity"], c["payload"]["caps"])
    with Recovery(
        j, task, compatibility(package), state_root=tmp_path / "learning" / task["identity"][7:]
    ) as owner:
        restored = owner.restore()
        assert restored["cuts"]["library"] == (
            bundle["library"]["identity"] if arm == "library" else None
        )
        assert (
            restored["states"]["selection"]["component_input"] == package["component_distillation"]
        )
        from oeis_learn.pipeline_validation.progressive_training import check_component_cut

        with pytest.raises(ValueError, match="cut differs"):
            check_component_cut(
                restored["states"],
                dict(restored["cuts"], library=digest("other")),
                package["component_distillation"],
            )

    if arm == "library" and seed == 1:
        from oeis_learn.generalization.progressive.component_mining import LibraryStore
        from oeis_learn.generalization.progressive.component_search import search_input
        from oeis_learn.pipeline_validation.progressive_training import distill_component

        store = LibraryStore(source_root, "library", purpose="acceptance", track="synthetic_strict")
        refs = [reference(source_root, f"source-{i}.json") for i in range(3)]
        next_library = store.publish(refs, previous=bundle["library"])
        next_bundle = search_input(store, next_library)
        next_target = distill_component(
            next_bundle, target["proposal"], target["member"], purpose="acceptance"
        )
        successor_package = seal(
            dict(
                package,
                component_distillation=freeze_distillation(
                    next_bundle, [next_target], arm="library"
                ),
            )
        )
        atomic_write(tmp_path / "successor-input.json", successor_package)
        successor_channel = seal(
            dict(
                channel,
                data=dict(channel["data"], artifact=reference(tmp_path, "successor-input.json")),
            )
        )
        atomic_write(tmp_path / "successor-channel.json", successor_channel)
        successor = record(
            "task",
            dict(c["payload"]["tasks"][1], campaign=c["identity"]),
            created_at=c["created_at"],
            purpose=c["purpose"],
            track=c["track"],
        )
        atomic_write(
            tmp_path / "control" / ("pause-" + digest("invocation-2")[7:] + ".json"),
            seal(dict(campaign=c["identity"], requested=True)),
        )
        third, _ = invoke_cpu(
            tmp_path,
            c,
            successor,
            successor_channel,
            ordinal=2,
            channel_path="successor-channel.json",
        )
        assert third["mechanical_status"] == "PARTIAL", third
        with Recovery(
            j,
            successor,
            compatibility(package),
            state_root=tmp_path / "learning" / successor["identity"][7:],
        ) as owner:
            assert owner.restore()["cuts"]["library"] == next_library["identity"]
            assert owner.base["tokens"] == second["metrics"]["charged_tokens"]
            from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate

            worker = RecoverableUpdate(owner, profile, update_policy, device="cpu")
            assert owner.restore()["ledger"].consumed == owner.base["tokens"]
            before = owner.state["checkpoint"]
            with pytest.raises(ValueError, match="boundary"):
                worker.adopt_component_cut(package["component_distillation"])
            assert owner.state["checkpoint"] == before
        fourth, _ = invoke_cpu(
            tmp_path,
            c,
            successor,
            successor_channel,
            ordinal=3,
            channel_path="successor-channel.json",
        )
        assert fourth["mechanical_status"] == "COMPLETE", fourth


def test_more_expensive_library_without_harder_task_yield_retains_baseline():
    from oeis_learn.generalization.progressive.component_distillation import compare_utility
    from tests.unit.test_progressive_metrics import rows

    data = []
    for seed in (1, 2):
        for arm in ("baseline", "candidate"):
            for stratum in ("composition", "unseen_structure", "retention"):
                for group in range(16):
                    data += rows(seed, arm, stratum, str(group), 2, 1)
    roster = [
        {k: r[k] for k in ("stratum", "group", "task")}
        for r in data
        if r["arm"] == "baseline" and r["seed"] == 1
    ]
    costs = dict(
        baseline=dict(discovery=0, validation=0, search=10, execution=10, training=80),
        candidate=dict(discovery=100, validation=50, search=10, execution=10, training=80),
    )
    result = compare_utility(
        data,
        roster,
        [1, 2],
        dict(claim="capability", bootstrap_seed=13, comparisons=1, links=[], costs=None),
        costs=costs,
        panel="components-harder",
    )
    assert result["verdict"] == "NO_MATERIAL_GAIN"
    assert result["total_cost_us"] == dict(baseline=100, candidate=250)
    assert result["cost_reduction"] == -1.5
    assert result["next_action"] == "RETAIN_BASELINE" and not result["compression_is_utility"]


def test_refuting_original_mined_row_quarantines_library_and_distillation(tmp_path):
    from oeis_learn.generalization.progressive.component_distillation import freeze_distillation
    from oeis_learn.generalization.progressive.quarantine import Journal
    from oeis_learn.generalization.progressive.contracts import load
    bundle, target = prepared(tmp_path)
    journal = Journal(tmp_path/'accounting')
    journal.training({'component_distillation':freeze_distillation(bundle,[target],arm='library')})
    row = load(tmp_path/'source-0.json')
    journal.training({'shards':[[row]]})
    affected = journal.refute([row['draw_identity']],reason='original target counterexample')['affected']
    assert bundle['library']['identity'] in affected
    assert target['identity'] in affected
    with pytest.raises(ValueError,match='quarantined'):
        journal.training({'component_distillation':freeze_distillation(bundle,[target],arm='library')})
