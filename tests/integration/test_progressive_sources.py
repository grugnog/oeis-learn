"""Actual four-input canonical update, restore, generation and native score."""

import copy
import torch
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.generalization.progressive.combinatorial import prepare
from oeis_learn.generalization.progressive.short_profile import admit, visible
from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate, short_update
from oeis_learn.pipeline_validation.progressive_evaluation import generate_short, score_short
from tests.helpers.progressive_fixture import caps, task
from tests.unit.test_progressive_training_policy import profile, policy
from tests.unit.test_bigint_reference import FACTORIAL

SOURCE = FACTORIAL.replace(
    "(return (get $b0))", "(return (select (eq $index (int 0)) (int 0) (get $b0)))"
)


def test_actual_short_update_restore_generation_and_native_score(tmp_path, fitting_client):
    row = prepare(["fixed_points"], moments=[1])["rows"][0]
    truth = {k: row[k] for k in ("indices", "values")}
    target = admit(
        SOURCE, truth, member="fixed-points-first", client=fitting_client, purpose="acceptance"
    )
    assert target is not None
    assert (
        admit(
            "(program (return (int 0)))",
            truth,
            member="bad",
            client=fitting_client,
            purpose="acceptance",
        )
        is None
    )
    from oeis_learn.pipeline_validation.progressive_device_acceptance import device, selected_profile
    p = selected_profile(profile(0.1))
    execution_device = device("acceptance")
    cfg = policy(p)
    from oeis_learn.generalization.progressive.contracts import seal

    t = seal(dict(task(caps=caps(tokens=10000, elapsed_us=60000000)), track="external_assisted"))
    j = BudgetJournal(tmp_path, t["payload"]["campaign"], caps(tokens=10000, elapsed_us=60000000))
    context = {k: {} for k in ("scheduler", "data", "replay", "history", "pins", "selection")}
    context["data"] = {"cursor": 0, "pending_batch": None}
    cuts = {
        k: digest(k)
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
    with Recovery(j, t, digest([p, cfg, "combinatorial-short/v1"])) as r:
        w = RecoverableUpdate(r, p, cfg, device=execution_device)
        w.initialize(context, cuts)
        before = [v.detach().clone() for v in w.runner.encoder.parameters()]
        r.begin(boot="boot", monotonic_us=0)
        metrics, _ = short_update(
            w, [target] * 16, update=1, context=context, cuts=cuts, boot="boot", monotonic_us=1
        )
        assert metrics["loss"] > 0
        assert any(not torch.equal(a, b) for a, b in zip(before, w.runner.encoder.parameters()))
        saved = r.restore()["states"]
        restored = RecoverableUpdate(r, p, cfg, device=execution_device)
        for name, model in [
            ("encoder", restored.runner.encoder),
            ("decoder", restored.runner.decoder),
        ]:
            assert all(
                torch.equal(v.cpu(), saved["weights"][name][k].cpu()) for k, v in model.state_dict().items()
            )
        assert restored.runner.state_dict()["scheduler"] == w.runner.state_dict()["scheduler"]
        assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a.cpu(), b.cpu())
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a:
                    equal(a[k], b[k])
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                assert a == b

        equal(restored.runner.state_dict(), saved["optimizer"])
        generated = generate_short(
            restored.runner.encoder,
            restored.runner.decoder,
            [visible(truth)],
            seed=13,
            max_tokens=16,
        )
        assert len(generated) == 1 and len(generated[0]["tokens"]) <= 16
        assert score_short(SOURCE, truth, client=fitting_client)["agreement"]
        assert not score_short("(program (return (int 0)))", truth, client=fitting_client)[
            "agreement"
        ]
        assert not score_short(SOURCE, truth, client=fitting_client)["primary_scaling_eligible"]
        from tests.contract.test_progressive_sources import source
        from oeis_learn.pipeline_validation.progressive_commands import (
            prepare_source_pilot,
            source_pilot_cycle,
        )

        sources = []
        roles = []
        for group in range(8):
            for variant in range(2):
                values = (
                    truth["values"]
                    if group == variant == 0
                    else [str(100 * group + 10 * variant + n * n + 1000) for n in range(9)]
                )
                sources.append(source(values=values, relations=[f"group:{group}"]))
                roles.append("training" if group < 4 else "evaluation")
        feasibility = prepare_source_pilot(sources, roles)
        assert feasibility["status"] == "ELIGIBLE"
        cycle = source_pilot_cycle(
            restored,
            [visible(truth)],
            [truth],
            client=fitting_client,
            seed=13,
            base_targets=[target] * 16,
            arm="without_source",
            feasibility=feasibility,
            sources=sources,
            roles=roles,
            transaction=dict(
                update=2,
                context=copy.deepcopy(restored.context),
                cuts=cuts,
                boot="boot",
                monotonic_us=2,
            ),
            max_tokens=1,
        )
        assert cycle["status"] == "COMPLETE" and not cycle["primary_scaling_eligible"]


def test_source_deficit_and_unsuitable_are_explicit():
    from tests.contract.test_progressive_sources import source
    from oeis_learn.generalization.progressive.sources import pilot_support

    s = source()
    result = pilot_support([s], ["training"])
    assert result["status"] == "SOURCE_DEFICIT"
    assert result["deficits"] == dict(training=4, evaluation=4)
    import pytest

    with pytest.raises(ValueError):
        pilot_support([s, s], ["training", "evaluation"])
