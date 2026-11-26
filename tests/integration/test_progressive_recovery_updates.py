"""Real model/AdamW/dropout transactions, including lost commit acknowledgements."""

import copy

import pytest
import torch

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate
from tests.helpers.progressive_fixture import caps, task
from tests.unit.test_progressive_training_policy import batch, policy, profile


@pytest.mark.parametrize(
    "point", ["reserved", "microbatch", "optimizer", "checkpoint", "committed", "matched_microbatch"]
)
def test_exact_model_retry_and_irreversible_charge(tmp_path, monkeypatch, point):
    treatment = point == "matched_microbatch"
    if treatment:
        point = "microbatch"
        from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
        p = profile()
        cfg = resolve_update_policy(p, objective="first_divergence", effective_batch=32, microbatch_size=7,
                                    weight_decay=0, schedule_horizon_tokens=4_000_000)
    else:
        p = profile(0.2)
        cfg = policy(p, microbatch_size=7)
    t = task(caps=caps(tokens=10000, elapsed_us=60000000))
    j = BudgetJournal(tmp_path, t["payload"]["campaign"], caps(tokens=10000, elapsed_us=60000000))
    context = {k: {} for k in ("scheduler", "data", "replay", "history", "pins", "selection")}
    context["data"] = {"cursor": 0, "pending_batch": None}
    context["scheduler"] = {"stream": {"shard": 0, "cursor": 0}, "archive": "old", "library": "old"}
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
    x, tokens, pad = batch(cfg["effective_batch"])
    preferences = {"preference_labels":[dict(position=0, positive=tokens[0,1].item(), negative=tokens[0,2].item())]
                   + [None]*(len(x)-1)} if treatment else {}
    charge = int(tokens[:, 1:].ne(pad).sum())
    after = copy.deepcopy(context)
    after["scheduler"]["stream"]["cursor"] = len(x)
    kwargs = {
        **preferences,
        "entries": ["source"] * len(x),
        "update": 1,
        "context": after,
        "cuts": cuts,
        "boot": "boot",
        "monotonic_us": 1,
    }
    with Recovery(j, t, digest([p, cfg])) as r:
        worker = RecoverableUpdate(r, p, cfg, device="cpu")
        worker.initialize(context, cuts)
        r.begin(boot="boot", monotonic_us=0)
        # Independent uninterrupted canonical update from the same initial cut.
        initial = r.restore()["states"]
        from oeis_learn.generalization.budgets import Ledger
        from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate

        gold = PolicyUpdate(p, cfg, device="cpu")
        gold.encoder.load_state_dict(initial["weights"]["encoder"])
        gold.decoder.load_state_dict(initial["weights"]["decoder"])
        gold.load_state_dict(initial["optimizer"])
        RecoverableUpdate._rng_restore(initial["rng"])
        gold.update(x, tokens, ledger=Ledger(10000).reserve("gold", charge), **preferences)
        expected = [
            v.detach().clone() for v in [*gold.encoder.parameters(), *gold.decoder.parameters()]
        ]
        expected_rng = torch.get_rng_state()
        RecoverableUpdate._rng_restore(initial["rng"])
        with monkeypatch.context() as m:

            def fail(*args, **kw):
                raise InterruptedError(point)

            if point == "reserved":
                m.setattr(worker.runner, "update", fail)
            elif point == "microbatch":
                forward = worker.runner.decoder.forward
                calls = []

                def interrupted(*args, **kw):
                    calls.append(1)
                    if len(calls) == 2:
                        raise InterruptedError(point)
                    return forward(*args, **kw)

                m.setattr(worker.runner.decoder, "forward", interrupted)
            elif point == "optimizer":
                m.setattr(r, "commit", fail)
            elif point == "checkpoint":
                append = r.append

                def failed_publication(**kw):
                    if "checkpoint" in kw:
                        raise InterruptedError(point)
                    return append(**kw)

                m.setattr(r, "append", failed_publication)
            else:
                commit = r.commit

                def lost_ack(*args, **kw):
                    commit(*args, **kw)
                    raise InterruptedError(point)

                m.setattr(r, "commit", lost_ack)
            with pytest.raises(InterruptedError):
                worker.step(x, tokens, **kwargs)
        r.close(boot="boot", monotonic_us=100)
    with Recovery(j, t, digest([p, cfg])) as r:
        r.begin(boot="new-boot", monotonic_us=1000)
        restored = RecoverableUpdate(r, p, cfg, device="cpu")
        _, record = restored.step(x, tokens, **dict(kwargs, boot="new-boot", monotonic_us=1001))
        actual = [*restored.runner.encoder.parameters(), *restored.runner.decoder.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert torch.equal(torch.get_rng_state(), expected_rng)
        receipt = r.restore()
        assert receipt["ledger"].consumed == charge * (1 if point == "committed" else 2)
        assert receipt["discarded_tokens"] == (0 if point == "committed" else charge)
        assert receipt["states"]["scheduler"] == after["scheduler"]
        before = j.snapshot()["consumed"]
        _, replay = restored.step(x, tokens, **dict(kwargs, boot="new-boot", monotonic_us=1002))
        assert replay == record and j.snapshot()["consumed"] == before
        with pytest.raises(ValueError, match="conflicting"):
            restored.step(
                x, tokens, **dict(kwargs, entries=["other"] * len(x), boot="new-boot", monotonic_us=1003)
            )
        r.close(boot="new-boot", monotonic_us=1100)


def test_optimizer_growth_rejected_before_new_generation_preserves_old_cut(tmp_path):
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger

    p = profile(0)
    cfg = policy(p, microbatch_size=7)
    t = task(caps=caps(tokens=10000, elapsed_us=60000000))
    j = BudgetJournal(tmp_path, t["payload"]["campaign"], caps(tokens=10000, elapsed_us=60000000))
    context = {k: {} for k in ("scheduler", "data", "replay", "history", "pins", "selection")}
    context["data"] = dict(cursor=0, pending_batch=None)
    cuts = dict.fromkeys(
        (
            "stream",
            "allocation",
            "archive",
            "library",
            "cost_journal",
            "selection",
            "milestones",
            "consumed",
        ),
        digest(None),
    )
    with Recovery(j, t, digest([p, cfg])) as recovery:
        worker = RecoverableUpdate(recovery, p, cfg, device="cpu")
        worker.initialize(context, cuts)
        original = recovery.state["checkpoint"]
        # Set the frozen envelope at the initial generation, before AdamW moments.
        captured = worker.capture(context)
        parts = [v for k, v in captured.items() if k not in {"rng", "data"}] + list(
            captured["rng"].values()
        )
        recovery.checkpoint_bytes = sum(
            len(GeneralizationCheckpointStore._encode(v)) for v in parts
        ) + (128 << 10)
        files = {str(p): p.read_bytes() for p in recovery.store.root.rglob("*") if p.is_file()}
        x, tokens, pad = batch()
        charge = int(tokens[:, 1:].ne(pad).sum())
        worker.runner.update(x, tokens, ledger=Ledger(10000).reserve("update", charge))
        after = copy.deepcopy(context)
        after["data"]["cursor"] = 1
        with pytest.raises(TimeoutError, match="checkpoint growth"):
            recovery.save(worker.capture(after), Ledger(10000, consumed=charge, updates=1), cuts)
        assert {
            str(p): p.read_bytes() for p in recovery.store.root.rglob("*") if p.is_file()
        } == files
        recovery.restore()
        assert recovery.state["checkpoint"] == original


def test_matched_generator_updates_restore_the_same_rows_clocks_and_initial_weights(tmp_path):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy

    p = profile()
    x, tokens, pad = batch(32)
    pos = tokens[0, 1].item()
    labels = [dict(position=0, positive=pos, negative=tokens[0, 2].item())] + [None] * 31
    count = int(tokens[:, 1:].ne(pad).sum())
    summaries = {}
    for arm in ("ordinary", "first_divergence"):
        cfg = resolve_update_policy(p, objective=arm, schedule_horizon_tokens=4_000_000,
                                    effective_batch=32, microbatch_size=7, weight_decay=0)
        t = task(arm=arm, caps=caps(tokens=10000))
        j = BudgetJournal(tmp_path / arm, t["payload"]["campaign"], caps(tokens=10000))
        context = {k: {} for k in ("scheduler", "data", "replay", "history", "pins", "selection")}
        context["data"] = {"cursor": 0, "pending_batch": None}
        cuts = {k: digest(k) for k in ("stream", "allocation", "archive", "library", "cost_journal",
                                       "selection", "milestones", "consumed")}
        args = dict(entries=["same-row"] * 32, update=1, context=context, cuts=cuts,
                    boot="test", monotonic_us=1, preference_labels=labels if arm == "first_divergence" else None)
        with Recovery(j, t, digest([p, cfg])) as r:
            worker = RecoverableUpdate(r, p, cfg, device="cpu")
            initial = model_snapshot(worker.runner.encoder, worker.runner.decoder)
            worker.initialize(context, cuts)
            r.begin(boot="test", monotonic_us=0)
            metrics, _ = worker.step(x, tokens, **args)
            last = model_snapshot(worker.runner.encoder, worker.runner.decoder)
        with Recovery(j, t, digest([p, cfg])) as r:
            worker = RecoverableUpdate(r, p, cfg, device="cpu")
            assert model_snapshot(worker.runner.encoder, worker.runner.decoder) == last
            assert worker.step(x, tokens, **args)[0] == metrics
            if arm == "first_divergence":
                with pytest.raises(ValueError, match="conflicting"):
                    worker.step(x, tokens, **dict(args, preference_labels=[None] * 32))
            restored = r.restore()
            summaries[arm] = (initial, last, restored["ledger"].consumed, restored["ledger"].updates)
    ordinary, treatment = summaries.values()
    assert ordinary[0] == treatment[0] and ordinary[1] != treatment[1]
    assert ordinary[2:] == treatment[2:] == (count, 1)


def test_rehearsal_measures_native_pair_signal_and_reloads_next_matched_update(tmp_path):
    from oeis_learn.generalization.progressive.training_rows import verified_training_pairs
    from oeis_learn.pipeline_validation.progressive_generator_probe import gradient, measured_updates, verify_sample
    from tests.unit.test_round028_training import training_rows

    rows = training_rows()
    labels = verified_training_pairs([rows])["pairs"]
    by_id = {r["draw_identity"]: r for r in rows}
    samples = [dict(positive=by_id[p["positive_row"]], negative=by_id[p["negative_row"]], label=p["label"])
               for p in labels]
    for s in samples:
        verify_sample(s)
    class Native:
        def execute(self, source, indices, **kw):
            return dict(outcome="ok", values=next(r["values"] for r in rows if r["source"] == source))
    result = gradient(profile(), samples, client=Native())
    assert result["native_verified_terms"] == 200 and result["gradient_l2"] > 0
    batches = [samples * 16, list(reversed(samples)) * 16]
    a, b = [measured_updates(profile(), batches, arm, tmp_path / arm, device="cpu", microbatch=7)
            for arm in ("ordinary", "first_divergence")]
    assert a["charged_tokens"] == b["charged_tokens"] and a["row_schedule"] == b["row_schedule"]
    assert a["initial"] == b["initial"] and a["final"] != b["final"] and b["preference_rows"] == 32
    assert all(v["full_restore"] and v["checkpoint_bytes"] > 0 and v["restore_us"] > 0 for v in (a, b))


def test_matched_checkpoint_replays_positive_token_exposure_and_rejects_changed_cut():
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.training_rows import matched_batch, matched_milestone, exposure_at
    from oeis_learn.generalization.progressive.contracts import seal
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts
    _, _, cuts = generator_design_and_cuts(1_000_000)
    cut = cuts["ordinary"]
    stream = seal(dict(order=["a", "b", "c"], registration=cut["registration"], seeds=cut["seeds"], pair=0))
    rows = {k:{"tokens":list(range(n))} for k,n in zip(stream["order"], (8,12,21))}
    planned = exposure_at(stream, rows, 3)
    direct = [matched_batch(stream, 32*i, 32) for i in range(3)]
    replay = digest(None)
    for selected in direct:
        replay = digest([replay, selected])
    assert planned["order_digest"] == replay and planned["useful_tokens"] == sum(len(rows[k]["tokens"]) for b in direct for k in b)
    pkg = {"matched_generator":stream, "profile":{"codec":cut["codec"], "model":{"seed":1}, "inputs":{}},
           "update_policy":{"objective":"ordinary", "identity":digest("policy")}}
    selection = {"matched_stream":stream["identity"], "initial_weights":cut["initial"]}
    state = {"selection":selection, "data":{"cursor":3}, "scheduler":{"controller":{"cursor":96}},
             "optimizer":{"policy":pkg["update_policy"]["identity"]}}
    restored = dict(states=state, ledger=SimpleNamespace(updates=3,consumed=planned["useful_tokens"]+100),
                    cuts={"selection":digest(selection), "consumed":replay}, discarded_tokens=100)
    record = matched_milestone(pkg, rows, {"identity":cut["parent"]}, restored, planned["useful_tokens"], 1_000_000)
    assert record["charged"]-record["useful_tokens"] == 100
    with pytest.raises(ValueError, match="independent matched exposure"):
        matched_milestone(pkg, rows, {"identity":cut["parent"]},
                          dict(restored, cuts=dict(restored["cuts"], consumed=digest("wrong"))), planned["useful_tokens"], 1_000_000)
