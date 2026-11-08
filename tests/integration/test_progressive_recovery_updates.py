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
    "point", ["reserved", "microbatch", "optimizer", "checkpoint", "committed"]
)
def test_exact_model_retry_and_irreversible_charge(tmp_path, monkeypatch, point):
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
    x, tokens, pad = batch()
    charge = int(tokens[:, 1:].ne(pad).sum())
    after = copy.deepcopy(context)
    after["scheduler"]["stream"]["cursor"] = 16
    kwargs = {
        "entries": ["source"] * 16,
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
        gold.update(x, tokens, ledger=Ledger(10000).reserve("gold", charge))
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
                x, tokens, **dict(kwargs, entries=["other"], boot="new-boot", monotonic_us=1003)
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
