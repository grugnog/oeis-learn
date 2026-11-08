"""Prepared support adapter and bounded canonical-learner acceptance rehearsal.

Research dispatch/24h entitlement belongs to the later registered training role.
This rehearsal never grants research authority or bypasses device qualification.
"""

import json
import random
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    validate,
    verify_refs,
)
from oeis_learn.generalization.progressive.curriculum import Controller
from oeis_learn.generalization.progressive.stream import load_shard
from oeis_learn.pipeline_validation.identity import confined, seal


class PreparedSupport:
    def __init__(self, root, shards, *, arm, refresh_updates, diagnostic=False):
        if (
            arm not in {"static", "refreshed"}
            or type(refresh_updates) is not int
            or refresh_updates < 1
        ):
            raise ValueError("fixed matched support policy required")
        if not shards:
            raise ValueError("initial sealed shard required")
        self.root = Path(root)
        self.shards = list(shards)
        self.arm = arm
        self.refresh_updates = refresh_updates
        self.rows = []
        for shard in shards:
            p = validate("stream", shard)["payload"]
            if p["source_policy"] != shards[0]["payload"]["source_policy"]:
                raise ValueError("shared structural source law required")
            if diagnostic and not p["native_verified"]:
                if shard["purpose"] != "acceptance":
                    raise ValueError("diagnostic shards cannot become research")
                blobs = verify_refs(root, p["rows"], max_bytes=256 << 20)
                rows = [json.loads(b) for b in blobs]
                if len(rows) != p["accepted"] or any(
                    seal(r) != r or r["membership"] != "training" for r in rows
                ):
                    raise ValueError("invalid diagnostic training shard")
            else:
                rows = load_shard(root, shard)
            if not rows:
                raise ValueError("nonempty verified training support required")
            self.rows.append(rows)
        initial = self.rows[0]

        def bin_of(row):
            return "frontier" if {"loop", "select"} & set(row["operations"]) else "familiar"

        self.bin_of = bin_of
        self.probes = {}
        for b in ("familiar", "frontier"):
            eligible = [r for r in initial if bin_of(r) == b]
            if len(eligible) < 2:
                raise ValueError("initial support/probe deficit in " + b)
            self.probes[b] = eligible[0]
        self.layout = {
            "bins": ["familiar", "frontier"],
            "probes": {
                b: {"role": "training", "id": r["draw_identity"]} for b, r in self.probes.items()
            },
        }
        self.excluded = {r["draw_identity"] for r in self.probes.values()}

    def batch(self, controller, count):
        selected = controller.select()["bin"]
        index = (
            0
            if self.arm == "static" or selected == "familiar"
            else controller.updates // self.refresh_updates
        )
        if index >= len(self.rows):
            raise TimeoutError("preallocated refreshed shards exhausted")
        available = [
            r
            for r in self.rows[index]
            if self.bin_of(r) == selected and r["draw_identity"] not in self.excluded
        ]
        if not available:
            raise ValueError("frozen bin support deficit")
        rows = [available[(controller.cursor + i) % len(available)] for i in range(count)]
        return [dict(r, values=r["values"][:20]) for r in rows], index


def acceptance_pilot(
    root,
    output,
    shards,
    recipe,
    *,
    arm,
    seed,
    updates=2,
    refresh_updates=1,
    batch_size=1,
    token_cap=16384,
    seconds=60,
    resume=False,
    fault=None,
    diagnostic_shards=False,
    target_tokens=None,
):
    """Real CPU SFT + exact full-state recovery, explicitly DIAGNOSTIC_ONLY."""
    began = time.monotonic()
    import numpy as np
    import torch

    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.fresh.bigint_worker import _batch_loss
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.progressive.baselines import load_random
    from oeis_learn.pipeline_validation.generalization_curriculum import observe_targets

    if (
        type(updates) is not int
        or not 1 <= updates <= 16
        or type(batch_size) is not int
        or not 1 <= batch_size <= 8
        or type(token_cap) is not int
        or not 1 <= token_cap <= 131072
        or type(seconds) not in (int, float)
        or not 0 < seconds <= 600
        or fault not in (None, "after_reserve")
    ):
        raise ValueError("bounded acceptance rehearsal required")
    if target_tokens is not None and (
        type(target_tokens) is not int or not 1 <= target_tokens <= token_cap
    ):
        raise ValueError("nominal target must fit the frozen execution ceiling")
    deadline = began + seconds
    support = PreparedSupport(
        root, shards, arm=arm, refresh_updates=refresh_updates, diagnostic=diagnostic_shards
    )
    config = {
        "arm": arm,
        "seed": seed,
        "updates": updates,
        "refresh_updates": refresh_updates,
        "batch_size": batch_size,
        "token_cap": token_cap,
        "seconds": seconds,
        "diagnostic_shards": diagnostic_shards,
        "target_tokens": target_tokens,
    }
    contract = digest(
        {"config": config, "shards": [s["identity"] for s in shards], "recipe": recipe}
    )
    target = confined(root, output, exists=False)
    if target.exists() and not resume:
        raise ValueError("existing pilot requires explicit exact resume")
    target.mkdir(parents=True, exist_ok=True)
    store = GeneralizationCheckpointStore(target / "checkpoints", contract, token_cap)
    e, d, counts = load_random(recipe, seed=seed)
    optimizer = torch.optim.AdamW(
        [*e.parameters(), *d.parameters()],
        lr=recipe["learning"]["learning_rate"],
        weight_decay=0.01,
    )
    c = Controller(support.layout, seed=seed)
    ledger = Ledger(token_cap)
    state = {
        "weights": {},
        "optimizer": {},
        "scheduler": {
            "controller": c.snapshot(),
            "initial_weights": model_snapshot(e, d),
            "pre_loss": None,
        },
        "rng": {},
        "data": {"cursor": 0, "pending_batch": None},
        "replay": {},
        "history": {},
        "pins": {},
        "selection": {},
    }
    if resume:
        pointer = load(target / "checkpoints/generalization-latest.json")
        restored = store.load(pointer)
        state = restored["states"]
        ledger = restored["ledger"]
        e.load_state_dict(state["weights"]["encoder"])
        d.load_state_dict(state["weights"]["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        c = Controller.restore(state["scheduler"]["controller"], support.layout)
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if ledger.exhausted:
            raise ValueError("terminal token ceiling cannot resume")
        if ledger.pending is not None and ledger.debited:
            # Unproven work stays spent; repeat the same batch with a fresh debit.
            c.commit(tokens=ledger.pending[1], success=False)
            ledger = Ledger(token_cap, consumed=ledger.consumed, updates=ledger.updates)
    pointer = None

    def save():
        nonlocal pointer
        state["weights"] = {"encoder": e.state_dict(), "decoder": d.state_dict()}
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"]["controller"] = c.snapshot()
        state["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "device": [],
        }
        state["data"]["cursor"] = c.updates
        pointer = store.save(state, ledger)

    missing = []
    while (
        c.updates < updates
        and (target_tokens is None or ledger.consumed < target_tokens)
        and time.monotonic() < deadline
    ):
        if c.block_position == 8:
            b = c.pending["bin"]
            probe = support.probes[b]
            after = observe_targets(e, d, [dict(probe, values=probe["values"][:20])], device="cpu")
            before = state["scheduler"]["pre_loss"]
            gain = max(
                -1, min(1, (before - after["loss_sum"] / after["tokens"]) / max(abs(before), 1))
            )
            c.feedback(role="training", probe=probe["draw_identity"], gain=gain)
            state["scheduler"]["pre_loss"] = None
        try:
            batch, index = support.batch(c, batch_size)
        except (TimeoutError, ValueError) as exc:
            missing.append(str(exc))
            break
        if state["data"]["pending_batch"] is not None and batch != state["data"]["pending_batch"]:
            raise ValueError("resumed consumed targets differ")
        if c.block_position == 0 and state["scheduler"]["pre_loss"] is None:
            probe = support.probes[c.pending["bin"]]
            observed = observe_targets(
                e, d, [dict(probe, values=probe["values"][:20])], device="cpu"
            )
            state["scheduler"]["pre_loss"] = observed["loss_sum"] / observed["tokens"]
        tokens = sum(len(r["tokens"]) for r in batch)
        ledger = ledger.reserve(
            digest([c.updates, ledger.consumed, [r["draw_identity"] for r in batch]]), tokens
        )
        state["data"]["pending_batch"] = batch
        save()
        if not ledger.debited:
            missing.append("token_ceiling")
            break
        if fault == "after_reserve":
            raise InterruptedError("acceptance fault after token reservation")
        if time.monotonic() >= deadline:
            missing.append("parent_deadline")
            break
        e.train()
        d.train()
        metrics = _batch_loss(e, d, batch, "cpu", optimizer=optimizer, observe=False)
        ledger = ledger.commit(ledger.pending[0])
        c.commit(tokens=tokens, success=True)
        c.set_stream_cut(
            shard=max(c.shard, index),
            cursor=c.cursor + len(batch),
            rejection_watermark=max(c.rejection_watermark, shards[index]["payload"]["draw_high"]),
        )
        state["data"]["pending_batch"] = None
        state["history"][str(c.updates)] = {
            "loss": metrics["loss"],
            "tokens": tokens,
            "shard": index,
            "consumed": [r["draw_identity"] for r in batch],
        }
        save()
    if pointer is None:
        save()
    elapsed = time.monotonic() - began
    result = seal(
        {
            "version": "progressive-diversity-acceptance/v1",
            "kind": "DIAGNOSTIC_ONLY",
            "qualified": False,
            "status": "COMPLETE"
            if c.updates == updates
            or (target_tokens is not None and ledger.consumed >= target_tokens)
            else "PARTIAL",
            "contract": contract,
            "arm": arm,
            "seed": seed,
            "matched_policy": digest(
                {
                    "config": {k: v for k, v in config.items() if k != "arm"},
                    "recipe": recipe,
                    "shards": [s["identity"] for s in shards],
                }
            ),
            "initial_shard": shards[0]["identity"],
            "initial_weights": state["scheduler"]["initial_weights"],
            "final_weights": model_snapshot(e, d),
            "charged_tokens": ledger.consumed,
            "useful_tokens": ledger.consumed - c.discarded_tokens,
            "discarded_tokens": c.discarded_tokens,
            "updates": c.updates,
            "controller": c.snapshot(),
            "checkpoint": pointer,
            "elapsed_us": int(elapsed * 1e6),
            "equal_wall_tokens_per_second": ledger.consumed / elapsed if not resume else None,
            "equal_wall_limit": "resumed invocation total time unavailable" if resume else None,
            "full_update_overshoot": max(0, ledger.consumed - target_tokens)
            if target_tokens is not None
            else 0,
            "nominal_target_tokens": target_tokens,
            "preparation_work_units": shards[-1]["payload"]["work_units"],
            "preparation_elapsed_us": None,
            "preparation_timing_reason": "role-level preparation receipt required for total cost",
            "missing": missing,
            "counts": counts,
            "history": state["history"],
        }
    )
    # Each invocation has a new immutable result. Latest model state remains owned
    # by the existing checkpoint store; result publication cannot reset its ledger.
    atomic_write(
        target / ("result-" + str(c.updates) + "-" + str(ledger.consumed) + ".json"), result
    )
    return result
