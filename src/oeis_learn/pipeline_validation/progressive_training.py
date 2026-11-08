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


def resolve_update_policy(
    profile,
    *,
    schedule_horizon_tokens,
    lr_multiplier=1,
    schedule="constant",
    effective_batch=16,
    microbatch_size=16,
    weight_decay=0.01,
    warmup_tokens=None,
):
    """Resolve learning semantics, not campaign/task execution authority.

    The LR for a logical update is evaluated at its reserved charged-token end
    coordinate. A failed attempt retains its debit but never advances the update
    clock; retries use their new cumulative charge. Warmup and decay horizons are
    frozen separately so a confirmation milestone cannot restart the schedule.
    """
    import math

    from oeis_learn.generalization.progressive.models import validate_profile

    p = validate_profile(profile)
    if (
        type(lr_multiplier) not in (int, float)
        or lr_multiplier not in (0.3, 1, 3)
        or type(weight_decay) not in (int, float)
        or weight_decay not in (0, 0.01, 0.1)
        or type(effective_batch) is not int
        or effective_batch not in (16, 32, 64)
        or type(microbatch_size) is not int
        or not 1 <= microbatch_size <= effective_batch
        or schedule not in ("constant", "warmup_cosine")
        or type(schedule_horizon_tokens) is not int
        or not 1 <= schedule_horizon_tokens <= 2**63 - 1
    ):
        raise ValueError("unsupported research update settings")
    if warmup_tokens is None:
        warmup_tokens = (schedule_horizon_tokens + 19) // 20 if schedule == "warmup_cosine" else 0
    if (
        type(warmup_tokens) is not int
        or (schedule == "constant" and warmup_tokens != 0)
        or (schedule == "warmup_cosine" and not 0 < warmup_tokens < schedule_horizon_tokens)
    ):
        raise ValueError("explicit compatible schedule horizon/warmup required")
    base = p["base_recipe"]["learning"]["learning_rate"]
    if type(base) not in (int, float) or not math.isfinite(base * lr_multiplier) or base <= 0:
        raise ValueError("positive finite verified baseline learning rate required")
    return seal(
        {
            "version": "progressive-update-policy/v1",
            "model_profile": p["identity"],
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": float(weight_decay),
            "baseline_lr": base,
            "lr_multiplier": float(lr_multiplier),
            "schedule": schedule,
            "schedule_horizon_tokens": schedule_horizon_tokens,
            "warmup_tokens": warmup_tokens,
            "final_multiplier": 0.1,
            "effective_batch": effective_batch,
            "microbatch_size": microbatch_size,
            "dropout": p["model"]["dropout"],
            "precision": "float32",
            "normalization": "mean-per-program",
            "clip_norm": 1.0,
            "coordinate": "reserved-charged-token-end",
            "checkpoint": "complete-update",
        }
    )


def validate_update_policy(profile, policy):
    keys = (
        "schedule_horizon_tokens",
        "lr_multiplier",
        "schedule",
        "effective_batch",
        "microbatch_size",
        "weight_decay",
        "warmup_tokens",
    )
    if not isinstance(policy, dict) or not set(keys) <= set(policy):
        raise ValueError("closed research update policy required")
    expected = resolve_update_policy(profile, **{k: policy[k] for k in keys})
    # Canonical serialization also distinguishes booleans from numeric constants.
    if digest(expected) != digest(policy):
        raise ValueError("research update policy/model binding changed")
    return expected


def learning_rate_at(policy, tokens):
    """Token coordinate, never a microbatch count or a restarted task clock."""
    import math

    if type(tokens) is not int or tokens < 0:
        raise ValueError("nonnegative integer schedule coordinate required")
    base = policy["baseline_lr"] * policy["lr_multiplier"]
    if policy["schedule"] == "constant":
        return base
    if policy["schedule"] != "warmup_cosine":
        raise ValueError("unsupported schedule")
    warmup, horizon = policy["warmup_tokens"], policy["schedule_horizon_tokens"]
    if tokens < warmup:
        return base * tokens / warmup
    fraction = min(1, (tokens - warmup) / (horizon - warmup))
    final = policy["final_multiplier"]
    return base * (final + (1 - final) * (1 + math.cos(math.pi * fraction)) / 2)


class TokenSchedule:
    """One prepare/step pair per complete logical update; JSON-compatible state."""

    def __init__(self, optimizer, policy):
        self.optimizer, self.policy = optimizer, policy
        self.tokens = self.updates = 0
        self.pending = None

    def prepare(self, tokens):
        if self.pending is not None or type(tokens) is not int or tokens <= self.tokens:
            raise ValueError("new cumulative charged-token coordinate required")
        self.pending = tokens
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate_at(self.policy, tokens)

    def step(self):
        if self.pending is None:
            raise ValueError("no logical update prepared")
        self.tokens = self.pending
        self.updates += 1
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("scheduler checkpoint requires complete update")
        return {
            "version": "progressive-token-schedule/v1",
            "policy": self.policy["identity"],
            "tokens": self.tokens,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "policy", "tokens", "updates"}
            or state["version"] != "progressive-token-schedule/v1"
            or state["policy"] != self.policy["identity"]
            or any(type(state[k]) is not int or state[k] < 0 for k in ("tokens", "updates"))
            or (state["updates"] == 0) != (state["tokens"] == 0)
            or state["updates"] > state["tokens"]
        ):
            raise ValueError("incompatible complete scheduler state")
        self.tokens, self.updates, self.pending = state["tokens"], state["updates"], None


class PolicyUpdate:
    """Research model/optimizer adapter over the canonical shared SFT operation.

    Callers own reserve persistence, complete model/RNG checkpoint publication and
    task entitlement. Supply an already debited Ledger; failures preserve that
    immutable debit and poison this adapter until a *new* adapter is restored
    from the prior full cut. No partial-gradient/weights-only resume is offered.
    T049/T050 own installed full-state recovery, not this policy adapter.
    """

    def __init__(self, profile, policy, *, device):
        import torch

        from oeis_learn.generalization.progressive.models import construct

        self.policy = validate_update_policy(profile, policy)
        self.encoder, self.decoder, self.counts = construct(profile, device=device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            [*self.encoder.parameters(), *self.decoder.parameters()],
            lr=learning_rate_at(self.policy, 0),
            weight_decay=self.policy["weight_decay"],
            betas=tuple(self.policy["betas"]),
            eps=self.policy["eps"],
            amsgrad=False,
            maximize=False,
            foreach=False,
            fused=False,
        )
        self.scheduler = TokenSchedule(self.optimizer, self.policy)
        self.active = self.failed = False

    def state_dict(self):
        import copy

        if self.active or self.failed:
            raise ValueError("checkpoint requires a complete successful update boundary")
        return copy.deepcopy(
            {
                "version": "progressive-update-state/v1",
                "policy": self.policy["identity"],
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
        )

    def load_state_dict(self, state):
        # This is only the optimizer/schedule slice of a full checkpoint. The
        # existing store/caller must restore weights, RNG, data and spent ledger.
        if self.active or self.failed or self.scheduler.updates or self.optimizer.state:
            raise ValueError("restore a new adapter from a complete checkpoint")
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "policy", "optimizer", "scheduler"}
            or state["version"] != "progressive-update-state/v1"
            or state["policy"] != self.policy["identity"]
        ):
            raise ValueError("incompatible optimizer policy")
        checked = TokenSchedule(self.optimizer, self.policy)
        checked.load_state_dict(state["scheduler"])
        groups = state["optimizer"].get("param_groups", [])
        current = self.optimizer.state_dict()["param_groups"]
        if len(groups) != len(current):
            raise ValueError("optimizer groups changed")
        for group, original in zip(groups, current):
            expected = dict(original, lr=learning_rate_at(self.policy, checked.tokens))
            if group != expected:
                raise ValueError("optimizer settings changed")
        import torch

        moments = state["optimizer"].get("state", {})
        if bool(moments) != bool(checked.updates):
            raise ValueError("optimizer and scheduler update counts disagree")
        for value in moments.values():
            if (
                set(value) != {"step", "exp_avg", "exp_avg_sq"}
                or any(
                    not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all())
                    for v in value.values()
                )
                or value["step"].numel() != 1
                or float(value["step"]) != checked.updates
            ):
                raise ValueError("invalid finite complete AdamW state")
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except BaseException:
            self.failed = True
            raise
        self.scheduler = checked

    def update(self, sequences, tokens, *, ledger, allocation=None):
        import torch

        from oeis_learn.decoder.bigint_codec import PAD_ID
        from oeis_learn.rl.sft_trainer import logical_update

        if self.active or self.failed:
            raise ValueError("restore a complete checkpoint before retry")
        n = self.policy["effective_batch"]
        if (
            len(sequences) != n
            or tokens.ndim != 2
            or tokens.shape[0] != n
            or tokens.shape[1] < 2
            or tokens.dtype != torch.long
            or not bool(tokens[:, 1:].ne(PAD_ID).any(dim=1).all())
        ):
            raise ValueError("exact effective batch with nonempty targets required")
        charge = int(tokens[:, 1:].ne(PAD_ID).sum())
        if (
            not ledger.debited
            or ledger.pending is None
            or ledger.pending[1] != charge
            or ledger.updates != self.scheduler.updates
            or ledger.consumed < self.scheduler.tokens + charge
        ):
            raise ValueError("matching full-update debit required before learning")
        if allocation is not None and (
            allocation.updates != ledger.updates
            or allocation.pending is None
            or allocation.block_position >= 8
        ):
            raise ValueError("selected allocation block with training feedback required")

        class AllocationClock:
            def step(self):
                allocation.commit(tokens=charge, success=True)

        self.active = True
        try:
            self.scheduler.prepare(ledger.consumed)
            self.encoder.train()
            self.decoder.train()
            result = logical_update(
                self.encoder,
                self.decoder,
                self.optimizer,
                sequences,
                tokens,
                device=self.device,
                pad_id=PAD_ID,
                microbatch_size=self.policy["microbatch_size"],
                scheduler=self.scheduler,
                allocation_clock=AllocationClock() if allocation is not None else None,
                observe=False,
                learning_metrics=True,
            )
            committed = ledger.commit(ledger.pending[0])
            return dict(
                result,
                examples=n,
                charged_tokens=charge,
                microbatches=(n + self.policy["microbatch_size"] - 1)
                // self.policy["microbatch_size"],
                learning_rate=self.optimizer.param_groups[0]["lr"],
            ), committed
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False


class RecoverableUpdate:
    """Canonical learner transactions under an already-owned recovery/task lease.

    The supervisor owns begin/close and termination proofs. This adapter performs
    no task dispatch and cannot create a new reservation after a terminal task.
    """

    def __init__(self, recovery, profile, policy, *, device):
        import copy

        self.recovery = recovery
        self.runner = PolicyUpdate(profile, policy, device=device)
        self.failed = False
        self.context = None
        if recovery.state["checkpoint"] is not None:
            restored = recovery.restore()
            state = restored["states"]
            self.runner.encoder.load_state_dict(state["weights"]["encoder"])
            self.runner.decoder.load_state_dict(state["weights"]["decoder"])
            self.runner.load_state_dict(state["optimizer"])
            self.context = copy.deepcopy(
                {k: v for k, v in state.items() if k not in {"weights", "optimizer", "rng"}}
            )
            self._rng_restore(state["rng"])

    @staticmethod
    def _rng_restore(rng):
        import numpy as np
        import torch

        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["device"]:
            if not torch.cuda.is_available():
                raise ValueError("checkpoint device RNG requires actual device")
            torch.cuda.set_rng_state_all(rng["device"])

    def capture(self, context):
        import copy

        import numpy as np
        import torch

        if set(context) != {"scheduler", "data", "replay", "history", "pins", "selection"}:
            raise ValueError("complete stream/allocation/archive context required")
        result = copy.deepcopy(context)
        result.update(
            weights={
                "encoder": self.runner.encoder.state_dict(),
                "decoder": self.runner.decoder.state_dict(),
            },
            optimizer=self.runner.state_dict(),
            rng={
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "device": torch.cuda.get_rng_state_all() if self.runner.device == "cuda" else [],
            },
        )
        return result

    def initialize(self, context, cuts):
        record = self.recovery.initialize(self.capture(context), cuts)
        self.context = context
        return record

    def step(
        self, sequences, tokens, *, entries, update, context, cuts, boot, monotonic_us, costs=None
    ):
        import copy

        from oeis_learn.decoder.bigint_codec import PAD_ID

        if self.failed:
            raise ValueError("restore committed full state before retry")
        batch = {
            "entries": entries,
            "inputs": digest(sequences),
            "targets": digest(tokens.tolist()),
        }
        prior = self.recovery.completed(update, batch)
        if prior is not None:
            restored = self.recovery.store.load(prior)
            return restored["states"]["history"][f"recovery-update/{update}"], prior
        count = int(tokens[:, 1:].ne(PAD_ID).sum())
        ledger = self.recovery.reserve(
            batch, count, boot=boot, monotonic_us=monotonic_us, costs=costs
        )
        try:
            metrics, ledger = self.runner.update(sequences, tokens, ledger=ledger)
            next_context = copy.deepcopy(context)
            next_context["history"][f"recovery-update/{update}"] = metrics
            next_context["data"] = {"cursor": ledger.updates, "pending_batch": None}
            record = self.recovery.commit(self.capture(next_context), ledger, cuts)
            self.context = next_context
            return metrics, record
        except BaseException:
            self.failed = True
            raise


class Progress:
    """Bounded <=60s sampler; overlapping allocator counters are never summed."""

    def __init__(self, output, *, interval=60):
        if type(interval) not in (int, float) or not 0 < interval <= 60:
            raise ValueError("progress interval must be at most sixty seconds")
        self.output = Path(output)
        self.interval = interval
        self.last = None
        self.ordinal = 0

    def sample(self, *, committed, charged, discarded, remaining, phase, device="cpu", force=False):
        import shutil
        from oeis_learn.tracking.foundation_metrics import resource_metrics, sensor_metrics

        now = time.monotonic()
        if not force and self.last is not None and now - self.last < self.interval:
            return None
        for n in (committed, charged, discarded):
            if type(n) is not int or n < 0:
                raise ValueError("exact nonnegative exposure required")
        if committed + discarded > charged:
            raise ValueError("exposure accounting differs")
        self.output.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(self.output)
        value = seal(
            dict(
                version="progressive-progress/v1",
                phase=phase,
                committed=committed,
                charged=charged,
                discarded=discarded,
                remaining=remaining,
                resources=resource_metrics(device),
                sensors=sensor_metrics(device),
                free_disk_bytes=disk.free,
                memory_accounting="overlapping counters reported separately",
                generation_tail=None,
                queue=None,
                fit=None,
                unavailable_reason="requires generator/native producer clocks",
            )
        )
        if self.ordinal >= 65536:
            raise ValueError("bounded progress inventory exhausted")
        atomic_write(self.output / f"progress-{self.ordinal:05d}.json", value)
        self.ordinal += 1
        self.last = now
        return value

    def preflight(
        self, *, bytes_required, inodes_required, remaining_us, finalization_us, research
    ):
        import os
        import shutil

        if any(
            type(v) is not int or v < 0
            for v in (bytes_required, inodes_required, remaining_us, finalization_us)
        ):
            raise ValueError("bounded resource preflight required")
        if research and finalization_us < 1800000000:
            raise ValueError("research publication reserve below thirty minutes")
        self.output.mkdir(parents=True, exist_ok=True)
        fs = os.statvfs(self.output)
        if (
            shutil.disk_usage(self.output).free < bytes_required
            or fs.f_favail < inodes_required
            or remaining_us <= finalization_us
        ):
            raise TimeoutError("storage/inodes/finalization reserve insufficient")


class Heartbeat:
    """Independent liveness sampling while a forward pass or checkpoint blocks."""

    def __init__(self, progress, *, device, initial):
        import threading

        self.progress = progress
        self.device = device
        self.state = dict(initial)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="discovery-heartbeat")

    def update(self, **values):
        with self.lock:
            self.state.update(values)
        if self.error is not None:
            raise RuntimeError("heartbeat publication failed") from self.error

    def emit(self):
        with self.lock:
            value = dict(self.state)
        self.progress.sample(**value, device=self.device, force=True)

    def _run(self):
        while not self.stop_event.wait(self.progress.interval):
            try:
                self.emit()
            except BaseException as exc:
                self.error = exc
                return

    def __enter__(self):
        self.emit()
        self.thread.start()
        return self

    def __exit__(self, kind, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("heartbeat did not stop")
        if self.error is not None and kind is None:
            raise RuntimeError("heartbeat publication failed") from self.error
        if kind is None:
            self.emit()
