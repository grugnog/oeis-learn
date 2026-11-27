"""Disposable next-update comparison from a complete, privately authenticated round028 cut."""

import os
from pathlib import Path
import time

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    atomic_write,
    digest,
    load,
    seal,
)
from oeis_learn.generalization.progressive.training_rows import matched_batch
from oeis_learn.generalization.progressive.velocity_trials import crossed, exposure, update_policy


def continuation_plan(package, parent):
    from .progressive_training import resolve_update_policy

    cut = parent["milestone"]
    stream = package["stream"]
    if (
        package != seal(package)
        or package["version"] != "round028-pilot-training/v1"
        or stream != seal(stream)
        or cut != seal(cut)
        or cut["arm"] != "ordinary"
        or cut["pair"] != stream["pair"]
        or cut["seeds"] != stream["seeds"]
        or cut["seeds"]["initialization"] != package["profile"]["model"]["seed"]
        or cut["registration"] != stream["registration"]
        or cut["stream"] != stream["identity"]
        or cut["parent"] != parent["checkpoint"]["sha256"]
        or parent["updates"] != cut["updates"]
    ):
        raise ValueError("authenticated ordinary donor package and complete parent required")
    rows = {r["draw_identity"]: r for r in package["rows"]}
    if (
        len(rows) != len(package["rows"])
        or not stream["order"]
        or not set(stream["order"]) <= rows.keys()
    ):
        raise ValueError("original ordered training rows required")
    prior = exposure(stream, rows, updates=cut["updates"], batch=32, discarded=cut["discarded"])
    if (
        not crossed(prior, cut["target"])
        or prior["charged_tokens"] != cut["charged"]
        or any(
            cut[k] != prior[k]
            for k in ("cursor", "order_digest", "last_update_tokens", "useful_tokens")
        )
    ):
        raise ValueError("donated token or row history differs from its complete useful-token look")
    old = resolve_update_policy(
        package["profile"],
        objective="ordinary",
        effective_batch=32,
        microbatch_size=package["microbatch"],
        weight_decay=0,
        schedule_horizon_tokens=4_000_000,
    )
    new = update_policy(package["profile"], "00", microbatch=package["microbatch"])
    metadata = {"identity", "version", "objective", "velocity_recipe", "matching_weight"}
    if {k: v for k, v in old.items() if k not in metadata} != {
        k: v for k, v in new.items() if k not in metadata
    } or parent["contract"] != digest([package["identity"], "ordinary", old["identity"]]):
        raise ValueError("donated update semantics differ from recipe 00")
    ids = matched_batch(stream, prior["cursor"], 32)
    count = sum(len(rows[k]["tokens"]) for k in ids)
    if count <= 0:
        raise ValueError("empty next positive update")
    return (
        seal(
            dict(
                parent=cut["identity"],
                checkpoint=cut["parent"],
                package=package["identity"],
                old_policy=old["identity"],
                new_policy=new["identity"],
                parent_updates=cut["updates"],
                prior_useful=cut["useful_tokens"],
                prior_charge=cut["charged"],
                update_tokens=count,
                rows=digest(ids),
                next_rows=digest(matched_batch(stream, prior["cursor"] + 32, 32)),
            )
        ),
        old,
        new,
    )


def _rng_digest(device):
    import hashlib
    import random
    import numpy as np
    import torch

    algorithm, keys, position, gaussian, cached = np.random.get_state()
    tensors = [torch.get_rng_state()]
    if device == "cuda":
        tensors.extend(torch.cuda.get_rng_state_all())
    return digest(
        [
            random.getstate(),
            [algorithm, keys.tolist(), position, gaussian, cached],
            [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in tensors],
        ]
    )


def compare_next_update(directory, package, parent, *, device):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.generalization.progressive.training_rows import matched_milestone
    from .progressive_generator_probe import batch
    from .progressive_pilot_worker import restore_inputs
    from .progressive_training import PolicyUpdate, RecoverableUpdate
    from .progressive_velocity_worker import optimizer_digest

    plan, old, new = continuation_plan(package, parent)
    cut, stream = parent["milestone"], package["stream"]
    rows = {r["draw_identity"]: r for r in package["rows"]}
    ids = matched_batch(stream, cut["cursor"], 32)
    count = plan["update_tokens"]

    def advance(policy):
        # Load each branch afresh. AdamW mutates its moments during the update.
        full = restore_inputs(directory, parent["checkpoint"], parent["contract"], parent["cap"])
        state, ledger = full["states"], full["ledger"]
        runner = PolicyUpdate(package["profile"], policy, device=device)
        if (
            ledger.pending is not None
            or ledger.consumed != cut["charged"]
            or ledger.updates != cut["updates"]
            or state["data"] != {"cursor": ledger.updates, "pending_batch": None}
            or state["history"]["look"] != cut["target"]
            or state["selection"]["initial_weights"]
            != model_snapshot(runner.encoder, runner.decoder)
            or (device == "cuda") != bool(state["rng"]["device"])
        ):
            raise ValueError("donation requires an original complete update, seed and RNG state")
        derived = matched_milestone(
            dict(matched_generator=stream, profile=package["profile"], update_policy=old),
            rows,
            {"identity": parent["checkpoint"]["sha256"]},
            dict(
                full,
                cuts=dict(selection=digest(state["selection"]), consumed=state["history"]["order"]),
                discarded_tokens=cut["discarded"],
            ),
            cut["useful_tokens"],
            cut["target"],
        )
        optimizer = state["optimizer"]
        schedule = optimizer["scheduler"]
        if (
            derived != cut
            or optimizer["policy"] != old["identity"]
            or schedule["policy"] != old["identity"]
            or schedule["updates"] != ledger.updates
            or schedule["tokens"] != ledger.consumed
        ):
            raise ValueError("original model, schedule, optimizer or data cursor differs")
        runner.encoder.load_state_dict(state["weights"]["encoder"])
        runner.decoder.load_state_dict(state["weights"]["decoder"])
        runner.load_state_dict(
            dict(
                optimizer,
                policy=policy["identity"],
                scheduler=dict(schedule, policy=policy["identity"]),
            )
        )
        RecoverableUpdate._rng_restore(state["rng"])
        del full, state, optimizer
        before = model_snapshot(runner.encoder, runner.decoder)
        seq, tokens = batch([rows[k] for k in ids], device)
        # The disposable child buys just one update; its absolute clock includes all prior charges.
        child = Ledger(ledger.consumed + count, consumed=ledger.consumed, updates=ledger.updates)
        _, child = runner.update(seq, tokens, ledger=child.reserve("donated-next-update", count))
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        schedule = runner.scheduler.state_dict()
        return dict(
            parent_weights=before,
            weights=model_snapshot(runner.encoder, runner.decoder),
            optimizer=optimizer_digest(runner),
            rng=_rng_digest(device),
            tokens=schedule["tokens"],
            updates=schedule["updates"],
            useful_tokens=cut["useful_tokens"] + count,
            next_rows=digest(matched_batch(stream, child.updates * 32, 32)),
        )

    reference = advance(old)
    candidate = advance(new)
    compared = seal(
        dict(
            plan=plan["identity"],
            parent_updates=cut["updates"],
            reference=reference,
            candidate=candidate,
            charged_tokens=2 * count,
        )
    )
    verify_comparison(compared, plan)
    return compared


def verify_comparison(compared, plan):
    from oeis_learn.generalization.progressive.contracts import _check, schema

    row = compared["candidate"]
    if (
        compared != seal(compared)
        or compared["plan"] != plan["identity"]
        or compared["reference"] != row
        or compared["charged_tokens"] != 2 * plan["update_tokens"]
        or compared["parent_updates"] != plan["parent_updates"]
        or row["updates"] != plan["parent_updates"] + 1
        or row["tokens"] != plan["prior_charge"] + plan["update_tokens"]
        or row["useful_tokens"] != plan["prior_useful"] + plan["update_tokens"]
        or row["next_rows"] != plan["next_rows"]
        or row["weights"] == row["parent_weights"]
    ):
        raise ValueError("disposable next-update parity failed")
    for key in ("weights", "parent_weights", "optimizer", "rng"):
        _check(schema("common")["$defs"]["hash"], row[key], {})


def main():
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_qualification import require_hip
    from .progressive_research_inputs import checked_file
    from .progressive_roles import ROUND029_ROLE_POLICY

    inputs = Path("/work/inputs")
    request = load(inputs / "request.json")
    if (
        request != seal(request)
        or request["version"] != "round029-donation-request/v1"
        or request["source"] != digest(source_snapshot())
        or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or request["role_policy"] != ROUND029_ROLE_POLICY
    ):
        raise ValueError("installed donation worker source, image or policy differs")
    package = load(checked_file(inputs, request["training"]), maximum=CAMPAIGN_DOCUMENT_BYTES)
    plan, _, _ = continuation_plan(package, request["parent"])
    if plan != request["plan"]:
        raise ValueError("registered donation or next batch differs")
    torch = require_hip()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic_ns()
    result = compare_next_update(inputs / "previous", package, request["parent"], device="cuda")
    atomic_write(
        Path("/work/output/producer.json"),
        seal(
            dict(
                version="round029-donation-result/v1",
                request=request["identity"],
                device="hip",
                comparison=result,
                elapsed_us=(time.monotonic_ns() - start) // 1000,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
            )
        ),
    )


if __name__ == "__main__":
    main()
