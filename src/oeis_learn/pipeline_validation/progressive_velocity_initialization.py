"""Qualify a durable zero-token cut on the exposed, previously priced ordinary projection."""

from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_donation_report import hashes
from .identity import confined
from .progressive_pilot_capacity import owned_allocations
from .progressive_research_inputs import checked_file

VERSION = "round029-ordinary-initialization/v1"
CAP = 1  # The checkpoint contract needs a positive cap. Every invocation targets zero.


def measure(view, plan, out, *, device):
    from .progressive_velocity_checkpoint import package
    from .progressive_velocity_worker import train_to

    if view != seal(view) or view["identity"] != plan["projection"]:
        raise ValueError("original exposed ordinary projection required")
    learner = package(view["streams"]["ordinary"], "00", plan["microbatch"], plan["origin"])
    if (
        learner["identity"] != plan["learner"]
        or learner["initial_weights"] != plan["initial_weights"]
        or view["fresh_start"] != plan["start"]
    ):
        raise ValueError("ordinary learner differs from the frozen qualification")
    directory = out / "learner"
    clocks = {"save": [], "load": []}
    states = []
    for _ in range(3):
        value = train_to(
            directory,
            learner,
            target=0,
            cap=CAP,
            device=device,
            checkpoint_timing=lambda op, us: clocks[op].append(us),
        )
        if any(
            value[k] != 0
            for k in (
                "updates",
                "useful_tokens",
                "charged_tokens",
                "discarded_tokens",
                "incremental_charged_tokens",
                "target",
                "overshoot",
                "cursor",
            )
        ):
            raise ValueError("zero-token boundary issued training work")
        if value["weights"] != plan["initial_weights"]:
            raise ValueError("reopened ordinary weights differ from the registered initialization")
        states.append(
            digest(
                {
                    k: value[k]
                    for k in (
                        "weights",
                        "optimizer",
                        "scheduler",
                        "next_rows",
                        "order_digest",
                        "checkpoint",
                        "rng",
                        "model",
                    )
                }
            )
        )
    if len(set(states)) != 1 or len(clocks["save"]) != 1 or len(clocks["load"]) != 2:
        raise ValueError("one initial save and two identical independent restores required")
    return seal(
        dict(
            plan=plan["identity"],
            state=states[0],
            restores=states[1:],
            clocks=clocks,
            checkpoint=value["checkpoint"],
            weights=value["weights"],
            tokens=0,
            updates=0,
            allocation=owned_allocations([directory]),
        )
    )


def verify_cut(plan, measured, out):
    """Verify every retained blob and the learner debit journal without importing a tensor runtime."""
    from oeis_learn.generalization.state import restore
    from oeis_learn.generalization.progressive.budgets import KEYS

    folder = confined(out, "learner")
    store = confined(folder, "checkpoint")
    pointer = load(confined(store, "generalization-latest.json"))
    if (
        measured != seal(measured)
        or measured["plan"] != plan["identity"]
        or measured["checkpoint"] != pointer
        or measured["restores"] != [measured["state"]] * 2
        or measured["weights"] != plan["initial_weights"]
        or measured["tokens"] != 0
        or measured["updates"] != 0
        or len(measured["clocks"]["save"]) != 1
        or len(measured["clocks"]["load"]) != 2
        or measured["allocation"] != owned_allocations([folder])
    ):
        raise ValueError("installed ordinary zero cut or its measured restores differ")
    hashes(measured["state"])
    for timings in measured["clocks"].values():
        for us in timings:
            integer(us, 1)
    contract = digest([plan["learner"], CAP])
    manifest = load(checked_file(store, pointer))
    restored = restore(manifest, contract=contract, cap=CAP, root=store)
    ledger = restored["ledger"]
    accounting = confined(folder, "accounting")
    events = confined(accounting, "budget.jsonl", exists=False)
    binding = seal(dict(campaign=contract, caps=dict(dict.fromkeys(KEYS, 0), tokens=CAP)))
    if (
        ledger.updates
        or ledger.consumed
        or ledger.pending
        or events.exists()
        and events.stat().st_size
        or load(confined(accounting, "budget-binding.json")) != binding
        or restored["states"]["data"] != dict(cursor=0, pending_batch=None)
    ):
        raise ValueError("ordinary initialization must have an empty optimizer ledger")
    return measured
