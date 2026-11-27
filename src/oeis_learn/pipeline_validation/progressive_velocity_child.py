"""Bridge a complete round029 optimizer cut into a separately funded continuation."""

from pathlib import Path

from oeis_learn.generalization.budgets import integer
from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import _check, digest, schema, seal
from oeis_learn.generalization.progressive.training_rows import matched_batch
from oeis_learn.generalization.progressive.velocity_trials import crossed, exposure, update_policy
from .identity import confined


def child_package(package, cut, *, parent_cap, registration):
    """Bind the predecessor, including charges discarded since its last completed update."""
    _check(schema("common")["$defs"]["hash"], registration, {})
    if (
        package != seal(package)
        or package["version"] not in {"round029-learner-input/v1", "round029-learner-input/v2"}
        or registration == package["registration"]
        or not crossed(cut, cut["target"])
        or cut["updates"] < 1
        or cut["overshoot"] != cut["useful_tokens"] - cut["target"]
    ):
        raise ValueError("separate child registration and complete observed parent look required")
    policy = package["policy"]
    if policy["schedule_horizon_tokens"] != 4_000_000:
        raise ValueError("the registered paired extension bridges the 4M schedule to 8M")
    following = update_policy(
        package["profile"],
        policy["velocity_recipe"],
        microbatch=policy["microbatch_size"],
        matching_weight=policy["matching_weight"],
        horizon=8_000_000,
    )
    parent = seal(
        dict(
            package=package["identity"],
            package_version=package["version"],
            registration=package["registration"],
            cap=integer(parent_cap, 1),
            policy=policy,
            **{
                k: cut[k]
                for k in (
                    "target",
                    "checkpoint",
                    "weights",
                    "optimizer",
                    "rng",
                    "scheduler",
                    "order_digest",
                    "cursor",
                    "updates",
                    "useful_tokens",
                    "discarded_tokens",
                    "charged_tokens",
                    "last_update_tokens",
                    "next_rows",
                    "overshoot",
                )
            },
        )
    )
    return seal(
        dict(
            package,
            version="round029-learner-child/v1",
            registration=registration,
            policy=following,
            parent=parent,
        )
    )


def parent_terms(child):
    """Rebuild the previous input from the identical stream, objective and original seed."""
    parent = child["parent"]
    previous = seal(
        {
            **{k: v for k, v in child.items() if k not in {"identity", "parent"}},
            "version": parent["package_version"],
            "registration": parent["registration"],
            "policy": parent["policy"],
        }
    )
    if (
        parent != seal(parent)
        or previous["identity"] != parent["package"]
        or child_package(
            previous, parent, parent_cap=parent["cap"], registration=child["registration"]
        )
        != child
        or integer(parent["charged_tokens"])
        != integer(parent["useful_tokens"]) + integer(parent["discarded_tokens"])
        or parent["charged_tokens"] > parent["cap"]
    ):
        raise ValueError("child no longer matches its complete funded predecessor")
    return parent


def restore_parent(directory, child, runner):
    """Open only the named parent generation and original journal. No original file is written."""
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from .progressive_training import RecoverableUpdate
    from .progressive_velocity_continuation import _rng_digest
    from .progressive_velocity_worker import optimizer_digest

    parent = parent_terms(child)
    if directory is None:
        raise ValueError(
            "original full parent checkpoint and debit journal required to start a child"
        )
    root = Path(directory)
    contract = digest([parent["package"], parent["cap"]])
    caps = dict(dict.fromkeys(KEYS, 0), tokens=parent["cap"])
    journal = BudgetJournal.inspect_existing(confined(root, "accounting"), contract, caps)
    store = GeneralizationCheckpointStore(confined(root, "checkpoint"), contract, parent["cap"])
    saved = store.load(parent["checkpoint"])
    state, ledger = saved["states"], saved["ledger"]
    size = child["policy"]["effective_batch"]
    history = exposure(
        child,
        child["rows"],
        updates=ledger.updates,
        batch=size,
        discarded=parent["discarded_tokens"],
    )
    if (
        journal["pending"]
        or journal["consumed"]["tokens"] != parent["charged_tokens"]
        or ledger.pending is not None
        or ledger.consumed > parent["charged_tokens"]
        or any(parent[k] != v for k, v in history.items())
        or state["selection"]
        != dict(order_digest=history["order_digest"], package=parent["package"])
        or state["data"] != dict(cursor=history["cursor"], pending_batch=None)
        or state["history"]["useful_tokens"] != history["useful_tokens"]
        or state["scheduler"] != dict(completed_updates=ledger.updates)
        or parent["next_rows"] != digest(matched_batch(child, history["cursor"], size))
        or (runner.device == "cuda") != bool(state["rng"]["device"])
    ):
        raise ValueError("parent exposure, random state, next rows or paid boundary differs")
    optimizer = state["optimizer"]
    schedule = optimizer["scheduler"]
    if (
        optimizer["policy"] != parent["policy"]["identity"]
        or schedule != parent["scheduler"]
        or schedule["policy"] != optimizer["policy"]
        or schedule["updates"] != ledger.updates
        or not history["useful_tokens"] <= schedule["tokens"] <= ledger.consumed
    ):
        raise ValueError("parent optimizer and scheduler disagree")
    runner.encoder.load_state_dict(state["weights"]["encoder"])
    runner.decoder.load_state_dict(state["weights"]["decoder"])
    runner.load_state_dict(
        dict(
            optimizer,
            policy=child["policy"]["identity"],
            scheduler=dict(schedule, policy=child["policy"]["identity"]),
        )
    )
    RecoverableUpdate._rng_restore(state["rng"])
    if (
        parent["weights"] != model_snapshot(runner.encoder, runner.decoder)
        or parent["optimizer"] != optimizer_digest(runner)
        or parent["rng"] != _rng_digest(runner.device)
    ):
        raise ValueError("authenticated parent tensors or random draws differ")
    return state["history"]["metrics"]
