"""Inspect trial 00's recorded parent and initialize a disposable child without training."""

import hashlib
import json
import runpy
import shutil
import tempfile
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import load, record, seal, verify_refs
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.milestones import milestone
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.generalization.progressive.trial_scopes import materialize
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.progressive_learner import compatibility
from oeis_learn.pipeline_validation.progressive_study import original
from oeis_learn.pipeline_validation.progressive_training import RecoverableUpdate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_state(expected, actual):
    import numpy as np
    import torch

    require(type(actual) is type(expected), "restored state type differs")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "restored state fields differ")
        for key in expected:
            compare_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        for left, right in zip(expected, actual, strict=True):
            compare_state(left, right)
    elif isinstance(expected, torch.Tensor):
        # Checkpoint tensors are decoded on CPU; a live HIP learner holds device tensors.
        torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, strict=True)
    else:
        require(actual == expected, "restored scalar differs")


def ledger_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_package(root, campaign, child):
    refs = campaign["payload"]["references"]
    (policy,) = [
        r["artifact"]
        for r in refs
        if r["kind"] == "policy"
        and r["artifact"]["sha256"] == child["policy"]
        and Path(r["artifact"]["path"]).name == child["id"] + ".json"
    ]
    request = json.loads(verify_refs(root, [policy])[0])
    require(request["operation"] == "train", "expected a frozen training request")
    (name,) = request["artifacts"]["training"]
    (channel,) = [
        r["artifact"] for r in refs if r["role"] == "learner" and r["artifact"]["path"] == name
    ]
    envelope = json.loads(verify_refs(root, [channel])[0])
    require(
        seal(envelope) == envelope and envelope["kind"] == "training_request",
        "invalid training channel",
    )
    data = envelope["data"]
    base = json.loads(verify_refs(root, [data["artifact"]], max_bytes=256 << 20)[0])
    require(compatibility(base) == data["compatibility"], "training base differs from its channel")
    return materialize(base, child)


def verify_parent_accounting(journal, task, state, base, receipt):
    """Bind the completed producer's separate supervisor receipt to its settled journal cut."""
    checkpoint = state["checkpoint"]
    require(
        receipt.get("version") == "progressive-final-accounting/v1"
        and seal(receipt) == receipt
        and receipt["task"] == task["identity"]
        and receipt["checkpoint"] == checkpoint["identity"]
        and receipt["useful_tokens"] == state["useful_tokens"]
        and state["pending"] is None,
        "final accounting differs from the completed parent checkpoint",
    )
    plan = task["payload"]
    prefix = plan["id"] + "/"
    checkpoint_marker = checkpoint["payload"]["cuts"]["cost_journal"]
    with journal.lock():
        events, pending, _, _, _ = journal._read()
        require(
            not any(k.startswith(prefix) for k in pending)
            and not any(k.startswith(prefix) for k in receipt["pending"]),
            "unsettled parent charge in journal or final accounting",
        )
        checkpoint_seen = receipt_seen = False
        for event in events:
            if receipt_seen and event["operation"].startswith(prefix):
                raise ValueError("parent operation after final accounting")
            if event["identity"] == checkpoint_marker:
                checkpoint_seen = True
            if event["identity"] == receipt["journal"]:
                require(checkpoint_seen, "final accounting predates the parent checkpoint")
                receipt_seen = True
        require(receipt_seen, "final accounting journal marker is absent")
    remaining = journal.task_remaining(task)
    expected = milestone(
        base["tokens"] + plan["caps"]["tokens"] - remaining["tokens"],
        ancestor_tokens=base["tokens"],
        ancestor_cost_us=base["ancestor_cost_us"],
        new_cost_us=plan["caps"]["elapsed_us"] - remaining["elapsed_us"],
    )
    require(
        receipt["remaining"] == remaining and receipt["milestone"] == expected,
        "parent journal charges differ from final accounting",
    )
    return receipt["milestone"]


def inspect_copy(root, work, campaign, proposal, *, targets):
    """Read inputs from root. All leases, bindings and child state live in work."""
    cid = campaign["identity"][7:]
    original_state = root / "run-state"
    control = original_state / "discovery-control" / cid
    require((control / "campaign.json").is_file(), "original controller binding required")
    shutil.copytree(control, work / "control")
    store = RunStore(work / "control", campaign)
    names = ("trial-00-1000000-train", "trial-00-2000000-train")
    frozen = {p["id"]: p for p in campaign["payload"]["tasks"]}
    plans = [store.resolved_plan(frozen[name]) for name in names]
    require(
        seal(proposal) == proposal and proposal["ordinal"] == 0, "sealed trial 00 proposal required"
    )
    require(all(plans), "missing saved screening choice")
    parent, child = plans
    require(
        tuple(p["token_target"] for p in plans) == targets, "screening token coordinates differ"
    )
    for p in plans:
        choice = p["trial_choices"][p["trial_choice"]]
        require(
            (p["trial"]["ordinal"], p["trial"]["phase"], p["trial"]["study"])
            == (0, "training", proposal["study"]),
            "screening task belongs to a different trial",
        )
        require(
            p["seed"] == proposal["seed"]
            and dict(choice["parameters"], shape=choice["parameters"].get("shape", "reference"))
            == dict(proposal["parameters"], shape=proposal["parameters"].get("shape", "reference")),
            "frozen screening parameters or seed differ",
        )
    scope = child["initializer"]["compatibility"]
    require(
        child["initializer"]["mode"] == "full_state"
        and child["initializer"]["parent_task"] == names[0]
        and parent["initializer"]["compatibility"] == scope
        and parent["trial_choice"] == child["trial_choice"],
        "expected trial 00's full-state parent",
    )
    tasks = [
        record(
            "task",
            dict(p, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        for p in plans
    ]
    checkpoint_ref = original(root, store, names[0], "train", "checkpoint_record")
    checkpoint = load(root / checkpoint_ref["path"])
    role = confined(root, store.state["outputs"][names[0]]["result"]["path"]).parent
    receipt = load(confined(role, "accounting-final.json"))
    package = training_package(root, campaign, child)
    accounting = original_state / "discovery-accounting" / cid
    learning = original_state / "discovery-learning" / cid / tasks[0]["identity"][7:]
    require(
        (accounting / "budget-binding.json").is_file() and (learning / "binding.json").is_file(),
        "original accounting and learner bindings required",
    )
    shutil.copytree(accounting, work / "accounting")
    shutil.copytree(learning, work / "parent")
    journal = BudgetJournal(
        work / "accounting",
        campaign["identity"],
        campaign["payload"]["caps"],
        journal_bytes=campaign["payload"]["storage"]["journal_bytes"],
    )
    prefix = ledger_hash(journal.path)
    require(ledger_hash(accounting / journal.path.name) == prefix, "ledger changed during the copy")
    watermark = journal.snapshot()["watermark"]
    with Recovery(journal, tasks[0], scope, state_root=work / "parent") as first:
        require(
            first.state["status"] == "COMPLETE" and first.state["checkpoint"] == checkpoint,
            "learner cut differs from the original completed producer",
        )
        costs = verify_parent_accounting(journal, tasks[0], first.state, first.base, receipt)
        before = first.restore()
        spent = before["ledger"].consumed
        updates = before["ledger"].updates
        require(
            spent == costs["tokens"] and spent >= targets[0] and updates > 0,
            "parent exposure differs from the settled screening look",
        )
        prior_us = costs["total_cost_us"]
        with Recovery(journal, tasks[1], scope, state_root=work / "child") as second:
            second.initialize_from(first, campaign=campaign)
            after = second.restore()
            require(
                second.base
                == {
                    "tokens": spent,
                    "updates": updates,
                    "useful": first.state["useful_tokens"],
                    "ancestor_cost_us": prior_us,
                },
                "inherited charges differ",
            )
            require(
                after["ledger"].consumed == spent
                and after["ledger"].updates == updates
                and after["cuts"] == dict(before["cuts"], cost_journal=watermark)
                and journal.task_remaining(tasks[1]) == child["caps"],
                "child advanced or lost a parent cut",
            )
            states = before["states"]
            require(
                states["optimizer"]["optimizer"]["state"]
                and states["scheduler"]["controller"]["cursor"] > 0,
                "trained optimizer and noninitial data position required",
            )
            device = "cuda" if states["rng"]["device"] else "cpu"
            worker = RecoverableUpdate(
                second, package["profile"], package["update_policy"], device=device
            )
            compare_state(states, worker.capture(worker.context))
        require(
            journal.task_remaining(tasks[0]) == receipt["remaining"]
            and ledger_hash(journal.path) == prefix,
            "restore changed prior accounting",
        )
    return {
        "parent": names[0],
        "child": names[1],
        "checkpoint": checkpoint["identity"],
        "device": device,
        "tokens": spent,
        "updates": updates,
        "prior_elapsed_us": prior_us,
        "final_accounting": receipt["identity"],
        "final_accounting_journal": receipt["journal"],
        "ledger_sha256": prefix,
        "cursor": states["scheduler"]["controller"]["cursor"],
        "parent_target": parent["token_target"],
        "child_target": child["token_target"],
        "child_updates": after["ledger"].updates - updates,
        "child_tokens": after["ledger"].consumed - spent,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="snapshot containing the original round027 run-state"
    )
    parser.add_argument(
        "--scratch", type=Path, required=True, help="existing directory outside run-state"
    )
    args = parser.parse_args()
    root, scratch = args.root.resolve(), args.scratch.resolve()
    require(
        not scratch.is_relative_to(root / "run-state"), "scratch must be outside the original state"
    )
    audit_path = (
        Path(__file__).resolve().parents[2]
        / "013-progressive-mathematical-discovery/validation/audit-round027-screening.py"
    )
    audit = runpy.run_path(str(audit_path))
    campaign = audit["saved"](
        root / "run-state/discovery/013-research-round-027/prepare-research/frozen/campaign.json"
    )
    require(
        campaign["identity"] == audit["CAMPAIGN"], "expected the original frozen round027 campaign"
    )
    trials, _ = audit["study_results"](root, campaign)
    with tempfile.TemporaryDirectory(dir=scratch) as work:
        result = inspect_copy(
            root, Path(work), campaign, trials[0]["proposal"], targets=(1_000_000, 2_000_000)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
