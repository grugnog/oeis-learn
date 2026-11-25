"""Inspect trial 00's recorded parent and initialize a disposable child without training."""

import hashlib
import json
import os
import re
import runpy
import shutil
import stat
import tempfile
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import load, record, seal, verify_refs
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.milestones import milestone
from oeis_learn.generalization.progressive.recovery import Recovery
from oeis_learn.generalization.progressive.storage_plan import checkpoint_bound
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


def copy_restore_inputs(work, sources, child_bound):
    """Preflight all inputs and one new generation, then recreate links only inside scratch."""
    require(
        all(not work.resolve().is_relative_to(source.resolve()) for source in sources.values()),
        "scratch must be outside the original inputs",
    )
    work.mkdir(parents=True, exist_ok=True)
    require(not any(work.iterdir()), "empty restore directory required")
    fs = os.statvfs(work)
    block = fs.f_frsize
    directories, links = [], []
    copies = {}
    allocated_bytes = {}
    copy_bytes = 0
    for name, source in sources.items():
        physical = {}
        pending = [(source, work / name)]
        while pending:
            path, dest = pending.pop()
            info = path.lstat()
            inode = (info.st_dev, info.st_ino)
            physical[inode] = info.st_blocks * 512
            if stat.S_ISDIR(info.st_mode):
                directories.append(dest)
                pending.extend((entry, dest / entry.name) for entry in path.iterdir())
            elif stat.S_ISREG(info.st_mode):
                if inode in copies:
                    links.append((copies[inode][1], dest))
                    continue
                copies[inode] = (path, dest)
            else:
                raise ValueError(f"restore requires regular files and directories: {path}")
            # An ordinary copy can expand a sparse file. Count each inode once at destination.
            logical = max(info.st_size, block if stat.S_ISDIR(info.st_mode) else 0)
            copy_bytes += max(physical[inode], (logical + block - 1) // block * block)
        allocated_bytes[name] = sum(physical.values())

    child_bytes, child_inodes = child_bound["checkpoint_bytes"], child_bound["checkpoint_inodes"]
    # Allow one filesystem block per child inode for allocation rounding and metadata.
    required_bytes = copy_bytes + child_bytes + child_inodes * block
    required_inodes = len(directories) + len(copies) + child_inodes
    free_bytes = fs.f_bavail * block
    require(
        required_bytes <= free_bytes and required_inodes <= fs.f_favail,
        f"restore needs {required_bytes} bytes and {required_inodes} inodes including the child; "
        f"scratch has {free_bytes} bytes and {fs.f_favail} inodes",
    )
    for directory in directories:
        directory.mkdir()
    for source, dest in copies.values():
        shutil.copy2(source, dest)
    for first_copy, dest in links:
        os.link(first_copy, dest)
    return {
        "source_allocated_bytes": allocated_bytes,
        "copy_bytes": copy_bytes,
        "child_checkpoint_bytes": child_bytes,
        "child_checkpoint_inodes": child_inodes,
        "required_bytes": required_bytes,
        "required_inodes": required_inodes,
        "available_bytes_before": free_bytes,
        "available_inodes_before": fs.f_favail,
    }


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
    require(
        not work.resolve().is_relative_to(original_state.resolve()),
        "scratch must be outside run-state",
    )
    control = confined(original_state, f"discovery-control/{cid}")
    accounting = confined(original_state, f"discovery-accounting/{cid}")
    names = ("trial-00-1000000-train", "trial-00-2000000-train")
    frozen = {p["id"]: p for p in campaign["payload"]["tasks"]}
    require(
        seal(proposal) == proposal and proposal["ordinal"] == 0, "sealed trial 00 proposal required"
    )
    # Locate only the selected learner for preflight. The copied RunStore checks the full
    # transition chain, and original() below authenticates this task and its producer result.
    tip = max(control.glob("transition-*.json"), default=None)
    require(tip is not None, "completed trial 00 controller transition required")
    transition = load(tip)
    require(seal(transition) == transition, "invalid controller transition")
    task_id = transition["state"]["outputs"][names[0]]["task"]
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", task_id), "invalid parent task identity")
    learning = confined(original_state, f"discovery-learning/{cid}/{task_id[7:]}")
    require(
        (control / "campaign.json").is_file()
        and (accounting / "budget-binding.json").is_file()
        and (learning / "binding.json").is_file(),
        "original controller, accounting and learner bindings required",
    )
    child_bound = checkpoint_bound(campaign["payload"]["storage"], frozen[names[1]])
    space = copy_restore_inputs(
        work, {"control": control, "accounting": accounting, "parent": learning}, child_bound
    )
    store = RunStore(work / "control", campaign)
    plans = [store.resolved_plan(frozen[name]) for name in names]
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
    require(tasks[0]["identity"] == task_id, "parent task changed during the copy")
    checkpoint = load(root / checkpoint_ref["path"])
    role = confined(root, store.state["outputs"][names[0]]["result"]["path"]).parent
    receipt = load(confined(role, "accounting-final.json"))
    package = training_package(root, campaign, child)
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
        with Recovery(
            journal,
            tasks[1],
            scope,
            state_root=work / "child",
            checkpoint_bytes=child_bound["checkpoint_bytes"],
        ) as second:
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
        "scratch_space": space,
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
