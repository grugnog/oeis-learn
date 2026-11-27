"""Check the selected trial 00 screening cut before its first child update."""

import os
import shutil
from types import SimpleNamespace

import pytest
from screening_restore import copy_restore_inputs, verify_parent_accounting
from test_linked_confirmation import compile_pair, exercise_pair, research_rows

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)
from oeis_learn.generalization.progressive.curriculum import Controller
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.generalization.progressive.trials import TrialJournal
from oeis_learn.generalization.progressive.tuning import BASELINE, CONTROL, propose
from oeis_learn.pipeline_validation.progressive_acceptance import fixture
from oeis_learn.pipeline_validation.progressive_acceptance_limits import measure, storage
from oeis_learn.pipeline_validation.progressive_learner import compatibility
from oeis_learn.pipeline_validation.progressive_roles import ROLE_POLICY
from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy


def test_restore_copy_preserves_only_private_hard_links(tmp_path):
    parent = tmp_path / "original"
    first, second = (parent / name for name in ("generation-1", "generation-2"))
    first.mkdir(parents=True)
    second.mkdir()
    weights = first / "weights.pt"
    weights.write_bytes(b"settled weights" * 8192)
    weights_copy = second / "weights.pt"
    os.link(weights, weights_copy)
    outside = tmp_path / "retained-weights.pt"
    os.link(weights, outside)
    independent = second / "optimizer.pt"
    independent.write_bytes(weights.read_bytes())

    work = tmp_path / "scratch"
    measured = copy_restore_inputs(
        work, {"parent": parent}, {"checkpoint_bytes": 1 << 20, "checkpoint_inodes": 64}
    )
    cloned = work / "parent/generation-1/weights.pt"
    alias = work / "parent/generation-2/weights.pt"
    assert cloned.samefile(alias) and cloned.stat().st_nlink == 2
    assert not cloned.samefile(weights)
    assert not cloned.samefile(work / "parent/generation-2/optimizer.pt")
    assert measured["source_allocated_bytes"]["parent"] == sum(
        path.stat().st_blocks * 512 for path in (parent, first, second, weights, independent)
    )
    cloned.write_bytes(b"changed on scratch")
    assert alias.read_bytes() == b"changed on scratch"
    assert weights.stat().st_nlink == 3
    assert (
        weights.read_bytes()
        == weights_copy.read_bytes()
        == outside.read_bytes()
        == independent.read_bytes()
    )


@pytest.mark.parametrize("short", ["bytes", "inodes"])
def test_restore_refuses_short_scratch_before_copy(tmp_path, monkeypatch, short):
    parent = tmp_path / "original"
    parent.mkdir()
    # Sparse input may expand on a filesystem that allocates every copied block.
    with (parent / "weights.pt").open("wb") as stream:
        stream.seek((256 << 10) - 1)
        stream.write(b"x")
    os.link(parent / "weights.pt", parent / "same-weights.pt")
    bound = {"checkpoint_bytes": 1 << 20, "checkpoint_inodes": 64}
    space = copy_restore_inputs(tmp_path / "measured", {"parent": parent}, bound)
    block = os.statvfs(tmp_path).f_frsize
    assert space["copy_bytes"] >= 256 << 10
    assert space["required_inodes"] == 2 + bound["checkpoint_inodes"]
    blocks = (space["required_bytes"] + block - 1) // block
    available = SimpleNamespace(
        f_frsize=block,
        f_bavail=blocks - (short == "bytes"),
        f_favail=space["required_inodes"] - (short == "inodes"),
    )
    monkeypatch.setattr(os, "statvfs", lambda _: available)
    work = tmp_path / "too-small"
    with pytest.raises(ValueError, match="including the child"):
        copy_restore_inputs(work, {"parent": parent}, bound)
    assert list(work.iterdir()) == []


def test_trial_00_screening_restores_before_first_update(tmp_path, monkeypatch):
    monkeypatch.setenv("OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE", "cpu")
    campaign, _, _, _ = fixture(tmp_path, interrupt=False, checkpoint_policy="hourly/v1")
    payload = campaign["payload"]
    study = digest("screening restore fixture")
    journal = TrialJournal(tmp_path / "study", study=study)
    proposal = propose(journal, 0)
    assert TrialJournal(journal.root, study=study).proposal(0) == proposal
    assert proposal["ordinal"] == 0 and proposal["parameters"] == BASELINE
    seed = proposal["seed"]

    rows = research_rows(seed)
    parent = rows["trial-00-1000000-train"]
    child = rows["trial-00-2000000-train"]
    assert child["parent"] == parent["id"] and parent["parent"] is None
    assert (parent["token_target"], child["token_target"]) == (1_000_000, 2_000_000)
    for row in (parent, child):
        assert row["trial"]["ordinal"] == 0 and row["trial"]["phase"] == "training"
        assert row["horizon"] == 4_000_000 and row["arm"] == "refreshed"

    # Preserve the real screening trajectory, finite choice and hourly persistence
    # while using the acceptance fixture's synthetic data and two parent updates.
    package = load(tmp_path / "training-input.json")
    profile = resolve_profile(
        package["profile"]["base_recipe"], seed=seed, dropout=BASELINE["dropout"]
    )
    native_image = load(tmp_path / "request-0.json")["image"]
    runtime = record(
        "runtime",
        {
            "scope": {
                "source": digest("fixture source"),
                "images": [native_image, digest("second fixture image")],
                "model": profile["identity"],
                "language": "bigint-body/v1",
                "precision": "fp32",
                "device": "cpu",
                "execution_policy": digest("local diagnostic worker"),
            },
            "qualification": [reference(tmp_path, "acceptance-limits.json")],
            "operations": ["train"],
            "role_policy": ROLE_POLICY,
            "fixture": True,
        },
        created_at=campaign["created_at"],
        track=campaign["track"],
        purpose=campaign["purpose"],
    )
    atomic_write(tmp_path / "screening-runtime.json", runtime)
    runtime_ref = reference(tmp_path, "screening-runtime.json")
    payload["runtime"] = runtime["identity"]
    payload["references"].append(
        {
            "name": "screening-runtime",
            "kind": "qualification",
            "role": "custodian",
            "artifact": runtime_ref,
        }
    )
    settings = (CONTROL, BASELINE)
    scopes = {}
    packages = {}
    for params in settings:
        policy = resolve_update_policy(
            profile,
            schedule_horizon_tokens=parent["horizon"],
            microbatch_size=16,
            **{k: v for k, v in params.items() if k != "dropout"},
        )
        scoped = seal(dict(package, profile=profile, update_policy=policy))
        packages[digest(params)] = scoped
        scopes[digest(params)] = {"compatibility": compatibility(scoped), "runtime": runtime_ref}
    assert scopes[digest(CONTROL)] != scopes[digest(BASELINE)]
    base = packages[digest(CONTROL)]
    atomic_write(tmp_path / "screening-input.json", base)
    channel = load(tmp_path / "training-channel.json")
    channel["data"].update(
        artifact=reference(tmp_path, "screening-input.json"), compatibility=compatibility(base)
    )
    atomic_write(tmp_path / "screening-channel.json", seal(channel))
    next(ref for ref in payload["references"] if ref["name"] == "training")["artifact"] = reference(
        tmp_path, "screening-channel.json"
    )

    compiled = compile_pair(
        tmp_path,
        (parent, child),
        seed,
        compatibility(base),
        study,
        settings,
        scopes,
        channel="screening-channel.json",
    )
    assert compiled[1]["initializer"]["mode"] == "full_state"
    assert compiled[1]["initializer"]["parent_task"] == compiled[0]["id"]
    measured = measure(profile)
    payload["storage"] = storage(tmp_path, measured, tasks=2)
    payload["caps"]["disk_bytes"] = payload["storage"]["required_bytes"]
    controller = Controller(package["layout"], seed=seed)
    row_index = 1 if controller.select()["bin"] == "familiar" else 3
    charge = BASELINE["effective_batch"] * len(package["shards"][0][row_index]["tokens"])
    for i, (plan, original, node) in enumerate(
        zip(payload["tasks"], compiled, payload["graph"]["nodes"], strict=True), 1
    ):
        request_ref = reference(
            tmp_path, tmp_path / "compiler/requests" / (original["id"] + ".json")
        )
        payload["references"].append(
            {"name": original["id"], "role": "custodian", "kind": "policy", "artifact": request_ref}
        )
        for key in ("id", "arm", "seed", "trial", "initializer", "trial_choices", "policy"):
            plan[key] = original[key]
        plan["token_target"] = 2 * i * charge
        for caps in (plan["caps"], node["caps"]):
            caps.update(tokens=8 * charge, disk_bytes=measured["task_bytes"])
    payload["caps"]["tokens"] = 16 * charge
    exercise_pair(tmp_path, campaign, charge, proposal=proposal, channel="screening-channel.json")


@pytest.mark.parametrize(
    "case",
    [
        "later task",
        "unsettled",
        "different settlement",
        "later parent charge",
        "earlier marker",
        "changed receipt charge",
        "wrong checkpoint",
        "wrong task",
        "receipt pending",
    ],
)
def test_settled_screening_parent_receipt(tmp_path, case):
    from oeis_learn.generalization.progressive.budgets import KEYS, BudgetJournal
    from oeis_learn.pipeline_validation.progressive_evidence import finalize_learning
    from tests.helpers.progressive_fixture import caps, task

    name = "trial-00-1000000-train"
    limit = caps(tokens=1_000_000, elapsed_us=8_000_000_000)
    parent = task(id=name, token_target=1_000_000, caps=limit)
    cid = parent["payload"]["campaign"]
    journal = BudgetJournal(tmp_path / "producer-journal", cid, limit)
    journal.task_remaining(parent)
    zero = dict.fromkeys(KEYS, 0)
    fee = journal.reserve(
        name, "train", 0, dict(zero, elapsed_us=7_200_000_000), boot="fixture", monotonic_us=0
    )
    earlier = journal.snapshot()["watermark"]
    update = journal.reserve(
        name, "update", 0, dict(zero, tokens=1_000_000), boot="fixture", monotonic_us=0
    )
    journal.commit(update["key"])
    checkpoint = seal({"payload": {"cuts": {"cost_journal": journal.snapshot()["watermark"]}}})
    state = {"checkpoint": checkpoint, "pending": None, "useful_tokens": 1_000_000}
    learning = tmp_path / "learning"
    binding = seal({"task": parent})
    atomic_write(learning / "binding.json", binding)
    atomic_write(
        learning / "event-00000.json", seal({"previous": binding["identity"], "state": state})
    )
    base = {"tokens": 0, "ancestor_cost_us": 0}

    # Fork behind the same checkpoint before the supervisor settles its elapsed charge.
    shutil.copytree(journal.root, tmp_path / "replay-journal")
    replay = BudgetJournal(tmp_path / "replay-journal", cid, limit)
    journal.commit(fee["key"], boot="fixture", monotonic_us=500_000_000, termination_proven=True)
    finalize_learning(journal, parent, learning, tmp_path / "role")
    receipt = load(tmp_path / "role/accounting-final.json")
    assert receipt["milestone"]["total_cost_us"] == 500_000_000

    if case == "different settlement":
        replay.commit(fee["key"])  # Legal, but charges all 7,200 seconds instead of the saved 500.
    elif case != "unsettled":
        replay.commit(fee["key"], boot="fixture", monotonic_us=500_000_000, termination_proven=True)
    if case == "later task":
        replay.reserve("unrelated", "profile", 1, zero, boot="fixture", monotonic_us=1)
        assert (
            verify_parent_accounting(replay, parent, state, base, receipt)["total_cost_us"]
            == 500_000_000
        )
        return
    if case == "later parent charge":
        extra = replay.reserve(name, "train", 1, zero, boot="fixture", monotonic_us=0)
        replay.commit(extra["key"])
    elif case == "earlier marker":
        receipt = seal(dict(receipt, journal=earlier))
    elif case == "changed receipt charge":
        receipt = seal(
            dict(receipt, milestone=seal(dict(receipt["milestone"], total_cost_us=7_200_000_000)))
        )
    elif case == "wrong checkpoint":
        receipt = seal(dict(receipt, checkpoint=digest("other weights")))
    elif case == "wrong task":
        receipt = seal(dict(receipt, task=digest("other trial")))
    elif case == "receipt pending":
        receipt = seal(dict(receipt, pending=[fee["key"]]))
    with pytest.raises(ValueError, match="parent|final accounting"):
        verify_parent_accounting(replay, parent, state, base, receipt)
