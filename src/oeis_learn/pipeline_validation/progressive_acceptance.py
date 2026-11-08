"""Short installed CPU lifecycle rehearsal. Never hardware/research authority."""

import os
import subprocess
import sys
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)

GPU_CASES = ("progressive-logical-update-restore", "progressive-two-task-continuation-pause")


def fixture(root, *, interrupt=True, first_cap_updates=8, weights_only=False, fail=False):
    """Hand-checkable mechanical sources, separate purpose and bounded horizons."""
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.curriculum import Controller
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy

    from .progressive_learner import compatibility

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = [
        "(program (return (add $index (int 1))))",
        "(program (return (add $index (int 2))))",
        "(program (return (select (lt $index (int 2)) $index (int 8))))",
        "(program (return (select (lt $index (int 3)) $index (int 9))))",
    ]
    rows = []
    for i, source in enumerate(sources):
        row, reason = verify_draw(
            {
                "source": source,
                "ordinal": i,
                "draw_identity": digest(["acceptance", i]),
                "source_policy": digest("hand-checked diagnostic"),
                "structure": structure_id(source),
                "detector_limit": "forced diagnostic sources; no autonomous discoveries",
            }
        )
        if row is None:
            raise ValueError(reason)
        rows.append(row)
    p = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), shape="smaller_width", seed=13, dropout=0.1
    )
    policy = resolve_update_policy(
        p, schedule_horizon_tokens=400000000, schedule="constant", microbatch_size=7
    )
    layout = {
        "bins": ["familiar", "frontier"],
        "probes": {
            "familiar": {"role": "training", "id": rows[0]["draw_identity"]},
            "frontier": {"role": "training", "id": rows[2]["draw_identity"]},
        },
    }
    if fail:
        rows[1] = seal(dict(rows[1], target_source="(program (return (int 99)))"))
    package = seal(
        {
            "version": "progressive-training-input/v1",
            "profile": p,
            "update_policy": policy,
            "shards": [rows],
            "layout": layout,
            "refresh_updates": 8,
            "scope": "DIAGNOSTIC_ONLY",
            "diagnostic_fault": {"slot": "first", "update": 1, "point": "after_reserve"}
            if interrupt
            else None,
        }
    )
    weights_ref = None
    if weights_only:
        import torch

        from oeis_learn.generalization.progressive.models import construct

        other = resolve_profile(
            resolve({"version": "bigint-bootstrap/v1"}), shape="smaller_width", seed=17, dropout=0.1
        )
        e, d, _ = construct(other, device="cpu")
        torch.save(
            {"encoder": e.state_dict(), "decoder": d.state_dict()}, root / "imported-weights.pt"
        )
        weights_ref = reference(root, "imported-weights.pt")
        package = seal(dict(package, weights=weights_ref))
    atomic_write(root / "training-input.json", package)
    channel = seal(
        {
            "version": "discovery-channel/v1",
            "role": "learner",
            "kind": "training_request",
            "data": {
                "membership": "training",
                "artifact": reference(root, "training-input.json"),
                "compatibility": compatibility(package),
                "profile": "bigint-body/v1",
            },
        }
    )
    atomic_write(root / "training-channel.json", channel)
    controller = Controller(layout, seed=13)
    charge = 16 * len(rows[1 if controller.select()["bin"] == "familiar" else 3]["tokens"])

    def caps(tokens, seconds=120):
        return {
            "tokens": tokens,
            "draws": 0,
            "rejections": 0,
            "attempts": 0,
            "native_work": 0,
            "elapsed_us": seconds * 1000000,
            "disk_bytes": 1 << 30,
            "archive_bytes": 0,
        }

    plans = []
    for i, name in enumerate(("first", "second")):
        plans.append(
            {
                "id": name,
                "stage": "capability" if i == 0 else "diversity",
                "kind": "training",
                "arm": "static",
                "seed": 13,
                "initializer": {
                    "mode": "weights_only" if weights_only else "random",
                    "artifact": weights_ref,
                    "compatibility": compatibility(package),
                }
                if i == 0
                else {
                    "mode": "full_state",
                    "artifact": None,
                    "compatibility": compatibility(package),
                    "parent_slot": "first",
                },
                "token_target": (3 if i == 0 else 5) * charge,
                "caps": caps(first_cap_updates * charge if i == 0 else 8 * charge),
                "startup_us": 1000000,
                "finalization_us": 4000000,
                "policy": digest("diagnostic-request"),
                "slot": name,
                "output": name,
            }
        )
    graph = {
        "selected_stages": ["capability", "diversity"],
        "nodes": [
            {
                "id": n,
                "stage": plans[i]["stage"],
                "kind": "routine",
                "dependencies": [] if i == 0 else ["first"],
                "maximum_visits": 1,
                "caps": caps(8 * charge),
            }
            for i, n in enumerate(("first", "second"))
        ],
        "edges": [
            {
                "source": "first",
                "target": "second",
                "condition": "COMPLETE",
                "action": "ADVANCE_REGISTERED",
            }
        ],
        "diagnostic_pool": caps(0, 0) | {"disk_bytes": 0},
    }
    request_refs = []
    for i, plan in enumerate(plans):
        name = f"request-{i}.json"
        atomic_write(
            root / name,
            {
                "operation": "train",
                "artifacts": {"training": ["training-channel.json"]},
                "image": "sha256:" + "a" * 64,
                "seconds": 120,
                "reservation": caps(0) | {"disk_bytes": 0},
            },
        )
        plan["policy"] = reference(root, name)["sha256"]
        request_refs.append(
            {"name": name, "role": "custodian", "kind": "policy", "artifact": reference(root, name)}
        )
    c = record(
        "campaign",
        {
            "blueprint": digest("diagnostic-lifecycle"),
            "runtime": digest("UNQUALIFIED_CPU_FIXTURE"),
            "graph": graph,
            "tasks": plans,
            "references": request_refs
            + (
                [
                    {
                        "name": "imported-weights",
                        "role": "learner",
                        "kind": "checkpoint",
                        "artifact": weights_ref,
                    }
                ]
                if weights_ref
                else []
            )
            + [
                {
                    "name": "training",
                    "role": "learner",
                    "kind": "prepared",
                    "artifact": reference(root, "training-channel.json"),
                }
            ],
            "caps": caps(16 * charge, 240) | {"disk_bytes": 2 << 30},
            "storage": {
                "checkpoint_bytes": 64 << 20,
                "atomic_generations": 2,
                "milestone_pins": 0,
                "shard_bytes": 1 << 20,
                "package_bytes": 4 << 20,
                "required_bytes": (133 << 20),
                "required_inodes": 1024,
                "available_bytes": 2 << 30,
                "available_inodes": 4096,
            },
            "target_policy": digest("diagnostic-no-scientific-authority"),
        },
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="synthetic_strict",
    )
    atomic_write(root / "campaign.json", c)
    tasks = [
        record(
            "task",
            dict(p, campaign=c["identity"]),
            created_at=c["created_at"],
            purpose=c["purpose"],
            track=c["track"],
        )
        for p in plans
    ]
    return c, tasks, channel, charge


def invoke_cpu(root, campaign, task, channel, *, ordinal, pythonpath=None):
    """Real isolated Python process calls the same installed role dispatcher."""
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.projections import describe, project

    from .progressive_roles import REGISTRY

    root = Path(root)
    accounting = root / "accounting"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    journal = BudgetJournal(accounting, campaign["identity"], campaign["payload"]["caps"])
    remaining = journal.task_remaining(task)
    name = "invocation-" + str(ordinal)
    output = root / name
    output.mkdir()
    project(
        root,
        output / "inputs",
        "train",
        {"training": [describe(root, "training-channel.json")]},
        registry=REGISTRY,
        authorized=[reference(root, "training-channel.json")],
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    start = time.monotonic_ns() // 1000
    cost = dict.fromkeys(remaining, 0)
    cost["elapsed_us"] = remaining["elapsed_us"]
    charge = journal.reserve(
        task["payload"]["id"], "train", ordinal, cost, boot=boot, monotonic_us=start
    )
    authority = seal(
        {
            "version": "progressive-supervisor/v1",
            "campaign": campaign,
            "task": task,
            "clock": {
                "reservation_key": charge["key"],
                "boot": boot,
                "start_us": start,
                "end_us": start + cost["elapsed_us"],
            },
            "invocation": name,
        }
    )
    atomic_write(output / "authority.json", authority)
    states = root / "learning"
    state = states / task["identity"].split(":")[1]
    state.mkdir(parents=True, exist_ok=True)
    parent_state = root / "no-predecessor"
    if task["payload"]["initializer"].get("parent_slot"):
        parent_plan = next(
            p
            for p in campaign["payload"]["tasks"]
            if p["slot"] == task["payload"]["initializer"]["parent_slot"]
        )
        parent = record(
            "task",
            dict(parent_plan, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        parent_state = states / parent["identity"].split(":")[1]
    code = """import sys
from pathlib import Path
from oeis_learn.pipeline_validation.progressive_role import execute
from oeis_learn.pipeline_validation.progressive_evidence import failure
try:
 r=execute('train',inputs=Path(sys.argv[1]),output=Path(sys.argv[2]),authority=Path(sys.argv[3]),accounting=Path(sys.argv[4]),control=Path(sys.argv[5]),state_root=Path(sys.argv[6]),predecessor_state=Path(sys.argv[7]))
except Exception as exc:
 r=failure(Path(sys.argv[2]),'train',exc,producer='installed-train')
raise SystemExit(0 if r['mechanical_status']=='COMPLETE' else 2 if r['mechanical_status']=='PARTIAL' else 1)
"""
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(output / "inputs"),
            str(output / "role"),
            str(output / "authority.json"),
            str(accounting),
            str(control),
            str(state),
            str(parent_state),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1, (remaining["elapsed_us"] // 1000000)),
    )
    # subprocess.run returns only after owned process termination. This is a
    # measured proof, not a worker timestamp or a simulated multi-hour interval.
    end = time.monotonic_ns() // 1000
    journal.commit(charge["key"], boot=boot, monotonic_us=end, termination_proven=True)
    atomic_write(
        output / "process.json",
        seal(
            {
                "returncode": process.returncode,
                "elapsed_us": end - start,
                "stdout": process.stdout[-4096:],
                "stderr": process.stderr[-4096:],
            }
        ),
    )
    return load(output / "role/result.json"), output
