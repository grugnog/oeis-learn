"""Price durable update accounting on exposed training row lengths under a CPU preparation grant."""

import hashlib
import json
import os
from pathlib import Path
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    physical_root_id,
    validate_grant,
)
from oeis_learn.generalization.progressive.velocity_trials import exposure, matched_batch
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_pilot_capacity import velocity_mounts
from .progressive_velocity_operations import funded_verification, replay_operation, run_operation
from .progressive_velocity_qualification import authorize_qualification
from .progressive_velocity_timing import batch_plan

VERSION = "round029-update-accounting-prices/v1"


def frame(order, lengths, batch, *, looks=(1_000_000, 4_000_000)):
    rows = {k: {"tokens": range(n)} for k, n in lengths.items()}
    cuts = []
    for look in looks:
        cells = batch_plan(order, rows, {}, batch, target=look)
        cuts.append(
            dict(
                target=look,
                updates=sum(c["updates_at_look"] for c in cells),
                useful_tokens=sum(c["tokens_at_look"] for c in cells),
            )
        )
    return seal(dict(batch=batch, stream=digest([order, lengths]), cuts=cuts))


def workload():
    package = Path(__file__).resolve().parents[1]
    files = (
        "pipeline_validation/progressive_velocity_ledger_prices.py",
        "pipeline_validation/progressive_velocity_timing.py",
        "generalization/progressive/budgets.py",
        "generalization/progressive/contracts.py",
        "generalization/progressive/storage.py",
        "generalization/progressive/training_rows.py",
        "generalization/progressive/velocity_trials.py",
    )
    return digest(
        {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files}
    )


def plan_for(revision, fresh, checkpoint, frames):
    return seal(
        dict(
            version=VERSION,
            candidate=revision,
            fresh=fresh,
            checkpoint=checkpoint,
            workload=workload(),
            batches=frames,
        )
    )


def source_inputs(root, fresh, cycles, revision):
    from . import progressive_velocity_broad_host as broad
    from . import progressive_velocity_checkpoint_host as checkpoints
    from .progressive_research_inputs import checked_file

    folder = confined(root, "run-state/discovery/" + SESSION)
    pointer = checkpoints.origin(folder)
    if (
        pointer["receipt"] != fresh["identity"]
        or fresh != seal(fresh)
        or cycles["origin"] != fresh["identity"]
    ):
        raise ValueError("bound fresh qualification and checkpoint matrix required")
    area = confined(folder, "qualification/" + pointer["grant"][7:])
    grant = load(confined(area, "grant.json"))
    cpu = broad.attempt(area, broad.coordinates(grant, "prepare"), grant)
    gpu = broad.attempt(area, broad.coordinates(grant, "measure"), grant)
    producer = load(confined(gpu, "producer.json"))
    req = load(confined(area, "device-request.json"))
    if (
        producer != seal(producer)
        or producer["identity"] != fresh["device"]["producer"]
        or req != seal(req)
        or req["identity"] != producer["request"]
    ):
        raise ValueError("original fresh device request required")
    path = checked_file(cpu / "native", req["projection"])
    view = load(path, maximum=req["projection"]["bytes"])
    if view != seal(view) or view["fresh_start"] != fresh["start"]["identity"]:
        raise ValueError("original positive stream differs")
    positive = view["streams"]["ordinary"]
    lengths = {k: len(v["tokens"]) for k, v in positive["rows"].items()}
    order = positive["order"][: len(lengths)]
    frames = {
        str(b): frame(order, lengths, b)
        for b in sorted({r["batch"] for r in cycles["methods"].values()})
    }
    for m in cycles["methods"]:
        cells = req["design"]["cells"][m]
        last = frames[str(cycles["methods"][m]["batch"])]["cuts"][-1]
        if last["updates"] != sum(c["updates_at_look"] for c in cells) or last[
            "useful_tokens"
        ] != sum(c["tokens_at_look"] for c in cells):
            raise ValueError("projected work differs from the paid optimizer frame")
    return plan_for(revision, fresh["identity"], cycles["receipt"], frames), order, lengths


def clock():
    return time.monotonic_ns() // 1000


def measured_price(total_us, intervals):
    worst = max(
        (s["loop_us"] + s["updates"] - 1) // s["updates"] for s in intervals if s["updates"]
    )
    return dict(
        expected_us=total_us,
        tail_us=total_us
        + sum(worst * s["updates"] - s["loop_us"] for s in intervals if s["updates"]),
    )


def measure(out, terms, order, lengths):
    rows = {k: {"tokens": range(n)} for k, n in lengths.items()}
    stream = dict(order=order)
    batch = terms["batch"]
    caps = dict(dict.fromkeys(KEYS, 0), tokens=terms["cuts"][-1]["useful_tokens"])
    area = out / "fixture-accounting"
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    chain = digest(None)
    useful = updates = 0
    began = clock()
    intervals = []
    for cut in [dict(target=0, updates=0, useful_tokens=0), *terms["cuts"]]:
        journal = BudgetJournal(area, terms["identity"], caps)
        if journal.snapshot()["consumed"]["tokens"] != useful:
            raise ValueError("resumed fixture debit differs")
        prior = exposure(stream, rows, updates=updates, batch=batch)
        if prior["order_digest"] != chain or prior["useful_tokens"] != useful:
            raise ValueError("resumed fixture row order differs")
        count = cut["updates"] - updates
        tick = clock()
        while useful < cut["target"]:
            ids = matched_batch(stream, updates * batch, batch)
            tokens = sum(lengths[k] for k in ids)
            debit = journal.reserve(
                "learner",
                "update",
                updates,
                dict(dict.fromkeys(KEYS, 0), tokens=tokens),
                boot=boot,
                monotonic_us=clock(),
            )
            journal.commit(debit["key"])
            chain = digest([chain, ids])
            updates += 1
            useful += tokens
        loop_us = clock() - tick
        replay = exposure(stream, rows, updates=updates, batch=batch)
        if (
            replay["useful_tokens"] != useful
            or replay["order_digest"] != chain
            or updates != cut["updates"]
            or useful != cut["useful_tokens"]
        ):
            raise ValueError("fixture did not cross the registered whole-update look")
        if count:
            intervals.append(dict(updates=count, loop_us=loop_us))
    elapsed = clock() - began
    result = seal(
        dict(
            frame=terms["identity"],
            updates=updates,
            useful_tokens=useful,
            order=chain,
            intervals=intervals,
            price=measured_price(elapsed, intervals),
            ledger_bytes=journal.path.stat().st_size,
        )
    )
    retain(out / "producer.json", result)
    return result


def verify(out, terms, order, lengths):
    result = load(confined(out, "producer.json"))
    cap = terms["cuts"][-1]["useful_tokens"]
    state = BudgetJournal.inspect_existing(
        out / "fixture-accounting",
        terms["identity"],
        dict(dict.fromkeys(KEYS, 0), tokens=cap),
        include_work=True,
    )
    n = terms["cuts"][-1]["updates"]
    if (
        result != seal(result)
        or result["frame"] != terms["identity"]
        or result["updates"] != n
        or result["useful_tokens"] != cap
        or state["pending"]
        or len(state["committed"]) != n
        or state["consumed"] != dict(dict.fromkeys(KEYS, 0), tokens=cap)
        or result["ledger_bytes"] != (out / "fixture-accounting/budget.jsonl").stat().st_size
    ):
        raise ValueError("settled fixture accounting differs from its registered frame")
    batch, chain = terms["batch"], digest(None)
    for i in range(n):
        ids = matched_batch(dict(order=order), i * batch, batch)
        charge = state["reserved_work"][f"learner/update/{i}"]
        if charge["tokens"] != sum(lengths[k] for k in ids):
            raise ValueError("fixture positive-token charge differs")
        chain = digest([chain, ids])
    intervals = result["intervals"]
    expected_counts = [
        c["updates"] - before
        for c, before in zip(
            terms["cuts"], [0, *(r["updates"] for r in terms["cuts"][:-1])], strict=True
        )
    ]
    if (
        result["order"] != chain
        or [r["updates"] for r in intervals] != [v for v in expected_counts if v]
        or any(type(v["loop_us"]) is not int or v["loop_us"] < 0 for v in intervals)
        or result["price"]["expected_us"] < sum(s["loop_us"] for s in intervals)
        or result["price"] != measured_price(result["price"]["expected_us"], intervals)
    ):
        raise ValueError("fixture order or measured price differs")
    return result


def coordinates(grant, batch):
    return dict(
        study=grant["identity"],
        stage="qualification",
        method="accounting-batch-" + batch,
        seed=29,
        look=4_000_000,
        panel="exposed_training",
        profile="accounting",
        input_mode="true",
        role="accounting",
        ordinal=0,
    )


def journal_peak(terms, block_size):
    # Bound each serialized RESERVE/COMMIT at the largest ordinal, full token cap,
    # a 128-byte boot id and full-length digests. Include both directories, files,
    # bindings, atomic staging and the result separately from the simulated journal.
    n = terms["cuts"][-1]["updates"]
    event = dict(
        event="RESERVE",
        index=2 * n,
        operation=f"learner/update/{n}",
        boot="x" * 128,
        start_us=2**63,
        end_us=2**63,
        cost=dict.fromkeys(KEYS, terms["cuts"][-1]["useful_tokens"]),
        previous=digest(None),
        identity=digest(None),
    )
    return 2 * n * (len(json.dumps(event)) + 1) + 16 * block_size


def operate(root, folder, plan, order, lengths, grant, *, journal=None, floor=0):
    area = confined(
        folder, "preparation/update-accounting/" + grant["identity"][7:], exists=journal is None
    )
    budget = lambda: BudgetJournal.inspect_existing(
        area / "accounting",
        grant["identity"],
        grant["resources"],
        include_work=True,
        include_windows=True,
    )
    results, errors = {}, {}
    for batch, terms in plan["batches"].items():
        coords = coordinates(grant, batch)
        req = seal(dict(plan=plan["identity"], frame=terms["identity"]))
        peak = journal_peak(terms, os.statvfs(area).f_frsize)
        work = dict(attempts=1, disk_bytes=peak)

        def check(out):
            return verify(out, terms, order, lengths)

        if journal is None:
            results[batch] = replay_operation(area, coords, req, grant, work, budget(), check)
            continue

        def guard():
            path = str(area.relative_to(root))
            owner = dict(
                owner="update_accounting",
                path=path,
                scope="one disposable CPU journal",
                retained=[],
                projected=dict(bytes=peak, inodes=16),
                observed=dict(bytes=0, inodes=0),
                revision=plan["candidate"],
                harm="incomplete preparation journal",
                rehearsal="T060",
                exhaustion="preserve the preceding batch receipt",
            )
            mounts = velocity_mounts(root, [owner], [dict(path=path, bytes=floor, inodes=0)])
            if any(m["deficit_bytes"] or m["deficit_inodes"] for m in mounts):
                errors[batch] = dict(stage="capacity", mounts=mounts)
                raise ValueError("insufficient mount capacity for one accounting batch")

        def execute(out, allowance):
            deadline = clock() + allowance
            try:
                funded_verification(
                    lambda directory: measure(directory, terms, order, lengths), out, deadline
                )
            finally:
                retain(
                    out / "cleanup.json", dict(acknowledged=True, within_bound=clock() <= deadline)
                )

        try:
            results[batch] = run_operation(
                area=area,
                grant=grant,
                journal=journal,
                coordinates=coords,
                request=req,
                cost=work,
                execute=execute,
                verify=check,
                before_dispatch=guard,
                installed=lambda out: (out / "producer.json").is_file(),
                max_attempts=2,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            errors.setdefault(batch, dict(stage="accounting", error=type(exc).__name__))
    if (area / "accounting/budget.jsonl").is_file():
        balance = budget()
    elif journal is None or any(area.glob("operations/*/attempt-*")):
        raise ValueError("original paid accounting journal required")
    else:
        balance = journal.snapshot()
    if balance["pending"] or balance["consumed"]["tokens"]:
        raise ValueError("CPU qualification has an inconsistent research debit")
    for op in results.values():
        key = f"{op['operation']['logical']}/dispatch/{op['attempt']}"
        if op["evidence"]["price"]["expected_us"] > balance["windows"][key]["charged_us"]:
            raise ValueError("measured CPU accounting exceeds its original debit")
    return seal(
        dict(
            plan=plan["identity"],
            grant=grant["identity"],
            errors=errors,
            batches={
                b: dict(receipt=op["identity"], **op["evidence"]) for b, op in results.items()
            },
            charged_work=balance["consumed"],
            remaining_us=balance["remaining"]["elapsed_us"],
        )
    )


def run(root, folder, plan, order, lengths, *, minutes, floor):
    parent = confined(folder, "preparation/update-accounting", exists=False)
    parent.mkdir(parents=True, exist_ok=True)
    block = os.statvfs(parent).f_frsize
    work = dict(
        attempts=2 * len(plan["batches"]),
        disk_bytes=2 * sum(journal_peak(t, block) for t in plan["batches"].values()),
    )
    grant = load(authorize_qualification(root, minutes, plan["identity"], work))
    area = confined(folder, "preparation/update-accounting/" + grant["identity"][7:], exists=False)
    retain(area / "plan.json", plan)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    result = operate(root, folder, plan, order, lengths, grant, journal=journal, floor=floor)
    if not result["errors"]:
        retain(area / "receipt.json", result)
    return result


def reopen(root, folder, plan, order, lengths, identity):
    area = confined(folder, "preparation/update-accounting/" + identity[7:])
    grant = load(confined(area, "grant.json"))
    auth = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, auth, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    if (
        load(confined(area, "plan.json")) != plan
        or grant["identity"] != identity
        or grant
        != load(
            confined(root, "run-state/discovery-authorizations/grant-" + identity[7:] + ".json")
        )
    ):
        raise ValueError("original CPU plan and independent grant required")
    replay = operate(root, folder, plan, order, lengths, grant)
    if replay != load(confined(area, "receipt.json")):
        raise ValueError("CPU prices differ from original paid operations")
    return replay


MARKER = "preparation/update-accounting-prices.json"


def record(root, folder, plan, order, lengths, result):
    if result["errors"] or result != reopen(root, folder, plan, order, lengths, result["grant"]):
        raise ValueError("both original paid CPU batches must reopen before publication")
    pointer = seal(
        dict(
            candidate=plan["candidate"],
            plan=plan["identity"],
            grant=result["grant"],
            receipt=result["identity"],
        )
    )
    path = confined(folder, MARKER, exists=False)
    if not path.exists():
        retain(path, pointer)
    elif load(path) != pointer:
        revise(path, pointer)


def inventory(root, fresh, cycles):
    folder = confined(root, "run-state/discovery/" + SESSION)
    marker = confined(folder, MARKER, exists=False)
    if not marker.exists():
        return None
    pointer = load(marker)
    plan, order, lengths = source_inputs(root, fresh, cycles, pointer["candidate"])
    if pointer != seal(pointer) or pointer["plan"] != plan["identity"]:
        raise ValueError("paid accounting scope, stream or workload differs")
    result = reopen(root, folder, plan, order, lengths, pointer["grant"])
    if result["identity"] != pointer["receipt"]:
        raise ValueError("original update-accounting price receipt required")
    return result
