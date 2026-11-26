"""Recover an interrupted physical role without granting a second allowance to an opened case."""

from collections import Counter

from oeis_learn.generalization.progressive.contracts import (
    digest,
    load,
    read_bound_record,
    reference,
    seal,
)
from .identity import confined
from .preparation_recovery import retain
from .progressive_panel_sources import require
from .progressive_round028_receipts import verify_role_mounts
from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE, clock_record
from oeis_learn.generalization.progressive.program_search_costing import CLOCK_FIELDS


def interrupted(runner, role, data, mounts):
    """Only a provably terminated original can release unused time to its successor."""
    area = runner.area
    assignment = area / "assignments" / (data["block"] + "-" + role + ".json")
    if not assignment.is_file():
        return None
    saved = load(confined(area, assignment))
    old = confined(area, "roles/" + saved["request"][7:])
    if (old / "boundary.json").exists():
        return None
    attempts = list(old.glob("attempt-*"))
    if not attempts:
        return None
    require(len(attempts) == 1 and seal(saved) == saved, "invalid original pilot attempt")
    attempt = confined(old, attempts[0])
    req = load(confined(old, "request.json"))
    data_path = confined(old, "data.json")
    require(
        saved["request"] == req["identity"]
        and seal(req) == req
        and req["role"] == role
        and req["image"] == runner.image
        and req["source"] == runner.registration["source"]
        and req["registration"] == runner.registration["identity"]
        and seal(data) == load(data_path, maximum=128 << 20)
        and req["data"] == seal(data)["identity"],
        "interrupted role parents changed",
    )
    output = confined(attempt, "output")
    # A normally finished role can still be adopted by NativeRoles.bind after the supervisor's end.
    end_path = attempt / "end.json"
    if end_path.is_file():
        ended = load(confined(attempt, end_path))
        require(
            ended == seal(ended) and ended["request"] == req["identity"], "invalid original end"
        )
        if ended["success"]:
            return None
    inspected = load(confined(output, "container-inspect.json"), maximum=128 << 20)
    verify_role_mounts(
        inspected,
        runner.image,
        {
            "/work/output": (output, True),
            "/work/inputs/request.json": (old / "request.json", False),
            "/work/inputs/data.json": (data_path, False),
            **{dest: (p, False) for p, dest in mounts},
        },
    )
    state = inspected[0]["State"]
    require(state["Running"] is False, "original pilot worker still owns its attempt")
    elapsed = load(confined(output, "gpu-lease.json"))["elapsed_us"] if runner.phase == "hip" else 0
    require(type(elapsed) is int and elapsed >= 0, "unproved original lease")
    if state["ExitCode"] == 0 and (output / "producer.json").is_file() and not end_path.exists():
        # Host interruption after engine inspection/cleanup but before recording its notification.
        retain(
            end_path, seal(dict(request=req["identity"], elapsed_us=max(1, elapsed), success=True))
        )
        return None
    return output, elapsed


def recover_choices(area, output, block, prompts, weight, recipe, arm, target):
    """A partial case stays unresolved. Each never-opened prompt retains its original full budget."""
    policy = CORRECT_FULL if target == 4_000_000 else DESCRIPTIVE
    records, clocks, unopened, unresolved = {}, {}, [], []
    model = None
    for p in prompts:
        directory = output / "case-boundaries" / digest(p["id"])[7:]
        if not directory.exists():
            unopened.append(p)
            continue
        confined(output, directory)
        choice = directory / "choice.json"
        if not choice.is_file():
            unresolved.append(p["id"])
            continue
        start = load(confined(directory, "start.json"))
        value = load(confined(directory, choice), maximum=128 << 20)
        c = value["clock"]
        require(
            start == seal(start)
            and value == seal(value)
            and start["prompt"] == value["prompt"] == digest(p)
            and start["model"] == value["model"] == c["snapshot"]
            and start["policy"] == value["policy"] == c["policy"] == policy
            and c["prompt"] == p["id"]
            and c == clock_record(**{k: c[k] for k in CLOCK_FIELDS}),
            "changed original case boundary",
        )
        model = model or start["model"]
        require(model == start["model"], "portfolio weights changed within the interrupted role")
        records[p["id"]] = value["record"]
        clocks[p["id"]] = c
    if not records:
        return None, unopened, unresolved
    ordered = sorted((r for r in prompts if r["id"] in records), key=lambda p: p["id"])
    clock = seal(
        dict(
            version="pipeline-portfolio-clocks/v1",
            policy=policy,
            source=digest(recipe),
            work=model,
            expected_portfolios=len(records),
            missing_portfolios=0,
            cost_complete=all(c["cost_complete"] for c in clocks.values()),
            terminations=dict(Counter(c["termination"] for c in clocks.values())),
            attempts=digest(sorted(clocks.items())),
            clocks=[clocks[k] for k in sorted(clocks)],
        )
    )
    candidate = seal(
        dict(
            version="discovery-candidates/v1",
            model=model,
            model_control="retained_D" if arm == "incumbent" else "earlier_checkpoint",
            checkpoint=weight["sha256"],
            recipe=digest(recipe),
            methods=["correct"],
            prompts=digest(ordered),
            records=records,
            clock=clock,
            counts={"recovered_portfolios": len(records)},
            elapsed_us=sum(round((c["finish"] - c["start"]) * 1e6) for c in clocks.values()),
        )
    )
    path = area / "recovered" / block / "candidates.json"
    retain(path, candidate)
    return (reference(area, path), path, candidate, ordered), unopened, unresolved


def measured_search(run, settle, progress, cpu, gpu, look, recipe, weight, weights, prompts, truth):
    from .progressive_round028_pilot import score_projection

    arm, target = look["arm"], look["target"]
    base = f"{arm}-{target}"
    original = {r["id"]: r for r in truth}
    pending = list(prompts)
    results = {}
    comparable = True
    segment = 0
    while pending:
        block = base if not segment else base + "-resume-" + str(segment)
        seconds = 60 * len(pending) + 120
        data = dict(
            block=block,
            arm=arm,
            target=target,
            rows=pending,
            recipe=recipe,
            weights=weight,
            seconds=seconds,
        )
        mounts = [(weights, "/work/inputs/weights.pt")]
        progress("pilot", arm=arm, target=target, role="search", cases=len(pending))
        previous = interrupted(gpu, "search", data, mounts)
        if previous is None:
            generated = run(gpu, "search", data, mounts)
            ref = generated["candidates"]
            value = read_bound_record(gpu.last_output, ref, version="discovery-candidates/v1")
            part = (ref, confined(gpu.last_output, ref["path"]), value, pending)
            pending = []
        else:
            output, lease = previous
            settle(block, "search", lease)
            part, pending, unknown = recover_choices(
                gpu.area, output, block, data["rows"], weight, recipe, arm, target
            )
            for key in unknown:
                results[key] = {"outcome": "unresolved", "complete": False}
            # A partial portfolio kept its original allowance; uncertainty is propagated by the reducer.
        if part:
            ref, path, value, finished = part
            rows = [original[p["id"]] for p in finished]
            progress("pilot", arm=arm, target=target, role="score", cases=len(rows))
            scored = run(
                cpu,
                "score",
                dict(
                    block=block,
                    rows=rows,
                    candidates=dict(ref, path="inputs/candidates.json"),
                    seconds=60 * len(rows) + 120,
                ),
                [(path, "/work/inputs/candidates.json")],
            )
            scores = read_bound_record(
                cpu.last_output, scored["scores"], version="discovery-scoring/v1"
            )
            batch, closed = score_projection(
                value,
                scores,
                ref["sha256"],
                finished,
                weight,
                recipe,
                CORRECT_FULL if target == 4_000_000 else DESCRIPTIVE,
            )
            require(not results.keys() & batch.keys(), "duplicate pilot case credit")
            results.update(batch)
            comparable &= closed
        segment += 1
    require(results.keys() == original.keys(), "missing registered pilot cases")
    return results, comparable
