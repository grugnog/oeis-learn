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
    require(seal(saved) == saved, "invalid original pilot assignment")
    old = confined(area, "roles/" + saved["request"][7:], exists=False)
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
    if state["Running"] is not False:
        cleanup = load(confined(output, "cleanup.json"))
        require(cleanup["acknowledged"] is True and cleanup["failure"] is None,
                "original pilot worker still owns its attempt")
    elapsed = load(confined(output, "gpu-lease.json"))["elapsed_us"] if runner.phase == "hip" else 0
    require(type(elapsed) is int and elapsed >= 0, "unproved original lease")
    if state["ExitCode"] == 0 and (output / "producer.json").is_file() and not end_path.exists():
        # Host interruption after engine inspection/cleanup but before recording its notification.
        retain(
            end_path,
            seal(
                dict(
                    request=req["identity"],
                    elapsed_us=elapsed,
                    supervisor_wall_unknown=True,
                    success=True,
                )
            ),
        )
        return None
    wall = ended["elapsed_us"] if end_path.is_file() else None
    return output, elapsed, wall


def recover_choices(area, output, block, prompts, weight, recipe, arm, target, inference_seed=None):
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
            and start.get("inference_seed") == value.get("inference_seed") == inference_seed
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
            **({"inference_seed": inference_seed} if inference_seed is not None else {}),
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


def measured_training(
    run, settle, gpu, arm, target, original_previous, training, package, registration
):
    """Resume only full-state generations; every attempted update remains charged across boots."""
    from oeis_learn.generalization.contracts import artifact_refs
    from oeis_learn.generalization.progressive.budgets import KEYS, BudgetJournal
    from oeis_learn.generalization.state import restore
    from .progressive_research_inputs import checked_file

    previous, mounts = original_previous
    attempt = 0
    charged = None
    while True:
        base = f"{arm}-{target}"
        block = base if not attempt else base + "-resume-" + str(attempt)
        data = dict(
            block=block,
            arm=arm,
            target=target,
            registration=registration,
            previous=previous,
            resume_charge=charged,
            training=dict(training, path="inputs/training.json"),
            package=package,
        )
        inputs = [(gpu.area / "training.json", "/work/inputs/training.json"), *mounts]
        failure = interrupted(gpu, "train", data, inputs)
        if failure is None:
            return run(gpu, "train", data, inputs)
        output, lease, wall = failure
        settle(block, "train", lease, wall)
        origin_path = output / "training-origin.json"
        if origin_path.exists():
            origin = load(confined(output, origin_path))
            require(
                origin == seal(origin)
                and (origin["registration"], origin["package"], origin["arm"], origin["target"])
                == (registration, package, arm, target),
                "interrupted training belongs to another request",
            )
            accrued = 0
            if (output / "accounting/budget-binding.json").exists():
                journal = BudgetJournal(
                    confined(output, "accounting"),
                    origin["contract"],
                    dict(dict.fromkeys(KEYS, 0), tokens=origin["cap"]),
                )
                accrued = journal.snapshot()["consumed"]["tokens"]
            charged = origin["starting_charge"] + accrued
            pointer = output / "checkpoint/generalization-latest.json"
            if pointer.exists():
                checkpoint = confined(output, "checkpoint")
                previous = load(confined(output, pointer))
                manifest = read_bound_record(checkpoint, previous)
                state = restore(
                    manifest, contract=origin["contract"], cap=origin["cap"], root=checkpoint
                )
                require(
                    state["ledger"].consumed <= charged,
                    "restored model outruns its irreversible charges",
                )
                mounts = [
                    (checked_file(checkpoint, r), "/work/inputs/previous/" + r["path"])
                    for r in [previous, *artifact_refs(manifest["states"])]
                ]
        else:
            require(not (output / "accounting").exists(), "lost pre-update training authority")
        attempt += 1


def qualification_role(runner, role, data, *, seconds, inputs):
    """Retry only exposed qualification work after proving termination of the original."""
    block = data["block"]
    ordinal = 0
    pending = data.get("rows", [])
    segments = []
    debits = []
    while True:
        name = block if ordinal == 0 else f"{block}-retry-{ordinal}"
        request = dict(data, block=name)
        allowance = seconds
        if role == "search" and segments:
            request.update(rows=pending, seconds=resident_seconds(len(pending)))
            allowance = request["seconds"] + 120
        request = seal(request)
        previous = interrupted(runner, role, request, inputs)
        if previous is None:
            result = runner(role, request, seconds=allowance, inputs=inputs)
            if not segments:
                return result
            value = read_bound_record(runner.last_output, result["candidates"], version="discovery-candidates/v1")
            segments.append(value)
            break
        output, lease, wall = previous
        failures = runner.area / "qualification-costs"
        debit = seal(dict(request=request["identity"], role=role, lease_us=lease, wall_us=wall))
        retain(failures / f"{name}-{role}.json", debit)
        debits.append(debit)
        if role == "search":
            part, unopened, unresolved = recover_choices(
                runner.area, output, name, request["rows"], request["weights"],
                request["recipe"], request["arm"], request["target"], request["inference_seed"]
            )
            if part:
                segments.append(part[2])
                missing = {p["id"] for p in unopened} | set(unresolved)
                pending = [p for p in pending if p["id"] in missing]
                if not pending:
                    break
        ordinal += 1
    # Join the verified per-case boundaries. An already sealed choice is never drawn again.
    first = segments[0]
    records, clocks = {}, {}
    for part in segments:
        require(all(part[k] == first[k] for k in ("model", "model_control", "checkpoint", "recipe", "methods"))
                and not records.keys() & part["records"].keys()
                and part["clock"]["inference_seed"] == data["inference_seed"]
                and part["clock"]["policy"] == first["clock"]["policy"], "qualification segment changed")
        records.update(part["records"])
        clocks.update({row["prompt"]: row for row in part["clock"]["clocks"]})
    ordered = sorted(data["rows"], key=lambda p: p["id"])
    require(set(records) == set(clocks) == {p["id"] for p in ordered}, "incomplete qualification recovery")
    clock = seal(dict(first["clock"], expected_portfolios=len(clocks), missing_portfolios=0,
        cost_complete=all(c["cost_complete"] for c in clocks.values()),
        terminations=dict(Counter(c["termination"] for c in clocks.values())),
        attempts=digest(sorted(clocks.items())), clocks=[clocks[k] for k in sorted(clocks)]))
    candidate = seal(dict(first, clock=clock, records=records, prompts=digest(ordered),
                          elapsed_us=sum(s["elapsed_us"] for s in segments), counts={"recovered_portfolios":len(records)}))
    dest = runner.area / "recovered" / (block + "-complete")
    retain(dest / "candidates.json", candidate)
    runner.last_output = dest
    runner.last_producer = seal(dict(recovered=candidate["identity"], segments=[p["identity"] for p in segments],
                                      debits=[d["identity"] for d in debits], elapsed_us=candidate["elapsed_us"]))
    return {"candidates": reference(dest, dest / "candidates.json")}


def measured_score(run, settle, cpu, data, mounts):
    # The CPU scorer can replay an already sealed choice. Its predecessor's time stays charged.
    block = data["block"]
    attempt = 0
    while True:
        name = block if not attempt else f"{block}-scoreretry-{attempt}"
        request = dict(data, block=name)
        previous = interrupted(cpu, "score", request, mounts)
        if previous is None:
            return run(cpu, "score", request, mounts)
        _, lease, wall = previous
        settle(name, "score", lease, wall)
        attempt += 1


def resident_seconds(cases):
    # A stalled resident still has an external containment deadline. Budget one
    # additional portfolio per case for encoding, native drainage and persistence,
    # plus the existing cold evaluation setup allowance. Each case owns its 60s.
    if type(cases) is not int or not 1 <= cases <= 160:
        raise ValueError("resident pilot roster must contain 1 to 160 cases")
    return 1800 + 120 * cases


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
        seconds = resident_seconds(len(pending))
        data = dict(
            block=block,
            arm=arm,
            target=target,
            rows=pending,
            recipe=recipe,
            inference_seed=look["seed"],
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
            output, lease, wall = previous
            settle(block, "search", lease, wall)
            part, pending, unknown = recover_choices(
                gpu.area, output, block, data["rows"], weight, recipe, arm, target, look["seed"]
            )
            comparable &= not unknown
            for key in unknown:
                results[key] = {"outcome": "unresolved", "complete": False}
            # A partial portfolio kept its original allowance; uncertainty is propagated by the reducer.
        if part:
            ref, path, value, finished = part
            rows = [original[p["id"]] for p in finished]
            progress("pilot", arm=arm, target=target, role="score", cases=len(rows))
            scored = measured_score(
                run,
                settle,
                cpu,
                dict(
                    block=block,
                    rows=rows,
                    candidates=dict(ref, path="inputs/candidates.json"),
                    seconds=resident_seconds(len(rows)),
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
                inference_seed=look["seed"],
            )
            require(not results.keys() & batch.keys(), "duplicate pilot case credit")
            results.update(batch)
            comparable &= closed
        segment += 1
    require(results.keys() == original.keys(), "missing registered pilot cases")
    return results, comparable
