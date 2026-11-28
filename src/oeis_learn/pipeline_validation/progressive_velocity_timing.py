"""Time registered optimizer work on exposed ordinary training slots, without making study cuts."""

from collections import defaultdict
import math
import os
from pathlib import Path
import time

from oeis_learn.generalization.budgets import Ledger, integer
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.shared_support import validated_rows
from oeis_learn.generalization.progressive.training_rows import matched_batch
from oeis_learn.generalization.progressive.velocity_trials import update_policy
from .progressive_velocity_matching import check_projection, _verified
from .progressive_velocity_worker import measure_calibration


def timing_projection(package, pairs, initial_weights, measurement):
    rows = package["rows"]
    shards = [rows[i : i + 4096] for i in range(0, len(rows), 4096)]
    indexed = validated_rows(shards, native_verified=True)
    stream = package["stream"]
    order = stream["order"]
    if (
        package != seal(package)
        or stream != seal(stream)
        or pairs != seal(pairs)
        or pairs["version"] != "round029-training-pairs/v1"
        or pairs["order"] != digest(order)
        or not order
        or len(set(order)) != len(order)
        or not set(order) <= indexed.keys()
        or stream["row_identities"] != digest([indexed[k]["identity"] for k in order])
    ):
        raise ValueError("original ordinary ordered training support and pair receipt required")
    projected = {
        k: dict(values=indexed[k]["values"][:20], tokens=indexed[k]["tokens"]) for k in order
    }
    donors = {}
    for k, pair in pairs["pairs"].items():
        if (
            k not in projected
            or pair["donor"] not in projected
            or pair["proof"]["status"] != "eligible"
        ):
            raise ValueError("pair outside the paid ordinary stream")
        donors[k] = projected[pair["donor"]]["values"]
    keys = matched_batch(stream, 0, 32 * ((len(order) + 31) // 32))
    return check_projection(
        seal(
            dict(
                version="round029-matching-projection/v1",
                plan=digest(["donor-update-timing/v1", package["identity"], measurement]),
                pair_proof=pairs["identity"],
                profile=package["profile"],
                initial_weights=initial_weights,
                rows=projected,
                donors=donors,
                order=keys,
            )
        )
    )


LOAD_BANDS = 3
SAMPLE_POINTS = ("typical", "slow")


def timing_limits():
    from oeis_learn.decoder.bigint_codec import MAX_BODY_TOKENS

    # Each ordinary recipe has at most two pair-support classes, three load
    # bands, and two observations in a band. The grant covers two dispatches.
    attempts_per_recipe = 2 * LOAD_BANDS * len(SAMPLE_POINTS) * 2
    recipes = [r for r in velocity_policy()["recipes"] if r["stream"] == "ordinary"]
    return dict(
        attempts=attempts_per_recipe * len(recipes),
        tokens=attempts_per_recipe * sum(r["batch"] for r in recipes) * MAX_BODY_TOKENS,
    )


def batch_plan(order, rows, donors, batch, *, target=4_000_000):
    """Represent every reachable batch start, including cycles that straddle the end of the stream."""
    if batch not in (16, 32) or len(set(order)) != len(order) or not order:
        raise ValueError("unique ordinary stream and registered batch size required")
    target = integer(target, 1)
    groups = defaultdict(list)
    tokens = {}
    for cursor in range(0, len(order), math.gcd(len(order), batch)):
        keys = [order[(cursor + i) % len(order)] for i in range(batch)]
        lengths = [len(rows[k]["tokens"]) for k in keys]
        tokens[cursor] = sum(lengths)
        groups[any(k in donors for k in keys)].append((max(lengths), tokens[cursor], cursor))
    cycle = sum(tokens.values())
    if min(tokens.values()) <= 0:
        raise ValueError("positive exposure required")
    # Charge the exact last whole optimizer batch that crosses the useful-token look.
    cycles, remainder = divmod(target - 1, cycle)
    issued = dict.fromkeys(tokens, cycles)
    cursor = 0
    while remainder >= 0:
        issued[cursor] += 1
        remainder -= tokens[cursor]
        cursor = (cursor + batch) % len(order)
    cells = []
    for has_pair, batches in sorted(groups.items()):
        batches.sort()
        # Within each donor-support class, split by padded length and token load. Use a
        # middle and largest registered load in each third, before observing any clocks.
        for i in range(LOAD_BANDS):
            subset = batches[len(batches) * i // LOAD_BANDS : len(batches) * (i + 1) // LOAD_BANDS]
            if subset:
                cells.append(
                    dict(
                        has_pair=has_pair,
                        batches=len(subset),
                        positive_tokens=sum(b[1] for b in subset),
                        updates_at_look=sum(issued[cursor] for _, _, cursor in subset),
                        tokens_at_look=sum(n * issued[cursor] for _, n, cursor in subset),
                        typical=subset[len(subset) // 2][2],
                        slow=subset[-1][2],
                    )
                )
    return cells


def timing_design(projection, count, microbatch):
    check_projection(projection)
    order = projection["order"][: integer(count, 1)]
    rows = projection["rows"]
    if (
        len(order) != len(rows)
        or set(order) != rows.keys()
        or len(set(order)) != count
        or matched_batch({"order": order}, 0, len(projection["order"])) != projection["order"]
    ):
        raise ValueError("calibration padding differs from original ordinary order")
    size = integer(microbatch, 1)
    plans = {str(b): batch_plan(order, rows, projection["donors"], b) for b in (16, 32)}
    methods = {}
    cost = dict(attempts=0, tokens=0)
    for r in velocity_policy()["recipes"]:
        if r["stream"] != "ordinary" or (r["matching"] and not projection["donors"]):
            continue
        batch = r["batch"]
        cursors = sorted({c[k] for c in plans[str(batch)] for k in SAMPLE_POINTS})
        work = {}
        for cursor in cursors:
            keys = matched_batch({"order": order}, cursor, batch)
            work[str(cursor)] = dict(
                positive_tokens=sum(len(rows[k]["tokens"]) for k in keys),
                eligible=sum(k in projection["donors"] for k in keys) if r["matching"] else 0,
                passes=(
                    sum(
                        any(k in projection["donors"] for k in keys[i : i + size])
                        for i in range(0, batch, size)
                    )
                    if r["matching"]
                    else 0
                ),
            )
        methods[r["id"]] = dict(batch=batch, matching=r["matching"], samples=work)
        cost["attempts"] += len(work)
        cost["tokens"] += sum(sample["positive_tokens"] for sample in work.values())
    return seal(
        dict(
            version="round029-update-timing-design/v1",
            projection=projection["identity"],
            count=count,
            microbatch=size,
            plans=plans,
            methods=methods,
            cost=cost,
        )
    )


def unit_price(cells, clocks):
    expected = tail = 0
    if set(clocks) != {c[k] for c in cells for k in SAMPLE_POINTS}:
        raise ValueError("timings must cover exactly the registered sample")
    for c in cells:
        mid = integer(clocks[c["typical"]], 1)
        upper = max(mid, integer(clocks[c["slow"]], 1))
        expected += c["updates_at_look"] * mid
        tail += c["updates_at_look"] * upper
    return dict(expected_us=expected, tail_us=tail)


def run_matrix(projection, design, *, device):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from .progressive_generator_probe import batch
    from .progressive_training import PolicyUpdate

    if design != timing_design(projection, design["count"], design["microbatch"]):
        raise ValueError("timing design differs from the frozen inputs")

    def now():
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        return time.monotonic_ns() // 1000

    calibration = measure_calibration(projection, device=device)
    _verified(calibration, projection)
    order = {"order": projection["order"][: design["count"]]}
    donors = projection["donors"]
    observations = {}
    for method, terms in design["methods"].items():
        if terms["matching"] and calibration["weight"] is None:
            observations[method] = dict(unavailable=calibration["reason"])
            continue
        start = now()
        policy = update_policy(
            projection["profile"],
            method,
            microbatch=min(terms["batch"], design["microbatch"]),
            matching_weight=calibration["weight"] if terms["matching"] else None,
        )
        runner = PolicyUpdate(projection["profile"], policy, device=device)
        startup = now() - start
        if model_snapshot(runner.encoder, runner.decoder) != projection["initial_weights"]:
            raise ValueError("timing cell did not start from the registered ordinary weights")
        ledger = Ledger(sum(w["positive_tokens"] for w in terms["samples"].values()))
        samples = {}
        for cursor in sorted(terms["samples"], key=int):
            keys = matched_batch(order, int(cursor), terms["batch"])
            terms_at = terms["samples"][cursor]
            start = now()
            seq, tokens = batch([projection["rows"][k] for k in keys], device)
            other = (
                [list(map(int, donors[k])) if k in donors else None for k in keys]
                if terms["matching"]
                else None
            )
            obs, ledger = runner.update(
                seq,
                tokens,
                ledger=ledger.reserve(cursor, terms_at["positive_tokens"]),
                matching_donors=other,
            )
            samples[cursor] = dict(
                elapsed_us=now() - start,
                positive_tokens=obs["charged_tokens"],
                eligible=obs.get("matching_eligible", 0),
                passes=obs.get("matching_passes", 0),
                auxiliary_forward_us=obs.get("matching_forward_us", 0),
            )
        observations[method] = dict(startup_us=startup, policy=policy["identity"], samples=samples)
        del runner
    return seal(
        dict(
            version="round029-update-timings/v1",
            design=design["identity"],
            calibration=calibration,
            methods=observations,
        )
    )


def matrix_prices(observed, projection, design, *, device):
    if (
        observed != seal(observed)
        or observed["version"] != "round029-update-timings/v1"
        or observed["design"] != design["identity"]
        or observed["methods"].keys() != design["methods"].keys()
    ):
        raise ValueError("observed matrix differs from the registered design")
    calibration = _verified(observed["calibration"], projection)
    if calibration["device"] != device:
        raise ValueError("calibration device differs from the registered worker")
    units = {}
    unavailable = {}
    measured_updates = measured_tokens = 0
    for method, terms in design["methods"].items():
        obs = observed["methods"][method]
        if terms["matching"] and calibration["weight"] is None:
            if obs != {"unavailable": calibration["reason"]}:
                raise ValueError("matching timing requires a valid measured loss weight")
            unavailable[method] = calibration["reason"]
            continue
        expected = update_policy(
            projection["profile"],
            method,
            microbatch=min(terms["batch"], design["microbatch"]),
            matching_weight=calibration["weight"] if terms["matching"] else None,
        )
        if (
            obs["policy"] != expected["identity"]
            or obs["samples"].keys() != terms["samples"].keys()
        ):
            raise ValueError("original update policy or ordered timing cells differ")
        clocks = {}
        for cursor, w in terms["samples"].items():
            point = obs["samples"][cursor]
            elapsed = integer(point["elapsed_us"], 1)
            auxiliary = integer(point["auxiliary_forward_us"])
            if (
                any(type(point[k]) is not int or point[k] != w[k] for k in w)
                or auxiliary > elapsed
                or (not w["passes"] and auxiliary)
            ):
                raise ValueError(
                    "timed update differs from its registered charge or matching opportunities"
                )
            clocks[int(cursor)] = elapsed
            measured_updates += 1
            measured_tokens += w["positive_tokens"]
        estimate = unit_price(design["plans"][str(terms["batch"])], clocks)
        init = integer(obs["startup_us"], 1)
        units[method] = dict(
            expected_us=estimate["expected_us"] + init, tail_us=estimate["tail_us"] + init
        )
    return dict(
        methods=units,
        unavailable=unavailable,
        look_work={
            batch: {k: sum(c[k] for c in cells) for k in ("tokens_at_look", "updates_at_look")}
            for batch, cells in design["plans"].items()
        },
        support_scope="ordinary positive stream with proven fixed-target matching pairs",
        calibration_us=calibration["elapsed_us"],
        calibration_batches=len(projection["order"]) // 32,
        calibration_eligible=calibration["eligible_rows"],
        calibration_passes=calibration["extra_passes"],
        update_count=measured_updates,
        sample_positive_tokens=measured_tokens,
    )


def main():
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.tracking.pipeline_profile import Recorder
    from .progressive_generator_probe import usage_summary
    from .progressive_qualification import require_hip
    from .progressive_research_inputs import checked_file
    from .progressive_roles import ROUND029_ROLE_POLICY

    inputs = Path("/work/inputs")
    request = load(inputs / "request.json")
    if (
        request != seal(request)
        or request["version"]
        != ("round029-timing-request/v2" if "runtime" in request else "round029-timing-request/v1")
        or request["source"] != digest(source_snapshot())
        or request["role_policy"] != ROUND029_ROLE_POLICY
        or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
    ):
        raise ValueError("installed timing source, image or role differs")
    ref = request["projection"]
    projection = load(checked_file(inputs, ref), maximum=ref["bytes"])
    torch = require_hip()
    torch.cuda.reset_peak_memory_stats()
    out = Path("/work/output")
    nodes = sorted(Path("/dev/dri").glob("renderD*"))
    start = time.monotonic_ns()
    cpu_start = time.process_time_ns()
    with Recorder(
        out / "telemetry",
        instrumented=True,
        render_node=nodes[0] if len(nodes) == 1 else None,
        host_counters=False,
    ) as recorder:
        with recorder.phase("ordinary-update-matrix"):
            observed = run_matrix(projection, request["design"], device="cuda")
            torch.cuda.synchronize()
    elapsed = (time.monotonic_ns() - start) // 1000
    cpu_us = (time.process_time_ns() - cpu_start) // 1000
    atomic_write(
        out / "producer.json",
        seal(
            dict(
                request=request["identity"],
                matrix=observed,
                device="hip",
                elapsed_us=elapsed,
                worker_cpu_us=cpu_us,
                telemetry=usage_summary(out / "telemetry", elapsed),
                peak_device_bytes=torch.cuda.max_memory_allocated(),
            )
        ),
    )


if __name__ == "__main__":
    main()
