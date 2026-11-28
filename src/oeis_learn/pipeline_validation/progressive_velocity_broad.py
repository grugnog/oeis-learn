"""Native preparation and optimizer sampling for the paired broader-composition treatments."""

from pathlib import Path
import time

from oeis_learn.generalization.budgets import Ledger, integer
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.training_rows import matched_batch, matching_pairs
from oeis_learn.generalization.progressive.velocity_trials import update_policy
from .progressive_velocity_matching import check_projection
from .progressive_velocity_timing import timing_design, unit_price

MAX_INPUT = 256 << 20


def shards(rows):
    return [rows[i : i + 4096] for i in range(0, len(rows), 4096)]


def projection(stream, rows, pairs, origin):
    order = stream["order"]
    unique = list(dict.fromkeys(order))
    if (
        stream != seal(stream)
        or pairs != seal(pairs)
        or pairs["order"] != digest(unique)
        or set(unique) != rows.keys()
    ):
        raise ValueError("broader positive order or matching support differs")
    # A draw can legitimately fill several matched slots. Price each slot once.
    keys = [digest([stream["identity"], i, draw]) for i, draw in enumerate(order)]
    positives, donors = {}, {}
    for slot, draw in zip(keys, order, strict=True):
        row = rows[draw]
        positives[slot] = dict(values=row["values"][:20], tokens=row["tokens"])
        if draw in pairs["pairs"]:
            pair = pairs["pairs"][draw]
            if pair["proof"]["status"] != "eligible" or pair["donor"] not in rows:
                raise ValueError("matching donor lacks its registered negative proof")
            donors[slot] = rows[pair["donor"]]["values"][:20]
    return check_projection(
        seal(
            dict(
                version="round029-matching-projection/v1",
                plan=stream["identity"],
                pair_proof=pairs["identity"],
                profile=origin["profile"],
                initial_weights=origin["initial_weights"],
                rows=positives,
                donors=donors,
                order=matched_batch(dict(order=keys), 0, 32 * ((len(keys) + 31) // 32)),
            )
        )
    )


def matching_views(data, stream, selected, *, client=None, recorded=None):
    seed = data["stream"]["seeds"]["training_families"]
    if data["pair_seed"] != seed or stream["seed"] != seed:
        raise ValueError("matching selection must use the registered trial stream seed")
    ordinary = {r["draw_identity"]: r for r in data["ordinary_rows"]}
    views, manifests, summary = {}, {}, {}
    remaining = data.get("native_pair_limit")
    for name, positives, rows in (
        ("ordinary", data["stream"], ordinary),
        ("broader", stream, selected),
    ):
        if name == "broader" and "frame" in data and not stream["available"]:
            continue
        order = list(dict.fromkeys(positives["order"]))
        pairs = matching_pairs(
            shards(list(rows.values())),
            order,
            seed=seed,
            client=client,
            recorded=recorded[name] if recorded is not None else None,
            native_limit=remaining,
        )
        if remaining is not None:
            remaining -= pairs["coverage"]["native_attempts"]
        manifests[name] = pairs
        views[name] = projection(positives, rows, pairs, data)
        summary[name] = dict(
            seed=seed,
            stream=positives["identity"],
            manifest=pairs["identity"],
            donors=digest(pairs["donor_map"]),
            coverage=pairs["coverage"],
            dispositions=pairs["dispositions"],
            eligible_unique=pairs["eligible_rows"],
            eligible_slots=len(views[name]["donors"]),
            eligible_families=pairs["eligible_families"],
        )
    return seal(
        dict(
            version="round029-paired-timing-input/v1",
            streams=views,
            support=summary,
            **({"fresh_start": data["start"]["identity"]} if "start" in data else {}),
        )
    ), manifests


def design(view, microbatch, calibration):
    if "streams" in view:
        if view != seal(view):
            raise ValueError("matching timing inputs differ")
        methods, cells = {}, {}
        timings = {
            name: timing_design(stream, len(stream["rows"]), microbatch)
            for name, stream in view["streams"].items()
        }
        recipes = [
            ("01", "broader", "00"),
            ("02", "ordinary", "02"),
            ("03", "broader", "02"),
        ]
        if "fresh_start" in view:
            recipes.extend((m, "ordinary", m) for m in ("00", "04", "05", "06"))
        for method, name, source_method in recipes:
            if name not in view["streams"]:
                continue
            timing = timings[name]
            if source_method not in timing["methods"]:
                continue
            if source_method == "02" and (
                not view["streams"]["ordinary"]["donors"]
                or not all(view["support"][k]["coverage"]["full"] for k in ("ordinary", name))
            ):
                continue
            methods[method] = timing["methods"][source_method]
            cells[method] = timing["plans"][str(methods[method]["batch"])]
        ordinary = view["streams"]["ordinary"]
        batches = len(ordinary["order"]) // 32
        calibration_tokens = sum(len(ordinary["rows"][k]["tokens"]) for k in ordinary["order"])
        samples = [s for m in methods.values() for s in m["samples"].values()]
        return seal(
            dict(
                version="round029-paired-timing-design/v1",
                projection=view["identity"],
                calibration=calibration,
                microbatch=microbatch,
                methods=methods,
                cells=cells,
                calibration_batches=batches,
                calibration_tokens=calibration_tokens,
                cost=dict(
                    attempts=batches + len(samples),
                    tokens=calibration_tokens + sum(s["positive_tokens"] for s in samples),
                ),
            )
        )
    ordinary = timing_design(view, len(view["rows"]), microbatch)
    methods = {"01": ordinary["methods"]["00"]}
    if calibration["weight"] is not None and "02" in ordinary["methods"]:
        methods["03"] = ordinary["methods"]["02"]
    samples = [s for m in methods.values() for s in m["samples"].values()]
    return seal(
        dict(
            version="round029-broad-timing-design/v1",
            projection=view["identity"],
            calibration=calibration,
            microbatch=microbatch,
            methods=methods,
            cells=ordinary["plans"]["32"],
            cost=dict(attempts=len(samples), tokens=sum(s["positive_tokens"] for s in samples)),
        )
    )


def supply_counts(stream, audit, data):
    return dict(
        ordinary=stream["ordinary"]["distinct"],
        broad=stream["broad"]["distinct"],
        ordinary_per_million=stream["ordinary"]["per_million"],
        broad_per_million=stream["broad"]["per_million"],
        shortage=stream["shortage"],
        positive_tokens=stream["broad"]["tokens"],
        slots=len(stream["order"]),
        eligible_draws=len(data["eligible_ordinals"]),
        stream=stream["identity"],
        audit=audit["identity"],
    )


def prepare(data, out, client):
    from .progressive_preparation import prepare_velocity_broad

    if "frame" in data:
        from . import progressive_velocity_fresh as fresh

        rows = fresh.native_rows(data, client)
        ordinary = fresh.ordinary_stream(data, rows)
        data = dict(data, ordinary_rows=rows, stream=ordinary)
    stream, selected, audit = prepare_velocity_broad(
        root=out,
        stream=data["stream"],
        shards=shards(data["ordinary_rows"]),
        law=data["stream"]["law"],
        ordinals=data["eligible_ordinals"],
        familiar=data["familiar"],
        calibration=data["difficulty"],
        protected=[],
        indexes=[],
        client=client,
    )
    stats = supply_counts(stream, audit, data)
    if "frame" in data:
        from .progressive_preparation import prepare_velocity_start

        start = prepare_velocity_start(
            data["profile"], ordinary, microbatch=data["microbatch"], broad=stream, device="cpu"
        )
        if start["initial_weights"] == data["previous_weights"]:
            raise ValueError("fresh model reproduced the rejected pilot initialization")
        package = seal(
            dict(request=data["identity"], stream=ordinary, ordinary_rows=rows, start=start)
        )
        atomic_write(out / "fresh-ordinary.json", package)
        data = dict(data, initial_weights=start["initial_weights"], start=start)
    if not stream["available"] and "frame" not in data:
        return seal(
            dict(
                request=data["identity"],
                disposition="no_breadth_at_matched_exposure",
                supply=stats,
                design=None,
            )
        )
    view, manifests = matching_views(data, stream, selected, client=client)
    plan = design(view, data["microbatch"], data["calibration"])
    atomic_write(out / "pairs.json", manifests)
    atomic_write(out / "projection.json", view)
    stats["matching"] = view["support"]
    return seal(
        dict(
            request=data["identity"],
            disposition="prepared" if stream["available"] else "fresh_ordinary_only",
            supply=stats,
            design=plan,
        )
    )


def run(view, plan, *, device):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from .progressive_generator_probe import batch
    from .progressive_training import PolicyUpdate

    if plan != design(view, plan["microbatch"], plan["calibration"]):
        raise ValueError("broader sample design differs from the issued projection")

    def now():
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        return time.monotonic_ns() // 1000

    calibration = plan["calibration"]
    extra = {}
    if "streams" in view:
        from .progressive_velocity_worker import measure_calibration
        from .progressive_velocity_matching import _verified

        ordinary = view["streams"]["ordinary"]
        calibration = _verified(measure_calibration(ordinary, device=device), ordinary)
        extra["calibration"] = calibration
    methods = {}
    for method, terms in plan["methods"].items():
        projected = (
            view["streams"]["broader" if method in ("01", "03") else "ordinary"]
            if "streams" in view
            else view
        )
        if terms["matching"] and calibration["weight"] is None:
            methods[method] = dict(unavailable=calibration.get("reason", "no ordinary calibration"))
            continue
        order = dict(order=projected["order"][: len(projected["rows"])])
        start = now()
        policy = update_policy(
            projected["profile"],
            method,
            microbatch=min(terms["batch"], plan["microbatch"]),
            matching_weight=calibration["weight"] if terms["matching"] else None,
        )
        runner = PolicyUpdate(projected["profile"], policy, device=device)
        startup = now() - start
        if model_snapshot(runner.encoder, runner.decoder) != projected["initial_weights"]:
            raise ValueError("broader optimizer has different initial weights")
        ledger = Ledger(sum(s["positive_tokens"] for s in terms["samples"].values()))
        samples = {}
        for cursor in sorted(terms["samples"], key=int):
            keys = matched_batch(order, int(cursor), terms["batch"])
            start = now()
            seq, tokens = batch([projected["rows"][k] for k in keys], device)
            donors = (
                [
                    list(map(int, projected["donors"][k])) if k in projected["donors"] else None
                    for k in keys
                ]
                if terms["matching"]
                else None
            )
            spent, ledger = runner.update(
                seq,
                tokens,
                ledger=ledger.reserve(cursor, terms["samples"][cursor]["positive_tokens"]),
                matching_donors=donors,
            )
            samples[cursor] = dict(
                elapsed_us=now() - start,
                positive_tokens=spent["charged_tokens"],
                eligible=spent.get("matching_eligible", 0),
                passes=spent.get("matching_passes", 0),
                auxiliary_forward_us=spent.get("matching_forward_us", 0),
            )
        methods[method] = dict(startup_us=startup, policy=policy["identity"], samples=samples)
        del runner
    return seal(dict(design=plan["identity"], methods=methods, **extra))


def prices(view, plan, observed):
    if (
        plan != design(view, plan["microbatch"], plan["calibration"])
        or observed != seal(observed)
        or observed["design"] != plan["identity"]
        or observed["methods"].keys() != plan["methods"].keys()
    ):
        raise ValueError("broader measurements differ from the frozen sample")
    calibration = plan["calibration"]
    if "streams" in view:
        from .progressive_velocity_matching import _verified

        calibration = _verified(observed["calibration"], view["streams"]["ordinary"])
    result = {}
    for method, terms in plan["methods"].items():
        row = observed["methods"][method]
        if terms["matching"] and calibration["weight"] is None:
            if row != dict(unavailable=calibration["reason"]):
                raise ValueError("matching optimizer requires a registered loss weight")
            continue
        projected = (
            view["streams"]["broader" if method in ("01", "03") else "ordinary"]
            if "streams" in view
            else view
        )
        expected = update_policy(
            projected["profile"],
            method,
            microbatch=min(terms["batch"], plan["microbatch"]),
            matching_weight=calibration["weight"] if terms["matching"] else None,
        )
        if (
            row["policy"] != expected["identity"]
            or row["samples"].keys() != terms["samples"].keys()
        ):
            raise ValueError("broader optimizer policy or sample differs")
        clocks = {}
        for cursor, sample in row["samples"].items():
            clock = integer(sample["elapsed_us"], 1)
            auxiliary = integer(sample["auxiliary_forward_us"])
            if (
                any(
                    type(sample[k]) is not int or sample[k] != v
                    for k, v in terms["samples"][cursor].items()
                )
                or auxiliary > clock
                or (not sample["passes"] and auxiliary)
            ):
                raise ValueError("broader positive work or auxiliary passes differ")
            clocks[int(cursor)] = clock
        startup = integer(row["startup_us"], 1)
        result[method] = {
            k: v + startup
            for k, v in unit_price(
                plan["cells"][method] if "streams" in view else plan["cells"], clocks
            ).items()
        }
    return result


def checked_request(data, action):
    common = {"identity", "role", "source", "image", "binding", "role_policy"}
    fields = {
        "prepare-broad": {
            "stream",
            "ordinary_rows",
            "familiar",
            "difficulty",
            "eligible_ordinals",
            "pair_seed",
            "native_pair_limit",
            "profile",
            "microbatch",
            "initial_weights",
            "calibration",
        },
        "time-broad": {"preparation", "physical", "projection", "design"},
    }
    fresh = action == "prepare-broad" and "frame" in data
    if fresh:
        fields[action] = (fields[action] - {"stream", "initial_weights"}) | {"frame", "previous_weights"}
    if (
        action not in fields
        or set(data) != common | fields[action]
        or data != seal(data)
        or data["role"] != action
    ):
        raise ValueError("broader qualification role requires its closed training-only request")
    if fresh:
        from . import progressive_velocity_fresh

        progressive_velocity_fresh.validate_inputs(data)
        if integer(data["native_pair_limit"]) > 2 * len(data["ordinary_rows"]):
            raise ValueError("fresh native pair selection exceeds its funded slots")
    elif action == "prepare-broad":
        from oeis_learn.generalization.progressive.shared_support import validated_rows

        stream = data["stream"]
        rows = validated_rows(shards(data["ordinary_rows"]), native_verified=True)
        keys = stream["order"]
        if (
            stream != seal(stream)
            or integer(data["native_pair_limit"]) > 2 * len(keys)
            or data["pair_seed"] != stream["seeds"]["training_families"]
            or set(keys) != rows.keys()
            or len(keys) != len(rows)
            or stream["row_identities"] != digest([rows[k]["identity"] for k in keys])
        ):
            raise ValueError(
                "broader native inputs must be the authenticated ordinary positive rows"
            )
    return data


def worker(action):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_roles import ROUND029_ROLE_POLICY
    import os

    out = Path("/work/output")
    data = load(Path("/work/inputs/request.json"), maximum=MAX_INPUT)
    checked_request(data, action)
    if (
        data != seal(data)
        or data["source"] != digest(source_snapshot())
        or data["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or data["role_policy"] != ROUND029_ROLE_POLICY
        or data["role"] != action
    ):
        raise ValueError("installed source, image or broader qualification role differs")
    if action == "prepare-broad":
        from oeis_learn.synthesis.client import SynthesisClient

        with SynthesisClient(data["identity"]) as client:
            result = prepare(data, out, client)
    elif action == "time-broad":
        from .progressive_qualification import require_hip
        from .progressive_generator_probe import usage_summary
        from oeis_learn.tracking.pipeline_profile import Recorder
        from .progressive_research_inputs import checked_file

        view = load(
            checked_file(Path("/work/inputs"), data["projection"]),
            maximum=data["projection"]["bytes"],
        )
        torch = require_hip()
        torch.cuda.reset_peak_memory_stats()
        nodes = sorted(Path("/dev/dri").glob("renderD*"))
        start, cpu = time.monotonic_ns(), time.process_time_ns()
        with Recorder(
            out / "telemetry",
            instrumented=True,
            render_node=nodes[0] if len(nodes) == 1 else None,
            host_counters=False,
        ) as recorder:
            with recorder.phase("broader-update-matrix"):
                observed = run(view, data["design"], device="cuda")
                torch.cuda.synchronize()
        elapsed = (time.monotonic_ns() - start) // 1000
        result = seal(
            dict(
                request=data["identity"],
                matrix=observed,
                elapsed_us=elapsed,
                worker_cpu_us=(time.process_time_ns() - cpu) // 1000,
                device="hip",
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                telemetry=usage_summary(out / "telemetry", elapsed),
            )
        )
    else:
        raise ValueError("unknown broader qualification role")
    atomic_write(out / "producer.json", result)
