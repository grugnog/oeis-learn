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
                raise ValueError("broader donor lacks its fixed-target proof")
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


def design(view, microbatch, calibration):
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
    if not stream["available"]:
        return seal(
            dict(
                request=data["identity"],
                disposition="no_breadth_at_matched_exposure",
                supply=stats,
                design=None,
            )
        )
    unique = list(dict.fromkeys(stream["order"]))
    # Every eligible proof is exact. Adjustable targets without a captured native
    # fit stay individually unresolved; the remaining targets still pay positive work.
    pairs = matching_pairs(shards(list(selected.values())), unique, seed=data["pair_seed"])
    view = projection(stream, selected, pairs, data)
    plan = design(view, data["microbatch"], data["calibration"])
    atomic_write(out / "pairs.json", pairs)
    atomic_write(out / "projection.json", view)
    stats.update(
        pair_dispositions=pairs["dispositions"],
        eligible_unique=pairs["eligible_rows"],
        eligible_slots=len(view["donors"]),
        eligible_families=pairs["eligible_families"],
    )
    return seal(dict(request=data["identity"], disposition="prepared", supply=stats, design=plan))


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

    order = dict(order=view["order"][: len(view["rows"])])
    methods = {}
    for method, terms in plan["methods"].items():
        start = now()
        policy = update_policy(
            view["profile"],
            method,
            microbatch=min(32, plan["microbatch"]),
            matching_weight=plan["calibration"]["weight"] if terms["matching"] else None,
        )
        runner = PolicyUpdate(view["profile"], policy, device=device)
        startup = now() - start
        if model_snapshot(runner.encoder, runner.decoder) != view["initial_weights"]:
            raise ValueError("broader optimizer has different initial weights")
        ledger = Ledger(sum(s["positive_tokens"] for s in terms["samples"].values()))
        samples = {}
        for cursor in sorted(terms["samples"], key=int):
            keys = matched_batch(order, int(cursor), 32)
            start = now()
            seq, tokens = batch([view["rows"][k] for k in keys], device)
            donors = (
                [list(map(int, view["donors"][k])) if k in view["donors"] else None for k in keys]
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
    return seal(dict(design=plan["identity"], methods=methods))


def prices(view, plan, observed):
    if (
        plan != design(view, plan["microbatch"], plan["calibration"])
        or observed != seal(observed)
        or observed["design"] != plan["identity"]
        or observed["methods"].keys() != plan["methods"].keys()
    ):
        raise ValueError("broader measurements differ from the frozen sample")
    result = {}
    for method, terms in plan["methods"].items():
        row = observed["methods"][method]
        expected = update_policy(
            view["profile"],
            method,
            microbatch=min(32, plan["microbatch"]),
            matching_weight=plan["calibration"]["weight"] if terms["matching"] else None,
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
        result[method] = {k: v + startup for k, v in unit_price(plan["cells"], clocks).items()}
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
            "profile",
            "microbatch",
            "initial_weights",
            "calibration",
        },
        "time-broad": {"preparation", "physical", "projection", "design"},
    }
    if (
        action not in fields
        or set(data) != common | fields[action]
        or data != seal(data)
        or data["role"] != action
    ):
        raise ValueError("broader qualification role requires its closed training-only request")
    if action == "prepare-broad":
        from oeis_learn.generalization.progressive.shared_support import validated_rows

        stream = data["stream"]
        rows = validated_rows(shards(data["ordinary_rows"]), native_verified=True)
        keys = stream["order"]
        if (
            stream != seal(stream)
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
