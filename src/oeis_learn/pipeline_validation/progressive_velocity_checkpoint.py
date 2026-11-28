"""Measure disposable whole-update checkpoint and restore cycles on the qualified ordinary stream."""

from pathlib import Path
import tempfile
import time

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import (
    _check,
    atomic_write,
    digest,
    load,
    schema,
    seal,
)
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_trials import exposure, update_policy

VERSION = "round029-checkpoint-matrix/v1"
METHODS = ("00", "04", "05", "06")


def package(ordinary, method, microbatch, origin):
    recipe = next(r for r in velocity_policy()["recipes"] if r["id"] == method)
    return seal(
        dict(
            version="round029-learner-input/v2",
            registration=origin,
            profile=ordinary["profile"],
            initial_weights=ordinary["initial_weights"],
            policy=update_policy(
                ordinary["profile"], method, microbatch=min(recipe["batch"], microbatch)
            ),
            rows=ordinary["rows"],
            order=ordinary["order"][: len(ordinary["rows"])],
            donors={},
        )
    )


def design(view, *, microbatch, origin):
    if view != seal(view) or not view.get("fresh_start"):
        raise ValueError("a bound fresh training projection is required")
    ordinary = view["streams"]["ordinary"]
    methods = {}
    for method in METHODS:
        learner = package(ordinary, method, microbatch, origin)
        size = learner["policy"]["effective_batch"]
        tokens = [
            exposure(learner, learner["rows"], updates=i, batch=size)["useful_tokens"]
            for i in (1, 2)
        ]
        methods[method] = dict(
            package=learner["identity"], batch=size, first=tokens[0], second=tokens[1]
        )
    return seal(
        dict(
            version=VERSION,
            projection=view["identity"],
            origin=origin,
            microbatch=microbatch,
            methods=methods,
            work=dict(attempts=2 * len(methods), tokens=sum(r["second"] for r in methods.values())),
        )
    )


def measure(view, plan, out, *, device):
    from .progressive_pilot_capacity import owned_allocations
    from .progressive_velocity_worker import train_to

    if plan != design(view, microbatch=plan["microbatch"], origin=plan["origin"]):
        raise ValueError("checkpoint treatment design differs")

    def clock():
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        return time.monotonic_ns() // 1000

    methods = {}
    for method, terms in plan["methods"].items():
        learner = package(view["streams"]["ordinary"], method, plan["microbatch"], plan["origin"])
        with tempfile.TemporaryDirectory(dir=out) as directory:
            folder = Path(directory)
            io = {"save": [], "load": []}

            def observe(kind, duration):
                io[kind].append(duration)

            def run(target):
                start = clock()
                cut = train_to(
                    folder,
                    learner,
                    target=target,
                    cap=terms["second"],
                    device=device,
                    checkpoint_timing=observe,
                )
                return cut, clock() - start

            first, first_us = run(terms["first"])
            saved = owned_allocations([folder / "checkpoint"])
            last, resumed_us = run(terms["second"])
            allocated = owned_allocations([folder / "checkpoint"])
            reopened, restore_us = run(terms["second"])
            initial_save, first_save, second_save = io["save"]
            resume_load, final_load = io["load"]
            keys = (
                "weights",
                "optimizer",
                "rng",
                "next_rows",
                "order_digest",
                "scheduler",
                "checkpoint",
            )
            if (
                first["updates"] != 1
                or first["useful_tokens"] != terms["first"]
                or last["updates"] != reopened["updates"]
                or last["charged_tokens"] != reopened["charged_tokens"]
            ):
                raise ValueError("checkpoint replay changed the original whole-update accounting")
            methods[method] = dict(
                package=learner["identity"],
                batch=terms["batch"],
                updates=last["updates"],
                useful_tokens=last["useful_tokens"],
                charged_tokens=last["charged_tokens"],
                written=digest({k: last[k] for k in keys}),
                restored=digest({k: reopened[k] for k in keys}),
                first_cut_us=first_us,
                resumed_cut_us=resumed_us,
                restore_us=restore_us,
                storage_us=dict(
                    initial_save=initial_save,
                    first_save=first_save,
                    second_save=second_save,
                    resume_load=resume_load,
                    final_load=final_load,
                ),
                checkpoint_bytes=max(saved["bytes"], allocated["bytes"]),
                checkpoint_inodes=max(saved["inodes"], allocated["inodes"]),
            )
        row = checked_row(plan, method, methods[method])
        atomic_write(out / f"recipe-{method}.json", treatment(plan, method, row))
    return seal(dict(design=plan["identity"], methods=methods))


def treatment(plan, method, row):
    return seal(dict(design=plan["identity"], method=method, measurement=row))


def checked_row(plan, method, row):
    settings = plan["methods"][method]
    clocks_and_space = (
        "first_cut_us",
        "resumed_cut_us",
        "restore_us",
        "checkpoint_bytes",
        "checkpoint_inodes",
    )
    if (
        set(row)
        != {
            "package",
            "batch",
            "updates",
            "useful_tokens",
            "charged_tokens",
            "written",
            "restored",
            "storage_us",
            *clocks_and_space,
        }
        or row["package"] != settings["package"]
        or row["batch"] != settings["batch"]
        or row["updates"] != 2
        or row["useful_tokens"] != settings["second"]
        or row["charged_tokens"] != settings["second"]
        or row["written"] != row["restored"]
    ):
        raise ValueError("saved and restored treatment boundaries differ")
    _check(schema("common")["$defs"]["hash"], row["written"], {})
    for field in clocks_and_space:
        integer(row[field], 1)
    io = row["storage_us"]
    for value in io.values():
        integer(value, 1)
    if (
        set(io) != {"initial_save", "first_save", "second_save", "resume_load", "final_load"}
        or io["initial_save"] + io["first_save"] > row["first_cut_us"]
        or io["second_save"] + io["resume_load"] > row["resumed_cut_us"]
        or io["final_load"] > row["restore_us"]
    ):
        raise ValueError("checkpoint storage clocks differ from their measured boundaries")
    return row


def checked_methods(plan, measured):
    if (
        measured != seal(measured)
        or measured["design"] != plan["identity"]
        or set(measured["methods"]) != set(METHODS)
    ):
        raise ValueError("checkpoint matrix differs from its registered design")
    for method, row in measured["methods"].items():
        checked_row(plan, method, row)
    return measured["methods"]


def worker():
    import os
    from .progressive_qualification import require_hip
    from .progressive_generator_probe import usage_summary
    from oeis_learn.tracking.pipeline_profile import Recorder
    from .progressive_roles import ROUND029_ROLE_POLICY
    from .progressive_research_inputs import checked_file
    from oeis_learn.experiments.source_snapshot import source_snapshot

    root, out = Path("/work/inputs"), Path("/work/output")
    req = load(root / "request.json")
    if (
        set(req)
        != {
            "identity",
            "version",
            "source",
            "image",
            "role_policy",
            "physical",
            "projection",
            "design",
        }
        or req != seal(req)
        or req["version"] != VERSION
        or req["source"] != digest(source_snapshot())
        or req["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or req["role_policy"] != ROUND029_ROLE_POLICY
    ):
        raise ValueError("checkpoint matrix source, request or physical worker differs")
    ref = req["projection"]
    view = load(checked_file(root, ref), maximum=ref["bytes"])
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
        with recorder.phase("checkpoint-matrix"):
            matrix = measure(view, req["design"], out, device="cuda")
            torch.cuda.synchronize()
    elapsed = (time.monotonic_ns() - start) // 1000
    checked_methods(req["design"], matrix)
    atomic_write(
        out / "producer.json",
        seal(
            dict(
                request=req["identity"],
                device="hip",
                matrix=matrix,
                elapsed_us=elapsed,
                worker_cpu_us=(time.process_time_ns() - cpu) // 1000,
                telemetry=usage_summary(out / "telemetry", elapsed),
                peak_device_bytes=torch.cuda.max_memory_allocated(),
            )
        ),
    )
