"""Read-only comparative role driver. The donor installs the frozen science wheel offline."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def training(work, request, package, device):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_velocity_worker import train_to

    destination = work / "output/learner"
    if package["identity"] != request["learner"]:
        raise ValueError("mounted training package differs")
    # Only the last complete generation and its entire irrevocable journal cross targets.
    parent_files = request["parent_files"]
    restart = work / "inputs/restart.json"
    if restart.exists():
        record = load(restart)
        if record != seal(record) or record["request"] != request["identity"]:
            raise ValueError("interrupted learner belongs to another funded target")
        parent_files = record["files"]
    for ref in parent_files:
        src = checked_file(work / "inputs/parent", ref)
        dst = destination / ref["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    cut = train_to(
        destination, package, target=request["target"], cap=request["cap"], device=device
    )
    store = destination / "checkpoint"
    manifest = load(checked_file(store, cut["checkpoint"]))
    return dict(cut=cut, weight=manifest["states"]["weights"])


def search(work, request, data, client, device, evaluation):
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.contracts import digest

    look = request["look"]
    if look.get("version") != "round029-visible-look/v1":
        raise ValueError("search requires the custodian's visible projection")
    method, recipe = look["method"], data["recipe"]
    model, weight = request["model"], request["weight"]
    if method == "D":
        from oeis_learn.fresh.bigint_diagnostics import model_snapshot

        encoder, decoder, _ = load_baseline(
            "retained_D", recipe, root=work, ref=weight, device=device
        )
        model = evaluation.model_receipt(
            look,
            profile=digest(recipe),
            weights=model_snapshot(encoder, decoder),
            checkpoint=weight["sha256"],
            origin=request["origin"],
        )
        del encoder, decoder
    probe = evaluation.velocity_probe(
        look, data["prompts"], mode=look["mode"], inference_seed=request["seed"], model=model
    )
    neural = model is not None
    ref = evaluation.generate(
        work,
        "generated",
        data["prompts"],
        recipe,
        checkpoint=weight,
        model_control=evaluation.model_control_for(look) if neural else "random",
        seed=look["seed"] if neural and not weight else None,
        reference_recipe=recipe if neural and weight and method != "D" else None,
        device=device,
        client=client,
        seconds=request["seconds"],
        methods=("correct",) if neural else (method,),
        registered_probe=probe,
        output_root=work / "output",
    )
    return dict(probe=probe, artifact=ref)


def perform(work, request, data, *, device, client=None, evaluation=None):
    from oeis_learn.generalization.progressive.contracts import reference
    if evaluation is None:
        from oeis_learn.pipeline_validation import progressive_evaluation as evaluation

    role = request["role"]
    if role == "train" and set(data) == {"package"}:
        return training(work, request, data["package"], device)
    if role == "search" and set(data) == {"prompts", "recipe"}:
        return search(work, request, data, client, device, evaluation)
    if role == "score" and set(data) == {"truth"} and device == "cpu":
        evaluation.score_velocity(
            work,
            request["candidates"],
            data["truth"],
            client=client,
            output="output/scored",
            probe=request["probe"],
            seconds=request["seconds"],
        )
        return dict(artifact=reference(work / "output", work / "output/scored/scores.json"))
    raise ValueError("invalid comparative role input projection")


def installed(work, wheel, wheel_hash, source, prefix):
    from contextlib import nullcontext
    from importlib.util import module_from_spec, spec_from_file_location
    from oeis_learn.experiments import source_snapshot as provenance
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.identity import file_hash
    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import MAX_INPUT
    from oeis_learn.synthesis.client import SynthesisClient

    if (
        not Path(provenance.__file__).resolve().is_relative_to(prefix)
        or digest(provenance.source_snapshot()) != source
    ):
        raise ValueError("installed science source differs")
    request = load(work / "inputs/request.json")
    data = load(work / "inputs/data.json", maximum=MAX_INPUT)
    adapter = work / "inputs/evaluation.py"
    if (
        request != seal(request)
        or request["version"] != "round029-comparative-wave/v1"
        or request["driver"] != digest([file_hash(Path(__file__)), file_hash(adapter)])
        or request["data"] != digest(data)
        or request["source"] != source
        or request["wheel"] != wheel_hash
        or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
        or os.environ["OEIS_QUALIFICATION_ENGINE"] != "podman"
    ):
        raise ValueError("comparative role differs from the registered runtime")
    # The accepted wheel supplies the model, fitter and scorer dependencies. The
    # controller binds this visible-probe adapter along with the standalone driver.
    spec = spec_from_file_location("oeis_learn.pipeline_validation._wave_evaluation", adapter)
    evaluation = module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    gpu = request["gpu"]
    client_context = (
        nullcontext(None) if request["role"] == "train" else SynthesisClient(request["identity"])
    )
    start = time.monotonic_ns()
    cpu = time.process_time_ns()
    with client_context as client:
        native = client.runtime_identity("bigint-body/v1") if client else None
        if gpu:
            from oeis_learn.pipeline_validation.progressive_qualification import require_hip
            from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
            from oeis_learn.tracking.pipeline_profile import Recorder, device_view

            torch = require_hip()
            torch.cuda.reset_peak_memory_stats()
            nodes = sorted(Path("/dev/dri").glob("renderD*"))
            with (
                Recorder(
                    work / "output/telemetry",
                    instrumented=True,
                    render_node=nodes[0] if len(nodes) == 1 else None,
                    host_counters=False,
                ) as recorder,
                recorder.phase(request["role"]),
            ):
                result = perform(work, request, data, device="cuda", client=client, evaluation=evaluation)
                torch.cuda.synchronize()
            elapsed = (time.monotonic_ns() - start) // 1000
            result.update(
                device="hip",
                device_view=device_view(),
                measured_us=elapsed,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                telemetry=usage_summary(work / "output/telemetry", elapsed),
            )
        else:
            result = perform(work, request, data, device="cpu", client=client, evaluation=evaluation)
    atomic_write(
        work / "output/producer.json",
        seal(
            dict(
                request=request["identity"],
                native=native,
                worker_us=(time.monotonic_ns() - start) // 1000,
                worker_cpu_us=(time.process_time_ns() - cpu) // 1000,
                **result,
            )
        ),
    )


def main(wheel_name, wheel_hash, source):
    import hashlib

    work = Path("/work")
    wheel = Path(wheel_name)
    if (
        wheel.parent != work / "inputs"
        or "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest() != wheel_hash
    ):
        raise ValueError("funded science wheel differs")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--no-cache-dir",
                "--target",
                tmp,
                str(wheel),
            ],
            check=True,
        )
        sys.path.insert(0, tmp)
        installed(work, wheel, wheel_hash, source, Path(tmp))


if __name__ == "__main__":
    main(*sys.argv[1:])
