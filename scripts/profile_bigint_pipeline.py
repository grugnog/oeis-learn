#!/usr/bin/env python3
"""Read-only parent / weights-only child HIP diagnostic; host is stdlib-only."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oeis_learn.fresh.common import confined, digest, file_hash, read, write  # noqa: E402


def workload():
    return dict(
        version="bigint-pipeline-profile/v1",
        qualification=False,
        resume=False,
        arms=["ordinary", "telemetry"],
        updates_per_arm=500,
        blocks=20,
        updates_per_block=25,
        batch_size=8,
        training_examples=128,
        validation_examples=256,
        prompts=32,
        groups=4,
        attempts=dict(correct=8, shuffled=1, absent=1, template=8),
        tokens=1024,
        portfolio_seconds=60,
        generation_group_seconds=240,
        scoring_group_seconds=180,
        cpu_allowance=16,
        sample_hz=1,
        stage_caps=dict(
            capability=120,
            warmup=180,
            updates_validation=300,
            portfolio_scoring=3600,
            traces=600,
            archive=300,
        ),
        external_seconds=5400,
        traces_bytes=128 << 20,
        archive_bytes=256 << 20,
        projection="schedule-weighted; diagnostic aggregate is not a training share",
        deferred=["T093"],
        negative_T096="preserved; no retry or learning authority",
    )


def prepare(run, inputs, output, *, image, source, render_node, scope='diverse32', replay=False):
    """Select only approved visible data and the selected checkpoint's weights."""
    parent = read(run / "run.json")
    if parent.get("version") != "bigint-run/v1" or parent.get("final_access") != "sealed":
        raise ValueError("sealed current bigint parent required")
    if read(run / "status.json").get("status") != "completed":
        raise ValueError("completed immutable parent required; profiling never resumes")
    recipe = parent["recipe"]
    if recipe["model"]["seed"] != 20261016 or recipe["inputs"]["profile"] != "hybrid-chunks/v1":
        raise ValueError("selected seed20261016 bigint checkpoint required")
    selected = read(run / "checkpoint-identity.json")["selected"]
    checkpoint = confined(run / "checkpoints", selected["path"])
    if file_hash(checkpoint) != selected["sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    record = read(checkpoint)
    if (
        record["version"] != "bigint-checkpoint/v1"
        or record["binding"]["contract"] != parent["contract_sha256"]
    ):
        raise ValueError("parent checkpoint contract mismatch")
    ref = record["files"]["weights"]
    weights = confined(run / "checkpoints", ref["path"])
    if file_hash(weights) != ref["sha256"] or ref["sha256"] != record["binding"]["weights"]:
        raise ValueError("selected weights mismatch")
    prepared = read(inputs / "prepared.json")
    if (
        prepared.get("identity") != digest({k: v for k, v in prepared.items() if k != "identity"})
        or prepared["identity"] != parent["inputs_identity"]
        or not prepared.get("native_admitted")
    ):
        raise ValueError("prepared data/parent identity mismatch")
    # Do not enumerate or mount the prepared private/final directory.
    example_file = inputs / "examples.json"
    if file_hash(example_file) != prepared["files"]["examples.json"]["sha256"]:
        raise ValueError("visible examples hash mismatch")
    examples = read(example_file)
    view = "concrete" if recipe["learning"]["target_view"] == "literal" else "parameterized"
    learner = examples["views"][view]
    if scope not in ('diverse32','transfer-screen'):raise ValueError('unknown pipeline cost scope')
    selected_rows=learner['training'] if scope=='diverse32' else learner['parameter_development']+learner['structural_development']
    ids=examples['tiny_ids'] if scope=='diverse32' else [r['id'] for r in selected_rows]
    byid={r['id']:r for r in selected_rows}
    expected=32 if scope=='diverse32' else 256
    if len(ids) != expected or len(set(ids)) != expected or any(k not in byid for k in ids):
        raise ValueError("complete diverse32 training panel required")
    prompts = [
        {k: byid[key][k] for k in ("id", "indices", "values", "horizon_indices", "family")}
        for key in ids
    ]
    if len({byid[k]["family"] for k in ids}) < 4:
        raise ValueError("diverse panel requires at least four structures")
    if replay:
        if scope!='diverse32':raise ValueError('replay uses only the frozen visible training panel')
        if replay=='batching/v1':
            from oeis_learn.pipeline_validation.replay_batching import prepare_inputs
        elif replay is True:
            from oeis_learn.pipeline_validation.replay import prepare_inputs
        else:raise ValueError('registered replay preparation mode required')
        return prepare_inputs(recipe,prompts,weights,output,image=image,source=source,render_node=render_node,
            parent=parent['contract_sha256'],parent_checkpoint=selected,data=prepared['identity'])
    training = [
        {k: r[k] for k in ("id", "indices", "values", "tokens")} for r in learner["training"][:128]
    ]
    validation = [
        {k: r[k] for k in ("id", "indices", "values", "tokens")}
        for split in ("parameter_development", "structural_development")
        for r in learner[split]
    ]
    if len(training) != 128 or len(validation) != 256:
        raise ValueError("complete128/256 synthetic update/validation records required")
    truth_file = inputs / "private/synthetic-truth.json"
    if file_hash(truth_file) != prepared["files"]["private/synthetic-truth.json"]["sha256"]:
        raise ValueError("synthetic training truth hash mismatch")
    alltruth = read(truth_file)
    truth = {k: alltruth[k] for k in ids}
    plan = workload()
    # Eight group-arm pairs: <=1920 generation +1440 scoring =3360s.
    if (
        8 * (plan["generation_group_seconds"] + plan["scoring_group_seconds"])
        > plan["stage_caps"]["portfolio_scoring"]
    ):
        raise ValueError("paired work caps exceed declared portfolio/scoring budget")
    manifest = dict(
        plan=plan,
        source_sha256=source,
        image=image,
        render_node=render_node,
        parent=parent["contract_sha256"],
        parent_checkpoint=selected,
        weights_sha256=ref["sha256"],
        data_sha256=prepared["identity"],
        recipe_sha256=digest(recipe),
        payload_sha256=digest(dict(prompts=prompts, training=training, validation=validation)),
        truth_sha256=digest(truth),
        prompt_ids=ids,
        training_ids=[r["id"] for r in training],
        validation_ids=[r["id"] for r in validation],
        seed=recipe["model"]["seed"],
        analysis_nonce=uuid.uuid4().hex,
        scope=scope,
        authority="weights-only diagnostic; final sealed; no resume/training admission",
    )
    manifest["identity"] = digest(manifest)
    gpu_input = output / "input"
    gpu_input.mkdir()
    shutil.copyfile(weights, gpu_input / "weights.pt")
    write(
        gpu_input / "work.json",
        dict(
            manifest=manifest,
            recipe=recipe,
            prompts=prompts,
            training=training,
            validation=validation,
        ),
    )
    write(output / "private/truth.json", truth)
    write(output / "manifest.json", manifest, immutable=True)
    return manifest


def archive(output, *, cap=256 << 20):
    """Whitelist reporting artifacts; exclude inputs, state, truth and caches."""
    output = Path(output)
    chosen, omitted, size, trace_bytes = [], [], 0, 0
    for p in sorted(
        output.rglob("*"), key=lambda p: (p.name not in {"summary.json", "manifest.json"}, str(p))
    ):
        rel = p.relative_to(output)
        if (
            not p.is_file()
            or p.is_symlink()
            or any(
                part in {"input", "private", ".runtime", "native-cache", "checkpoints"}
                for part in rel.parts
            )
            or p.suffix not in {".json", ".jsonl", ".html", ".log"}
        ):
            continue
        record = dict(path=str(rel), bytes=p.stat().st_size, sha256=file_hash(p))
        is_trace = "traces" in rel.parts
        if size + record["bytes"] > cap - (1 << 20) or (
            is_trace and trace_bytes + record["bytes"] > 128 << 20
        ):
            omitted.append(dict(record, reason="archive/trace byte cap"))
            continue
        size += record["bytes"]
        trace_bytes += record["bytes"] if is_trace else 0
        chosen.append((p, record))
    if not any(r["path"] == "summary.json" for _, r in chosen):
        raise ValueError("mandatory summary missing")
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p, record in chosen:
            z.write(p, record["path"])
        z.writestr(
            "archive-manifest.json",
            json.dumps(dict(files=[r for _, r in chosen], omissions=omitted)),
        )
    return target


def container_stage(host, image, source, output, *, mode, seconds, gpu, argv=None):
    from oeis_learn.container_runtime import private_runtime_environment
    from oeis_learn.tracking.pipeline_profile import character_device

    output.mkdir(parents=True, exist_ok=True)
    name = "oeis-pipeline-" + uuid.uuid4().hex
    mounts = [(source, "/work/input", False), (output, "/work/output", True)]
    expected_devices = (
        {node: character_device(node) for node in ("/dev/kfd", host.policy["render_node"])}
        if gpu
        else {}
    )
    started = time.monotonic()
    command = host.engine.command(
        image=image,
        name=name,
        mounts=mounts,
        auto_remove=False,
        argv=argv or [
            "python",
            "/opt/oeis-build/scripts/profile_bigint_pipeline.py",
            "--inside",
            mode,
            "--output",
            "/work/output",
        ],
        gpu=gpu,
        render_node=host.policy.get("render_node") if gpu else None,
        memory_bytes=host.policy["memory_bytes"],
        cpus=host.policy["cpus"],
        tmp_bytes=2 << 30,
        workdir="/opt/oeis-build",
        environment={
            **private_runtime_environment(output),
            "PYTHONPATH": "/opt/oeis-build/src:/opt/oeis-build",
            "OMP_NUM_THREADS": "2",
            "OEIS_EXPECTED_PIPELINE_DEVICES": json.dumps(expected_devices, sort_keys=True),
        },
    )
    cleanup_state={}
    try:
        if argv is None:
            launch(command, output, seconds=seconds)
        else:
            from oeis_learn.pipeline_validation.supervision import launch as supervised
            supervised(command,output,seconds=seconds,stream_cap=host.policy.get('pipeline_budgets',{}).get('stream_bytes',8<<20),cleanup_state=cleanup_state,grace_seconds=.25)
    except BaseException as exc:
        if argv is not None and not isinstance(exc,KeyboardInterrupt):
            try:
                failed=json.loads((output/'summary.json').read_text())
            except (OSError,ValueError):failed={}
            if failed.get('fatal')=='RESOURCE_BREACH':
                from oeis_learn.pipeline_validation.runner import ResourceBreach
                raise ResourceBreach(failed.get('error','role resource breach')) from exc
        raise
    finally:
        cleanup_started=time.monotonic()
        if argv is not None:
            cleanup_state.setdefault('deadline',cleanup_started+5)
        try:
            result = host.engine.call(["inspect", name], capture_output=True, timeout=10 if argv is None else max(.001,min(2,cleanup_state['deadline']-time.monotonic())))
            inspection = json.loads(result.stdout)
            write(output / "container-inspect.json", inspection)
            if argv is not None and inspection[0].get('State',{}).get('OOMKilled'):
                from oeis_learn.pipeline_validation.runner import ResourceBreach
                raise ResourceBreach('compute container exceeded its memory envelope')
            config = inspection[0]["HostConfig"]
            binds = {
                m["Destination"]: m["RW"]
                for m in inspection[0].get("Mounts", [])
                if m["Type"] == "bind"
            }
            from oeis_learn.tracking.pipeline_profile import audit_device_view

            method = audit_device_view(
                config.get("Devices") or [], read(output / "devices.json"), expected_devices
            )
            if (
                config.get("NetworkMode") != "none"
                or not config.get("ReadonlyRootfs")
                or config.get("Privileged")
                or binds != {"/work/input": False, "/work/output": True}
                or config.get("DeviceRequests")
            ):
                if argv is not None:
                    from oeis_learn.pipeline_validation.runner import ResourceBreach
                    raise ResourceBreach('actual role isolation audit failed')
                raise RuntimeError("actual role isolation audit failed")
            write(
                output / "isolation-audit.json",
                dict(
                    status="PASS",
                    device_evidence=method,
                    expected_devices=expected_devices,
                    version="pipeline-isolation-audit/v1",
                ),
            )
        finally:
            start = time.monotonic()
            if argv is None:
                host.engine.remove(name)
            else:
                from oeis_learn.pipeline_validation.supervision import reclaim
                reclaim(host.engine,name,output,cleanup_state)
            elapsed = time.monotonic() - (start if argv is None else cleanup_started)
            write(
                output / "cleanup.json",
                dict(seconds=elapsed, allowance_seconds=5, within_bound=elapsed <= 5),
            )
            if argv is not None and elapsed>5:
                from oeis_learn.pipeline_validation.runner import ResourceBreach
                raise ResourceBreach('pipeline container reclamation exceeded five-second envelope')
            write(
                output / "host-timing.json",
                dict(
                    seconds_through_cleanup=time.monotonic() - started,
                    allowance_seconds=seconds,
                    reason="includes engine launch/inspect/cleanup; compare inspect State timestamps for container lifetime",
                ),
            )
            if elapsed > 5:
                raise RuntimeError("role cleanup exceeded5seconds")


def launch(command, output, *, seconds):
    """External watchdog survives blocked torch/native calls; bounded log/disk use."""
    print(f"RUN pipeline role; logs: {output}", flush=True)
    started, last_progress = time.monotonic(), time.monotonic()
    with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= seconds:
                    raise TimeoutError("profile role external deadline")
                if any(
                    (output / name).stat().st_size > 8 << 20
                    for name in ("stdout.log", "stderr.log")
                ):
                    raise RuntimeError("profile role log cap8MiB")
                if disk_bytes(output) > 8 << 30:
                    raise RuntimeError("profile role disk watchdog8GiB")
                try:
                    process.wait(timeout=min(2, max(0.01, seconds - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - last_progress >= 30:
                    print(
                        f"  running {int(time.monotonic() - started)}s; {output}/progress.json",
                        flush=True,
                    )
                    last_progress = time.monotonic()
            if process.returncode:
                raise RuntimeError(
                    f"profile role exit{process.returncode}; inspect {output}/summary.json and stderr.log"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def disk_bytes(output):
    size = 0
    for p in output.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                size += p.stat().st_size
        except FileNotFoundError:
            pass  # Atomic progress/cache replacement during the watchdog scan.
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("auto", "podman", "docker"), default="auto")
    p.add_argument("--image")
    p.add_argument("--gpu-base-image")
    p.add_argument("--render-node")
    p.add_argument("--memory-gib", type=float)
    p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--free-disk-gib", type=int, default=5)
    p.add_argument("--host-free-memory-gib", type=float, default=1)
    p.add_argument("--run", help="completed original diverse32 run directory")
    p.add_argument("--inputs", help="original prepared bigint data directory")
    p.add_argument("--output", required=True)
    p.add_argument("--inside", choices=("gpu", "scorer", "preflight"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inside:
        from oeis_learn.fresh.pipeline_profile import inside

        return inside(Path("/work/input"), Path(args.output), args.inside)
    if not args.run or not args.inputs:
        p.error(
            "--run and --inputs are required; no training, preparation, resume or final scope exists"
        )
    if args.cpus != 16:
        p.error("this frozen profile requests16 CPU-equivalents; effective quota is reported")
    from foundation_local import Host, relative
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    run, inputs, output = (ROOT / relative(x) for x in (args.run, args.inputs, args.output))
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("fresh output required; reruns never overwrite evidence")
    if (
        output == run
        or run in output.parents
        or output in run.parents
        or output == inputs
        or output in inputs.parents
        or inputs in output.parents
    ):
        raise ValueError("diagnostic output must be independent of parent/data")
    output.mkdir(parents=True)
    summary = dict(
        version="bigint-pipeline-profile/v1",
        status="FAIL",
        qualified=False,
        T093="DEFERRED",
        T094="pending actual HIP evidence review",
        T096="original negative unchanged",
        stages={},
    )
    start = time.monotonic()
    try:
        host = Host(args)
        image = host.ensure()  # Build is outside the bounded measurement run.
        host.select_gpu()
        manifest = prepare(
            run,
            inputs,
            output,
            image=image,
            source=digest(collect_snapshot(ROOT)),
            render_node=host.policy["render_node"],
        )
        start = time.monotonic()
        write(output / "launch-policy.json", host.policy)
        with host.gpu_lock():
            # Same engine/image/device/mount isolation, before the expensive model
            # workload.420s includes cold engine startup; tiny device work is120s.
            container_stage(
                host,
                image,
                output / "input",
                output / "preflight",
                mode="preflight",
                seconds=420,
                gpu=True,
            )
            summary["stages"]["preflight"] = read(output / "preflight/summary.json")
            if summary["stages"]["preflight"].get("status") != "COMPLETE":
                raise RuntimeError("device/trace preflight did not complete; benchmark unstarted")
            container_stage(
                host,
                image,
                output / "input",
                output / "gpu",
                mode="gpu",
                seconds=min(3100, 4800 - (time.monotonic() - start)),
                gpu=True,
            )
        summary["stages"]["gpu"] = read(output / "gpu/summary.json")
        score_input = output / "private/scorer-input"
        score_input.mkdir()
        shutil.copyfile(output / "private/truth.json", score_input / "truth.json")
        shutil.copyfile(output / "gpu/selections.json", score_input / "selections.json")
        write(
            score_input / "work.json",
            dict(
                manifest=manifest,
                recipe=read(output / "input/work.json")["recipe"],
                selections_sha256=file_hash(score_input / "selections.json"),
            ),
        )
        container_stage(
            host,
            image,
            score_input,
            output / "scorer",
            mode="scorer",
            seconds=min(1600, 5100 - (time.monotonic() - start)),
            gpu=False,
        )
        summary["stages"]["scorer"] = read(output / "scorer/summary.json")
        from oeis_learn.fresh.pipeline_report import report

        summary["analysis"] = report(output)
        summary["status"] = (
            "COMPLETE"
            if all(x.get("status") == "COMPLETE" for x in summary["stages"].values())
            else "PARTIAL"
        )
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for name in ("preflight", "gpu", "scorer"):
            path = output / name / "summary.json"
            if path.is_file() and name not in summary["stages"]:
                summary["stages"][name] = read(path)
        summary["seconds_excluding_build"] = time.monotonic() - start
        write(output / "summary.json", summary)
        target = archive(output)
        print(f"Results: {output}\nArchive: {target}", flush=True)
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
