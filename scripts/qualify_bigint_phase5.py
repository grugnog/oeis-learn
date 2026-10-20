#!/usr/bin/env python3
"""Portable Phase 5 qualification and evidence collection; host is stdlib-only.

Default runs CPU/native/retirement and measurements. --gpu additionally requires
actual HIP and measures the production model. New outputs preserve earlier runs.
No Phase 6 screen or final-cohort evaluation is scheduled.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "arithmetic": [
        "tests/contract/test_bigint_build.py",
        "tests/contract/test_bigint_language.py",
        "tests/integration/test_bigint_runtime.py",
        "tests/integration/test_bigint_decoding.py",
        "tests/unit/test_bigint_reference.py",
        "tests/unit/test_bigint_encoder.py",
        "tests/integration/test_native_phase7.py",
        "tests/unit/test_bigint_benchmark.py",
        "tests/contract/test_bigint_performance.py",
    ],
    "fitting": [
        "tests/contract/test_native_fitting.py",
        "tests/unit/test_constant_solver.py",
        "tests/unit/test_dixon_diophantine_solver.py",
        "tests/unit/test_mersenne61_filter.py",
        "tests/unit/test_foundation_grounding.py",
        "tests/integration/test_symbolic_grounding_pipeline.py",
        "tests/integration/test_recurrence_grounding.py",
    ],
    "containment": [
        "tests/integration/test_synthesis_worker.py",
        "tests/integration/test_bigint_workflow.py",
        "tests/unit/test_synthesis_service_limits.py",
        "tests/unit/test_synthesis_state.py",
        "tests/contract/test_bigint_workflow.py",
        "tests/unit/test_container_runtime.py",
        "tests/unit/test_container_validation.py",
        "tests/unit/test_foundation_roles.py",
    ],
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def junit(path):
    cases = ET.parse(path).findall(".//testcase")
    counts = {
        tag: sum(c.find(tag) is not None for c in cases) for tag in ("failure", "error", "skipped")
    }
    if not cases or any(counts.values()):
        raise ValueError("mandatory qualification cases empty, failed or skipped: " + str(counts))
    return dict(tests=len(cases), **counts, sha256=sha(path))


def inside(output, group):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(version="phase5-suite/v1", group=group, status="FAIL", qualified=False)
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="phase5-pytest-") as scratch:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *GROUPS[group],
                    "-m",
                    "not gpu and not docker",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp=" + scratch,
                    # Engine access belongs to this host utility's actual launch/
                    # inspect/cleanup audit, never a socket inside the test role.
                    "--deselect=tests/unit/test_foundation_roles.py::test_actual_container_role_views_and_reclamation",
                    "--junitxml=" + str(output / "results.xml"),
                ],
                check=False,
            )
        report["tests"] = junit(output / "results.xml")
        if result.returncode:
            raise ValueError("pytest returned failure despite JUnit")
        from oeis_learn.fresh.bigint_config import packaged_artifacts
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        from oeis_learn.synthesis.contracts import identity

        report.update(
            artifacts=packaged_artifacts(),
            source_sha256=identity(collect_snapshot(ROOT)),
            status="PASS",
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write(output / "summary.json", report)
    return 0 if report["status"] == "PASS" else 1


def archive(output, *, cap=64 << 20):
    """Reports only: no weights, prepared data, private truths or runtime caches."""
    output = Path(output)
    target = output.with_suffix(".zip")
    selected, omitted, size = [], [], 0
    for path in sorted(
        output.rglob("*"),
        key=lambda p: (p != output / "summary.json", p.name != "summary.json", str(p)),
    ):
        relative = path.relative_to(output)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(
                p in {".runtime", "private", "input", "inputs", "pytest", "checkpoints"}
                for p in relative.parts
            )
            or (
                path.suffix not in {".json", ".xml", ".log", ".yaml"}
                and relative != Path("learnability.zip")
            )
        ):
            continue
        if relative.parts[0] == "learnability-run" and (
            len(relative.parts) != 2
            or path.name
            not in {
                "run.json",
                "status.json",
                "summary.json",
                "checkpoint-identity.json",
                "comparison.json",
            }
        ):
            continue
        record = dict(path=str(relative), bytes=path.stat().st_size, sha256=sha(path))
        if size + record["bytes"] > cap - (1 << 20):
            omitted.append(dict(record, reason="archive byte cap"))
        else:
            selected.append((path, record))
            size += record["bytes"]
    if not any(p.name == "summary.json" for p, _ in selected):
        raise ValueError("mandatory summary absent from result archive")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, record in selected:
            z.write(path, record["path"])
        z.writestr(
            "archive-manifest.json",
            json.dumps(dict(files=[r for _, r in selected], omissions=omitted), indent=2),
        )
    return target


def launch(command, output, *, seconds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), "logs:", output, flush=True)
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        start = time.monotonic()
        try:
            while True:
                remaining = seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds)
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(
                        "  running",
                        int(time.monotonic() - start),
                        "seconds;",
                        output / "stdout.log",
                        flush=True,
                    )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    if code:
        raise RuntimeError(f"exit {code}; inspect {output}/stdout.log and stderr.log")


def container_stage(host, image, output, argv, *, gpu=False, seconds=3600):
    from oeis_learn.container_runtime import private_runtime_environment

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "oeis-phase5-" + uuid.uuid4().hex
    if gpu:
        host.select_gpu()
    command = host.engine.command(
        image=image,
        name=name,
        mounts=[(output, "/work/output", True)],
        argv=argv,
        gpu=gpu,
        render_node=host.policy.get("render_node") if gpu else None,
        memory_bytes=host.policy["memory_bytes"],
        cpus=host.policy["cpus"],
        tmp_bytes=2 << 30,
        workdir="/opt/oeis-build",
        auto_remove=False,
        environment={
            **private_runtime_environment(output),
            "PYTHONPATH": "/opt/oeis-build/src:/opt/oeis-build",
            "OMP_NUM_THREADS": "2",
            "OEIS_REQUIRE_BIGINT_BUILD": "1",
            "OEIS_REQUIRE_NATIVE": "1",
            "OEIS_QUALIFICATION_IMAGE": image,
        },
    )
    try:
        with host.gpu_lock() if gpu else nullcontext():
            launch(command, output, seconds=seconds)
            inspected = host.engine.call(["inspect", name], capture_output=True, timeout=30)
            inspection = json.loads(inspected.stdout)
            write(output / "container-inspect.json", inspection)
            record = inspection[0]
            policy = record["HostConfig"]
            if (
                policy.get("NetworkMode") != "none"
                or not policy.get("ReadonlyRootfs")
                or policy.get("Privileged")
                or any(
                    m.get("Destination") != "/work/output"
                    for m in record.get("Mounts", [])
                    if m.get("Type") == "bind"
                )
            ):
                raise RuntimeError("actual role network/root/mount isolation audit failed")
    finally:
        start = time.monotonic()
        host.engine.remove(name)
        elapsed = time.monotonic() - start
        write(
            output / "cleanup.json",
            dict(seconds=elapsed, allowance_seconds=5, within_bound=elapsed <= 5),
        )
        if elapsed > 5:
            raise RuntimeError("role cleanup exceeded declared5second allowance")


def shape_manifest(inputs):
    """Read only public prepared records; never read a scoring/private payload."""
    inputs = Path(inputs)
    prepared = json.loads((inputs / "prepared.json").read_text())
    examples = json.loads((inputs / "examples.json").read_text())
    rows = [r for rs in examples["views"]["concrete"].values() for r in rs]
    return dict(
        prepared_sha256=sha(inputs / "prepared.json"),
        split_sha256=prepared["split_sha256"],
        counts=prepared["counts"],
        tiny_count=len(examples["tiny_ids"]),
        exclusions=prepared["exclusions"],
        native_admitted=prepared["native_admitted"],
        stages={stage: sum(r["stage"] == stage for r in rows) for stage in ("A", "B", "C", "D")},
        max_input_bits=max(abs(int(v)).bit_length() for r in rows for v in r["values"]),
        max_tokens=max(
            len(r["tokens"])
            for view in examples["views"].values()
            for rs in view.values()
            for r in rs
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("auto", "docker", "podman"), default="auto")
    p.add_argument("--image")
    p.add_argument("--gpu-base-image")
    p.add_argument("--render-node")
    p.add_argument("--memory-gib", type=float)
    p.add_argument("--cpus", type=int)
    p.add_argument("--free-disk-gib", type=int, default=5)
    p.add_argument("--host-free-memory-gib", type=float, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--checkpoint-run", help="actual retained matched run; never substituted with fixtures"
    )
    p.add_argument("--inputs", help="already prepared complete paired data")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="fetch pinned bulk data and prepare complete paired inputs",
    )
    p.add_argument("--config", default="configs/foundation/bigint_smoke.yaml")
    p.add_argument(
        "--learnability",
        action="store_true",
        help="after HIP/data/cost gates, run the registered bounded diverse32 gate",
    )
    p.add_argument("--inside", choices=tuple(GROUPS), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inside:
        return inside(args.output, args.inside)
    if args.learnability and not args.gpu:
        p.error("--learnability requires --gpu; CPU fallback prohibited")
    if args.inputs and args.prepare:
        p.error("choose existing --inputs or --prepare")
    from foundation_local import Host, relative
    from oeis_learn.container_validation import validate

    output = ROOT / relative(args.output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("use a new output suffix; previous evidence is immutable")
    output.mkdir(parents=True)
    report = dict(
        version="phase5-qualification/v1",
        status="FAIL",
        qualified=False,
        gpu_requested=args.gpu,
        stages={},
        engine=args.engine,
        pending={
            "T092": "actual HIP on eventual image required",
            "T093": "actual matched checkpoint not supplied",
            "T095": "complete paired dataset and matching non-scoring cost preflight required",
            "T096": "diverse32 training gate not run",
            "T098": "all prerequisite evidence must pass",
        },
    )
    code = 1
    try:
        host = Host(args)
        image = host.ensure()
        report.update(image=image, runtime=host.policy["runtime"], engine=host.engine.name)
        write(output / "summary.json", report)
        for group in GROUPS:
            stage = output / group
            container_stage(
                host,
                image,
                stage,
                [
                    "python",
                    "scripts/qualify_bigint_phase5.py",
                    "--inside",
                    group,
                    "--output",
                    "/work/output",
                ],
            )
            result = json.loads((stage / "summary.json").read_text())
            if result.get("status") != "PASS":
                raise RuntimeError("mandatory group failed: " + group)
            report["stages"][group] = result
            write(output / "summary.json", report)
        validate(host, "bigint-retirement", output / "retirement")
        report["stages"]["retirement"] = json.loads(
            (output / "retirement/validation.json").read_text()
        )
        public = [
            sys.executable,
            str(ROOT / "scripts/foundation_run.py"),
            "--engine",
            host.engine.name,
            "--image",
            image,
        ]
        inputs = ROOT / relative(args.inputs) if args.inputs else None
        if args.prepare:
            inputs = output / "inputs"
            launch(
                [*public, "data", "fetch", "--config", relative(args.config)],
                output / "fetch-command",
                seconds=7200,
            )
            launch(
                [
                    *public,
                    "prepare",
                    "--config",
                    relative(args.config),
                    "--output",
                    relative(str(inputs)),
                ],
                output / "prepare-command",
                seconds=7500,
            )
        if inputs:
            launch(
                [*public, "check", "--inputs", relative(str(inputs))],
                output / "check-command",
                seconds=600,
            )
            write(output / "performance/shapes.json", shape_manifest(inputs))
        config = ROOT / relative(args.config)
        (output / "performance").mkdir(exist_ok=True)
        (output / "performance/measurement-config.yaml").write_bytes(config.read_bytes())
        argv = [
            "python",
            "scripts/benchmark_bigint.py",
            "--output",
            "/work/output",
            "--config",
            "/work/output/measurement-config.yaml",
        ]
        if args.gpu:
            validate(host, "bigint-gpu", output / "gpu")
            report["stages"]["gpu"] = json.loads((output / "gpu/validation.json").read_text())
            report["pending"].pop("T092")
            argv += ["--neural", "--device", "cuda"]
            if inputs:
                argv += ["--shape-manifest", "/work/output/shapes.json"]
        container_stage(host, image, output / "performance", argv, gpu=args.gpu, seconds=3600)
        report["stages"]["performance"] = json.loads(
            (output / "performance/benchmark.json").read_text()
        )
        performance = report["stages"]["performance"]
        if inputs and performance.get("preflight_eligible"):
            report["pending"].pop("T095")
        if args.learnability:
            if not inputs or not performance.get("preflight_eligible"):
                raise RuntimeError(
                    "learnability blocked: complete prepared data and passing measured HIP preflight required"
                )
            run = output / "learnability-run"
            launch(
                [
                    *public,
                    "learnability",
                    "--config",
                    relative(str(output / "performance/measured-recipe.yaml")),
                    "--inputs",
                    relative(str(inputs)),
                    "--view",
                    "diverse",
                    "--seed",
                    "20261016",
                    "--run",
                    relative(str(run)),
                ],
                output / "learnability-command",
                seconds=7500,
            )
            launch(
                [
                    *public,
                    "pack",
                    "--run",
                    relative(str(run)),
                    "--output",
                    relative(str(output / "learnability.zip")),
                ],
                output / "pack-command",
                seconds=300,
            )
            report["stages"]["learnability"] = dict(
                run=relative(str(run)),
                checkpoint=json.loads((run / "checkpoint-identity.json").read_text()),
                summary=json.loads((run / "summary.json").read_text()),
            )
            report["pending"]["T096"] = (
                "run complete; inspect actual full-denominator31/32 and control reports before promotion"
            )
        if args.checkpoint_run:
            launch(
                [
                    sys.executable,
                    str(ROOT / "scripts/foundation_run.py"),
                    "--engine",
                    host.engine.name,
                    "--image",
                    image,
                    "diagnose",
                    "--run",
                    relative(args.checkpoint_run),
                    "--fitting",
                    "paired",
                    "--output",
                    relative(str(output / "checkpoint-rescue")),
                ],
                output / "checkpoint-command",
                seconds=3600,
            )
            report["pending"].pop("T093")
        report["status"] = "PASS"
        code = 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        write(output / "summary.json", report)
        target = archive(output)
        print("Results:", output / "summary.json", "Archive:", target, flush=True)
    return code


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
