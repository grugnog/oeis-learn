#!/usr/bin/env python3
"""Portable Phase 5 qualification and evidence collection; host is stdlib-only.

Default runs CPU/native/retirement and measurements. --gpu additionally requires
actual HIP and measures the production model. New outputs preserve earlier runs.
--fitting-only runs fitting/admission regressions without data or GPU work.
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
        "tests/unit/test_bigint_runtime_profile.py",
        "tests/unit/test_bigint_host_lookups.py",
        "tests/unit/test_bigint_input_load_profile.py",
        "tests/unit/test_bigint_first_byte.py",
        "tests/unit/test_bigint_growth_reservation.py",
        "tests/unit/test_bigint_engine_policy.py",
        "tests/unit/test_bigint_generation.py",
        "tests/unit/test_pipeline_profile.py",
        "tests/unit/test_bigint_diagnostic_evidence.py",
        "tests/unit/test_bigint_diagnostic_collector.py",
        "tests/unit/test_bigint_cost_scopes.py",
        "tests/unit/test_bigint_scoped_runner.py",
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
        "tests/unit/test_bigint_soak.py",
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
    from oeis_learn.fresh.cost_policy import prepared_shape

    return prepared_shape(inputs)


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
        "--engine-adoption-only",
        action="store_true",
        help="CPU arithmetic/fitting/containment plus production/zero comparison; no data or training",
    )
    p.add_argument(
        "--profile-input-load",
        action="store_true",
        help="nested transfer timings; requires CPU --runtime-profile-only",
    )
    p.add_argument(
        "--compare-host-lookups",
        action="store_true",
        help="add the pre-cache ordinary CPU baseline; requires --runtime-profile-only",
    )
    p.add_argument(
        "--compare-memory-growth",
        action="store_true",
        help="qualification-only zero/64KiB comparison; requires runtime-profile-only and profile-input-load",
    )
    p.add_argument(
        "--runtime-profile-only",
        action="store_true",
        help="paired CPU execution-stage profiling;600seconds work, archive and no dataset/neural/fitting work",
    )
    p.add_argument(
        "--fitting-only",
        action="store_true",
        help="target CPU native fitting/admission; archive results without measurements or training",
    )
    p.add_argument(
        "--performance-only",
        action="store_true",
        help="target HIP parity and measured cost with existing inputs; never train",
    )
    p.add_argument(
        "--checkpoint-run", help="actual retained matched run; never substituted with fixtures"
    )
    p.add_argument("--inputs", help="already prepared complete paired data")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="fetch pinned bulk data and prepare complete paired inputs",
    )
    p.add_argument("--config")
    p.add_argument("--cost-scope", choices=("diverse32", "transfer-screen"), default="diverse32")
    p.add_argument(
        "--learnability",
        action="store_true",
        help="retired: use run_bigint_diverse32.py with a reviewed scoped preflight",
    )
    p.add_argument("--inside", choices=tuple(GROUPS), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.engine_adoption_only:
        if (
            args.gpu
            or args.inside
            or args.runtime_profile_only
            or args.fitting_only
            or args.performance_only
            or args.inputs
            or args.prepare
            or args.checkpoint_run
            or args.learnability
            or args.compare_host_lookups
            or args.config
        ):
            p.error(
                "--engine-adoption-only excludes GPU, other scopes, inputs/preparation, checkpoints and training"
            )
        args.profile_input_load = True
        args.compare_memory_growth = True
    if args.compare_memory_growth and (
        not (args.runtime_profile_only or args.engine_adoption_only)
        or not args.profile_input_load
        or args.compare_host_lookups
        or args.inside
    ):
        p.error(
            "--compare-memory-growth requires isolated CPU --runtime-profile-only --profile-input-load"
        )
    if args.profile_input_load and (
        not (args.runtime_profile_only or args.engine_adoption_only) or args.inside
    ):
        p.error("--profile-input-load requires the CPU --runtime-profile-only scope")
    if args.compare_host_lookups and (not args.runtime_profile_only or args.inside):
        p.error("--compare-host-lookups requires the CPU --runtime-profile-only scope")
    if args.inside:
        return inside(args.output, args.inside)
    args.config = args.config or (
        "configs/foundation/bigint_smoke_phase5.yaml"
        if args.cost_scope == "diverse32"
        else "configs/foundation/bigint_literal_screen.yaml"
    )
    if args.learnability:
        p.error(
            "review the scoped preflight first; use scripts/run_bigint_diverse32.py --reviewed-preflight PATH"
        )
    if args.runtime_profile_only and (
        args.gpu
        or args.performance_only
        or args.fitting_only
        or args.inputs
        or args.prepare
        or args.checkpoint_run
    ):
        p.error(
            "--runtime-profile-only excludes GPU, other scopes, inputs, preparation and checkpoints"
        )
    if args.inputs and args.prepare:
        p.error("choose existing --inputs or --prepare")
    if args.fitting_only and (
        args.gpu or args.performance_only or args.inputs or args.prepare or args.checkpoint_run
    ):
        p.error("--fitting-only excludes GPU, measurements, inputs, preparation and checkpoints")
    if args.performance_only and (
        not args.gpu or not args.inputs or args.learnability or args.prepare or args.checkpoint_run
    ):
        p.error(
            "--performance-only requires --gpu --inputs and excludes preparation/training/checkpoints"
        )
    from foundation_local import Host, relative
    from oeis_learn.container_validation import validate
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.synthesis.contracts import identity

    output = ROOT / relative(args.output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("use a new output suffix; previous evidence is immutable")
    output.mkdir(parents=True)
    report = dict(
        version="phase5-qualification/v1",
        status="FAIL",
        qualified=False,
        gpu_requested=args.gpu,
        compare_host_lookups=args.compare_host_lookups,
        profile_input_load=args.profile_input_load,
        **({"compare_memory_growth": True} if args.compare_memory_growth else {}),
        scope=(
            "engine-adoption-only"
            if args.engine_adoption_only
            else "runtime-profile-only"
            if args.runtime_profile_only
            else "fitting-only"
            if args.fitting_only
            else "performance-only"
            if args.performance_only
            else "full"
        ),
        stages={},
        engine=args.engine,
        pending={
            "T091": "both Docker and Podman evidence required; this invocation tests one engine",
            "T092": "actual HIP on eventual image required",
            "T093": "DEFERRED by user; outside current Phase5 gate",
            "T094": "actual neural measurements and any >5x slowdown review required",
            "T095": "complete paired dataset and matching non-scoring cost preflight required",
            "T096": "diverse32 training gate not run",
            "T097": "review public command/export evidence including interrupted cases",
            "T098": "all prerequisite evidence must pass",
        },
    )
    report["expected_source_sha256"] = identity(collect_snapshot(ROOT))
    if args.engine_adoption_only:
        from oeis_learn.synthesis.engine_policy import production_policy

        report["expected_engine_policy"] = dict(
            manifest=production_policy(), sha256=identity(production_policy())
        )
    if args.performance_only:
        report["pending"].update(
            {
                "T089": "arithmetic not rerun in targeted measurement scope",
                "T090": "fitting qualification not rerun in targeted scope",
            }
        )
    if args.fitting_only:
        report["pending"]["T089"] = "arithmetic not rerun in targeted fitting scope"
    code = 1
    try:
        host = Host(args)
        image = host.ensure()
        report.update(image=image, runtime=host.policy["runtime"], engine=host.engine.name)
        write(output / "summary.json", report)
        groups = (
            ()
            if args.runtime_profile_only or args.performance_only
            else ("fitting",)
            if args.fitting_only
            else GROUPS
        )
        for group in groups:
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
                seconds=3000 if args.engine_adoption_only else 3600,
            )
            result = json.loads((stage / "summary.json").read_text())
            report["stages"][group] = result
            write(output / "summary.json", report)
            if (
                args.engine_adoption_only
                and result.get("artifacts", {}).get("engine_policy")
                != report["expected_engine_policy"]
            ):
                raise RuntimeError("group engine policy mismatch: " + group)
            if result.get("status") != "PASS":
                raise RuntimeError("mandatory group failed: " + group)
            if result.get("source_sha256") != report["expected_source_sha256"]:
                raise RuntimeError(
                    "image source differs from this checkout; rebuild the current source"
                )
        if args.runtime_profile_only or args.engine_adoption_only:
            if args.runtime_profile_only:
                report["pending"].update(
                    T089="arithmetic not rerun in runtime profiling scope",
                    T090="fitting not rerun in runtime profiling scope",
                )
            stage = output / "runtime-profile"
            container_stage(
                host,
                image,
                stage,
                [
                    "python",
                    "scripts/benchmark_bigint.py",
                    "--runtime-profile",
                    *(["--compare-host-lookups"] if args.compare_host_lookups else []),
                    *(["--profile-input-load"] if args.profile_input_load else []),
                    *(["--compare-memory-growth"] if args.compare_memory_growth else []),
                    "--output",
                    "/work/output",
                ],
                seconds=600,
            )
            result = json.loads((stage / "benchmark.json").read_text())
            report["stages"]["runtime-profile"] = result
            if args.compare_memory_growth and result.get("compare_memory_growth") is not True:
                raise RuntimeError("requested memory growth comparison evidence missing")
            if args.profile_input_load and result.get("profile_input_load") is not True:
                raise RuntimeError("requested nested input-load evidence missing")
            if args.compare_host_lookups and result.get("compare_host_lookups") is not True:
                raise RuntimeError("requested host lookup comparison evidence missing")
            if result.get("status") != "PASS":
                raise RuntimeError("runtime profiling failed; inspect stage report and logs")
            if result.get("source_sha256") != report["expected_source_sha256"]:
                raise RuntimeError(
                    "profiling image source differs from this checkout; rebuild current source"
                )
            report["status"] = "PASS"
            return 0
        if args.fitting_only:
            report["status"] = "PASS"
            return 0  # finally publishes the bounded archive, including on failure.
        if not args.performance_only:
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
            "--cost-scope",
            args.cost_scope,
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
        if performance.get("source_sha256") != report["expected_source_sha256"]:
            raise RuntimeError("measurement image source differs from this checkout")
        if args.gpu and not performance.get("slowdown_investigations"):
            report["pending"].pop("T094")
        elif performance.get("slowdown_investigations"):
            report["pending"]["T094"] = (
                "review measured >5x small-value execution slowdowns before long runs"
            )
        if inputs and performance.get("preflight_eligible"):
            report["pending"]["T095"] = (
                "matching scoped receipt passed; review both scope receipts and preregistration before closing T095"
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
        negative = bool(args.gpu and inputs and not performance.get("preflight_eligible"))
        report["status"] = "BLOCKED" if negative else "PASS"
        code = 2 if negative else 0
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
