#!/usr/bin/env python3
"""Temporary Phase 2 qualification utility; portable host uses only the stdlib.

Uses the existing foundation engine/image/device/mount policies. Does not add
production CLI suites or launch training. Output directories are immutable.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CPU_TESTS = [
    "tests/contract/test_bigint_build.py",
    "tests/contract/test_bigint_contracts.py",
    "tests/contract/test_bigint_language.py",
    "tests/contract/test_native_fitting.py",
    "tests/integration/test_bigint_runtime.py",
    "tests/integration/test_bigint_decoding.py",
    "tests/unit/test_bigint_reference.py",
    "tests/unit/test_bigint_encoder.py",
    "tests/unit/test_synthesis_service_limits.py",
    "tests/unit/test_synthesis_state.py",
    "tests/unit/test_hybrid_ablation.py",
    "tests/unit/test_bigint_bootstrap_data.py",
    "tests/unit/test_bigint_search_policy.py",
    "tests/unit/test_bigint_learning_policy.py",
    "tests/unit/test_synthesis_provenance.py",
    "tests/unit/test_synthesis_archive.py",
    "tests/unit/test_bigint_qualification.py",
    "tests/unit/test_bigint_diagnostic_evidence.py",
    "tests/unit/test_bigint_diagnostic_collector.py",
    "tests/unit/test_bigint_runtime_profile.py",
    "tests/unit/test_bigint_host_lookups.py",
    "tests/unit/test_bigint_input_load_profile.py",
    "tests/unit/test_bigint_first_byte.py",
    "tests/unit/test_bigint_growth_reservation.py",
    "tests/unit/test_bigint_engine_policy.py",
]


def inside(output, gpu):
    import hashlib
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="FAIL",
        gpu_requested=gpu,
        torch=torch.__version__,
        hip=torch.version.hip,
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        source_revision=os.environ.get("OEIS_SOURCE_REVISION"),
        started=time.time(),
    )
    try:
        if gpu and not (torch.version.hip and torch.cuda.is_available()):
            raise RuntimeError("actual HIP required; CPU fallback prohibited")
        tests = ["tests/integration/test_bigint_gpu.py"] if gpu else CPU_TESTS
        # Pytest creates private (0700) scratch directories. Keep those out of
        # evidence mounted by a different UID; publish only the reports below.
        with tempfile.TemporaryDirectory(prefix="oeis-bigint-phase2-") as scratch:
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    *tests,
                    "--basetemp=" + str(Path(scratch) / "pytest"),
                    "--junitxml=" + str(output / "tests.xml"),
                ],
                check=False,
            )
        tree = ET.parse(output / "tests.xml")
        totals = {
            k: sum(int(s.attrib.get(k, 0)) for s in tree.iter("testsuite"))
            for k in ("tests", "failures", "errors", "skipped")
        }
        report["tests"] = totals
        if (
            run.returncode
            or totals["failures"]
            or totals["errors"]
            or totals["skipped"]
            or not totals["tests"]
        ):
            raise RuntimeError("qualification requires tests with zero failures/errors/skips")
        build = Path("/opt/oeis-build-info/bigint-build.json")
        report["native_build"] = json.loads(build.read_text())
        files = sorted(
            p
            for parent in ("src", "crates", "scripts")
            for p in (ROOT / parent).rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".rs", ".toml", ".lock"}
            and "target" not in p.parts
        )
        report["source_files"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        }
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report["finished"] = time.time()
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"native_build", "source_files"}}, indent=2
        )
    )
    return 0 if report["status"] == "PASS" else 1


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
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inside:
        return inside(args.output, args.gpu)
    from foundation_local import Host, relative
    from oeis_learn.container_runtime import private_runtime_environment

    output = ROOT / relative(args.output)
    if output.exists():
        raise ValueError("use a new evidence directory; existing evidence is immutable")
    output.mkdir(parents=True)
    host = Host(args)
    image = host.ensure()
    name = "oeis-bigint-phase2-" + uuid.uuid4().hex
    if args.gpu:
        host.select_gpu()
    with host.gpu_lock() if args.gpu else nullcontext():
        argv = [
            "python",
            "scripts/qualify_bigint_phase2.py",
            "--inside",
            "--output",
            "/work/output",
        ]
        if args.gpu:
            argv.append("--gpu")
        command = host.engine.command(
            image=image,
            name=name,
            mounts=[(output, "/work/output", True)],
            argv=argv,
            gpu=args.gpu,
            render_node=host.policy["render_node"] if args.gpu else None,
            memory_bytes=host.policy["memory_bytes"],
            cpus=host.policy["cpus"],
            tmp_bytes=2 << 30,
            workdir="/opt/oeis-build",
            auto_remove=False,
            environment={
                **private_runtime_environment(output),
                "PYTHONPATH": "/opt/oeis-build/src:/opt/oeis-build",
                "OMP_NUM_THREADS": str(host.policy["cpus"]),
                "OEIS_REQUIRE_BIGINT_BUILD": "1",
                "OEIS_REQUIRE_BIGINT_GPU": "1" if args.gpu else "0",
                "OEIS_TEST_OUTPUT": "/work/output/gpu",
            },
        )
        print(f"Image: {image}\nLogs/evidence: {output}", flush=True)
        code = 1
        try:
            with (
                (output / "stdout.log").open("wb") as stdout,
                (output / "stderr.log").open("wb") as stderr,
            ):
                code = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=1800
                ).returncode
        finally:
            host.engine.remove(name)
            (output / "host.json").write_text(
                json.dumps(
                    dict(image=image, runtime=host.policy["runtime"], returncode=code), indent=2
                )
                + "\n"
            )
        print(f"{'PASS' if code == 0 else 'FAIL'}: {output / 'summary.json'}")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
