"""Resumable portable validation. Only the host wrapper calls the container engine."""

import json
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from .container_runtime import private_runtime_environment
from .fresh.common import digest, file_hash, lock, read, write
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
CPU_NODES = sorted(
    {
        str(p.relative_to(ROOT))
        for area in ("unit", "contract", "integration")
        for pattern in (
            "test_foundation*.py",
            "test_fresh*.py",
            "test_container*.py",
            "test_extended_fresh.py",
            "test_bootstrap_*.py",
            "test_connected_sampler.py",
        )
        for p in (ROOT / "tests" / area).glob(pattern)
        if p.name
        not in {
            "test_foundation_gpu.py",
            "test_foundation_learning_gpu.py",
            "test_fresh_gpu.py",
            "test_bootstrap_gpu.py",
        }
    }
) + [
    "tests/test_dev_images.py",
    "tests/integration/test_foundation_gpu.py::test_unavailable_gpu_never_falls_back",
]
GPU_NODES = [
    "tests/integration/test_portable_gpu.py",
    "tests/integration/test_fresh_gpu.py",
    "tests/integration/test_bootstrap_gpu.py",
    "tests/integration/test_foundation_evaluation.py::test_gpu_checkpoint_loading_and_prefix_only_generation",
    "tests/unit/test_foundation_sampler.py::test_complete_effective_constructors_and_real_forward_backward[cuda]",
]


def _stages(host, suite, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = host.ensure()
    root = Path(host.policy["workspace"])
    code = digest(
        {
            str(p.relative_to(root)): file_hash(p)
            for parent in ("src", "scripts", "tests")
            for p in sorted((root / parent).rglob("*.py"))
        }
    )
    gpu = suite == "gpu"
    if gpu:
        host.select_gpu()
    stages = [
        (
            "environment",
            [
                "python",
                "-c",
                "import json,torch; from oeis_learn.sandbox.pipeline import Runtime; r=Runtime(backend='rust').evaluate('i256.const 7',[0,1]); assert r.outcome is None and r.outputs==r.reference_outputs; "
                + (
                    "assert torch.version.hip and torch.cuda.is_available(); x=torch.ones((16,16),device='cuda'); assert (x@x).sum().item()==4096; "
                    if gpu
                    else ""
                )
                + "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'gpu_available':torch.cuda.is_available(),'devices':[{'name':torch.cuda.get_device_properties(i).name,'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}))",
            ],
        ),
        (
            suite + "-regressions",
            [
                "python",
                "-m",
                "pytest",
                "-q",
                *(GPU_NODES if gpu else CPU_NODES),
                *([] if gpu else ["-m", "not gpu and not docker"]),
                "--basetemp=/work/output/pytest",
                "--junitxml=/work/output/results.xml",
            ],
        ),
    ]
    if gpu:
        stages.append(
            (
                "gpu-learning",
                [
                    "python",
                    "-m",
                    "oeis_learn.evaluation.learning_gate",
                    "--hardware",
                    "--output",
                    "/work/output",
                ],
            )
        )
    results = {}
    for name, argv in stages:
        out = output / name
        out.mkdir(exist_ok=True)
        receipt = out / "receipt.json"
        identity = digest(
            {
                "image": image,
                "code": code,
                "suite": suite,
                "argv": argv,
                "runtime": host.policy["runtime"],
                "observed_environment": results.get("environment", {}).get("files"),
            }
        )
        if receipt.exists() and not (gpu and name == "environment"):
            try:
                saved = read(receipt)
            except (OSError, ValueError):
                saved = {}
            files = saved.get("files", {})
            if (
                saved.get("identity") == identity
                and saved.get("status") == "PASS"
                and files
                and all((out / p).is_file() and file_hash(out / p) == h for p, h in files.items())
            ):
                print(f"SKIP {name}: verified prior success", flush=True)
                results[name] = saved
                continue
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        container = "oeis-validation-" + uuid.uuid4().hex
        command = host.engine.command(
            image=image,
            name=container,
            mounts=[(inputs, "/work/input", False), (out, "/work/output", True)],
            auto_remove=False,
            argv=argv,
            gpu=gpu,
            render_node=host.policy.get("render_node") if gpu else None,
            memory_bytes=host.policy["memory_bytes"],
            cpus=host.policy["cpus"],
            workdir="/opt/oeis-build",
            tmp_bytes=2 << 30,
            labels={
                "oeis.run": digest([host.policy["workspace"], str(output), "validation"])[7:39]
            },
            environment={
                **private_runtime_environment(out),
                "PYTHONPATH": "/opt/oeis-build/src",
                "OMP_NUM_THREADS": str(host.policy["cpus"]),
                "OEIS_TEST_OUTPUT": "/work/output/gpu",
            },
        )
        # A new launch must produce new evidence, never consume stale JUnit.
        for stale in out.glob("*.xml"):
            stale.unlink()
        process = None
        print(f"RUN {name}; log: {out / 'stdout.log'}", flush=True)
        try:
            try:
                with (
                    (out / "stdout.log").open("wb") as stdout,
                    (out / "stderr.log").open("wb") as stderr,
                ):
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                    started = time.monotonic()
                    cap = 7200 if name == "gpu-learning" else 1800 if gpu else 3600
                    try:
                        while True:
                            left = cap - (time.monotonic() - started)
                            if left <= 0:
                                raise subprocess.TimeoutExpired(command, cap)
                            try:
                                returncode = process.wait(timeout=min(30, left))
                                if returncode:
                                    raise subprocess.CalledProcessError(returncode, command)
                                break
                            except subprocess.TimeoutExpired:
                                print(
                                    f"  running {int(time.monotonic() - started)}s; log: {out / 'stdout.log'}",
                                    flush=True,
                                )
                    finally:
                        try:
                            host.engine.remove(container)
                        finally:
                            if process.poll() is None:
                                process.kill()
                                process.wait(timeout=5)
            except BaseException:
                # Popen itself may fail before a process object is available.
                if process is None:
                    host.engine.remove(container)
                raise
            counts = {}
            if name.endswith("-regressions") and not (out / "results.xml").is_file():
                raise RuntimeError("test process produced no JUnit evidence")
            if (out / "results.xml").exists():
                cases = list(ET.parse(out / "results.xml").getroot().iter("testcase"))
                counts = {
                    "tests": len(cases),
                    "skipped": sum(c.find("skipped") is not None for c in cases),
                }
                if not cases or (gpu and counts["skipped"]):
                    raise RuntimeError(
                        "GPU validation requires actual execution of every selected test"
                    )
            files = {
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file()
                and (p.suffix in {".xml", ".json"} or p.name.endswith(".log"))
                and p != receipt
                and "pytest" not in p.relative_to(out).parts
                and ".runtime" not in p.relative_to(out).parts
            }
            result = {
                "status": "PASS",
                "identity": identity,
                "files": files,
                "counts": counts,
                "gpu": gpu,
                "qualified": False,
            }
            write(receipt, result)
            results[name] = result
        except BaseException as exc:
            write(
                receipt,
                {"status": "FAIL", "identity": identity, "error": str(exc), "qualified": False},
            )
            raise RuntimeError(
                f"{name} failed; inspect {out / 'stderr.log'} and {out / 'stdout.log'}"
            ) from exc
    return image, results


def validate(host, suite, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    token = digest([host.policy["workspace"], str(output), "validation"])[7:39]
    with lock(output / "validation.lock"):
        (output / "validation.json").unlink(missing_ok=True)
        # Do not overlap regression kernels with training in this workspace.
        with host.gpu_lock() if suite == "gpu" else nullcontext():
            host.engine.cleanup(token)
            try:
                image, results = _stages(host, suite, output)
            finally:
                host.engine.cleanup(token)
        if suite == "gpu":
            host.pipeline("check")
        write(
            output / "validation.json",
            {"suite": suite, "image": image, "results": results, "qualified": False},
        )
        print(json.dumps({"status": "PASS", "output": str(output), "qualified": False}))
