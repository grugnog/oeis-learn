"""G8 owning-path acceptance, bound to implementation and executable evidence.

CPU correctness is useful evidence. It is explicitly insufficient to claim GPU
cache equivalence, mixed precision stability or measured training throughput.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, canonical_bytes
from oeis_learn.evaluation.repair_gate import ROOT
from oeis_learn.sandbox.pipeline import digest_bytes

SUITES = {
    "cache": ["tests/unit/test_foundation_cache.py"],
    "policy": [
        "tests/unit/test_foundation_policy.py",
        "tests/unit/test_kl_regularization.py",
        "tests/unit/test_prompt_weighting.py",
        "tests/unit/test_partitioned_entropy.py",
        "tests/unit/test_sampler_determinism.py",
        "tests/integration/test_egca_training_step.py",
        "tests/integration/test_co_training_step.py",
    ],
    "encoder_structure": [
        "tests/unit/test_foundation_integer_features.py",
        "tests/unit/test_foundation_structural_splits.py",
        "tests/unit/test_foundation_models.py",
        "tests/unit/test_tri_stream_encoder.py",
    ],
    "archive_curriculum": [
        "tests/unit/test_foundation_archive_curriculum.py",
        "tests/unit/test_foundation_provenance.py",
        "tests/unit/test_elite_buffer_replay.py",
        "tests/unit/test_symple_curriculum.py",
        "tests/unit/test_adag_allocator.py",
        "tests/unit/test_curriculum_scheduler.py",
        "tests/unit/test_curriculum_gates.py",
        "tests/unit/test_curriculum_orchestrator.py",
    ],
    "trace": [
        "tests/unit/test_foundation_trace.py",
        "tests/unit/test_execution_tracer.py",
        "tests/unit/test_egca_credit_assignment.py",
        "tests/contract/test_credit_attribution_contract.py",
        "tests/integration/test_egca_coverage_attribution.py",
    ],
    "search_rounds_resume": [
        "tests/unit/test_foundation_search.py",
        "tests/unit/test_foundation_self_training.py",
        "tests/unit/test_foundation_learning_gate.py",
        "tests/contract/test_foundation_learning_cli.py",
        "tests/integration/test_foundation_qualified_resume.py",
    ],
}


def runtime_identity():
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "devices": [
            {
                "name": torch.cuda.get_device_properties(i).name,
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def implementation_identity():
    from oeis_learn.experiments.source_snapshot import qualification_identity

    return qualification_identity("G8")


def check_junit(path, suite=None):
    cases = ET.parse(path).findall(".//testcase")
    if not cases or any(
        c.find("failure") is not None
        or c.find("error") is not None
        or c.find("skipped") is not None
        for c in cases
    ):
        raise ValueError("G8 owning suite failed, skipped or empty")
    if suite is not None:
        expected = {Path(p).stem for p in SUITES[suite]}
        found = {c.attrib.get("classname", "").split(".")[-1] for c in cases}
        if not expected <= found:
            raise ValueError("incomplete G8 owning-module coverage")
    return len(cases)


def check_hardware_junit(path):
    """Require every declared architecture/length and mixed-precision case."""
    count = check_junit(path)
    expected = {
        f"test_gpu_fp32_cached_full_prefix[{length}-{position}]"
        for length in (1, 2, 513, 1024)
        for position in ("sinusoidal", "rope", "prefix_ancestor")
    } | {
        f"test_gpu_fp16_scaler_finite_optimizer_update[{position}]"
        for position in ("sinusoidal", "rope", "prefix_ancestor")
    }
    cases = ET.parse(path).findall(".//testcase")
    if (
        count != len(expected)
        or {case.attrib.get("name") for case in cases} != expected
        or any(
            case.attrib.get("classname") != "tests.integration.test_foundation_learning_gpu"
            for case in cases
        )
    ):
        raise ValueError("incomplete G8 accelerator coverage")
    return count


def _read_evidence(root, reference, *, junit=False):
    from oeis_learn.experiments.artifacts import ArtifactPath

    fields = {"path", "sha256", "cases"} if junit else {"path", "sha256"}
    if not isinstance(reference, dict) or set(reference) != fields:
        raise ValueError("invalid G8 artifact reference")
    path = ArtifactPath(root, reference["path"]).as_path()
    raw = path.read_bytes()
    if digest_bytes(raw) != reference["sha256"]:
        raise ValueError("G8 evidence changed")
    return raw


def evidence_references(report):
    """Enumerate software plus process/JUnit closure for immutable transport."""
    refs = list(report["checks"].values())
    if report.get("hardware") is not None:
        refs.append(report["hardware"]["process"])
        if report["hardware"]["junit"] is not None:
            refs.append(report["hardware"]["junit"])
    return refs


def require_g8(report, *, evidence_root, diagnostic=False):
    if (
        set(report)
        != {
            "gate",
            "version",
            "implementation_sha256",
            "software_status",
            "hardware_status",
            "checks",
            "runtime",
            "hardware",
            "report_sha256",
        }
        or report["gate"] != "G8"
        or report["version"] != "g8/v2"
        or report["report_sha256"] != digest(report, "report_sha256")
        or report["implementation_sha256"] != implementation_identity()
        or report["software_status"] != "PASS"
        or report["runtime"] != runtime_identity()
        or set(report["checks"]) != set(SUITES)
    ):
        raise ValueError("complete current G8 software evidence required")
    root = Path(evidence_root).resolve()
    for name, record in report["checks"].items():
        if set(record) != {"path", "sha256", "cases"}:
            raise ValueError("invalid G8 artifact reference")
        path = (root / record["path"]).resolve()
        if (
            root not in path.parents
            or digest_bytes(path.read_bytes()) != record["sha256"]
            or check_junit(path, name) != record["cases"]
        ):
            raise ValueError("G8 evidence changed")
    status, hardware = report["hardware_status"], report["hardware"]
    if status == "PENDING":
        if hardware is not None:
            raise ValueError("unexecuted G8 hardware must have no execution record")
    elif status in ("PASS", "FAIL"):
        if not isinstance(hardware, dict) or set(hardware) != {"process", "junit"}:
            raise ValueError("durable G8 hardware process diagnostics required")
        process = _read_evidence(root, hardware["process"])
        from oeis_learn.experiments.artifacts import load_json

        process = load_json(process)
        if (
            set(process) != {"requested", "launched", "returncode", "stdout", "stderr", "error"}
            or process["requested"] is not True
            or type(process["launched"]) is not bool
            or not isinstance(process["stdout"], str)
            or not isinstance(process["stderr"], str)
        ):
            raise ValueError("invalid G8 process diagnostics")
        if status == "FAIL":
            if not isinstance(process["error"], str) or not process["error"]:
                raise ValueError("failed G8 hardware requires a failure reason")
            raise ValueError("G8 hardware qualification failed: " + process["error"])
        if (
            process["launched"] is not True
            or type(process["returncode"]) is not int
            or process["returncode"] != 0
            or process["error"] is not None
        ):
            raise ValueError("G8 accelerator process did not pass")
        record = hardware["junit"]
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "cases"}:
            raise ValueError("G8 accelerator JUnit required")
        _read_evidence(root, record, junit=True)
        if check_hardware_junit(root / record["path"]) != record["cases"]:
            raise ValueError("G8 accelerator evidence mismatch")
    else:
        raise ValueError("invalid G8 hardware status")
    if not diagnostic:
        if status != "PASS":
            raise ValueError(
                "G8 hardware qualification remains pending; CPU diagnostics cannot authorize it"
            )
        if not torch.cuda.is_available():
            raise ValueError("qualified accelerator unavailable in this execution environment")
    return report["report_sha256"]


def run_qualification(output, *, hardware=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks, errors = {}, []
    for name, suite in SUITES.items():
        path = output / (name + ".xml")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *suite, "--junitxml=" + str(path.resolve())],
            cwd=ROOT,
            check=False,
        )
        try:
            if result.returncode:
                raise ValueError(f"{name}: pytest failed")
            checks[name] = {
                "path": path.name,
                "sha256": digest_bytes(path.read_bytes()),
                "cases": check_junit(path, name),
            }
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    hardware_record, hardware_status = None, "PENDING"
    if hardware:
        from oeis_learn.experiments.artifacts import atomic_write, ArtifactPath

        path = output / "gpu.xml"
        # A failed launch must never reuse a passing JUnit from an earlier run.
        if path.is_file():
            path.unlink()
        process = dict(
            requested=True, launched=False, returncode=None, stdout="", stderr="", error=None
        )
        junit = None
        hardware_status = "FAIL"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/integration/test_foundation_learning_gpu.py",
                    "--junitxml=" + str(path.resolve()),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            process.update(
                launched=True,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if path.is_file():
                junit = dict(path=path.name, sha256=digest_bytes(path.read_bytes()), cases=None)
            if result.returncode:
                raise ValueError(f"hardware pytest exited with status {result.returncode}")
            count = check_hardware_junit(path)
            junit["cases"] = count
            hardware_status = "PASS"
        except (ValueError, OSError, ET.ParseError) as exc:
            process["error"] = f"{type(exc).__name__}: {exc}"
        payload = canonical_bytes(process)
        atomic_write(
            ArtifactPath(output, "gpu-process-" + digest_bytes(payload)[7:] + ".json"), payload
        )
        hardware_record = dict(
            process=dict(
                path="gpu-process-" + digest_bytes(payload)[7:] + ".json",
                sha256=digest_bytes(payload),
            ),
            junit=junit,
        )
    report = {
        "gate": "G8",
        "version": "g8/v2",
        "implementation_sha256": implementation_identity(),
        "software_status": "PASS" if not errors else "FAIL",
        "hardware_status": hardware_status,
        "checks": checks,
        "runtime": runtime_identity(),
        "hardware": hardware_record,
    }
    report["report_sha256"] = digest(report)
    from oeis_learn.tracking.foundation_metrics import replace_json

    replace_json(output / "g8.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    report = run_qualification(args.output, hardware=args.hardware)
    return (
        0
        if report["software_status"] == "PASS"
        and (not args.hardware or report["hardware_status"] == "PASS")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
