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
from oeis_learn.evaluation.repair_gate import ROOT, implementation_identity as g7_identity
from oeis_learn.sandbox.pipeline import digest_bytes

SUITES = {
    "cache": ["tests/unit/test_foundation_cache.py"],
    "policy": [
        "tests/unit/test_foundation_policy.py",
        "tests/unit/test_kl_regularization.py",
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
        "tests/unit/test_elite_buffer_replay.py",
        "tests/unit/test_symple_curriculum.py",
        "tests/unit/test_adag_allocator.py",
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
    files = sorted(
        {ROOT / path for suite in SUITES.values() for path in suite}
        | {
            ROOT / "tests/integration/test_foundation_learning_gpu.py",
            ROOT / "configs/foundation/qualified.yaml",
        }
    )
    return digest(
        {
            "g7_implementation": g7_identity(),
            "owning_tests": {str(p.relative_to(ROOT)): digest_bytes(p.read_bytes()) for p in files},
        }
    )


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
        or report["version"] != "g8/v1"
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
    if not diagnostic:
        if report["hardware_status"] != "PASS" or report["hardware"] is None:
            raise ValueError(
                "G8 hardware qualification remains pending; CPU diagnostics cannot authorize it"
            )
        path = (root / report["hardware"]["path"]).resolve()
        if (
            root not in path.parents
            or digest_bytes(path.read_bytes()) != report["hardware"]["sha256"]
            or check_junit(path) != report["hardware"]["cases"]
        ):
            raise ValueError("G8 accelerator evidence mismatch")
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
    hardware_record = None
    if hardware:
        path = output / "gpu.xml"
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
        )
        if not result.returncode:
            try:
                hardware_record = {
                    "path": path.name,
                    "sha256": digest_bytes(path.read_bytes()),
                    "cases": check_junit(path),
                }
            except ValueError:
                pass
    report = {
        "gate": "G8",
        "version": "g8/v1",
        "implementation_sha256": implementation_identity(),
        "software_status": "PASS" if not errors else "FAIL",
        "hardware_status": "PASS" if hardware_record else "PENDING",
        "checks": checks,
        "runtime": runtime_identity(),
        "hardware": hardware_record,
    }
    report["report_sha256"] = digest(report)
    (output / "g8.json").write_bytes(canonical_bytes(report))
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
