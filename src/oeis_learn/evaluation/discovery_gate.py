"""G9 executable evidence with unavailable/failed/unsupported distinctions."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, canonical_bytes
from oeis_learn.sandbox.streaming import identity as logical_identity

ROOT = Path(__file__).resolve().parents[3]
SUITES = {
    "logical_python": ["tests/unit/test_foundation_extended_wat.py"],
    "discovery_certificates": [
        "tests/integration/test_foundation_discovery.py",
        "tests/contract/test_foundation_discovery_cli.py",
    ],
    "owning_discovery": [
        "tests/unit/test_vector_search_normalized.py",
        "tests/unit/test_pslq_solver.py",
        "tests/unit/test_numerical_validator.py",
        "tests/unit/test_relation_identity.py",
        "tests/unit/test_vicreg_loss.py",
        "tests/integration/test_pslq_discovery_e2e.py",
        "tests/integration/test_discovery_pipeline.py",
        "tests/contract/test_discovery_report_contract.py",
    ],
    "logical_native": ["tests/integration/test_foundation_extended_wat.py"],
}


def implementation_identity():
    files = set()
    for directory in (
        "src/oeis_learn/discovery",
        "src/oeis_learn/sandbox",
        "src/oeis_learn/decoder",
        "src/oeis_learn/evaluation",
        "src/oeis_learn/cli",
        "crates/oeis_wasm_evaluator",
    ):
        files.update(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in (".py", ".rs", ".toml", ".lock", ".wat")
        )
    files.update(ROOT / p for suite in SUITES.values() for p in suite)
    files.add(ROOT / "scripts/verify_phase9.py")
    return digest({str(p.relative_to(ROOT)): digest(p.read_text()) for p in sorted(files)})


def junit(path, suite):
    try:
        cases = ET.parse(path).findall(".//testcase")
    except ET.ParseError as exc:
        raise ValueError("invalid G9 JUnit evidence") from exc
    if not cases or any(
        any(c.find(tag) is not None for tag in ("failure", "error", "skipped")) for c in cases
    ):
        raise ValueError("mandatory G9 cases empty, failed or skipped")
    if not {Path(p).stem for p in SUITES[suite]} <= {
        c.attrib.get("classname", "").split(".")[-1] for c in cases
    }:
        raise ValueError("missing owning G9 suite")
    return len(cases)


def run_qualification(output, *, native=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    available = importlib.util.find_spec("oeis_wasm_evaluator") is not None
    for name, suites in SUITES.items():
        if name == "logical_native" and (not native or not available):
            checks[name] = {
                "status": "unavailable",
                "reason": "mandatory CPU Docker native build/parity pending",
            }
            continue
        path = output / (name + ".xml")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *suites, "--junitxml=" + str(path.resolve())],
            cwd=ROOT,
            check=False,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        try:
            if result.returncode:
                raise ValueError("pytest failed")
            count = junit(path, name)
            checks[name] = {
                "status": "PASS",
                "path": path.name,
                "sha256": digest(path.read_text()),
                "cases": count,
            }
        except (ValueError, OSError) as exc:
            checks[name] = {"status": "FAIL", "reason": str(exc)}
    runtime = {"python": logical_identity("(pure (return (const 0)))")}
    if native and available:
        runtime["rust"] = logical_identity("(pure (return (const 0)))", "rust")
    report = {
        "gate": "G9",
        "version": "g9/v1",
        "implementation_sha256": implementation_identity(),
        "checks": checks,
        "runtime": runtime,
        "status": "PASS"
        if all(c["status"] == "PASS" for c in checks.values())
        else "FAIL"
        if any(c["status"] == "FAIL" for c in checks.values())
        else "PENDING",
        "unsupported_capabilities": ["general_WASM_theorems", "unrestricted_invariant_synthesis"],
        "gpu_required": False,
    }
    report["report_sha256"] = digest(report)
    (output / "g9.json").write_bytes(canonical_bytes(report))
    return report


def require_g9(report, *, evidence_root):
    if (
        set(report)
        != {
            "gate",
            "version",
            "implementation_sha256",
            "checks",
            "runtime",
            "status",
            "unsupported_capabilities",
            "gpu_required",
            "report_sha256",
        }
        or report["gate"] != "G9"
        or report["version"] != "g9/v1"
        or report["status"] != "PASS"
        or report["report_sha256"] != digest(report, "report_sha256")
        or report["implementation_sha256"] != implementation_identity()
        or set(report["checks"]) != set(SUITES)
        or report["gpu_required"] is not False
    ):
        raise ValueError("current complete executable G9 evidence required")
    expected = {
        "python": logical_identity("(pure (return (const 0)))"),
        "rust": logical_identity("(pure (return (const 0)))", "rust"),
    }
    if report["runtime"] != expected:
        raise ValueError("G9 runtime/source/build identity mismatch")
    root = Path(evidence_root).resolve()
    for name, reference in report["checks"].items():
        if set(reference) != {"status", "path", "sha256", "cases"} or reference["status"] != "PASS":
            raise ValueError("required G9 capability failed/unavailable")
        path = (root / reference["path"]).resolve()
        if (
            root not in path.parents
            or digest(path.read_text()) != reference["sha256"]
            or junit(path, name) != reference["cases"]
        ):
            raise ValueError("G9 saved evidence changed")
    return report["report_sha256"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    report = run_qualification(a.output, native=a.native)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
