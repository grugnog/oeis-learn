#!/usr/bin/env python3
"""Bounded spec-013 CPU/native acceptance inside the pinned qualification image.

The existing socket-free container_stage launcher owns engine isolation and cleanup.
This script cannot launch research, grant HIP qualification or waive skipped cases.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from oeis_learn.pipeline_validation.progressive_cli import CPU_TESTS
from oeis_learn.generalization.progressive.contracts import atomic_write, reference
from oeis_learn.pipeline_validation.identity import digest, seal
from oeis_learn.experiments.source_snapshot import source_snapshot

# T019 specifies the future T024/T025 reducer. Expected failures are not runtime
# acceptance; retain their explicit disposition without accepting skips elsewhere.
DEFERRED_SPECIFICATIONS = ("tests/unit/test_progressive_metrics.py",)
HISTORICAL_TESTS = (
    "tests/contract/test_generalization_authority.py",
    "tests/contract/test_generalization_roles.py",
    "tests/contract/test_foundation_packaging.py",
    "tests/unit/test_generalization_contracts.py",
    "tests/unit/test_generalization_budgets.py",
    "tests/unit/test_generalization_operational_budgets.py",
    "tests/unit/test_generalization_execution.py",
    "tests/unit/test_generalization_qualification.py",
    "tests/unit/test_generalization_exclusions.py",
    "tests/unit/test_generalization_metrics.py",
    "tests/unit/test_generalization_legacy.py",
    "tests/unit/test_generalization_host.py",
    "tests/integration/test_generalization_registration.py",
    "tests/unit/test_foundation_phase10_integration.py::test_review_dispositions_bind_current_package_implementation",
    "tests/unit/test_pipeline_contracts.py::Contracts::test_current_dispositions_use_the_package_implementation_serializer",
)
TESTS = tuple(p for p in CPU_TESTS if p not in DEFERRED_SPECIFICATIONS) + HISTORICAL_TESTS
NATIVE_CASE = "test_native_witness_acceptance_uses_packaged_service"
WHEEL_CASE = "test_wheel_preserves_every_bound_source_resource"


def reconcile(path, *, requested=TESTS):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 << 20:
        raise ValueError("bounded mandatory JUnit report required")
    cases = ET.parse(path).findall(".//testcase")
    counts = {
        tag: sum(c.find(tag) is not None for c in cases) for tag in ("failure", "error", "skipped")
    }
    if not cases or any(counts.values()):
        raise ValueError("mandatory foundation cases empty, failed or skipped: " + str(counts))
    modules = {c.attrib.get("classname", "") for c in cases}
    for node in requested:
        module = node.split("::")[0].removesuffix(".py").replace("/", ".")
        if not any(m == module or m.startswith(module + ".") for m in modules):
            raise ValueError("missing requested test module: " + module)
    names = {c.attrib.get("name") for c in cases}
    if not {NATIVE_CASE, WHEEL_CASE} <= names:
        raise ValueError("actual native witness and installed-wheel cases required")
    return dict(tests=len(cases), **counts)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progressive-foundation.json").exists():
        raise ValueError("fresh foundation acceptance output required")
    report = dict(
        version="progressive-foundation-acceptance/v1",
        status="FAIL",
        qualified=False,
        scope="T018 CPU/native software acceptance",
        requested=list(TESTS),
        deferred_specifications=list(DEFERRED_SPECIFICATIONS),
        counts=None,
        native_case=NATIVE_CASE,
        wheel_case=WHEEL_CASE,
        error=None,
    )
    try:
        if any(
            os.environ.get(key) != "1"
            for key in ("OEIS_REQUIRE_NATIVE", "OEIS_REQUIRE_BIGINT_BUILD")
        ):
            raise ValueError("mandatory native image mode required")
        report["source"] = digest(source_snapshot())
        xml = output / "progressive-foundation.xml"
        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-m",
                "not gpu and not docker",
                "--basetemp=" + str(output / ".runtime/pytest"),
                "-o",
                "tmp_path_retention_policy=failed",
                "--junitxml=" + str(xml),
                *TESTS,
            ],
            timeout=780,
        ).returncode
        if code:
            raise ValueError("mandatory foundation pytest exit " + str(code))
        report["counts"] = reconcile(xml)
        report["junit"] = reference(output, xml)
        report["status"] = "PASS"
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    atomic_write(output / "progressive-foundation.json", seal(report))
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
