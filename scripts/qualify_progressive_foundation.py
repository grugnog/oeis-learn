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

# T048 includes an explicit third-seed routing specification for later T053.
# Their independent gold/Torch/Optuna tests remain mandatory in CPU_TESTS.
DEFERRED_SPECIFICATIONS = ()
HISTORICAL_TESTS = (
    "tests/unit/test_pac_support_audit.py",
    "tests/integration/test_pac_structural_publication.py",
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
PYTEST_TIMEOUT_SECONDS = 3000
NATIVE_CASE = "test_native_witness_acceptance_uses_packaged_service"
WHEEL_CASE = "test_wheel_preserves_every_bound_source_resource"


# Keep the three measured expensive modules on separate runners. Distribute the
# rest deterministically; no test or per-case allowance is removed.
PARTITIONS = ('lifecycle', 'tuning', 'construction', 'contracts')
ANCHORS = {
    'tests/integration/test_progressive_acceptance.py': 'lifecycle',
    'tests/integration/test_progressive_tuning.py': 'tuning',
    'tests/unit/test_progressive_research_bindings.py': 'construction',
}


def partition_tests(name):
    if name not in PARTITIONS:
        raise ValueError('unknown acceptance partition')
    groups = {key: [] for key in PARTITIONS}
    for i, node in enumerate(TESTS):
        groups[ANCHORS.get(node, PARTITIONS[i % len(PARTITIONS)])].append(node)
    return tuple(groups[name])


def required_cases(requested):
    owners = {
        NATIVE_CASE: 'tests/unit/test_progressive_coverage.py',
        WHEEL_CASE: 'tests/contract/test_foundation_packaging.py',
    }
    return {case for case, module in owners.items() if module in requested}


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
    if not required_cases(requested) <= names:
        raise ValueError("actual native witness and installed-wheel cases required")
    return dict(tests=len(cases), **counts)


def run(output, *, partition=None):
    requested = TESTS if partition is None else partition_tests(partition)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progressive-foundation.json").exists():
        raise ValueError("fresh foundation acceptance output required")
    report = dict(
        version="progressive-foundation-acceptance/v1",
        status="FAIL",
        qualified=False,
        scope="T018 CPU/native software acceptance",
        requested=list(requested),
        deferred_specifications=list(DEFERRED_SPECIFICATIONS),
        counts=None,
        native_case=NATIVE_CASE,
        wheel_case=WHEEL_CASE,
        error=None,
    )
    if partition is not None:
        report.update(version='progressive-foundation-partition/v1', partition=partition,
                      scope='Partial CPU/native acceptance; all four partitions required')
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
                "-vv",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "-m",
                "not gpu and not docker",
                "--basetemp=" + str(output / ".runtime/pytest"),
                "-o",
                "tmp_path_retention_policy=failed",
                "--junitxml=" + str(xml),
                *requested,
            ],
            timeout=PYTEST_TIMEOUT_SECONDS,
            env=dict(os.environ, OEIS_PROGRESSIVE_ACCEPTANCE_EXPORT=str(output / "progressive-acceptance.zip")),
        ).returncode
        if code:
            raise ValueError("mandatory foundation pytest exit " + str(code))
        report["counts"] = reconcile(xml, requested=requested)
        report["junit"] = reference(output, xml)
        if 'tests/integration/test_progressive_acceptance.py' in requested:
            report["integrated"] = reference(output, "progressive-acceptance.zip")
        report["status"] = "PASS"
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    atomic_write(output / "progressive-foundation.json", seal(report))
    return 0 if report["status"] == "PASS" else 1


def merge(partitions, output):
    """Reconcile every original shard before publishing full acceptance."""
    import copy
    from oeis_learn.generalization.progressive.contracts import load, verify_refs
    partitions, output = Path(partitions), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'progressive-foundation.json').exists():
        raise ValueError('fresh merged acceptance output required')
    combined = ET.Element('testsuites')
    sources = set()
    integrated = None
    for name in PARTITIONS:
        folder = partitions / name
        report = load(folder / 'progressive-foundation.json')
        if (seal(report) != report or report.get('version') != 'progressive-foundation-partition/v1'
            or report.get('partition') != name or report.get('status') != 'PASS'
            or report.get('requested') != list(partition_tests(name))):
            raise ValueError('complete original partition required: ' + name)
        sources.add(report['source'])
        xml = verify_refs(folder, [report['junit']])[0]
        parsed = ET.fromstring(xml)
        if reconcile(folder / report['junit']['path'], requested=partition_tests(name)) != report['counts']:
            raise ValueError('partition counts differ from original JUnit')
        for suite in ([parsed] if parsed.tag == 'testsuite' else parsed.findall('testsuite')):
            combined.append(copy.deepcopy(suite))
        if 'tests/integration/test_progressive_acceptance.py' in partition_tests(name):
            verify_refs(folder, [report['integrated']])
            integrated = reference(output.parent, folder / report['integrated']['path'])
    if sources != {digest(source_snapshot())}:
        raise ValueError('partition source closures differ from current source')
    xml = output / 'progressive-foundation.xml'
    ET.ElementTree(combined).write(xml)
    counts = reconcile(xml)
    record = seal(dict(version='progressive-foundation-acceptance/v1', status='PASS',
        qualified=False, scope='T018 CPU/native software acceptance', source=sources.pop(),
        requested=list(TESTS), counts=counts, junit=reference(output, xml),
        partitions=list(PARTITIONS), integrated=integrated))
    atomic_write(output / 'progressive-foundation.json', record)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--partition', choices=PARTITIONS)
    group.add_argument('--merge', help='parent directory containing all original partitions')
    args = parser.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.output)
    return run(args.output, partition=args.partition)


if __name__ == '__main__':
    raise SystemExit(main())
