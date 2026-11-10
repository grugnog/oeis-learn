"""Installed diagnostic cases with independent processes and durable outcomes."""
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from oeis_learn.generalization.progressive.contracts import atomic_write


def pytest_cases(nodes, output, *, env, seconds, cwd):
    from .progressive_acceptance import _bounded_process
    output = Path(output)
    began = time.monotonic()
    suite = ET.Element('testsuite', name='progressive-device')
    outcomes = []
    blocked = None
    for index, node in enumerate(nodes):
        case = output / 'cases' / str(index)
        case.mkdir(parents=True, exist_ok=False)
        remaining = seconds - (time.monotonic() - began)
        entries = []
        error = blocked
        if not error and remaining > 1:
            try:
                code = _bounded_process([sys.executable, '-m', 'pytest', '-vv', '-ra',
                    '--tb=long', '-o', 'addopts=', '-o', 'faulthandler_timeout=120',
                    '--basetemp', str(output/'private'/str(index)),
                    '--junitxml', str(case/'junit.xml'), node],
                    cwd=cwd, env=env, log=case/'process.log',
                    seconds=remaining/(len(nodes)-index), maximum=4<<20)
                path = case/'junit.xml'
                if path.stat().st_size > 4<<20:
                    raise ValueError('case XML bound')
                entries = ET.parse(path).findall('.//testcase')
                if code or len(entries) != 1:
                    raise ValueError('case failed or wrong case count')
            except (OSError, ValueError, TimeoutError, ET.ParseError) as exc:
                error = traceback.format_exc()
                # Fresh process and owned-worker cleanup precede a health check.
                # A poisoned/reset device blocks later device cases only.
                remaining = seconds-(time.monotonic()-began)
                try:
                    if remaining <= 1 or _bounded_process([sys.executable, '-c',
                        'from oeis_learn.pipeline_validation.progressive_qualification import require_hip; t=require_hip(); t.cuda.synchronize()'],
                        cwd=cwd, env=env, log=case/'health.log',
                        seconds=min(30,remaining), maximum=1<<20):
                        blocked = 'DEVICE_UNAVAILABLE_OR_BUDGET_EXHAUSTED'
                except (OSError, ValueError, TimeoutError):
                    blocked = 'DEVICE_HEALTH_PROBE_FAILED'
        elif not error:
            error = 'BATCH_DEADLINE_EXHAUSTED'
        if error:
            module, name = node.split('::', 1)
            entry = ET.Element('testcase', classname=module[:-3].replace('/','.'), name=name)
            ET.SubElement(entry,'error', message=error[-2048:]).text = error
            entries = [entry]
        suite.extend(entries)
        outcomes.append(dict(node=node,status='FAILED' if error or any(
            e.find(t) is not None for e in entries for t in ('error','failure','skipped')) else 'PASS', error=error))
        atomic_write(case/'outcome.json', outcomes[-1])
        # An interrupted batch leaves its completed original cases available.
        temp = output/'cases.xml.tmp'
        ET.ElementTree(suite).write(temp, encoding='utf-8', xml_declaration=True)
        os.replace(temp, output/'cases.xml')
    atomic_write(output/'case-outcomes.json', outcomes)
    return int(any(c['status'] != 'PASS' for c in outcomes))
