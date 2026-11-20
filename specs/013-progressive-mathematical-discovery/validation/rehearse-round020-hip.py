"""Rehearse the four failed fixtures on round020's image without issuing acceptance."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

ORIGINAL = 'f08c2d16943bda3774385589e3b40ae65859ec5a'
REQUEST = 'sha256:f4866393db8558751d598f06d09652f357742a0752a27d9101479a8dc23f813c'
WHEEL = 'sha256:67e42cbf7c31cbf9e8b5433685fe62dc22d5d2de3b4c4eb62effdfe7b2c8feb9'
FIXTURES = {
    'tests/helpers/progressive_host_cycle.py': 'sha256:a27fb40cd1dd4a05939c0e174b2c9a3eb892e3224a19835767580b8dc1e17018',
    'tests/integration/test_progressive_tuning.py': 'sha256:7c38e1444be320459b00077e45debcb4c488ef2320be8abfe5e7e120cd65c343',
}
SESSION = Path('run-state/discovery/013-research-round-020')
OUTPUT = Path('run-state/discovery-tools/round020-hip-rehearsal')


def inside():
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation.generalization_execution import verify_execution
    from oeis_learn.pipeline_validation.identity import file_hash, seal
    from oeis_learn.pipeline_validation.progressive_case_batch import pytest_cases
    from oeis_learn.pipeline_validation.progressive_device_acceptance import NODES
    from oeis_learn.pipeline_validation.progressive_qualification import hardware_facts
    from oeis_learn.pipeline_validation.research_limits import DEVICE_ACCEPTANCE_SECONDS

    inputs, output, build = Path('/work/inputs'), Path('/work/output'), Path('/opt/oeis-build')
    request = load(inputs / 'request.json')
    if (seal(request) != request or request['identity'] != REQUEST or hardware_facts() != request['hardware']
            or verify_execution() != request['execution_policy']
            or os.environ['OEIS_QUALIFICATION_IMAGE'] != request['image']):
        raise ValueError('round020 selected device and numerical policy required')
    if file_hash(inputs / 'package.whl') != WHEEL:
        raise ValueError('original installed wheel differs')
    installed = output / 'installed'
    with zipfile.ZipFile(inputs / 'package.whl') as wheel:
        wheel.extractall(installed)
    tree = output / 'fixture-tree'
    tree.mkdir()
    for path in build.iterdir():
        if path.name != 'tests':
            (tree / path.name).symlink_to(path)
    shutil.copytree(build / 'tests', tree / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    for name, expected in FIXTURES.items():
        if file_hash(inputs / name) != expected:
            raise ValueError('reviewed diagnostic fixture differs: ' + name)
        shutil.copyfile(inputs / name, tree / name)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (installed, tree))),
               OEIS_PROGRESSIVE_ACCEPTANCE_DEVICE='actual_HIP',
               OEIS_PROGRESSIVE_ACCEPTANCE_REQUEST=str(inputs / 'request.json'),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    cases = [NODES[i] for i in (1, 2, 3, 6)]
    outcome = pytest_cases(cases, output, env=env, seconds=DEVICE_ACCEPTANCE_SECONDS, cwd=tree)
    if outcome == 0:
        # Only disposable pytest basetemps owned by this successful rehearsal.
        # Keep per-case XML, outcomes, logs, input hashes and supervisor records.
        try:
            shutil.rmtree(output / 'private')
        except PermissionError as exc:
            print(f"Four cases passed. Temporary pytest files remain in {output / 'private'}: {exc}",
                  file=sys.stderr, flush=True)
    return outcome


def rehearse(root, revision):
    from oeis_learn.experiments.source_snapshot import collect_snapshot, file_digest
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.identity import confined, digest, file_hash
    from oeis_learn.pipeline_validation.progressive_handoff import stage
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import read, write
    from oeis_learn.pipeline_validation.research_limits import CLEANUP_SECONDS, DEVICE_ACCEPTANCE_SECONDS

    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('full reviewed Git revision required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != ORIGINAL:
        raise ValueError('run this rehearsal from the qualified round020 checkout')
    folder = confined(root, SESSION)
    device = folder / 'prepare-research/acceptance/device'
    request_path = confined(folder, device / 'integrated/request.json')
    request = read(request_path)
    wheel = confined(folder, device / 'package.whl')
    if (request['identity'] != REQUEST or digest(collect_snapshot(root)) != request['source']
            or file_hash(wheel) != WHEEL):
        raise ValueError('original selected request, source and wheel required')
    journal_path = confined(folder, 'journal.json')
    if file_hash(journal_path) != 'sha256:f6aff949d0ac7c6f0180f824a001e944dfbe6988a36131e0ca0e95f341540d4d':
        raise ValueError('round020 journal changed since the reviewed diagnostic bundle')
    frozen = read(journal_path)['steps']['prepare-research/freeze']
    for ref in frozen['outputs']:
        checked_file(root, ref)
    free = shutil.disk_usage(root).free
    original_required = read(folder / 'prepare-research/storage-preflight.json')['required_bytes']
    print(f"Verified {len(frozen['outputs'])} freeze outputs. Filesystem has {free:,} bytes free. "
          f"Round020 required {original_required:,} bytes at freeze.", flush=True)
    fixtures = {}
    for name, expected in FIXTURES.items():
        raw = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=root)
        if file_digest(raw) != expected:
            raise ValueError('unexpected fixture revision: ' + name)
        fixtures[name] = raw
    binding = read(folder / 'session.json')['binding']
    host = Workflow(SimpleNamespace(root=root, folder=folder),
                    SimpleNamespace(**binding, phase='prepare-research')).host()
    output = confined(root, OUTPUT, exists=False)
    if output.exists():
        raise ValueError('Rehearsal already exists. Review its preserved case-outcomes.json and process.json at ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    print('Diagnostic logs and case outcomes: ' + str(output), flush=True)
    inputs = [(request_path, '/work/inputs/request.json'), (wheel, '/work/inputs/package.whl'),
              (Path(__file__).resolve(), '/work/inputs/rehearse.py')]
    for name, raw in fixtures.items():
        path = output / 'reviewed' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        inputs.append((path, '/work/inputs/' + name))
    write(output / 'scope.json', dict(scope='DIAGNOSTIC_ONLY', revision=revision, fixtures=FIXTURES,
                                      request=REQUEST, wheel=WHEEL, image=request['image']))
    try:
        stage(host, root, request['image'], output, ['python', '/work/inputs/rehearse.py', '--inside'],
              gpu=True, seconds=DEVICE_ACCEPTANCE_SECONDS + CLEANUP_SECONDS, readonly_inputs=inputs)
    finally:
        archive_path = output / 'diagnostics.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for pattern in ('scope.json', 'case-outcomes.json', 'cases.xml', 'process.json', 'cleanup.json',
                            'stdout.log', 'stderr.log', 'cases/*/process.log', 'cases/*/junit.xml',
                            'cases/*/outcome.json', 'cases/*/health.log'):
                for path in sorted(output.glob(pattern)):
                    archive.write(path, path.relative_to(output))
        print('Share diagnostics: ' + str(archive_path), flush=True)
    print('Four HIP fixtures passed. Case results: ' + str(output / 'case-outcomes.json'), flush=True)
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--inside']:
        raise SystemExit(inside())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    sys.path.insert(0, str(root / 'src'))
    raise SystemExit(rehearse(root, args.revision))
