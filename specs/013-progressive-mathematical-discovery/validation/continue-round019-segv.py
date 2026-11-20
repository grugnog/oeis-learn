#!/usr/bin/env python3
"""Resume the reviewed round019 diagnostic-thread crash on its qualified checkout."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
from types import SimpleNamespace

SESSION = 'run-state/discovery/013-research-round-019'
PHASE = 'prepare-research'
PREFIX = PHASE + '/'
# Fingerprints from the uploaded round019 storage-continuation bundle. A second
# crash requires separate review even if its supervisor also reports SIGSEGV.
STORAGE = 'sha256:d5bef8632acb656c7a31da12b722f2cb69baeab6511eef5231e245675353d849'
CRASH_ROW = 'sha256:1cfd6cebacb7560ac3fbe34d8e37016a3c3b234a068101f7f18383ddf655567d'
DEBIT = 'sha256:4a6f1908a4a0801900da4dd8cda61628091bc385414814fb6a15eb306a2dbc5c'
LOGS = {
    'command.json': 'aa99dcd955b4677bebc46eb0d0d30f6498a36612dcd04a904da8979ceb943e5f',
    'process.json': 'bcd2798fa36a1ea52ed31a46e4bfac1a44b0da2be25c685d948faf02ebf03a2a',
    'stderr.log': '6737d92e83386bbbd97546f8f7c9c26166062ae47f45a3bd0045755d01d3dd5f',
    'stdout.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}


def storage_helper(s):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    folder = s.folder / PHASE / 'storage-continuation'
    previous = read(folder / 'continuation.json')
    row = s.journal['steps'][PREFIX + 'storage-continuation-enroll']
    if (previous['identity'] != STORAGE or row.get('status') != 'COMPLETE'
            or row.get('binding') != previous['script'] or row.get('value') != previous):
        raise ValueError('reviewed storage continuation required')
    path = checked_file(s.root, previous['script'])
    spec = importlib.util.spec_from_file_location('round019_storage', path)
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    # This checks the enrolled first helper, source snapshot, active owner,
    # completed authorities and every retained output against the session.
    storage.original_helper(s)
    archive = s.journal['steps'][PREFIX + 'storage-continuation-archive']
    if archive.get('status') != 'COMPLETE' or archive.get('binding') != STORAGE:
        raise ValueError('missing reviewed storage archive')
    return storage, previous


def owner_ref(root, output, original):
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    if output.is_symlink() or {p.name for p in output.iterdir()} != {'owner.json'}:
        raise ValueError('expected owner-only crash output; partial campaign requires review')
    owner = read(output / 'owner.json')
    if (owner.get('version') != 'discovery-owner/v1' or owner.get('output') != original
            or not re.fullmatch('[0-9a-f]{32}', owner.get('invocation', ''))):
        raise ValueError('freeze owner differs')
    return reference(root, output / 'owner.json')


def recovery_record(s, previous, script):
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.identity import digest, seal
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    base = s.folder / PHASE
    old = base / 'storage-continuation'
    row = s.journal['steps'][PREFIX + 'freeze']
    debit = read(old / 'continuation-allowance.json')
    if digest(row) != CRASH_ROW or debit['identity'] != DEBIT or debit['binding'] != row['binding']:
        raise ValueError('only the reviewed and charged diagnostic-thread crash may resume')
    logs = []
    for name, expected in LOGS.items():
        ref = reference(s.root, old / 'logs/0000' / name)
        if ref['sha256'] != 'sha256:' + expected:
            raise ValueError('crash evidence changed: ' + name)
        logs.append(ref)
    prior = read(checked_file(s.root, previous['original_allowance']))
    checked_file(s.root, previous['retry_allowance'])
    previous_seconds = previous['previous_seconds'] + debit['spent_seconds']
    remaining = min(86400, prior['seconds'] - previous_seconds)
    if remaining <= 120:
        raise ValueError('parent preparation allowance exhausted')
    args = row['binding']['arguments']
    return seal(dict(version='round019-segv-continuation/v1', source=s.binding['source'],
                     previous=STORAGE, original_binding=row['binding'], script=reference(s.root, script),
                     allowance=reference(s.root, old / 'continuation-allowance.json'),
                     previous_seconds=previous_seconds, remaining_seconds=remaining,
                     owner=owner_ref(s.root, base / 'frozen', args['output']), crash_logs=logs))


def check_capacity(s, args, storage):
    from oeis_learn.container_runtime import Engine
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, load
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    base = s.folder / PHASE
    images = read(base / 'qualification/qualification.json')['images']
    engine = Engine(args.engine)
    for image in images:
        if engine.image(image) != image:
            raise ValueError('qualified image unavailable: ' + image)
    report = storage.capacity(s.root, load(base / 'prepared/prepared.json', maximum=CAMPAIGN_DOCUMENT_BYTES))
    print(json.dumps(report, indent=2), flush=True)
    return report


def execute(s, args, script, proceed):
    from oeis_learn.pipeline_validation.identity import confined, seal
    from oeis_learn.pipeline_validation.preparation_recovery import Allowance, retain
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow, command
    from oeis_learn.pipeline_validation.progressive_research_launch import readiness
    from oeis_learn.pipeline_validation.progressive_workstation_state import read, Blocked
    from oeis_learn.pipeline_validation.supervision import launch
    storage, previous = storage_helper(s)
    base = s.folder / PHASE
    folder = confined(s.root, str((base / 'segv-continuation').relative_to(s.root)), exists=False)
    row = s.journal['steps'][PREFIX + 'freeze']
    if row.get('status') != 'COMPLETE':
        record = recovery_record(s, previous, script)
        report = check_capacity(s, args, storage)
        if report['scope'] != 'full' or report['status'] != 'READY':
            return 2
        if not proceed:
            print('Reviewed crash, images and storage verified. Use --continue to retry freeze.', flush=True)
            return 0
        binding = seal(dict(continuation=record['identity'], arguments=record['original_binding']['arguments']))
        output = base / 'frozen'
        def freeze():
            # Preserve evidence before allowing the producer to create a fresh
            # owner at the same campaign path. Never reuse an attempt directory.
            folder.mkdir()
            retain(folder / 'continuation.json', record)
            retain(folder / 'storage-preflight.json', report)
            archived = folder / 'original-crashed-freeze'
            if owner_ref(s.root, output, binding['arguments']['output']) != record['owner']:
                raise ValueError('freeze output changed before archive')
            output.rename(archived)
            logs = folder / 'logs/0000'
            with Allowance(folder / 'continuation-allowance.json', binding, record['remaining_seconds']).attempt() as (left, ordinal):
                if ordinal != 0:
                    raise ValueError('prior crash recovery attempt requires review')
                argv = command(s.root, args, 'freeze', **binding['arguments'])
                try:
                    launch(argv, logs, seconds=left-120, cleanup_seconds=120,
                           stream_cap=4 << 20, allowed_returncodes=(0, 2, 3))
                finally:
                    storage.collect_trace(output, folder)
                result = read(output / 'result.json')
                process = json.loads((logs / 'process.json').read_bytes())
                if result['mechanical_status'] == 'BLOCKED':
                    raise Blocked(str(result['reason']) + ': ' + ', '.join(result['missing_evidence']))
                if process['returncode'] != 0 or result['operation'] != 'freeze' or result['mechanical_status'] != 'COMPLETE':
                    raise ValueError('canonical freeze did not complete')
            return dict(operation='freeze', output=binding['arguments']['output'], exit_code=0), [
                archived / 'owner.json', folder / 'continuation.json', folder / 'storage-preflight.json',
                folder / 'continuation-allowance.json', logs / 'process.json', output / 'result.json', output / 'campaign.json']
        if folder.exists():
            raise ValueError('existing diagnostic-crash recovery retained for review')
        if s.step('freeze', freeze, binding=binding, recover=True) is None:
            return s.code()
    elif not proceed:
        print('Freeze already complete.', flush=True)
        return 0

    workflow = Workflow(s, args)
    def relative(path):
        return str(path.relative_to(s.root))
    setup = s.journal['steps'][PREFIX + 'execution-preflight']['value']['setup_seconds']
    accepted = workflow.run_command('acceptance', 'acceptance', campaign=relative(base / 'frozen/campaign.json'),
                                   output=relative(base / 'acceptance'), seconds=setup['acceptance'],
                                   dependencies=(PREFIX + 'freeze',))
    if accepted is None:
        return s.code()
    def ready():
        return readiness(workflow, *(s.journal['steps'][PREFIX + name]['value'] for name in
                                     ('registered-data', 'native-census', 'sensitivity', 'materials')))
    s.step('readiness', ready, dependencies=(PREFIX + 'acceptance',))
    workflow.run_after_prepare()
    return s.code()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--continue', dest='proceed', action='store_true')
    options = parser.parse_args()
    root, script = Path(options.root).resolve(), Path(__file__).resolve()
    if not script.is_relative_to(root / 'run-state'):
        parser.error('extract this helper into run-state at the qualified checkout')
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    from oeis_learn.pipeline_validation.progressive_workstation_state import NAMES, Session, TelemetryCollector, read, telemetry
    binding = read(root / SESSION / 'session.json')['binding']
    args = SimpleNamespace(phase=PHASE, engine=binding['engine'], memory_gib=binding['memory_gib'],
                           cpus=binding['cpus'], render_node=binding.get('render_node'))
    s = Session(root, SESSION, binding, PHASE)
    NAMES.update({'continuation.json', 'continuation-allowance.json', 'freeze-archive.json',
                  'original-journal.json', 'storage-preflight.json', 'disk-inventory.json'})
    stop = threading.Event()
    def observe():
        while not stop.wait(30):
            try:
                s.observer.observe(telemetry(root))
            except OSError as exc:
                s.observer.failed(exc)
    def interrupt(*_):
        raise KeyboardInterrupt()
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with s.own():
            s.observer = TelemetryCollector(s.folder)
            worker = threading.Thread(target=observe, daemon=True)
            worker.start()
            try:
                return execute(s, args, script, options.proceed)
            finally:
                stop.set()
                worker.join()
    except KeyboardInterrupt:
        return 130
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
