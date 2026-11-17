#!/usr/bin/env python3
"""Run-specific, stdlib continuation; copy outside the qualified source closure."""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

SESSION = 'run-state/discovery/013-research-round-019'
COMMIT = 'b616b116dad59a05ad1911687583d8ab1626af13'
SOURCE = 'sha256:71d69aec9f4a6902e0cb47e9326b666f78dc71d9d63ff9f66c1f355a891883b8'
RESULT = 'sha256:b89a3d2ad711f3e4a49201f6fb57e41b450defe44cba60f05e31ae4fa6a04fd5'
PREFIX = 'prepare-research/'
MAX_SECONDS = 24 * 3600


def snapshot_argv(argv):
    """Keep canonical arguments and imports; use CPython's safe stack dumper."""
    if argv[1] != '-S':
        raise ValueError('canonical stdlib Python invocation required')
    bootstrap = ('import faulthandler,runpy,sys;from pathlib import Path;'
                 'faulthandler.enable();faulthandler.dump_traceback_later(300,repeat=True);'
                 'sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(sys.argv[0]).parent));'
                 'runpy.run_path(sys.argv[0],run_name="__main__")')
    return [argv[0], '-S', '-c', bootstrap, *argv[2:]]


def allocation(prior, process):
    values = [prior.get('seconds'), prior.get('spent_seconds'), process.get('elapsed_seconds')]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('finite original elapsed accounting required')
    if prior.get('active') is not None or not 0 < values[1] < values[0]:
        raise ValueError('closed original preparation allowance required')
    left = values[0] - values[1] - values[2]
    if left <= 120:
        raise ValueError('original preparation allowance exhausted')
    return min(MAX_SECONDS, left)


def timeout_evidence(row, result, process, recorded, expected):
    if (row.get('status') != 'FAILED' or row.get('history') or result.get('identity') != RESULT
            or result.get('mechanical_status') != 'INTERRUPTED'
            or result.get('producer_error', {}).get('type') != 'KeyboardInterrupt'
            or process.get('returncode') != 130 or process.get('outcome') != 'FAIL'
            or process.get('cleanup_failure') or process.get('omitted_bytes') != {'stdout': 0, 'stderr': 0}
            or process.get('failure') != 'TimeoutError: external deadline: ' + str(recorded.get('argv'))
            or recorded.get('argv', [])[1:] != expected[1:]
            or recorded.get('seconds') != 7080 or recorded.get('cleanup_seconds') != 120
            or not 7080 <= process.get('elapsed_seconds', 0) < 7200):
        raise ValueError('only the confirmed round019 clean freeze timeout may continue')


def authenticate(s):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=s.root, text=True).strip()
    if head != COMMIT or s.binding['source'] != SOURCE or digest(collect_snapshot(s.root)) != SOURCE:
        raise ValueError('keep the exact b616b116 qualified checkout; do not pull or edit source')
    owner = read(s.folder/'preparation-owner.json')
    pointer = s.root/'run-state/discovery-preparation-revisions'/('runtime-'+owner['receipt']['census'][7:]+'.json')
    if read(pointer) != owner or owner['receipt']['recipient'] != SESSION:
        raise ValueError('round019 is no longer the active preparation owner')
    for key, row in s.journal['steps'].items():
        if row['status'] == 'COMPLETE':
            for ref in row.get('outputs', []):
                checked_file(s.root, ref)
        if row['status'] in {'RUNNING', 'INTERRUPTED'} and not key.startswith('run-research/'):
            raise ValueError('reconcile interrupted operation first: '+key)
        if key.startswith(PREFIX) and row['status'] != 'COMPLETE' and key not in {
                PREFIX+'freeze', PREFIX+'acceptance', PREFIX+'readiness'}:
            raise ValueError('original preparation did not complete: '+key)
    q = read(s.folder/'prepare-research/qualification/qualification.json')
    images = [json.loads((s.folder/'prepare-research'/('image-'+name)/'image.json').read_bytes())['image']
              for name in ('qualification', 'gpu')]
    if q['images'] != images:
        raise ValueError('original qualified image bindings differ')
    for name in ('base-qualification', 'qualify', 'prepare', 'execution-preflight'):
        if s.journal['steps'].get(PREFIX+name, {}).get('status') != 'COMPLETE':
            raise ValueError('missing completed authority: '+name)


def copy_trace(output, destination):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    if not (output/'result.json').exists():
        return
    ref = read(output/'result.json').get('producer_error', {}).get('local_traceback')
    if ref:
        if ref['bytes'] > 64 << 10:
            raise ValueError('unexpectedly large host traceback; original remains private')
        path = checked_file(output, ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())


def continue_campaign(s, args, script):
    from oeis_learn.generalization.progressive.contracts import reference, seal
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow, command
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.preparation_recovery import Allowance, retain
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.supervision import launch
    from oeis_learn.pipeline_validation.progressive_research_launch import readiness
    authenticate(s)
    base = s.folder/'prepare-research'
    folder = base/'freeze-continuation'
    relative = lambda p: str(p.relative_to(s.root))
    arguments = dict(blueprint=relative(base/'calibration-reuse/blueprint.json'),
                     qualification=relative(base/'qualification'), prepared=relative(base/'prepared'),
                     output=relative(base/'frozen'))
    argv = command(s.root, args, 'freeze', **arguments)
    binding = reference(s.root, script)
    def enroll():
        process = json.loads((base/'logs/freeze/process.json').read_bytes())
        recorded = json.loads((base/'logs/freeze/command.json').read_bytes())
        result = read(base/'frozen/result.json')
        timeout_evidence(s.journal['steps'][PREFIX+'freeze'], result, process, recorded, argv)
        if any((base/'frozen'/n).exists() for n in ('campaign.json', 'runtime.json', 'tasks')):
            raise ValueError('partially published freeze requires separate review')
        prior = read(base/'prepare-allowance.json')
        expected = seal(dict(scope=s.binding, operation='prepare', arguments={
            k: v for k, v in dict(arguments, output=relative(base/'prepared')).items() if k != 'prepared'}))
        if prior['binding'] != expected:
            raise ValueError('original preparation allowance scope differs')
        originals = [reference(s.root, p) for p in sorted((base/'frozen').rglob('*')) if p.is_file()]
        value = seal(dict(version='round019-freeze-continuation/v1', script=binding,
            source=SOURCE, remaining_seconds=allocation(prior, process),
            prior_allowance=reference(s.root, base/'prepare-allowance.json'),
            previous_freeze_seconds=process['elapsed_seconds'], originals=originals,
            no_requalification=True, no_calibration_execution=True, scientific_budgets_changed=False))
        copy_trace(base/'frozen', folder/'original-freeze-traceback.log')
        retain(folder/'continuation-original-journal.json', seal(s.journal))
        retain(folder/'continuation.json', value)
        return value, [folder/'continuation.json', folder/'continuation-original-journal.json']
    enrolled = s.step('freeze-continuation-enroll', enroll, binding=binding)
    if enrolled is None:
        return
    checked_file(s.root, enrolled['script'])
    checked_file(s.root, enrolled['prior_allowance'])
    def archive():
        target = folder/'original-failed-freeze'
        if not target.exists():
            for ref in enrolled['originals']:
                checked_file(s.root, ref)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(base/'frozen', target)
        elif (base/'frozen').exists():
            raise ValueError('ambiguous failed freeze archive')
        refs = [dict(r, path=relative(target/Path(r['path']).relative_to(base.relative_to(s.root)/'frozen')))
                for r in enrolled['originals']]
        for ref in refs:
            checked_file(s.root, ref)
        value = seal(dict(version='round019-freeze-archive/v1', originals=enrolled['originals'], archived=refs))
        retain(folder/'freeze-archive.json', value)
        return value, [folder/'freeze-archive.json', *(s.root/r['path'] for r in refs)]
    if s.step('freeze-continuation-archive', archive, binding=enrolled['identity']) is None:
        return
    bound = seal(dict(continuation=enrolled['identity'], arguments=arguments))
    def freeze():
        with Allowance(folder/'continuation-allowance.json', bound, enrolled['remaining_seconds']).attempt() as (left, ordinal):
            logs = folder/'logs'/str(ordinal).zfill(4)
            try:
                launch(snapshot_argv(argv), logs, seconds=left-120, cleanup_seconds=120, stream_cap=4 << 20)
            finally:
                try:
                    copy_trace(base/'frozen', folder/'retry-freeze-traceback.log')
                except (OSError, ValueError, KeyError) as exc:
                    # Collection must not replace the original subprocess failure.
                    (folder/'traceback-collector-error.log').write_text(type(exc).__name__+': '+str(exc)[:4096]+'\n')
            if read(base/'frozen/result.json').get('mechanical_status') != 'COMPLETE':
                raise ValueError('canonical freeze did not complete')
            return dict(operation='freeze', output=arguments['output'], exit_code=0), [
                logs/'process.json', base/'frozen/result.json', base/'frozen/campaign.json']
    row = s.journal['steps'][PREFIX+'freeze']
    replacing = row.get('binding') != bound and row.get('status') == 'FAILED' and not row.get('history')
    if s.step('freeze', freeze, binding=bound, recover=replacing) is None:
        return
    w = Workflow(s, args)
    setup = s.journal['steps'][PREFIX+'execution-preflight']['value']['setup_seconds']
    w.run_command('acceptance', 'acceptance', campaign=relative(base/'frozen/campaign.json'),
                  output=relative(base/'acceptance'), seconds=setup['acceptance'], dependencies=(PREFIX+'freeze',))
    def ready():
        values = [s.journal['steps'][PREFIX+n]['value'] for n in
                  ('registered-data', 'native-census', 'sensitivity', 'materials')]
        return readiness(w, *values)
    s.step('readiness', ready, dependencies=(PREFIX+'acceptance',))
    w.run_after_prepare()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    options = p.parse_args(argv)
    root = Path(options.root).resolve()
    script = Path(__file__).resolve()
    if not script.is_relative_to(root/'run-state'):
        p.error('copy this helper into run-state; retain the qualified b616b116 checkout')
    sys.path.insert(0, str(root/'src'))
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, read, NAMES, TelemetryCollector, telemetry
    binding = read(root/SESSION/'session.json')['binding']
    args = SimpleNamespace(phase='prepare-research', engine=binding['engine'], memory_gib=binding['memory_gib'],
                           cpus=binding['cpus'], render_node=binding.get('render_node'))
    s = Session(root, SESSION, binding, 'prepare-research')
    s.observer = TelemetryCollector(s.folder)
    NAMES.update({'continuation.json', 'continuation-original-journal.json', 'continuation-allowance.json', 'freeze-archive.json'})
    stop = threading.Event()
    def observe():
        while not stop.wait(30):
            try:
                s.observer.observe(telemetry(root))
            except OSError as exc:
                s.observer.failed(exc)
    def interrupt(*_):
        raise KeyboardInterrupt()
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with s.own():
            threading.Thread(target=observe, daemon=True).start()
            continue_campaign(s, args, script)
            return s.code()
    except KeyboardInterrupt:
        return 130
    finally:
        stop.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
