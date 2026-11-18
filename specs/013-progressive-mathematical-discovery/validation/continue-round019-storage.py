#!/usr/bin/env python3
"""Check round019 disk capacity; optionally resume the reviewed storage block.

Run outside the pinned source tree. This host-only adapter never substitutes
its diagnostic storage estimate for canonical freeze/acceptance validation.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

SESSION = 'run-state/discovery/013-research-round-019'
PREFIX = 'prepare-research/'
BLOCKED_RESULT = 'sha256:738bb6996b2d3c7254072ccb2140fac3955cd184b9a430d42786d9d9214f1b83'
MASKED_ERROR = "'NoneType' object has no attribute 'get'"


def original_helper(s):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.generalization.progressive.contracts import seal
    row = s.journal['steps'].get(PREFIX+'freeze-continuation-enroll', {})
    receipt = row.get('value', {})
    if row.get('status') != 'COMPLETE' or seal(receipt) != receipt or row.get('binding') != receipt.get('script'):
        raise ValueError('original round019 continuation enrollment required')
    path = checked_file(s.root, receipt['script'])
    spec = importlib.util.spec_from_file_location('original_round019_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.authenticate(s)
    return module, receipt


def capacity(root, prepared):
    """Same storage arithmetic as pinned freeze, without its trial validation.

    Check the task-partition lower bound first. Only if that fits, measure the
    full projection and checkpoint closure using canonical storage functions.
    Every measured input is hash checked; this report grants no execution right.
    """
    from oeis_learn.generalization.progressive.contracts import verify_refs, seal
    from oeis_learn.generalization.progressive.storage import checkpoint_size, projection_size, JOURNAL_BYTES
    from oeis_learn.generalization.progressive.storage_plan import task_checkpoints
    from oeis_learn.generalization.progressive.trial_scopes import variants
    from oeis_learn.generalization.progressive.campaign import initializer_origin
    from oeis_learn.pipeline_validation.progressive_research_resources import uses_task_checkpoint_bounds
    q = prepared['payload']
    task_bytes = sum(t['caps']['disk_bytes'] for t in q['task_plans'])
    fixed = task_bytes + JOURNAL_BYTES + q['shard_bytes'] + q['package_bytes']
    def report(required, inodes, scope, **details):
        free = shutil.disk_usage(root).free
        available_inodes = os.statvfs(root).f_favail
        return seal(dict(version='round019-storage-preflight/v1', prepared=prepared['identity'],
            status='READY' if required <= free and inodes <= available_inodes and scope == 'full' else 'BLOCKED',
            scope=scope, required_bytes=required, available_bytes=free,
            additional_bytes=max(0, required-free), required_inodes=inodes,
            available_inodes=available_inodes, additional_inodes=max(0, inodes-available_inodes),
            canonical_validation_still_required=True, **details))
    if fixed > shutil.disk_usage(root).free or q['required_inodes'] > os.statvfs(root).f_favail:
        return report(fixed, q['required_inodes'], 'lower_bound', task_store_bytes=task_bytes)
    refs = q['references']
    unique = {}
    for ref in [r['artifact'] for r in refs] + [q['runtime'], q['checkpoint'], q['target_policy']]:
        if ref['path'] in unique and unique[ref['path']] != ref:
            raise ValueError('conflicting input identity')
        unique[ref['path']] = ref
    # Deduplicate measurements, never replace canonical scientific validation.
    producer_refs = {}
    origins = {}
    for plan in q['task_plans']:
        for variant in variants(plan):
            budget = variant.get('iteration_budget')
            if budget:
                ref = budget['producer']
                if ref['path'] in producer_refs and producer_refs[ref['path']] != ref:
                    raise ValueError('conflicting calibration producer')
                producer_refs.pop(ref['path'], None)
                producer_refs[ref['path']] = ref
        initializer = plan['initializer']
        if initializer['artifact'] is not None:
            key = json.dumps({k: initializer.get(k) for k in ('mode', 'artifact', 'origin')}, sort_keys=True)
            origins[key] = initializer
    producers = []
    for ref in producer_refs.values():
        value = json.loads(verify_refs(root, [ref])[0])
        if seal(value) != value or value.get('status') != 'PASS':
            raise ValueError('original complete calibration required for storage estimate')
        producers.append(value)
    measured, checkpoint_inodes = checkpoint_size(root, q['checkpoint'], checkpoint_root=q.get('checkpoint_root'), research=True)
    measured = max([measured, *(p['checkpoint_bytes'] for p in producers)])
    checkpoint_inodes = max([checkpoint_inodes, *(p['checkpoint_inodes'] for p in producers)])
    resource_refs = [r['artifact'] for r in refs if r['name'] == 'research_resource_policy']
    scoped = len(resource_refs) == 1 and uses_task_checkpoint_bounds(json.loads(verify_refs(root, resource_refs)[0]))
    bounds = task_checkpoints(q['task_plans'], producers, q['milestone_pins']) if scoped else {}
    if bounds:
        measured = max(measured, max(b['checkpoint_bytes'] for b in bounds.values()))
        checkpoint_inodes = max(checkpoint_inodes, max(b['checkpoint_inodes'] for b in bounds.values()))
    for plan in q['task_plans']:
        bound = bounds.get(plan['id'], dict(checkpoint_bytes=measured, milestone_pins=q['milestone_pins']))
        if plan['kind'] == 'training' and bound['checkpoint_bytes']*(2+bound['milestone_pins']) > plan['caps']['disk_bytes']:
            raise ValueError('task checkpoint storage deficit: '+plan['id'])
    closure = {}
    for initializer in origins.values():
        _, found = initializer_origin(root, initializer, track=prepared['track'], purpose=prepared['purpose'])
        closure.update({r['path']: r for r in found})
    projection_bytes, projection_inodes = projection_size(root, list(unique.values()))
    projection_bytes += sum(r['bytes'] for r in closure.values()) + measured
    projection_inodes += sum(len(Path(r['path']).parts)+1 for r in closure.values())
    required_inodes = max(q['required_inodes'], len(q['task_plans'])*(checkpoint_inodes*(2+q['milestone_pins'])+256)+projection_inodes+16384)
    return report(fixed+projection_bytes, required_inodes, 'full', task_store_bytes=task_bytes,
                  projection_bytes=projection_bytes, checkpoint_bytes=measured)


def reviewed_failure(row, result, process, stdout, allowance):
    """Only the uploaded clean storage block, never a generic failed operation."""
    if (row.get('status') != 'FAILED' or row.get('reason') != MASKED_ERROR or len(row.get('history', [])) != 1
            or result.get('identity') != BLOCKED_RESULT or result.get('producer_error') is not None
            or result.get('mechanical_status') != 'BLOCKED' or result.get('reason') != 'FREEZE_DEFICIT'
            or result.get('missing_evidence') != ['storage_bytes'] or result != stdout
            or process.get('returncode') != 2 or process.get('cleanup_failure') or process.get('outcome') != 'FAIL'
            or process.get('omitted_bytes') != {'stdout': 0, 'stderr': 0}
            or not process.get('failure', '').startswith('RuntimeError: exit 2: ')
            or allowance.get('active') is not None or len(allowance.get('attempts', [])) != 1
            or allowance.get('binding') != row.get('binding')
            or allowance['attempts'][0].get('outcome') != 'AttributeError'
            or allowance['attempts'][0].get('error') != MASKED_ERROR):
        raise ValueError('only the reviewed clean round019 storage block may resume')
    spent, elapsed = allowance.get('spent_seconds'), process.get('elapsed_seconds')
    if (any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in (spent, elapsed))
            or not elapsed <= spent <= elapsed+2 or spent >= allowance['seconds']):
        raise ValueError('original charged interval differs')


def remaining_seconds(prior, first, retry):
    """Debit both failed freezes before reallocating unused parent setup time."""
    values = [prior.get('seconds'), prior.get('spent_seconds'), first.get('previous_freeze_seconds'), retry.get('spent_seconds')]
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in values):
        raise ValueError('finite original accounting required')
    if prior.get('active') is not None or retry.get('active') is not None:
        raise ValueError('closed original allowances required')
    remaining = values[0]-sum(values[1:])
    if remaining <= 120:
        raise ValueError('original preparation time exhausted')
    return min(24*3600, remaining)


def collect_trace(output, folder):
    """Collection errors are secondary; null producer_error is normal for blocks."""
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    try:
        if not (output/'result.json').exists():
            return
        error = read(output/'result.json').get('producer_error')
        if error is None:
            return
        if not isinstance(error, dict):
            raise ValueError('producer_error must be null or an object')
        ref = error.get('local_traceback')
        if ref:
            if ref['bytes'] > 64 << 10:
                raise ValueError('host traceback exceeds diagnostic bound')
            (folder/'freeze-traceback.log').write_bytes(checked_file(output, ref).read_bytes())
    except Exception as exc:
        try:
            (folder/'collector-error.log').write_text(type(exc).__name__+': '+str(exc)[:4096]+'\n')
        except OSError:
            print('Traceback collector failed: '+str(exc), file=sys.stderr, flush=True)


def inspect_storage(s, args, folder):
    """Read-only engine accounting and workspace inventory, packaged on failure."""
    from oeis_learn.pipeline_validation.supervision import launch
    from oeis_learn.pipeline_validation.progressive_workstation_state import write, read
    from oeis_learn.pipeline_validation.discovery_cleanup import inventory
    output = folder/('inspection-'+str(time.time_ns()))
    output.mkdir(parents=True)
    commands = [('engine-disk', [args.engine, 'system', 'df', '-v']),
                ('engine-store', [args.engine, 'info', '--format', '{{json .Store}}'])]
    images = read(s.folder/'prepare-research/qualification/qualification.json')['images']
    for i, image in enumerate(images):
        commands.append(('image-history-'+str(i), [args.engine, 'history', '--no-trunc', '--format', 'json', image]))
    for name, argv in commands:
        try:
            launch(argv, output/name, seconds=120, cleanup_seconds=10, stream_cap=4 << 20)
        except Exception as exc:
            (output/(name+'-collector-error.log')).write_text(type(exc).__name__+': '+str(exc)[:4096]+'\n')
    try:
        write(output/'disk-inventory.json', inventory(s.root))
    except Exception as exc:
        (output/'inventory-collector-error.log').write_text(type(exc).__name__+': '+str(exc)[:4096]+'\n')
    print('Read-only storage diagnostics: '+str(output), flush=True)


def execute(s, args, script, *, proceed=False, inspect=False):
    from oeis_learn.generalization.progressive.contracts import load, seal, reference, CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow, command
    from oeis_learn.pipeline_validation.progressive_workstation_state import read, write, Blocked
    from oeis_learn.pipeline_validation.preparation_recovery import retain, Allowance
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_research_launch import readiness
    from oeis_learn.pipeline_validation.supervision import launch
    helper, first = original_helper(s)
    base = s.folder/'prepare-research'
    folder = base/'storage-continuation'
    folder.mkdir(parents=True, exist_ok=True)
    relative = lambda p: str(p.relative_to(s.root))
    if inspect:
        inspect_storage(s, args, folder)
    row = s.journal['steps'][PREFIX+'freeze']
    previous = s.journal['steps'].get(PREFIX+'storage-continuation-enroll', {}).get('value')
    if previous and row.get('binding', {}).get('continuation') == previous['identity'] and row['status'] != 'COMPLETE':
        print('Retaining new freeze failure for review: '+row.get('reason', row['status']), flush=True)
        return s.code()
    if row['status'] != 'COMPLETE':
        print('Checking storage before expensive freeze validation...', flush=True)
        report = capacity(s.root, load(base/'prepared/prepared.json', maximum=CAMPAIGN_DOCUMENT_BYTES))
        path = folder/('check-'+str(time.time_ns()))/'storage-preflight.json'
        write(path, report)
        print(json.dumps(report, sort_keys=True, indent=2), flush=True)
        if report['status'] != 'READY':
            print('No freeze/training started. Free additional space and rerun this command.', flush=True)
            return 2
        if not proceed:
            print('Storage fits now. Add --continue to retry canonical freeze and continue into training.', flush=True)
            return 0
        def enroll():
            old = base/'freeze-continuation'
            process = json.loads((old/'logs/0000/process.json').read_bytes())
            result = read(base/'frozen/result.json')
            retry = read(old/'continuation-allowance.json')
            reviewed_failure(row, result, process, json.loads((old/'logs/0000/stdout.log').read_bytes()), retry)
            if any((base/'frozen'/n).exists() for n in ('campaign.json', 'runtime.json', 'tasks')):
                raise ValueError('partially published campaign requires review')
            prior = read(checked_file(s.root, first['prior_allowance']))
            if retry['seconds'] != first['remaining_seconds'] or row['binding']['continuation'] != first['identity']:
                raise ValueError('original continuation scope differs')
            value = seal(dict(version='round019-storage-continuation/v1', source=s.binding['source'],
                script=reference(s.root, script), original_script=first['script'],
                original_allowance=first['prior_allowance'], retry_allowance=reference(s.root, old/'continuation-allowance.json'),
                remaining_seconds=remaining_seconds(prior, first, retry),
                previous_seconds=prior['spent_seconds']+first['previous_freeze_seconds']+retry['spent_seconds'],
                original_binding=row['binding'], originals=[reference(s.root, p) for p in sorted((base/'frozen').rglob('*')) if p.is_file()],
                scientific_budgets_changed=False, qualification_repeated=False))
            retain(folder/'original-journal.json', seal(s.journal))
            retain(folder/'continuation.json', value)
            return value, [folder/'original-journal.json', folder/'continuation.json']
        enrolled = s.step('storage-continuation-enroll', enroll, binding=reference(s.root, script))
        if enrolled is None:
            return s.code()
        for ref in (enrolled['original_allowance'], enrolled['retry_allowance'], enrolled['original_script']):
            checked_file(s.root, ref)
        def archive():
            target = folder/'original-blocked-freeze'
            if not target.exists():
                for ref in enrolled['originals']:
                    checked_file(s.root, ref)
                shutil.move(base/'frozen', target)
            elif (base/'frozen').exists():
                raise ValueError('ambiguous blocked publication archive')
            refs = [dict(r, path=relative(target/Path(r['path']).relative_to(base.relative_to(s.root)/'frozen'))) for r in enrolled['originals']]
            for ref in refs:
                checked_file(s.root, ref)
            value = seal(dict(version='round019-storage-archive/v1', originals=enrolled['originals'], archived=refs))
            retain(folder/'freeze-archive.json', value)
            return value, [folder/'freeze-archive.json', *(s.root/r['path'] for r in refs)]
        if s.step('storage-continuation-archive', archive, binding=enrolled['identity']) is None:
            return s.code()
        arguments = dict(blueprint=relative(base/'calibration-reuse/blueprint.json'), qualification=relative(base/'qualification'),
                         prepared=relative(base/'prepared'), output=relative(base/'frozen'))
        bound = seal(dict(continuation=enrolled['identity'], arguments=arguments))
        def freeze():
            with Allowance(folder/'continuation-allowance.json', bound, enrolled['remaining_seconds']).attempt() as (left, ordinal):
                logs = folder/'logs'/str(ordinal).zfill(4)
                try:
                    launch(helper.snapshot_argv(command(s.root, args, 'freeze', **arguments)), logs,
                           seconds=left-120, cleanup_seconds=120, stream_cap=4 << 20, allowed_returncodes=(0, 2, 3))
                finally:
                    collect_trace(base/'frozen', folder)
                result = read(base/'frozen/result.json')
                if result['mechanical_status'] != 'COMPLETE':
                    message = result.get('reason', '')+': '+', '.join(result.get('missing_evidence', []))
                    raise (Blocked if result['mechanical_status'] == 'BLOCKED' else RuntimeError)(message)
                return dict(operation='freeze', output=arguments['output'], exit_code=0), [
                    logs/'process.json', base/'frozen/result.json', base/'frozen/campaign.json']
        replacing = row.get('binding') == enrolled['original_binding'] and row.get('reason') == MASKED_ERROR
        if s.step('freeze', freeze, binding=bound, recover=replacing) is None:
            return s.code()
    elif not proceed:
        print('Freeze already complete; add --continue for canonical acceptance/run/resume.', flush=True)
        return 0
    w = Workflow(s, args)
    setup = s.journal['steps'][PREFIX+'execution-preflight']['value']['setup_seconds']
    w.run_command('acceptance', 'acceptance', campaign=relative(base/'frozen/campaign.json'), output=relative(base/'acceptance'),
                  seconds=setup['acceptance'], dependencies=(PREFIX+'freeze',))
    def ready():
        return readiness(w, *(s.journal['steps'][PREFIX+n]['value'] for n in ('registered-data', 'native-census', 'sensitivity', 'materials')))
    s.step('readiness', ready, dependencies=(PREFIX+'acceptance',))
    w.run_after_prepare()
    return s.code()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--continue', dest='proceed', action='store_true', help='retry only after storage fits; then canonical acceptance and run/resume')
    p.add_argument('--inspect-storage', action='store_true', help='collect read-only Podman layer and workspace disk accounting')
    options = p.parse_args(argv)
    root, script = Path(options.root).resolve(), Path(__file__).resolve()
    if not script.is_relative_to(root/'run-state'):
        p.error('copy into run-state; retain the qualified b616b116 checkout and original helper')
    os.chdir(root)
    sys.path.insert(0, str(root/'src'))
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session, read, NAMES, TelemetryCollector, telemetry
    binding = read(root/SESSION/'session.json')['binding']
    args = SimpleNamespace(phase='prepare-research', engine=binding['engine'], memory_gib=binding['memory_gib'], cpus=binding['cpus'], render_node=binding.get('render_node'))
    s = Session(root, SESSION, binding, 'prepare-research')
    s.observer = TelemetryCollector(s.folder)
    NAMES.update({'continuation.json', 'original-journal.json', 'continuation-allowance.json', 'freeze-archive.json', 'storage-preflight.json', 'disk-inventory.json'})
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
            return execute(s, args, script, proceed=options.proceed, inspect=options.inspect_storage)
    except KeyboardInterrupt:
        return 130
    finally:
        stop.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
