"""Full-size, training-only generator rehearsal against the installed round023 image."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import math
import shutil
import zipfile
import sys
import threading
from types import SimpleNamespace

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import validate_campaign
from oeis_learn.generalization.progressive.contracts import atomic_write, record, reference, seal
from oeis_learn.generalization.progressive.projections import describe
from oeis_learn.generalization.progressive.storage import usage
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.preparation_recovery import retain
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json
from oeis_learn.pipeline_validation.progressive_roles import current_runtime, invoke
from oeis_learn.pipeline_validation.progressive_workstation import Workflow

SESSION = 'run-state/discovery/013-research-round-023'
BASE = SESSION + '/prepare-research'
ORIGINAL = 'sha256:16c2f375dd323f5e8bdcb319216763c43b10fbf5b4593d3d5b168277f7921b1b'
TASK = 'capability-retained_D-00-generate'
SUPPORT = 'sha256:ce39ac65480892439a8c394726f17f016c6b4704a709a0be9f9d923655b73467'
OUT = 'run-state/discovery-tools/round023-generator-rehearsal'
MIB = 1 << 20
EXTRA = 1024 * MIB  # Diagnostic ceiling for this one invocation.
CACHE = 77727804  # Observed bytes in the failed worker's private runtime.


def audit(campaign, base):
    from oeis_learn.generalization.progressive.storage_plan import checkpoint_bound
    from oeis_learn.generalization.progressive.trial_scopes import variants

    payload = campaign['payload']
    training = _json(base / 'prepared/resource-estimate.json')['training']
    evaluation = _json(base / 'evaluation-timing/evaluation-feasibility.json')['tasks']
    rows = training + evaluation
    timing = {r['task']: r for r in rows}
    metadata = len(json.dumps(dict(campaign=campaign), indent=2).encode()) + (2 << 20) + (64 << 10)
    tasks = []
    for plan in payload['tasks']:
        caps = plan['caps']
        row = dict(task=plan['id'],
                   disk_bytes=caps['disk_bytes'],
                   elapsed_us=caps['elapsed_us'],
                   tokens=caps['tokens'],
                   attempts=caps['attempts'],
                   archive_bytes=caps['archive_bytes'])
        if plan['id'] in timing:
            r = timing[plan['id']]
            if plan['kind'] == 'training':
                row['padded_time_slack_us'] = r['available_work_us'] - r['estimated_work_us']
                bound = checkpoint_bound(payload['storage'], plan)
                row['checkpoint_bytes_reserved'] = bound['checkpoint_bytes'] * (2 + bound['milestone_pins'])
                row['choices'] = len(variants(plan))
            else:
                row['padded_time_slack_us'] = r['available_us'] - r['estimated_us']
        tasks.append(row)
    return dict(task_count=len(tasks),
                training_count=len(training),
                evaluation_count=len(evaluation),
                metadata_floor_bytes=metadata,
                observed_gpu_runtime_bytes=CACHE,
                per_task_inodes=(payload['storage']['required_inodes'] - 16384) // len(tasks),
                journal_bytes=payload['storage']['journal_bytes'],
                package_bytes=payload['storage']['package_bytes'],
                training_seconds=sum(t['caps']['elapsed_us']
                                     for t in payload['tasks'] if t['kind'] == 'training') / 1e6,
                negative_timing_slack=[r['task'] for r in tasks if r.get('padded_time_slack_us', 0) < 0],
                tasks=tasks)


def review_original(root, campaign):
    """Bind the disposition and both original debits before spending diagnostic time."""
    control = confined(root, 'run-state/discovery-control/' + ORIGINAL[7:])
    accounting = confined(root, 'run-state/discovery-accounting/' + ORIGINAL[7:])
    last = _json(control / 'transition-0005.json')
    if (last['identity'] != 'sha256:00cd7f8b5a3851934ccded7a04d6792522ffc655b2892dcf84e4a23d58e765c6'
            or sorted(p.name for p in control.glob('transition-*.json'))
            != [f'transition-{i:04d}.json'
                for i in range(6)] or (control / 'assessment-exposure.json').exists() or sorted(
                    p.name
                    for p in (root / SESSION / 'run-research').glob('execution-*')) != ['execution-0000']):
        raise ValueError('round023 advanced beyond the reviewed evidence')
    tip = last['identity']
    for i in reversed(range(6)):
        transition = _json(control / f'transition-{i:04d}.json')
        if seal(transition) != transition or transition['identity'] != tip:
            raise ValueError('round023 transition chain differs')
        tip = transition['previous']
    ledger = (accounting / 'budget.jsonl').read_bytes().splitlines()
    tip = 'sha256:e2a423540cfe1d3c59431f04ef290cff3be64d85ee899641fe1c5466d1180a97'
    if len(ledger) != 4:
        raise ValueError('round023 accounting advanced')
    for line in reversed(ledger):
        entry = json.loads(line)
        if seal(entry) != entry or entry['identity'] != tip:
            raise ValueError('round023 budget chain differs')
        tip = entry['previous']
    for cut in last['state']['outputs'].values():
        path = checked_file(root, cut['result'])
        for value in _json(path)['metrics'].values():
            if isinstance(value, dict) and set(value) == {'bytes', 'sha256', 'path'}:
                checked_file(path.parent, value)
    journal = _json(root / SESSION / 'journal.json')
    charged = math.ceil(
        sum(
            a.get('elapsed_seconds', 0) for row in journal['steps'].values()
            for a in [*row.get('history', []), row]))
    available = _json(root / BASE / 'prepare-allowance.json')['seconds']
    return dict(campaign=campaign['identity'],
                original_results=last['state']['outputs'],
                replacement_task=TASK,
                charged_generator=json.loads(ledger[-1])['cost'],
                authorized_replacements=1,
                original_preparation_seconds=available,
                charged_journal_seconds=charged,
                carry_forward_seconds=available - charged)


def inspect_partitions(root, campaign, report):
    """Check static inputs and report unresolved predecessor sizes explicitly."""
    from oeis_learn.generalization.progressive.storage import projection_size
    from functools import lru_cache
    base = root / BASE
    p = campaign['payload']
    refs = {r['artifact']['path']: r['artifact'] for r in p['references']}
    requests = {r['artifact']['sha256']: r['artifact'] for r in p['references'] if r['kind'] == 'policy'}

    @lru_cache(None)
    def static(path, channel):
        ref = refs[path]
        if channel in ('model', 'training'):
            checked_file(root, ref)
            return projection_size(root, [ref])[0]
        return ref['bytes']

    for plan, row in zip(p['tasks'], report['tasks'], strict=True):
        request = _json(checked_file(root, requests[plan['policy']]))
        operation = request['operation']
        descriptors = request['artifacts']
        size = sum(static(v, ch) for ch, items in descriptors.items() for v in items if isinstance(v, str))
        slots = [v for items in descriptors.values() for v in items if isinstance(v, dict)]
        row.update(operation=operation,
                   static_projection_estimate=size,
                   dynamic_inputs=len(slots),
                   envelopes_bytes=sum(refs[v]['bytes'] for items in descriptors.values() for v in items
                                       if isinstance(v, str)),
                   members=len(descriptors.get('visible', descriptors.get('score_truth', []))),
                   dynamic_sources=[(v['task'], v['output']) for v in slots])
        fixed = report['metadata_floor_bytes'] + size
        if operation in ('generate', 'train'):
            fixed += CACHE
        fixed += row.get('checkpoint_bytes_reserved', 0)
        row['estimated_slack_before_dynamic_inputs_and_output'] = row['disk_bytes'] - fixed
    sampled = _json(base / 'evaluation-timing/cases/13/producer.json')
    report['two_member_candidate_bytes'] = sampled['producer']['bytes']
    report['two_member_scores_bytes'] = sampled['scoring']['bytes']
    report['static_storage_flags'] = [
        r['task'] for r in report['tasks'] if r['estimated_slack_before_dynamic_inputs_and_output'] < 0
    ]
    report['original_debits'] = review_original(root, campaign)
    return report


def training_prompts(support, count):
    from oeis_learn.generalization.progressive.cohorts import validate_prompt
    from oeis_learn.generalization.progressive.training_rows import validate_row

    if seal(support) != support or support['diagnostic']:
        raise ValueError('original training support required')
    unique = {r['draw_identity']: r for shard in support['shards'] for r in shard}
    ordered = sorted(unique.values(), key=lambda r: (-max(len(v) for v in r['values'][:20]), r['identity']))
    prompts = []
    for row in ordered[:count]:
        validate_row(row, native_verified=True)
        value = dict(id=row['draw_identity'], indices=row['indices'][:20], values=row['values'][:20])
        validate_prompt(value)
        prompts.append(value)
    if len(prompts) != count:
        raise ValueError('insufficient distinct training prompts')
    return prompts


def configure(root, out, campaign, prompts):
    """A distinct journal and request. Mount only training prompts and registered D model channels."""
    payload = campaign['payload']
    plan, = [p for p in payload['tasks'] if p['id'] == TASK]
    req, = [r for r in payload['references'] if r['name'] == 'requests/' + TASK]
    request = _json(checked_file(root, req['artifact']))
    if req['artifact']['sha256'] != plan['policy'] or len(
            request['artifacts']['visible']) != len(prompts) or len(prompts) != 64:
        raise ValueError('original 64-member generation request required')
    artifacts = {'model': [describe(root, p) for p in request['artifacts']['model']], 'visible': []}
    if Counter(a.kind for a in artifacts['model']) != {'recipe': 1, 'checkpoint': 1}:
        raise ValueError('original all-methods model channels required')
    for i, prompt in enumerate(prompts):
        path = out / 'training-prompts' / f'{i}.json'
        retain(
            path,
            seal(dict(version='discovery-channel/v1', kind='visible_prompt', role='generator',
                      data=prompt)))
        descriptor = describe(root, path)
        artifacts['visible'].append(descriptor)
        payload['references'].append(
            dict(name=f'rehearsal-training-{i}', kind='prepared', role='generator',
                 artifact=descriptor.ref))
    request['artifacts']['visible'] = [a.ref['path'] for a in artifacts['visible']]
    path = out / 'request.json'
    retain(path, request)
    req['artifact'] = reference(root, path)
    plan['policy'] = req['artifact']['sha256']
    plan['output'] = str((out / 'worker').relative_to(root))
    plan['caps']['disk_bytes'] += EXTRA
    slot, = [n for n in payload['graph']['nodes'] if n['id'] == plan['slot']]
    slot['caps']['disk_bytes'] += EXTRA
    payload['caps']['disk_bytes'] += EXTRA
    for name in ('required_bytes', 'task_store_bytes'):
        payload['storage'][name] += EXTRA
    diagnostic = validate_campaign(seal(campaign))
    task = record('task',
                  dict(plan, campaign=diagnostic['identity']),
                  created_at=diagnostic['created_at'],
                  purpose=diagnostic['purpose'],
                  track=diagnostic['track'])
    return diagnostic, task, request, artifacts


def measure(root, out, campaign):
    base = root / BASE
    support_ref = _json(base / 'materials/materials.json')['support']
    if support_ref['sha256'] != SUPPORT:
        raise ValueError('reviewed immutable training corpus required')
    support = _json(checked_file(root, support_ref))
    diagnostic, task, req, artifacts = configure(root, out, campaign, training_prompts(support, 64))
    runtime = _json(base / 'frozen/runtime.json')
    if runtime['identity'] != diagnostic['payload']['runtime'] or seal(runtime) != runtime:
        raise ValueError('original frozen runtime required')
    binding = _json(root / SESSION / 'session.json')['binding']
    host = Workflow(SimpleNamespace(root=root, folder=root / SESSION),
                    SimpleNamespace(**binding, phase='prepare-research')).host()
    inspections = out / 'inspections'
    inspections.mkdir(exist_ok=True)
    inspection = inspections / str(len(list(inspections.iterdir()))).zfill(4)
    inspection.mkdir()
    verified = current_runtime(host,
                               root,
                               inspection,
                               runtime,
                               producer_root=checked_file(root,
                                                          runtime['payload']['qualification'][0]).parent)
    ledger = root / 'run-state/discovery-accounting' / diagnostic['identity'][7:]
    journal = BudgetJournal(ledger,
                            diagnostic['identity'],
                            diagnostic['payload']['caps'],
                            journal_bytes=diagnostic['payload']['storage']['journal_bytes'])
    from oeis_learn.pipeline_validation.progressive_quarantine_control import require_clean_inputs
    original_account = root / 'run-state/discovery-accounting' / ORIGINAL[7:]
    require_clean_inputs(root, SimpleNamespace(root=original_account), artifacts)
    require_clean_inputs(root, journal, artifacts)
    worker = out / 'worker'
    stop = threading.Event()
    peak = [0, 0]

    def sample():
        while not stop.wait(5):
            try:
                used = usage([worker / 'role'])
            except FileNotFoundError:
                continue
            for i in (0, 1):
                peak[i] = max(peak[i], used[i])

    retain(
        out / 'scope.json',
        seal(
            dict(original=ORIGINAL,
                 diagnostic=diagnostic['identity'],
                 image=req['image'],
                 task=task['identity'],
                 members=sorted(a.identity for a in artifacts['visible']),
                 support=support_ref,
                 seconds=req['seconds'])))
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        invoke(host,
                        req['image'],
                        root,
                        'generate',
                        artifacts,
                        output=str(worker.relative_to(root)),
                        campaign=diagnostic,
                        task=task,
                        journal=journal,
                        ordinal=0,
                        reservation=req['reservation'],
                        seconds=req['seconds'],
                        runtime=verified)
    finally:
        stop.set()
        monitor.join()
        if worker.exists():
            retain(out / 'sampled-peak.json', dict(bytes=peak[0], inodes=peak[1]))
    return finish(root, out)


def finish(root, out):
    worker = out / 'worker'
    scope = _json(out / 'scope.json')
    if seal(scope) != scope or scope['original'] != ORIGINAL or scope['support']['sha256'] != SUPPORT:
        raise ValueError('original rehearsal scope required')
    result = _json(worker / 'role/result.json')
    process = _json(worker / 'supervision/process.json')
    cleanup = _json(worker / 'supervision/cleanup.json')
    if (seal(result) != result or result['operation'] != 'generate' or process['returncode'] != 0
            or process.get('cleanup_failure') or not cleanup['acknowledged']
            or not cleanup['within_bound']):
        raise ValueError('completed generator and acknowledged cleanup required')
    if result['mechanical_status'] != 'COMPLETE':
        raise RuntimeError('generator did not complete: ' + result['reason'])
    candidates = checked_file(worker / 'role', result['metrics']['candidates'])
    data = _json(candidates, maximum=512 << 20)
    visible = [_json(p) for p in sorted((out / 'training-prompts').glob('*.json'))]
    if (sorted(v['identity'] for v in visible) != scope['members'] or any(seal(v) != v for v in visible)
            or len(data['records']) != 64 or set(data['records']) != {v['data']['id']
                                                                      for v in visible}
            or not data['clock']['cost_complete'] or data['clock']['expected_portfolios'] != 256):
        raise ValueError('incomplete training-only member roster')
    for ref in result['metrics'].values():
        checked_file(worker / 'role', ref)
    # Match the serialized envelope that the next scoring projection must read.
    wrapped = seal(dict(version='discovery-channel/v1', kind='candidate', role='scorer', data=data))
    scoring_input_bytes = len(
        (json.dumps(wrapped, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    used = usage([worker / 'role'])
    authority = _json(worker / 'authority/supervisor.json')
    if (seal(authority) != authority or authority['task']['identity'] != scope['task']
            or authority['campaign']['identity'] != scope['diagnostic']):
        raise ValueError('generator authority differs from rehearsal')
    ledger = root / 'run-state/discovery-accounting' / scope['diagnostic'][7:] / 'budget.jsonl'
    entries = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    reserved = authority['reservation']
    if (len(entries) != 2 or entries[0] != reserved or entries[1]['event'] != 'COMMIT'
            or entries[1]['operation'] != reserved['operation'] or entries[1]['previous'] != reserved['identity']
            or any(seal(e) != e for e in entries)):
        raise ValueError('completed original diagnostic debit required')
    charge = authority['storage']
    sampled = _json(out / 'sampled-peak.json') if (out / 'sampled-peak.json').exists() else {}
    peak = [sampled.get('bytes', 0), sampled.get('inodes', 0)]
    task = authority['task']
    limit_report = _json(out / 'workflow-limits.json')
    # Readiness wraps the generator's evaluation a second time as public evidence.
    # Account for the indentation and ownership reference as the consumer will.
    evaluation_ref = result['metrics']['evaluation']
    evaluation = _json(checked_file(worker / 'role', evaluation_ref), maximum=512 << 20)
    public = seal(dict(version='discovery-owned-output/v1', task=TASK, task_record=scope['task'],
                       operation='generate', output='evaluation', record=evaluation,
                       artifact=reference(root, worker / 'role' / evaluation_ref['path'])))
    envelope = seal(dict(version='discovery-channel/v1', kind='public_evidence', role='verifier', data=public))
    evaluation_bytes = len((json.dumps(envelope, sort_keys=True, indent=2) + '\n').encode())
    # Scale each estimate by its producer's frozen roster.
    per_member = {
        'candidates': math.ceil(scoring_input_bytes / 64),
        'evaluation': math.ceil(evaluation_bytes / 64),
        'scores': limit_report['two_member_scores_bytes'],
    }
    tasks = {r['task']: r for r in limit_report['tasks']}
    dynamic_bytes = {}
    for name, row in tasks.items():
        size = 0
        for source, output in row['dynamic_sources']:
            if output in per_member:
                size += 4096 + tasks[source]['members'] * per_member[output]
            elif output in ('census', 'coverage'):
                ref = limit_report['original_debits']['original_results'][source]['result']
                producer = _json(checked_file(root, ref))
                size += 4096 + 2 * producer['metrics'][output]['bytes']
        dynamic_bytes[name] = size
    projected = {k: row['envelopes_bytes'] + dynamic_bytes[k] for k, row in tasks.items()}
    return seal(
        dict(version='round023-generator-rehearsal/v1',
             status='PASS',
             scope='DIAGNOSTIC_ONLY',
             original=ORIGINAL,
             diagnostic=scope['diagnostic'],
             task=TASK,
             image=scope['image'],
             members=64,
             max_bytes=task['payload']['caps']['disk_bytes'],
             prior=charge,
             ledger=reference(root, ledger),
             output_bytes=used[0],
             estimated_dynamic_envelope_bytes=dynamic_bytes,
             peak_projected_input_envelopes=max(projected.values()),
             projected_input_cap_flags=[name for name, n in projected.items() if n > 512 * MIB],
             sampled_peak_bytes=max(used[0], peak[0]),
             sampled_peak_inodes=max(used[1], peak[1]),
             limits=reference(root, out / 'workflow-limits.json'),
             candidates_bytes=candidates.stat().st_size,
             scoring_channel_bytes=scoring_input_bytes,
             result=reference(root, worker / 'role/result.json'),
             process=reference(root, worker / 'supervision/process.json'),
             cleanup=reference(root, worker / 'supervision/cleanup.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    base = ROOT / BASE
    campaign = _json(base / 'frozen/campaign.json')
    if seal(campaign) != campaign or campaign['identity'] != ORIGINAL:
        raise ValueError('original round023 campaign required')
    out = confined(ROOT, OUT, exists=False)
    out.mkdir(parents=True, exist_ok=True)
    control = confined(ROOT, 'run-state/discovery-control/' + ORIGINAL[7:])
    with (out / 'rehearsal.lock').open('a') as lock, (control / 'run.lock').open('rb') as original_lock:
        for handle in (lock, original_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = inspect_partitions(ROOT, campaign, audit(campaign, base))
        retain(out / 'workflow-limits.json', report)
        print(f"Reviewed {report['task_count']} frozen tasks. Limits: {out/'workflow-limits.json'}",
              flush=True)
        if args.audit_only:
            share(out)
            return
        try:
            run_or_recover(out, campaign)
        finally:
            share(out)


def share(out):
    name = out / 'diagnostics.zip'
    tmp = out / 'diagnostics.tmp'
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ('scope.json', 'workflow-limits.json', 'sampled-peak.json', 'result.json', 'failure.json',
                   'worker/supervision/process.json', 'worker/supervision/cleanup.json'):
            if (out / fn).is_file(): z.write(out / fn, fn)
    tmp.replace(name)
    print('Share diagnostics:', name, flush=True)


def run_or_recover(out, campaign):
    if (out / 'result.json').is_file():
        result = _json(out / 'result.json')
        if seal(result) != result or result['original'] != ORIGINAL or result['status'] != 'PASS':
            raise ValueError('rehearsal receipt differs')
        for field in ('result', 'process', 'cleanup', 'ledger', 'limits'):
            checked_file(ROOT, result[field])
    else:
        if (out / 'worker/role/result.json').exists():
            result = finish(ROOT, out)
            retain(out / 'result.json', result)
            return
        if (out / 'worker').exists():
            raise ValueError('prior worker evidence exists; preserve it for review before another dispatch')
        if shutil.disk_usage(out).free < EXTRA + (2 << 30): raise ValueError('at least 3 GiB free required')
        print('Starting one 64-member training-only retained-D rehearsal.', flush=True)
        try:
            result = measure(ROOT, out, campaign)
        except Exception as exc:
            from oeis_learn.pipeline_validation.progressive_workstation_state import write
            import traceback
            write(
                out / 'failure.json',
                dict(type=type(exc).__name__,
                     stack=[
                         dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                         for f in traceback.extract_tb(exc.__traceback__)
                     ]))
            raise
        atomic_write(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
