"""One separately charged generator replacement after the reviewed round023 storage failure."""
from contextlib import contextmanager
import fcntl
import json
import math

from oeis_learn.generalization.progressive.contracts import reference
from .identity import confined, digest, seal
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file, _json
from .progressive_workstation_state import Blocked, read
from . import progressive_round022_successor as predecessor

SESSION = predecessor.SUCCESSOR
SUCCESSOR = 'run-state/discovery/013-research-round-024'
BASE = SESSION + '/prepare-research'
RUN = SESSION + '/run-research/execution-0000'
CAMPAIGN = '16c2f375dd323f5e8bdcb319216763c43b10fbf5b4593d3d5b168277f7921b1b'
CONTROL = 'run-state/discovery-control/' + CAMPAIGN
ACCOUNTING = 'run-state/discovery-accounting/' + CAMPAIGN
REHEARSAL = 'run-state/discovery-tools/round023-generator-rehearsal'
TASK = 'capability-retained_D-00-generate'
WORKER = RUN + '/task-8221a277f0e4830d3a0ebd41ff7e21eda9758f9feeb82f9e8b7b1fdcf68afaec'
VERSION = 'progressive-preparation-owner-proof/v12'
NAME = 'round023_generator_replacement'
REVIEWED = 'sha256:8affc96df94ee28f8329109141183969b71b50c59a59661469a6193843ebe07d'
MIB = 1 << 20


def originals(root):
    c, a = confined(root, CONTROL), confined(root, ACCOUNTING)
    paths = [confined(root, SESSION + '/' + n) for n in predecessor.SESSION_FILES]
    paths.extend(confined(root, RUN + '/' + n) for n in ('run.json', 'result.json'))
    for directory, pattern in ((c, 'transition-*.json'), (a, 'budget*.json*'), (a, 'task-binding-*.json'),
                               (a / 'storage-inventory', '*.json')):
        paths.extend(directory.glob(pattern))
    last = read(c / 'transition-0005.json')['state']
    outputs = {name: cut['result'] for name, cut in last['outputs'].items()}
    paths.append(checked_file(root, outputs['capability-language']))
    paths.extend(
        confined(root, WORKER + '/supervision/' + n)
        for n in ('process.json', 'cleanup.json', 'stderr.log'))
    refs = sorted((reference(root, p) for p in paths), key=lambda r: r['path'])
    if (digest(refs) != REVIEWED
            or sorted(p.name
                      for p in (root / SESSION / 'run-research').glob('execution-*')) != ['execution-0000']
            or (c / 'assessment-exposure.json').exists() or (root / SESSION / 'assess').exists()):
        raise Blocked('round023 differs from the reviewed failed launch')
    for ref in outputs.values():
        path = checked_file(root, ref)
        for artifact in read(path)['metrics'].values():
            if isinstance(artifact, dict) and set(artifact) == {'bytes', 'sha256', 'path'}:
                checked_file(path.parent, artifact)
    return outputs


def rehearsal(root):
    base = confined(root, REHEARSAL)
    receipt = read(base / 'result.json')
    worker = base / 'worker'
    expected = {
        'original': 'sha256:' + CAMPAIGN,
        'members': 64,
        'task': TASK,
        'status': 'PASS',
        'scope': 'DIAGNOSTIC_ONLY',
        'version': 'round023-generator-rehearsal/v1'
    }
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise Blocked('completed training-only full-size rehearsal required')
    for field, path in [('result', worker / 'role/result.json'),
                        ('process', worker / 'supervision/process.json'),
                        ('cleanup', worker / 'supervision/cleanup.json')]:
        if checked_file(root, receipt[field]) != path:
            raise Blocked('original rehearsal evidence required')
    result = read(worker / 'role/result.json')
    proc = _json(worker / 'supervision/process.json')
    cleanup = _json(worker / 'supervision/cleanup.json')
    if (proc['returncode'] != 0 or proc.get('cleanup_failure') or not cleanup['acknowledged']
            or not cleanup['within_bound'] or result['mechanical_status'] != 'COMPLETE'
            or result['operation'] != 'generate'):
        raise Blocked('successful original generator and cleanup required')
    candidate = checked_file(worker / 'role', result['metrics']['candidates'])
    if receipt['candidates_bytes'] != candidate.stat().st_size or receipt['diagnostic'] == receipt['original']:
        raise Blocked('separate diagnostic generator result required')
    scope = read(base / 'scope.json')
    authority = _json(worker / 'authority/supervisor.json')
    ledger = 'run-state/discovery-accounting/' + scope['diagnostic'][7:] + '/budget.jsonl'
    if receipt['ledger'] != reference(root, ledger):
        raise Blocked('diagnostic ledger changed after rehearsal')
    prereq = _json(checked_file(root, scope['support']))
    if scope['support'][
            'sha256'] != 'sha256:ce39ac65480892439a8c394726f17f016c6b4704a709a0be9f9d923655b73467':
        raise Blocked('original training-only support required')
    by_id = {row['draw_identity']: row for shard in prereq['shards'] for row in shard}
    prompts = []
    for path in sorted((base / 'training-prompts').glob('*.json')):
        envelope = read(path)
        v = envelope['data']
        source = by_id.get(v['id'], {})
        if (source.get('membership') != 'training' or source.get('native_verified') is not True
                or v != dict(id=source['draw_identity'],
                             indices=source['indices'][:20],
                             values=source['values'][:20])):
            raise Blocked('rehearsal input outside registered training support')
        prompts.append(envelope)
    data = _json(candidate, maximum=512 << 20)
    original = _json(root / BASE / 'frozen/campaign.json')
    req, = [r['artifact'] for r in original['payload']['references'] if r['name'] == 'requests/' + TASK]
    req = _json(checked_file(root, req))
    models = [_json(confined(root, p)) for p in req['artifacts']['model']]
    expected_inputs = digest(sorted([*prompts, *models], key=lambda v: v['identity']))
    if (len(prompts) != 64 or len(data['records']) != 64
            or set(data['records']) != {v['data']['id']
                                        for v in prompts} or not data['clock']['cost_complete']
            or data['clock']['expected_portfolios'] != 256 or authority['inputs'] != expected_inputs
            or authority['operation'] != 'generate'
            or authority['reservation']['cost'] != req['reservation'] or receipt['image'] != req['image']
            or scope['diagnostic'] != authority['campaign']['identity']
            or receipt['diagnostic'] != scope['diagnostic']
            or scope['task'] != authority['task']['identity'] or seal(authority) != authority):
        raise Blocked('incomplete or unbound training-only generator rehearsal')
    return receipt


def comparison_estimates(root, receipt, audit):
    """Replay the flagged comparisons against their frozen method and original HIP samples."""
    from .progressive_outputs import COMPARISON_TASKS

    plans = {row['task']: row for row in audit['tasks']}
    dynamic = receipt['estimated_dynamic_envelope_bytes']
    if set(dynamic) != set(plans):
        raise Blocked('incomplete rehearsal projection inventory')
    flags = {k for k, row in plans.items() if row['envelopes_bytes'] + dynamic[k] > 512 * MIB}
    if flags != set(receipt['projected_input_cap_flags']) or flags - COMPARISON_TASKS:
        raise Blocked('unresolved rehearsal projection outside the reviewed comparison readers')
    if not flags:
        return dict(estimates={}, sample_maxima={})
    timing = root / BASE / 'evaluation-timing'
    index = read(timing / 'evaluation-cases.json')
    request = read(timing / 'request.json')
    if (index['identity'] != 'sha256:2ea7f16e5e7931cfe4cbf8adab27f02bb500e68521b7a707d16329220685eb7f'
            or index['request'] != request['identity']):
        raise Blocked('reviewed round023 HIP timing inventory required')
    samples = []
    for item in index['cases']:
        case = request['cases'][item['index']]
        if case['methods'] != ['correct']:
            continue
        sample = read(checked_file(timing, item['producer']))
        if (sample['case'] != case['identity'] or sample['request'] != request['identity']
                or sample['samples'] != 2 or sample['status'] != 'PASS' or sample['device'] != 'actual_HIP'):
            raise Blocked('complete original two-member primary HIP timing sample required')
        samples.append(sample)
    if len(samples) != 12:
        raise Blocked('all 12 primary timing cases required')
    maxima = {key: max(r[field]['bytes'] for r in samples)
              for key, field in [('evaluation', 'producer'), ('scores', 'scoring')]}
    campaign = _json(root / BASE / 'frozen/campaign.json')
    if campaign['identity'] != 'sha256:' + CAMPAIGN or seal(campaign) != campaign:
        raise Blocked('original round023 frozen campaign required')
    estimates = primary_comparison_sizes(root, campaign['payload'], plans, flags, maxima)
    return dict(estimates=estimates, sample_maxima=maxima)


def primary_comparison_sizes(root, payload, plans, flags, maxima):
    by_id = {t['id']: t for t in payload['tasks']}
    artifacts = {r['artifact']['path']: r['artifact'] for r in payload['references']}
    policies = {r['artifact']['sha256']: r['artifact'] for r in payload['references'] if r['kind'] == 'policy'}
    primary, = [r['artifact'] for r in payload['references'] if r['name'] == 'private/channels/primary-portfolio']
    channel = read(checked_file(root, primary))
    if (channel['kind'] != 'oeis_generation' or channel['role'] != 'generator'
            or channel['data'] != dict(methods=['correct'], forced=None)):
        raise Blocked('original primary-only generation policy required')

    def request(name, op):
        plan = by_id[name]
        if any(choice['policy'] != plan['policy'] for choice in plan.get('trial_choices', [])):
            raise Blocked('comparison trial choices have different request policies')
        value = _json(checked_file(root, policies[plan['policy']]))
        if value['operation'] != op:
            raise Blocked('comparison producer operation differs')
        return value['artifacts']

    estimates = {}
    for name in sorted(flags):
        slots = request(name, 'publish')['public']
        sources = [(v['task'], v['output']) for v in slots if isinstance(v, dict)]
        row = plans[name]
        if (row['operation'] != 'publish' or sources != [tuple(v) for v in row['dynamic_sources']]
                or len(sources) != 8 or len(set(sources)) != 8):
            raise Blocked('four original paired comparison cells required')
        total = sum(artifacts[v]['bytes'] for v in slots if isinstance(v, str))
        for (gen, gen_output), (score, score_output) in zip(sources[::2], sources[1::2], strict=True):
            g, s = request(gen, 'generate'), request(score, 'score')
            count = len(g['visible'])
            if (gen_output != 'evaluation' or score_output != 'scores' or primary['path'] not in g['model']
                    or s['candidates'] != [dict(task=gen, operation='generate', output='candidates', kind='candidate')]
                    or count != len(s['score_truth']) or count != plans[gen]['members']
                    or count != plans[score]['members'] or count <= 0
                    or by_id[gen]['caps']['attempts'] != count * 8):
                raise Blocked('matching primary-only generation, scoring and member counts required')
            for output in (gen_output, score_output):
                # Multi-member outputs share one header. Repeating the entire two-member
                # file overcounts it. Add 50% for sample variation and 8 KiB per compact
                # ownership envelope (original path, task identity, hashes and metadata).
                total += (maxima[output] * ((count + 1) // 2) * 3 + 1) // 2 + 8192
        if total > 512 * MIB:
            raise Blocked(f'{name} primary comparison estimate {total} exceeds the 512 MiB reader cap')
        estimates[name] = total
    return estimates


def proof(root, receipt, session, journal):
    with (confined(root, CONTROL) / 'run.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _proof(root, receipt, session, journal)


def _proof(root, receipt, session, journal):
    outputs = originals(root)
    old = confined(root, SESSION)
    if (read(old / 'preparation-owner.json')['receipt'] != receipt or session != read(old / 'session.json')
            or journal != read(old / 'journal.json')):
        raise Blocked('original round023 owner and journal required')
    r = rehearsal(root)
    limit_report = confined(root, REHEARSAL + '/workflow-limits.json')
    if checked_file(root, r['limits']) != limit_report:
        raise Blocked('limit audit differs from completed rehearsal')
    audit = _json(limit_report)
    if audit['negative_timing_slack'] or audit['original_debits']['campaign'] != 'sha256:' + CAMPAIGN:
        raise Blocked('original complete workflow limit audit required')
    comparison = comparison_estimates(root, r, audit)
    spent = math.ceil(
        sum(
            v.get('elapsed_seconds', 0) for row in journal['steps'].values()
            for v in [*row.get('history', []), row]))
    ledger = reference(root, ACCOUNTING + '/budget.jsonl')
    debit = json.loads(checked_file(root, ledger).read_bytes().splitlines()[-1])
    return seal(
        dict(version=VERSION,
             predecessor=receipt,
             successor=SUCCESSOR,
             reviewed_evidence=REVIEWED,
             original_ledger=ledger,
             comparison_projection=comparison,
             original_outputs=outputs,
             replacement_operation=debit['operation'],
             charged_generator=debit['cost'],
             original_prepare_seconds=read(old / 'prepare-research/prepare-allowance.json')['seconds'],
             charged_round023_seconds=spent,
             additional_prepare_seconds=21600,
             rehearsal=reference(root, REHEARSAL + '/result.json'),
             diagnostic_campaign=r['diagnostic'],
             limit_report=reference(root, limit_report),
             original_costs_retained=True,
             scientific_allowance_transferred=False,
             current_source_requalification_required=True))


@contextmanager
def preparation_window(root, session):
    """One six-hour wall-clock envelope across retries; stop before the cleanup reserve."""
    import signal
    from .preparation_recovery import Allowance

    from . import progressive_round024_successor as random_control
    from . import progressive_round025_successor as learner
    predecessor = {root/learner.SUCCESSOR: learner.CAMPAIGN, root/SUCCESSOR: CAMPAIGN, root/random_control.SUCCESSOR: random_control.CAMPAIGN}.get(session.folder)
    if predecessor is None:
        yield
        return
    allowance = Allowance(session.folder / 'replacement-preparation-allowance.json',
                          dict(predecessor='sha256:' + predecessor, session=str(session.folder.relative_to(root))), 21600)
    with allowance.attempt() as (remaining, _):
        if remaining <= 120:
            raise Blocked(f'{session.folder.name} preparation allowance exhausted')

        def expired(*_):
            raise KeyboardInterrupt(f'{session.folder.name} preparation deadline reached')

        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining - 120)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def setup_limits(setup, evidence):
    if seal(evidence) != evidence or evidence['reviewed_evidence'] != REVIEWED or evidence[
            'additional_prepare_seconds'] != 21600:
        raise Blocked('reviewed independent round024 preparation allowance required')
    return dict(setup, prepare=21600, freeze=21600)


def check_inputs(root, plans, refs, proof):
    original = _json(confined(root, BASE + '/frozen/campaign.json'))['payload']

    def scopes(tasks, references):
        by_path = {r['artifact']['path']: r['artifact'] for r in references}
        by_name = {r['name']: r['artifact'] for r in references}
        by_id = {p['id']: p for p in tasks}
        values = []
        for name in ('capability-census', 'capability-language', TASK):
            task = by_id[name]
            ref = by_name['requests/' + name]
            if task['policy'] != ref['sha256']:
                raise Blocked('request binding differs')
            req = _json(checked_file(root, ref))
            channels = {}
            for key, paths in req['artifacts'].items():
                if key == 'model':
                    models = []
                    for path in paths:
                        value = _json(checked_file(root, by_path[path]))
                        data = value['data'].copy()
                        artifact = data.pop('artifact')
                        checked_file(root, artifact)
                        models.append(
                            (value['kind'], value['role'], data, artifact['sha256'], artifact['bytes']))
                    channels[key] = models
                else:
                    channels[key] = [by_path[path]['sha256'] for path in paths]
            values.append((task['seed'], task['arm'], task['initializer'], req['operation'], req['seconds'],
                           req['reservation'], channels))
        return values

    if scopes(plans, refs) != scopes(original['tasks'], original['references']):
        raise Blocked('generator or carried audit/census inputs, settings or reservations differ')

    def science(tasks):
        keys = ('tokens', 'draws', 'attempts', 'rejections', 'native_work', 'elapsed_us', 'archive_bytes')
        return [(p['id'], p['token_target'], [p['caps'][k] for k in keys]) for p in tasks]

    if science(plans) != science(original['tasks']) or proof['charged_generator']['attempts'] != 3072:
        raise Blocked('approved replacement must preserve all scientific work and time limits')

    comparison = proof.get('comparison_projection', {})
    if comparison.get('estimates'):
        rows = _json(checked_file(root, proof['limit_report']))['tasks']
        primary_comparison_sizes(root, dict(tasks=plans, references=refs),
                                 {r['task']: r for r in rows}, comparison['estimates'],
                                 comparison['sample_maxima'])


def additions(root, proof):
    """Extra operational space above the previous measured per-task partition."""
    audit = _json(checked_file(root, proof['limit_report']))
    measured = _json(checked_file(root, proof['rehearsal']))
    result = {}
    for row in audit['tasks']:
        op = row['operation']
        # Include campaign authority serialization, supervision and temporary publication.
        fixed = audit['metadata_floor_bytes'] + 32 * MIB
        inputs = row['static_projection_estimate']
        if op == 'generate':
            # Original measured rule already reserves the checkpoint for each generator.
            inputs = max(0, inputs - (row['disk_bytes'] - 128 * MIB))
        if op in ('generate', 'train'):
            fixed += 2 * max(audit['observed_gpu_runtime_bytes'], measured['sampled_peak_bytes'])
        # Candidates are wrapped for scoring/publication. Allow two copies at peak.
        if row['dynamic_inputs'] and op in ('score', 'publish'):
            fixed += 2 * measured['estimated_dynamic_envelope_bytes'][row['task']]
        result[row['task']] = max(0, math.ceil((fixed + inputs - 128 * MIB) / MIB)) * MIB
    return result


def preflight(root, evidence):
    from shutil import disk_usage
    from .progressive_workstation_state import write
    original = read(root / BASE / 'storage-preflight.json')
    extra = sum(additions(root, evidence).values())
    required = original['required_bytes'] + extra
    free = disk_usage(root).free
    target = root / SUCCESSOR / 'prepare-research/replacement-storage-estimate.json'
    write(
        target,
        seal(
            dict(required_estimate_bytes=required,
                 additional_task_reservations=extra,
                 available_now_bytes=free,
                 original=original['identity'],
                 canonical_freeze_required=True)))
    if free < required:
        raise Blocked(
            f'round024 storage estimate exceeds current free space by {required-free} bytes: {target}')
    return read(target), [target]


def expand(payload, extra):
    plans = payload['task_plans']
    if set(extra) != {p['id'] for p in plans}:
        raise Blocked('complete original task storage inventory required')
    slots = {n['id']: n for n in payload['graph']['nodes']}
    for p in plans:
        n = extra[p['id']]
        p['caps']['disk_bytes'] += n
        slots[p['slot']]['caps']['disk_bytes'] += n
        payload['caps']['disk_bytes'] += n


@contextmanager
def research_scope(root, campaign, path, store):
    refs = campaign['payload']['references']
    matches = [r['artifact'] for r in refs if r['name'] == NAME]
    expected = confined(root, SUCCESSOR + '/prepare-research/frozen/campaign.json', exists=False)
    if confined(root, path) != expected and not matches:
        yield
        return
    proof_path = SUCCESSOR + '/preparation-owner-proof.json'
    if confined(root, path) != expected or matches != [reference(root, proof_path)]:
        raise Blocked('generator authorization belongs only to round024')
    with (confined(root, CONTROL) / 'run.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = read(root / proof_path)
        if evidence != _proof(root,
                              read(root / SESSION / 'preparation-owner.json')['receipt'],
                              read(root / SESSION / 'session.json'), read(root / SESSION / 'journal.json')):
            raise Blocked('original generator proof differs')
        check_inputs(root, campaign['payload']['tasks'], refs, evidence)
        retain(
            confined(root,
                     'run-state/discovery-preparation-revisions/generator-replacement-' + CAMPAIGN +
                     '.json',
                     exists=False), seal(dict(proof=evidence['identity'], successor=campaign['identity'])))
        from .progressive_outputs import commit
        for name, op in [('capability-census', 'census'), ('capability-language', 'audit-language')]:
            ref = evidence['original_outputs'][name]
            if store.state['tasks'].get(name) == 'COMPLETE':
                if store.state['outputs'][name]['result'] != ref:
                    raise Blocked('carried result differs')
                continue
            task = store.next_task()
            if task is None or task['payload']['id'] != name:
                raise Blocked('only the original census and audit may be carried')
            commit(root, store, task, checked_file(root, ref).parent, op)
            store.boundary(task, 'COMPLETE', store.state['invocation'])
        yield
