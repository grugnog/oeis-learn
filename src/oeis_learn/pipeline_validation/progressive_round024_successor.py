"""One reviewed random-control replacement following a real, charged local fit expiry."""
from contextlib import contextmanager
import fcntl
import json
import math
from functools import lru_cache

from oeis_learn.generalization.progressive.contracts import digest, reference, seal
from . import progressive_round023_successor as previous
from .identity import confined
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file, _json
from .progressive_workstation_state import Blocked, read

SESSION = previous.SUCCESSOR
SUCCESSOR = 'run-state/discovery/013-research-round-025'
BASE = SESSION + '/prepare-research'
RUN = SESSION + '/run-research/execution-0000'
CAMPAIGN = 'ea3266af2f6950610ce472af244f735575c38db21243edc9b3a9d77d0267cf91'
CONTROL = 'run-state/discovery-control/' + CAMPAIGN
ACCOUNTING = 'run-state/discovery-accounting/' + CAMPAIGN
REHEARSAL = 'run-state/discovery-tools/round024-random-rehearsal'
WORKER = RUN + '/task-4f5688b05f4130b30c716ccb8623d2779d4bee1c1a40e1f983b1301c3cc4f8c1'
TASK = 'capability-random-00-generate'
VERSION = 'progressive-preparation-owner-proof/v13'
NAME = 'round024_random_generator_replacement'
REVIEWED = 'sha256:20ee64838d39cf48187d0ce23f2e4522f45d70d7e517086538462c7446fe1e32'
HIP_PASS = 'sha256:0156d39f1c2935039c908106efeae39365226f362949fac7c0eabb6c74039c58'
CARRIED = ('capability-census', 'capability-language', *(f'capability-retained_D-{i:02d}-{op}'
                                                         for i in range(8) for op in ('generate', 'score')))


def require(condition, message):
    if not condition:
        raise Blocked(message)


def reviewed(root):
    account, control = confined(root, ACCOUNTING), confined(root, CONTROL)
    paths = [
        root / RUN / 'run.json', root / RUN / 'result.json', root / WORKER / 'role/result.json',
        root / WORKER / 'role/work-debits.jsonl', account / 'budget-binding.json', account / 'budget.jsonl'
    ]
    paths.extend(control.glob('transition-*.json'))
    paths.extend(account.glob('task-binding-*.json'))
    paths.extend((account / 'storage-inventory').glob('*.json'))
    refs = sorted((reference(root, p) for p in paths), key=lambda r: r['path'])
    require(
        digest(refs) == REVIEWED and not (control / 'assessment-exposure.json').exists()
        and not (root / SESSION / 'assess').exists()
        and sorted(p.name
                   for p in (root / SESSION / 'run-research').glob('execution-*')) == ['execution-0000'],
        'round024 differs from the reviewed terminal invocation or has exposed assessment')
    supervisor = root / WORKER / 'supervision'
    checked_file(
        supervisor,
        dict(path='process.json',
             bytes=264,
             sha256='sha256:66562216b8cebcc116f867cd1ee496667e92caae4ce1a9d28548ea6d93167974'))
    checked_file(
        supervisor,
        dict(path='cleanup.json',
             bytes=132,
             sha256='sha256:047e76bb1255f00d99da59b7cb3ef8f162eff7c66c3f68a8c4aa46b64b75a2f3'))
    state = read(control / 'transition-0038.json')['state']
    outputs = state['outputs']
    require(
        set(outputs) == set(CARRIED)
        and state['tasks'] == dict(dict.fromkeys(CARRIED, 'COMPLETE'), **{TASK: 'FAILED'})
        and not state['choices'] and not state['decisions'], 'unexpected round024 terminal state')
    for cut in outputs.values():
        p = checked_file(root, cut['result'])
        for value in read(p)['metrics'].values():
            if isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}:
                checked_file(p.parent, value)
    ledger = [json.loads(line) for line in (account / 'budget.jsonl').read_bytes().splitlines()]
    charges = {e['operation'].split('/')[0]: e for e in ledger if e['event'] == 'COMMIT'}
    require(set(charges) == set(CARRIED[2:]) | {TASK}, 'complete reviewed round024 charges required')
    return outputs, charges


def rehearsal(root):
    out = confined(root, REHEARSAL)
    result, proof = read(out / 'result.json'), read(out / 'stage/proof.json')
    require(result['identity'] == HIP_PASS and result == seal(dict(proof, status='PASS')),
            'reviewed four-member random-model HIP pass required')
    checked_file(
        out / 'stage',
        dict(path='process.json',
             bytes=265,
             sha256='sha256:990c9b707bf94168aceb2f7f37d5bb7f4c64857f17ee94ca96ffe8199e693a2e'))
    checked_file(
        out / 'stage',
        dict(path='cleanup.json',
             bytes=133,
             sha256='sha256:af3c725e91943be7ce1c89038f8690cec29dc258911bd9143e56d5510e82ed76'))
    for key in ('debits', 'events', 'result', 'account'):
        checked_file(out / 'stage', proof[key])
    return reference(root, out / 'result.json')


@contextmanager
def lease(root):
    with confined(root, CONTROL + '/run.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def proof(root, receipt, session, journal):
    with lease(root):
        return _proof(root, receipt, session, journal)


def _proof(root, receipt, session, journal):
    checked_file(
        root,
        dict(path='src/oeis_learn/generalization/progressive/budgets.py',
             bytes=25645,
             sha256='sha256:64819084108419e7ec6885199210cc06deb2252a95263f4421596c69363f7b0b'))
    old = confined(root, SESSION)
    require(
        session == read(old / 'session.json') and journal == read(old / 'journal.json')
        and receipt == read(old / 'preparation-owner.json')['receipt']
        and read(old / 'preparation-owner-proof.json')['identity']
        == 'sha256:722db0a6321ec4b0a8b15a2f03e6d3854af1af9fd637fc8f5691a14211baa99f',
        'original round024 ownership required')
    outputs, charges = reviewed(root)
    spent = math.ceil(
        sum(
            a.get('elapsed_seconds', 0) for step in journal['steps'].values()
            for a in [*step.get('history', []), step]))
    return seal(
        dict(version=VERSION,
             predecessor=receipt,
             successor=SUCCESSOR,
             reviewed_evidence=REVIEWED,
             original_campaign='sha256:' + CAMPAIGN,
             original_outputs=outputs,
             charges=charges,
             original_ledger=reference(root, ACCOUNTING + '/budget.jsonl'),
             rehearsal=rehearsal(root),
             charged_round024_seconds=spent,
             original_costs_retained=True,
             scientific_allowance_transferred=False,
             replacement=TASK,
             authorized_replacements=1,
             current_source_requalification_required=True,
             additional_prepare_seconds=21600))


def setup_limits(setup, evidence):
    require(
        evidence['version'] == VERSION and evidence['reviewed_evidence'] == REVIEWED
        and seal(evidence) == evidence and evidence['additional_prepare_seconds'] == 21600,
        'bounded round025 setup authorization required')
    return dict(setup, prepare=21600, freeze=21600)


def additions(root):
    return previous.additions(root, read(root / SESSION / 'preparation-owner-proof.json'))


def input_scopes(root, tasks, refs, verify):
    by_name = {r['name']: r['artifact'] for r in refs}
    by_path = {r['artifact']['path']: r['artifact'] for r in refs}
    by_id = {t['id']: t for t in tasks}
    scopes = {}
    for name in (*CARRIED, TASK):
        plan = by_id[name]
        req = by_name['requests/' + name]
        require(req['sha256'] == plan['policy'], 'task request binding differs: ' + name)
        request = _json(verify(**req))
        channels = {}
        for kind, items in request['artifacts'].items():
            channel = []
            for item in items:
                if isinstance(item, dict):
                    channel.append(item)  # Ordered dynamic slot, including its exact producer.
                elif kind == 'model':
                    envelope = read(verify(**by_path[item]))
                    data = dict(envelope['data'])
                    ref = data.pop('artifact')
                    verify(**ref)
                    channel.append((envelope['kind'], envelope['role'], data, ref['bytes'], ref['sha256']))
                else:
                    ref = by_path[item]
                    verify(**ref)
                    channel.append((ref['bytes'], ref['sha256']))
            channels[kind] = channel
        scopes[name] = (plan['seed'], plan['arm'], plan['initializer'], plan['stage'], plan['slot'], {
            k: v
            for k, v in request.items() if k not in ('artifacts', 'image')
        }, channels)
    return scopes


def check_inputs(root, plans, refs, evidence):

    @lru_cache(None)
    def verify(path, bytes, sha256):
        return checked_file(root, dict(path=path, bytes=bytes, sha256=sha256))

    original = _json(root / BASE / 'frozen/campaign.json')
    require(original['identity'] == 'sha256:' + CAMPAIGN and seal(original) == original,
            'round024 campaign differs')
    old = original['payload']

    def science(tasks):
        keys = ('tokens', 'draws', 'attempts', 'rejections', 'native_work', 'elapsed_us', 'archive_bytes')
        return [(p['id'], p['token_target'], [p['caps'][k] for k in keys]) for p in tasks]

    require(
        science(plans) == science(old['tasks']) and evidence['replacement'] == TASK and input_scopes(
            root, plans, refs, verify) == input_scopes(root, old['tasks'], old['references'], verify),
        'round025 changed scientific limits or carried/replacement inputs')

    old_proof = read(root / SESSION / 'preparation-owner-proof.json')
    comparison = old_proof['comparison_projection']
    rows = _json(checked_file(root, old_proof['limit_report']))['tasks']
    previous.primary_comparison_sizes(root, dict(tasks=plans, references=refs),
                                      {row['task']: row
                                       for row in rows}, comparison['estimates'], comparison['sample_maxima'])


def carry(root, store, journal, evidence):
    from .progressive_outputs import commit
    from oeis_learn.generalization.progressive.contracts import record
    plans = {p['id']: p for p in store.campaign['payload']['tasks']}
    for name in CARRIED:
        cut = evidence['original_outputs'][name]
        task = record('task',
                      dict(plans[name], campaign=store.campaign['identity']),
                      created_at=store.campaign['created_at'],
                      purpose=store.campaign['purpose'],
                      track=store.campaign['track'])
        if name in evidence['charges']:
            old = evidence['charges'][name]
            # Mirror paid work under its successor task cap. A resumed import
            # completes the same reservation, without issuing another dispatch.
            journal.task_remaining(task)
            charge = journal.reserve(name,
                                     'carry-round024',
                                     0,
                                     old['cost'],
                                     boot=old['boot'],
                                     monotonic_us=old['start_us'])
            journal.commit(charge['key'])
        expected = dict(cut, task=task['identity'])
        if store.state['tasks'].get(name) == 'COMPLETE':
            require(store.state['outputs'].get(name) == expected, 'carried output changed: ' + name)
            continue
        require(store.next_task() == task, 'unexpected carry order: ' + name)
        commit(root, store, task, checked_file(root, cut['result']).parent, cut['operation'])
        store.boundary(task, 'COMPLETE', store.state['invocation'])


@contextmanager
def research_scope(root, campaign, path, store, journal):
    matches = [r['artifact'] for r in campaign['payload']['references'] if r['name'] == NAME]
    expected = root / SUCCESSOR / 'prepare-research/frozen/campaign.json'
    if not matches and confined(root, path) != expected:
        yield
        return
    proof_path = SUCCESSOR + '/preparation-owner-proof.json'
    require(
        confined(root, path) == expected and matches == [reference(root, proof_path)],
        'single authorized round025 campaign required')
    with lease(root):
        old = root / SESSION
        evidence = read(root / proof_path)
        require(
            evidence == _proof(root,
                               read(old / 'preparation-owner.json')['receipt'], read(old / 'session.json'),
                               read(old / 'journal.json')), 'round024 evidence changed')
        check_inputs(root, campaign['payload']['tasks'], campaign['payload']['references'], evidence)
        grant = 'run-state/discovery-preparation-revisions/random-replacement-' + CAMPAIGN + '.json'
        retain(confined(root, grant, exists=False),
               seal(dict(proof=evidence['identity'], successor=campaign['identity'])))
        carry(root, store, journal, evidence)
        status = store.state['tasks'].get(TASK)
        if status is None:
            first = store.next_task()
            require(first is not None and first['payload']['id'] == TASK,
                    'replacement must be the first fresh dispatch')
        if status in {'FAILED', 'PARTIAL', 'SKIPPED'}:
            raise Blocked('authorized random-control replacement failed; preserve its evidence')
        yield
