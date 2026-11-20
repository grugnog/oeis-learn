"""Reviewed one-audit replacement after round022 failed before language-worker dispatch."""
from contextlib import contextmanager
import fcntl
import json
import math

from oeis_learn.generalization.progressive.contracts import reference
from .identity import confined, digest, seal
from .progressive_workstation_state import Blocked, read
from .progressive_research_inputs import checked_file, _json

SESSION = 'run-state/discovery/013-research-round-022'
SUCCESSOR = 'run-state/discovery/013-research-round-023'
VERSION = 'progressive-preparation-owner-proof/v11'
NAME = 'round022_language_audit_replacement'
CAMPAIGN = '6cacb68eb0d50375a08e3f1f3bdb0bce758c05a691e18c779a5a28ef71c4a0ea'
CONTROL = 'run-state/discovery-control/' + CAMPAIGN
ACCOUNTING = 'run-state/discovery-accounting/' + CAMPAIGN
RUN = SESSION + '/run-research/execution-0000'
BASE = SESSION + '/prepare-research'
# Digest of the path, size and byte hash of each original, including both
# hash chains, original census result and payload, and pre-dispatch traceback.
REVIEWED = 'sha256:c9bc58988790ddddff31a1137fa94d31dda2a8e1016c3f7abda84abcf0cc4c38'
SESSION_FILES = ('session.json', 'journal.json', 'preparation-owner.json', 'preparation-owner-proof.json',
                 'panel-data-reuse.json', 'prepare-research/preparation-data-reuse.json',
                 'prepare-research/frozen/campaign.json', 'prepare-research/prepare-allowance.json',
                 'prepare-research/acceptance/attempts/0000/acceptance.json')
RESULT = RUN + '/task-af94eacfbe5292f288c93c260ade57e9e1c1ff2e5a69d23461ab5d6252f19092/role/result.json'


def reviewed_records(root):
    control = confined(root, CONTROL)
    account = confined(root, ACCOUNTING)
    refs = [reference(root, SESSION + '/' + name) for name in SESSION_FILES]
    for name in ('run.json', 'result.json', 'private/host-traceback.txt'):
        refs.append(reference(root, RUN + '/' + name))
    refs.extend(reference(root, path) for path in (RESULT, RESULT.replace('result.json', 'census.json')))
    for directory, pattern in ((control, 'transition-*.json'), (account, 'budget*.json*'),
                               (account, 'task-binding-*.json'), (account/'storage-inventory', '*.json')):
        refs.extend(reference(root, path) for path in sorted(directory.glob(pattern)))
    if digest(refs) != REVIEWED:
        raise Blocked('round022 launch, budget or original evidence differs from the reviewed records')
    old = confined(root, SESSION)
    if (set(p.name for p in (old/'run-research').glob('execution-*') if p.is_dir()) != {'execution-0000'}
            or (old/'assess').exists() or (control/'assessment-exposure.json').exists()):
        raise Blocked('round022 has a later execution or assessment')
    failed = confined(root, RUN+'/task-bca6ccf45133267aa7f0d4a5afe15b662778a2b7b3877a193293e7943bce94ba', exists=False)
    if any((failed/name).exists() for name in ('supervision', 'role', 'authority')):
        raise Blocked('round022 language worker dispatch evidence found')
    return read(control/'transition-0003.json')['state']['outputs']['capability-census']['result']


def proof(root, receipt, session, journal):
    with predecessor_lease(root):
        return _proof(root, receipt, session, journal)


def _proof(root, receipt, session, journal):
    census = reviewed_records(root)
    owner = read(confined(root, SESSION + '/preparation-owner.json'))
    if (receipt != owner['receipt']
            or read(checked_file(root, owner['proof'])) != read(confined(root, SESSION+'/preparation-owner-proof.json'))
            or session != read(confined(root, SESSION+'/session.json'))
            or journal != read(confined(root, SESSION+'/journal.json'))):
        raise Blocked('round022 original preparation owner required')
    elapsed = sum(attempt.get('elapsed_seconds', 0) for row in journal['steps'].values()
                  for attempt in [*row.get('history', []), row])
    remaining = read(confined(root, BASE+'/prepare-allowance.json'))['seconds'] - math.ceil(elapsed)
    if remaining <= 120:
        raise Blocked('preparation allowance exhausted')
    ledger = reference(root, ACCOUNTING+'/budget.jsonl')
    audit = json.loads(checked_file(root, ledger).read_bytes().splitlines()[-1])
    return seal(dict(version=VERSION, predecessor=receipt, successor=SUCCESSOR, reviewed_evidence=REVIEWED,
                     census=census, original_ledger=ledger, replacement_operation=audit['operation'],
                     replacement_reservation=audit['cost'], historical_campaign='sha256:'+CAMPAIGN,
                     remaining_prepare_seconds=remaining, charged_round022_seconds=math.ceil(elapsed),
                     original_costs_retained=True, scientific_allowance_transferred=False,
                     current_source_requalification_required=True))


def setup_limits(setup, evidence):
    if seal(evidence) != evidence or evidence['reviewed_evidence'] != REVIEWED or evidence['version'] != VERSION:
        raise Blocked('reviewed round022 proof required')
    return dict(setup, prepare=evidence['remaining_prepare_seconds'], freeze=86400)


def check_inputs(root, plans, references, evidence):
    """Match the original two census channels and 512 witnesses by bytes and order."""
    original = _json(confined(root, BASE+'/frozen/campaign.json'))['payload']
    def inputs(name, tasks, refs):
        task, = [t for t in tasks if t['id'] == name]
        policy, = [r['artifact'] for r in refs if r['name'] == 'requests/'+name]
        if task['policy'] != policy['sha256']:
            raise Blocked('frozen role request differs')
        request = _json(checked_file(root, policy))
        paths = {r['artifact']['path']: r['artifact'] for r in refs if r['role'] == 'custodian'}
        sources = [paths[path]['sha256'] for path in request['artifacts']['sources']]
        return (request['operation'], request['seconds'], request['reservation'], sources,
                task['caps'], task['arm'], task['seed'], task['stage'])

    for name in ('capability-census', 'capability-language'):
        current = inputs(name, plans, references)
        if current != inputs(name, original['tasks'], original['references']):
            raise Blocked('successor census or language audit inputs or allowance differ')
        if name == 'capability-language' and current[2] != evidence['replacement_reservation']:
            raise Blocked('one original-size audit reservation required')


@contextmanager
def predecessor_lease(root):
    with confined(root, CONTROL+'/run.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def research_scope(root, campaign, campaign_path, store):
    """Bind one accepted successor to the original public census and pending historical debit."""
    refs = campaign['payload']['references']
    matches = [r['artifact'] for r in refs if r['name'] == NAME]
    expected = SUCCESSOR+'/prepare-research/frozen/campaign.json'
    if not matches and confined(root, campaign_path) != root/expected:
        yield
        return
    path = confined(root, campaign_path)
    if path != root/expected or matches != [reference(root, SUCCESSOR+'/preparation-owner-proof.json')]:
        raise Blocked('audit authorization belongs only to the accepted round023 campaign')
    with predecessor_lease(root):
        evidence = read(checked_file(root, matches[0]))
        if evidence != _proof(root, read(root/SESSION/'preparation-owner.json')['receipt'],
                             read(root/SESSION/'session.json'), read(root/SESSION/'journal.json')):
            raise Blocked('successor evidence differs from reviewed round022 launch')
        check_inputs(root, campaign['payload']['tasks'], refs, evidence)
        old_plans = _json(confined(root, BASE+'/frozen/campaign.json'))['payload']['tasks']
        def scientific_limits(plans):
            keys = ('tokens', 'draws', 'attempts', 'rejections', 'native_work')
            return [(p['id'], p['token_target'], [p['caps'][k] for k in keys]) for p in plans]
        if scientific_limits(campaign['payload']['tasks']) != scientific_limits(old_plans):
            raise Blocked('replacement must preserve the frozen scientific work limits')
        from .preparation_recovery import retain
        # A different publication cannot claim the single replacement grant,
        # even if the operator supplies a fresh execution or output directory.
        grant = root/'run-state/discovery-preparation-revisions'/('audit-replacement-'+CAMPAIGN+'.json')
        retain(confined(root, grant, exists=False), seal(dict(predecessor=evidence['historical_campaign'],
                                proof=evidence['identity'], successor=campaign['identity'])))
        if store.state['tasks'].get('capability-census') == 'COMPLETE':
            if store.state['outputs']['capability-census']['result'] != evidence['census']:
                raise Blocked('successor census must retain its original producer')
        else:
            from .progressive_outputs import commit
            task = store.next_task()
            if task is None or task['payload']['id'] != 'capability-census' or store.state['tasks']:
                raise Blocked('only the original outcome-blind census may be carried forward')
            commit(root, store, task, checked_file(root, evidence['census']).parent, 'census')
            store.boundary(task, 'COMPLETE', store.state['invocation'])
        yield
