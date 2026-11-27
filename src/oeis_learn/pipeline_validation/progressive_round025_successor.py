"""Import the audited capability stage and its charges before corrected learner dispatch."""
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import json
import math

from oeis_learn.generalization.progressive.contracts import digest, reference, seal
from . import progressive_round023_successor as generator
from . import progressive_round024_successor as previous
from .identity import confined
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file, _json
from .progressive_workstation_state import read
from .progressive_round024_successor import require

SESSION = previous.SUCCESSOR
SUCCESSOR = 'run-state/discovery/013-research-round-026'
RETRY = 'run-state/discovery/013-research-round-027'
RETRY_VERSION = 'progressive-preparation-owner-proof/v15'
BASE = SESSION + '/prepare-research'
RUN = SESSION + '/run-research/execution-0000'
CAMPAIGN = 'eaf72f0eb07d1459970c41c43f277d917c752948ef29b89c55099a5fb1bf0757'
CONTROL = 'run-state/discovery-control/' + CAMPAIGN
ACCOUNT = 'run-state/discovery-accounting/' + CAMPAIGN
TASK = 'diversity-static-20261016'
VERSION = 'progressive-preparation-owner-proof/v14'
NAME = 'round025_capability_reuse'
AUDIT = 'sha256:0226d41d3b714a072a0d44677133ce56bb7c150400b9269dcf9434e9b0632410'
CARRIED = (*previous.CARRIED, *(f'capability-random-{i:02d}-{op}' for i in range(8)
                                for op in ('generate', 'score')), 'capability-readiness')


@contextmanager
def lease(root):
    with confined(root, CONTROL + '/run.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def reviewed(root):
    control, account = confined(root, CONTROL), confined(root, ACCOUNT)
    require(
        sorted(p.name for p in (root / SESSION / 'run-research').glob('execution-*')) == ['execution-0000']
        and not (control / 'assessment-exposure.json').exists() and not (root / SESSION / 'assess').exists(),
        'original single unassessed round025 invocation required')
    run = read(root / RUN / 'run.json')
    require(
        run['identity'] == 'sha256:8aa3d3308d634f77629b6cb4ea5c15bd28232eba232ef04da122e68ac0377f03'
        and read(root / RUN / 'result.json')['identity']
        == 'sha256:9ebfdc5b7dcbf429a06c5446f24c43d83d9ec1af2e44e719f666c3d0e8755395',
        'original pre-learner failure required')
    paths = sorted(control.glob('transition-*.json'))
    require(len(paths) == 72, 'expected 72 original campaign transitions')
    tip = 'sha256:' + CAMPAIGN
    for i, path in enumerate(paths):
        v = read(path)
        require(path.name == f'transition-{i:04d}.json' and v['previous'] == tip,
                'round025 transitions differ')
        tip = v['identity']
    require(tip == 'sha256:aa669ea2c648811033756ec207903281063e969bc6a25f77ccb6ea0120b24a3b',
            'round025 transitions differ from audited stop')
    for files, expected in ((account.glob('task-binding-*.json'),
                             'sha256:daa7029bd0586d2cdd8a2302775eb330fbbd63977e4f53bbb9de5c69f097b9df'),
                            ((account / 'storage-inventory').glob('*.json'),
                             'sha256:525566f5ad8eea3c377c4b7c65bd57b632a0c5d27381221dac3adbf96fd3871b')):
        require(
            digest([reference(root, p) for p in sorted(files)]) == expected,
            'original accounting evidence differs')
    require(
        read(account / 'budget-binding.json')['identity'] ==
        'sha256:c2450d6d941090580754079453519fe36e10bb9ab5efb13bc1e5e15705cf73a6', 'budget binding differs')
    ledger_ref = dict(path=ACCOUNT + '/budget.jsonl',
                      bytes=34581,
                      sha256='sha256:86c4c0462417f9a4c0c34c850f90a76880108066f8ffda49c50fb38c6d5eaeb1')
    events = [json.loads(line) for line in checked_file(root, ledger_ref).read_bytes().splitlines()]
    charges = {e['operation'].split('/')[0]: e for e in events if e['event'] == 'COMMIT'}
    # This exact ledger was separately reconciled: 33 pairs, no open reservation, zero learner tokens.
    require(set(charges) == set(CARRIED[2:]), 'round025 capability charges differ')
    outputs = v['state']['outputs']
    require(
        digest(outputs) == 'sha256:72b1ffc6fab3ff5ab729eb85b27ab829b06999795639ac04375562db95302e7c'
        and v['state']['tasks'] == dict(dict.fromkeys(CARRIED, 'COMPLETE'), **{TASK: 'FAILED'}),
        'audited capability roster differs')
    for cut in outputs.values():
        path = checked_file(root, cut['result'])
        for item in read(path)['metrics'].values():
            if isinstance(item, dict) and set(item) == {'bytes', 'sha256', 'path'}:
                checked_file(path.parent, item)
    return outputs, charges, ledger_ref


def _proof(root, receipt, session, journal):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot = collect_snapshot(root)
    # Exclude only this proof module to avoid a self-referential digest.
    # The new qualification still binds the complete source, including this file.
    snapshot.pop('src/oeis_learn/pipeline_validation/progressive_round025_successor.py')
    require(
        digest(snapshot) in {
            'sha256:841b6b7d3111c1f1aee356240a19f420b9173d9d54bec9d702712099e1537365',
            'sha256:aedb2608f1ceefd072e67b708f196620554881a60448b4f52c79d1772d5f3d1d',
        },
        'reviewed round026 implementation required')
    old = confined(root, SESSION)
    require(
        receipt == read(old / 'preparation-owner.json')['receipt']
        and receipt['identity'] == 'sha256:11664dc67d5bb9190cd6ccc722abe456271160fafd782c55c961c8b1c7b4134c'
        and session == read(old / 'session.json')
        and session['identity'] == 'sha256:4ef96a533de1179863b2488f1e1557d149bb255844d22fa4bdcb4cbb6ae637b6'
        and journal == read(old / 'journal.json')
        and journal['steps']['run-research/invocation-0000']['status'] == 'FAILED'
        and read(old / 'preparation-owner-proof.json')['identity']
        == 'sha256:93875a08f65437cd56f14d6cb3fdde1b7fd6b0d2e959479c7b2b975753d41fd5',
        'round025 owner or invocation differs')
    outputs, charges, ledger = reviewed(root)
    spent = math.ceil(
        sum(
            a.get('elapsed_seconds', 0) for step in journal['steps'].values()
            for a in [*step.get('history', []), step]))
    return seal(
        dict(version=VERSION,
             predecessor=receipt,
             successor=SUCCESSOR,
             reviewed_audit=AUDIT,
             original_campaign='sha256:' + CAMPAIGN,
             original_outputs=outputs,
             charges=charges,
             original_ledger=ledger,
             charged_round025_seconds=spent,
             original_costs_retained=True,
             scientific_allowance_transferred=False,
             current_source_requalification_required=True,
             replacement=TASK,
             authorized_replacements=1,
             additional_prepare_seconds=21600))


def proof(root, receipt, session, journal):
    with lease(root):
        return _proof(root, receipt, session, journal)


def remaining_setup(root):
    path = root / SUCCESSOR / 'replacement-preparation-allowance.json'
    record = read(path)
    require(record['version'] == 'progressive-operational-allowance/v1'
            and record['binding'] == dict(predecessor='sha256:' + CAMPAIGN, session=SUCCESSOR)
            and record['seconds'] == 21600 and record['active'] is None and record['attempts']
            and 0 < record['spent_seconds'] <= 21600
            and math.isclose(sum(a['charged_seconds'] for a in record['attempts']), record['spent_seconds']),
            'settled original round026 preparation allowance required')
    return 21600 - math.ceil(record['spent_seconds'])


def registration_failure(root, receipt, session, journal, ancestor):
    folder = root / SUCCESSOR
    require(session == read(folder/'session.json') and session['identity'] ==
            'sha256:f788f8ea98fab054dcb1ca64ceca94d0793f39f0aabf5edd1cc53ec1dcd0c589'
            and journal == read(folder/'journal.json') and journal['identity'] ==
            'sha256:b179ae3fae6e8c662e2be8c7ca985749434a223b1516166991e8b9d485cef5d6'
            and receipt == read(folder/'preparation-owner.json')['receipt'] and receipt['identity'] ==
            'sha256:6b4575b81579b52fb4356b6eb2b9cd97c37db3ccd96a29ffab9e364c114bf41a'
            and ancestor == read(folder/'preparation-owner-proof.json') and ancestor['identity'] ==
            'sha256:3480b895e3dea0fa9dedaf92863d66c24cd45f4cb30befc3613f849d581f1462',
            'original round026 CPU qualification failure and capability proof required')
    require(not any((folder/name).exists() for name in
                    ('run-research','assess','prepare-research/frozen/campaign.json',
                     'prepare-research/acceptance/attempts')), 'round026 advanced beyond qualification')
    stage = folder/'prepare-research/base-qualification/pipeline/stages/cpu.semantic/bigint-cpu-regressions'
    for name, size, sha in (
            ('results.xml', 469256, '57c4f53a2b246fb35ed40588235530ca66ee5950cfcd944dd826134d39c6322c'),
            ('process.json', 455, '75cbaf0feb2f71305f1870ceabcfe955ff6b5d06a3f091ce485f1a4f6b3052c7'),
            ('cleanup.json', 130, '5bbc6f7e8e31e9ef90e5669ae9c59a450446afb8e8f227844b5c5ccea0b3deb2')):
        checked_file(stage, dict(path=name, bytes=size, sha256='sha256:'+sha))
    left = remaining_setup(root)
    require(left > 120, 'round026 preparation allowance exhausted')
    return seal(dict(ancestor, version=RETRY_VERSION, predecessor=receipt, successor=RETRY,
                     original_proof=ancestor['identity'], round026_journal=journal['identity'],
                     round026_allowance=reference(root,folder/'replacement-preparation-allowance.json'),
                     charged_round026_seconds=21600-left, additional_prepare_seconds=left))


def _retry_proof(root, receipt, session, journal):
    old = root/SESSION
    ancestor = _proof(root, read(old/'preparation-owner.json')['receipt'],
                      read(old/'session.json'), read(old/'journal.json'))
    return registration_failure(root, receipt, session, journal, ancestor)


def retry_proof(root, receipt, session, journal):
    with lease(root):
        return _retry_proof(root, receipt, session, journal)


def setup_limits(setup, evidence):
    if evidence['version'] == RETRY_VERSION:
        left = evidence['additional_prepare_seconds']
        require(seal(evidence) == evidence and evidence['successor'] == RETRY
                and evidence['reviewed_audit'] == AUDIT and 120 < left < 21600
                and left + evidence['charged_round026_seconds'] == 21600,
                'bounded round027 preparation allowance required')
        return dict(setup, prepare=left, freeze=left)
    require(
        evidence['version'] == VERSION and evidence['reviewed_audit'] == AUDIT and seal(evidence) == evidence
        and evidence['additional_prepare_seconds'] == 21600, 'bounded round026 setup required')
    return dict(setup, prepare=21600, freeze=21600)


def check_inputs(root, plans, refs, evidence):

    @lru_cache(None)
    def verify(path, bytes, sha256):
        return checked_file(root, dict(path=path, bytes=bytes, sha256=sha256))

    campaign = _json(confined(root, BASE + '/frozen/campaign.json'))
    require(
        seal(campaign) == campaign and campaign['identity'] == 'sha256:' + CAMPAIGN
        and evidence['reviewed_audit'] == AUDIT, 'audited original campaign required')
    old = campaign['payload']
    require(
        previous.input_scopes(root, plans, refs, verify,
                              names=CARRIED) == previous.input_scopes(root,
                                                                      old['tasks'],
                                                                      old['references'],
                                                                      verify,
                                                                      names=CARRIED),
        'capability inputs, controls, or requests changed')
    require([p['id'] for p in plans] == [p['id'] for p in old['tasks']], 'frozen research roster changed')

    def requests(references):
        return {r['artifact']['sha256']: r['artifact'] for r in references if r['kind'] == 'policy'}

    original, current = requests(old['references']), requests(refs)
    for a, b in zip(old['tasks'], plans, strict=True):
        require(
            all(a[k] == b[k] for k in ('stage', 'slot', 'arm', 'kind', 'token_target'))
            and all(a['caps'][k] == b['caps'][k] for k in a['caps'] if k != 'disk_bytes'),
            'scientific task limits changed: ' + a['id'])
        if a['kind'] != 'training':
            continue
        before, after = [
            _json(verify(**mapping[p['policy']])) for p, mapping in ((a, original), (b, current))
        ]
        normalized = dict(
            before,
            image=after['image'],
            artifacts={
                'training':
                [s.replace(SESSION + '/', evidence['successor'] + '/', 1) for s in before['artifacts']['training']]
            },
            reservation=dict(before['reservation'], native_work=0))
        require(
            before['operation'] == 'train' and before['reservation']['native_work'] == 1000000
            and after == normalized and all(v == 0
                                            for k, v in after['reservation'].items() if k != 'elapsed_us'),
            'unexpected learner request change: ' + a['id'])
    # Recompute the retained comparison reader-cap proof using the original measured bounds.
    predecessor = read(root / previous.SESSION / 'preparation-owner-proof.json')
    comp = predecessor['comparison_projection']
    rows = _json(checked_file(root, predecessor['limit_report']))['tasks']
    generator.primary_comparison_sizes(root, dict(tasks=plans, references=refs), {r['task']: r
                                                                                  for r in rows},
                                       comp['estimates'], comp['sample_maxima'])


@contextmanager
def research_scope(root, campaign, path, store, journal):
    matches = [r['artifact'] for r in campaign['payload']['references'] if r['name'] == NAME]
    retry_path = root / RETRY / 'prepare-research/frozen/campaign.json'
    recipient = RETRY if confined(root, path) == retry_path else SUCCESSOR
    expected = root / recipient / 'prepare-research/frozen/campaign.json'
    if not matches and confined(root, path) != expected:
        yield
        return
    proof_path = recipient + '/preparation-owner-proof.json'
    require(
        confined(root, path) == expected and matches == [reference(root, proof_path)],
        'single round026 campaign required')
    with lease(root):
        predecessor, build = (SUCCESSOR, _retry_proof) if recipient == RETRY else (SESSION, _proof)
        old = root / predecessor
        evidence = read(root / proof_path)
        require(
            evidence['successor'] == recipient and evidence == build(root,
                               read(old / 'preparation-owner.json')['receipt'], read(old / 'session.json'),
                               read(old / 'journal.json')), 'round025 evidence changed')
        check_inputs(root, campaign['payload']['tasks'], campaign['payload']['references'], evidence)
        retain(
            confined(root,
                     'run-state/discovery-preparation-revisions/learner-replacement-' + CAMPAIGN + '.json',
                     exists=False), seal(dict(proof=evidence['identity'], successor=campaign['identity'])))
        previous.carry(root, store, journal, evidence, names=CARRIED, operation='carry-round025')
        status = store.state['tasks'].get(TASK)
        require(status not in {'FAILED', 'PARTIAL', 'SKIPPED'},
                'authorized diversity learner replacement already ended')
        if status is None:
            first = store.next_task()
            require(first and first['payload']['id'] == TASK,
                    'first fresh dispatch must be the original diversity learner')
        yield
