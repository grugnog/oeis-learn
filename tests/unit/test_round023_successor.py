from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

from oeis_learn.generalization.progressive import projections
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from oeis_learn.pipeline_validation import progressive_round023_successor as successor
from oeis_learn.pipeline_validation.progressive_quarantine_control import require_clean_inputs
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

HELPER = Path(__file__).resolve(
).parents[2] / 'specs/013-progressive-mathematical-discovery/validation/rehearse-round023-generator.py'


@pytest.mark.parametrize(('operation', 'channel', 'role', 'kind'),
                         [('score', 'candidates', 'scorer', 'candidate'),
                          ('publish', 'public', 'verifier', 'public_evidence')])
def test_megabyte_result_routes_through_quarantine_and_projection(tmp_path, operation, channel, role, kind):
    path = tmp_path / 'record.json'
    atomic_write(
        path,
        seal(dict(version='discovery-channel/v1', kind=kind, role=role,
                  data=dict(content='x' * (1 << 20)))))
    item = projections.describe(tmp_path, path)
    channels = {k: [] for k in REGISTRY[operation].channels}
    channels[channel].append(item)
    require_clean_inputs(tmp_path, SimpleNamespace(root=tmp_path / 'account'), channels)
    mounted = projections.project(tmp_path,
                                  tmp_path / 'view',
                                  operation,
                                  channels,
                                  registry=REGISTRY,
                                  authorized=[item.ref])
    assert next(mounted[channel].glob('*.json')).read_bytes() == path.read_bytes()


def test_large_private_prompt_still_blocked(tmp_path):
    path = tmp_path / 'record.json'
    atomic_write(
        path,
        seal(
            dict(version='discovery-channel/v1',
                 kind='visible_prompt',
                 role='generator',
                 data='x' * (1 << 20))))
    with pytest.raises(ValueError, match='bounded regular JSON'):
        projections.channel_value(tmp_path, projections.describe(tmp_path, path))


def test_adjustment_reconciles_only_disk_and_uses_observed_rehearsal(tmp_path):
    rows = [
        dict(task='g',
             operation='generate',
             disk_bytes=555837713,
             static_projection_estimate=421619985,
             dynamic_inputs=0),
        dict(task='s',
             operation='score',
             disk_bytes=128 << 20,
             static_projection_estimate=10000,
             dynamic_inputs=1)
    ]
    atomic_write(tmp_path / 'audit',
                 dict(metadata_floor_bytes=60000000, observed_gpu_runtime_bytes=77727804, tasks=rows))
    atomic_write(tmp_path / 'rehearsal', dict(sampled_peak_bytes=101000000, estimated_dynamic_envelope_bytes=dict(g=0, s=14000000)))
    extra = successor.additions(
        tmp_path, dict(limit_report=reference(tmp_path, 'audit'),
                       rehearsal=reference(tmp_path, 'rehearsal')))
    assert extra['g'] > 4592919
    caps = dict(disk_bytes=3000, tokens=42, attempts=3072)
    payload = dict(caps=caps.copy(),
                   graph=dict(nodes=[dict(id='a', caps=caps.copy())]),
                   task_plans=[dict(id=r['task'], slot='a', caps=caps.copy()) for r in rows])
    successor.expand(payload, extra)
    assert payload['caps'] == payload['graph']['nodes'][0]['caps'] == dict(
        caps, disk_bytes=3000 + sum(extra.values()))
    assert all(p['caps'] == dict(caps, disk_bytes=3000 + extra[p['id']]) for p in payload['task_plans'])
    with pytest.raises(ValueError, match='complete original'):
        successor.expand(payload, {})


def test_six_hour_approval_preserves_scope():
    evidence = seal(
        dict(version=successor.VERSION,
             reviewed_evidence=successor.REVIEWED,
             additional_prepare_seconds=21600))
    assert successor.setup_limits({}, evidence) == dict(prepare=21600, freeze=21600)
    with pytest.raises(ValueError):
        successor.setup_limits({}, seal(dict(evidence, additional_prepare_seconds=21601)))


def test_other_round_cannot_claim_grant(tmp_path):
    atomic_write(tmp_path / 'elsewhere.json', {})
    campaign = dict(payload=dict(references=[dict(name=successor.NAME, artifact={})]))
    with pytest.raises(ValueError, match='only to round024'):
        with successor.research_scope(tmp_path, campaign, 'elsewhere.json', None):
            pytest.fail('unexpected scientific dispatch')


def test_rehearsal_excludes_held_out_rows_and_binds_order():
    helper = runpy.run_path(str(HELPER))
    with pytest.raises(ValueError):
        helper['training_prompts'](seal(dict(diagnostic=True, shards=[])), 64)
    with pytest.raises(ValueError):
        helper['training_prompts'](seal(dict(diagnostic=False, shards=[])), 64)


def test_preparation_elapsed_across_retries(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    import signal
    timers = []
    monkeypatch.setattr(signal, 'setitimer', lambda which, seconds: timers.append(seconds))
    folder = tmp_path / successor.SUCCESSOR
    folder.mkdir(parents=True)
    for _ in range(2):
        with successor.preparation_window(tmp_path, SimpleNamespace(folder=folder)):
            pass
    value = read(folder / 'replacement-preparation-allowance.json')
    assert value['seconds'] == 21600 and len(value['attempts']) == 2
    assert 0 < timers[2] <= timers[0] <= 21600 - 120


def request_set(root, name, *, prompts=('a', 'b'), weight='weights', attempts=3072):
    plans, refs = [], []

    def add(label, data):
        path = root / name / label
        atomic_write(path, data)
        ref = reference(root, path)
        refs.append(dict(name=label, artifact=ref))
        return ref

    weight_ref = add('weights', {'fixture': weight})
    model = add(
        'model',
        seal(dict(role='generator', kind='checkpoint', data=dict(artifact=weight_ref,
                                                                 compatibility='same'))))
    for task, op in [('capability-census', 'census'), ('capability-language', 'audit-language'),
                     (successor.TASK, 'generate')]:
        sources = [add(task + '/' + p, dict(prompt=p))['path'] for p in prompts]
        channels = dict(visible=sources, model=[model['path']]) if op == 'generate' else dict(
            sources=sources)
        req = add('requests/' + task,
                  dict(operation=op, seconds=600, reservation=dict(attempts=attempts), artifacts=channels))
        caps = dict.fromkeys(
            ('tokens', 'draws', 'attempts', 'rejections', 'native_work', 'elapsed_us', 'archive_bytes'), 0)
        plans.append(
            dict(id=task,
                 arm='original',
                 seed=1,
                 initializer={},
                 policy=req['sha256'],
                 token_target=0,
                 caps=caps))
    return plans, refs


@pytest.mark.parametrize(
    'change',
    [{}, dict(prompts=('b', 'a')),
     dict(weight='changed'), dict(attempts=3073)])
def test_carry_and_replacement_bind_order_and_checkpoint(tmp_path, change):
    plans, refs = request_set(tmp_path, 'old')
    atomic_write(tmp_path / successor.BASE / 'frozen/campaign.json',
                 dict(payload=dict(tasks=plans, references=refs)))
    args = (*request_set(tmp_path, 'new', **change), dict(charged_generator=dict(attempts=3072)))
    if change:
        with pytest.raises(ValueError, match='inputs, settings or reservations'):
            successor.check_inputs(tmp_path, *args)
    else:
        successor.check_inputs(tmp_path, *args)


def test_recover_finished_worker_and_refuse_changed_debit(tmp_path, monkeypatch):
    import json
    helper = runpy.run_path(str(HELPER))
    run = helper['run_or_recover']
    monkeypatch.setitem(run.__globals__, 'ROOT', tmp_path)
    out = tmp_path / helper['OUT']
    worker = out / 'worker'
    role = worker / 'role'
    prompts = [seal(dict(data=dict(id=str(i)))) for i in range(64)]
    for i, prompt in enumerate(prompts):
        atomic_write(out / 'training-prompts' / f'{i}.json', prompt)
    diagnostic = 'sha256:' + '1' * 64
    task = seal(dict(payload=dict(caps=dict(disk_bytes=1 << 30))))
    scope = seal(dict(original=helper['ORIGINAL'], support=dict(sha256=helper['SUPPORT']),
                      diagnostic=diagnostic, task=task['identity'], image='immutable',
                      members=sorted(p['identity'] for p in prompts)))
    atomic_write(out / 'scope.json', scope)
    ledger = tmp_path / 'run-state/discovery-accounting' / diagnostic[7:] / 'budget.jsonl'
    ledger.parent.mkdir(parents=True)
    reserved = seal(dict(event='RESERVE', operation='generator'))
    debit = seal(dict(event='COMMIT', previous=reserved['identity'], operation='generator'))
    ledger.write_text('\n'.join(json.dumps(v) for v in (reserved, debit)) + '\n')
    atomic_write(worker / 'authority/supervisor.json', seal(dict(
        campaign=dict(identity=diagnostic), task=task, reservation=reserved, storage={})))
    atomic_write(role / 'candidates.json', dict(records={str(i): {} for i in range(64)},
                                              clock=dict(cost_complete=True, expected_portfolios=256)))
    atomic_write(role / 'evaluation.json', {'public': True})
    atomic_write(role / 'result.json', seal(dict(operation='generate', mechanical_status='COMPLETE',
        metrics={k: reference(role, role / (k + '.json')) for k in ('candidates', 'evaluation')})))
    atomic_write(worker / 'supervision/process.json', dict(returncode=0))
    atomic_write(worker / 'supervision/cleanup.json', dict(acknowledged=True, within_bound=True))
    atomic_write(out / 'workflow-limits.json', dict(two_member_scores_bytes=500, tasks=[
        dict(task='g', members=64, envelopes_bytes=100, dynamic_sources=[]),
        dict(task='s', members=64, envelopes_bytes=10, dynamic_sources=[['g', 'candidates']])]))
    run(out, {})
    receipt = (out / 'result.json').read_bytes()
    report = json.loads(receipt)
    assert report['estimated_dynamic_envelope_bytes']['s'] >= report['scoring_channel_bytes']
    assert report['projected_input_cap_flags'] == []
    run(out, {})
    assert (out / 'result.json').read_bytes() == receipt
    ledger.write_text('')
    with pytest.raises(ValueError, match='identity/size differs'):
        run(out, {})


def test_research_grant_carries_both_completed_producers_once(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    tasks = [('capability-census', 'census'), ('capability-language', 'audit-language')]
    base = tmp_path / successor.SESSION
    for filename, content in [('preparation-owner', dict(receipt={})), ('session', {}), ('journal', {})]:
        write(base / (filename + '.json'), seal(content))
    outputs = {}
    for name, operation in tasks:
        path = base / name / 'result.json'
        atomic_write(path, seal(dict(mechanical_status='COMPLETE', operation=operation)))
        outputs[name] = reference(tmp_path, path)
    proof = seal(dict(original_outputs=outputs))
    path = tmp_path / successor.SUCCESSOR / 'preparation-owner-proof.json'
    atomic_write(path, proof)
    campaign = seal(dict(payload=dict(references=[dict(name=successor.NAME, artifact=reference(tmp_path,path))],
                                      tasks=[])))
    frozen = tmp_path / successor.SUCCESSOR / 'prepare-research/frozen/campaign.json'
    atomic_write(frozen, campaign)
    lock = tmp_path / successor.CONTROL / 'run.lock'
    lock.parent.mkdir(parents=True)
    lock.touch()
    monkeypatch.setattr(successor, '_proof', lambda *args: proof)
    monkeypatch.setattr(successor, 'check_inputs', lambda *args: None)

    class Store:
        def append(self, state):
            self.state = state

        def next_task(self):
            name = tasks[len(self.state['tasks'])][0]
            return dict(identity=name, payload=dict(id=name))

        def boundary(self, task, status, invocation):
            self.state['tasks'][task['payload']['id']] = status

    store = Store()
    store.campaign = campaign
    store.state = dict(tasks={}, outputs={}, invocation='test')
    for _ in range(2):
        with successor.research_scope(tmp_path, campaign, frozen, store):
            assert store.state['tasks'] == {name: 'COMPLETE' for name, _ in tasks}
            assert {n: cut['result'] for n, cut in store.state['outputs'].items()} == outputs
    with pytest.raises(ValueError, match='original recovery publication differs'):
        with successor.research_scope(tmp_path, seal(dict(campaign, other=True)), frozen, store):
            pytest.fail('second campaign claimed replacement')


def test_primary_comparison_forecast_uses_whole_two_member_files(tmp_path):
    """Four 640-member cells fit with 50% sample headroom. Larger output still blocks."""
    publish = 'diversity-comparison'
    refs, tasks, rows, slots = [], [], {}, []

    def put(name, value, kind='policy'):
        path = tmp_path / (name + '.json')
        atomic_write(path, value)
        ref = reference(tmp_path, path)
        refs.append(dict(name=name, kind=kind, artifact=ref))
        return ref

    primary = put('private/channels/primary-portfolio',
                  seal(dict(kind='oeis_generation', role='generator', data=dict(methods=['correct'], forced=None))),
                  'prepared')
    template = put('template', dict(version='template'), 'prepared')
    for i in range(4):
        gen, score = f'generate-{i}', f'score-{i}'
        for name, op, output in [(gen, 'generate', 'evaluation'), (score, 'score', 'scores')]:
            inputs = dict(visible=['v'] * 640, model=[primary['path']]) if op == 'generate' else dict(
                score_truth=['t'] * 640, candidates=[dict(task=gen, operation='generate', output='candidates', kind='candidate')])
            policy = put(name, dict(operation=op, artifacts=inputs))['sha256']
            tasks.append(dict(id=name, policy=policy, caps=dict(attempts=5120), trial_choices=[dict(policy=policy)]))
            rows[name] = dict(members=640)
            slots.append(dict(task=name, operation=op, output=output, kind='public_evidence'))
    policy = put(publish, dict(operation='publish', artifacts=dict(public=[template['path'], *slots])))['sha256']
    tasks.append(dict(id=publish, policy=policy))
    rows[publish] = dict(operation='publish', dynamic_sources=[(v['task'], v['output']) for v in slots])
    args = tmp_path, dict(tasks=tasks, references=refs), rows, [publish]
    maxima = dict(evaluation=228968, scores=10563)
    result = successor.primary_comparison_sizes(*args, maxima)
    assert result[publish] == 459899520 + 8 * 8192 + template['bytes']
    with pytest.raises(ValueError, match='exceeds the 512 MiB'):
        successor.primary_comparison_sizes(*args, dict(maxima, evaluation=300000))
    rows['score-0']['members'] -= 1
    with pytest.raises(ValueError, match='member counts'):
        successor.primary_comparison_sizes(*args, maxima)
    rows['score-0']['members'] += 1
    tasks[0]['trial_choices'][0]['policy'] = 'other'
    with pytest.raises(ValueError, match='different request policies'):
        successor.primary_comparison_sizes(*args, maxima)


def test_unreviewed_flag_still_blocks(tmp_path):
    audit = dict(tasks=[dict(task='capability-language', envelopes_bytes=1)])
    receipt = dict(estimated_dynamic_envelope_bytes={'capability-language': 512 << 20},
                   projected_input_cap_flags=['capability-language'])
    with pytest.raises(ValueError, match='outside the reviewed'):
        successor.comparison_estimates(tmp_path, receipt, audit)


def test_compact_projection_keeps_exact_original_and_byte_accounting(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager
    from oeis_learn.generalization.progressive import storage
    from oeis_learn.generalization.progressive.contracts import load

    charged = []
    @contextmanager
    def guard(path, size):
        charged.append((path, size))
        yield
    monkeypatch.setattr(storage, 'writing', guard)
    value = seal(dict(version='discovery-channel/v1', role='verifier', kind='public_evidence',
                      data=seal(dict(record=seal(dict(records={'p': ['a', 'b']})), artifact={'sha256': 'original'}))))
    plain, compact = (tmp_path / s for s in ('plain.json', 'compact.json'))
    atomic_write(plain, value)
    atomic_write(compact, value, compact=True)
    assert plain.read_bytes() == (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    assert load(plain) == load(compact) == seal(value)
    assert compact.stat().st_size < plain.stat().st_size
    assert charged == [(p, p.stat().st_size) for p in (plain, compact)]
    with pytest.raises(ValueError, match='fresh publication'):
        atomic_write(compact, value, compact=True)


@pytest.mark.parametrize("success,complete,outcome", [
    (True, True, "success"), (True, False, "success"),
    (False, True, "failure"), (False, False, "unresolved"),
])
def test_compact_comparison_round_trips_through_publication(tmp_path, monkeypatch, success, complete, outcome):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation import progressive_outputs as outputs
    from oeis_learn.pipeline_validation.progressive_publication import decision_from_outputs

    names = ['gen', 'score', 'diversity-comparison']
    plans = [dict(id=n, slot='stage', policy='frozen') for n in names]
    campaign = dict(identity='campaign', created_at='fixed',
                    payload=dict(tasks=plans, graph=dict(nodes=[dict(id='stage', dependencies=[])])))
    task = dict(identity='consumer', purpose='research', track='synthetic_strict', payload=plans[-1])
    store = SimpleNamespace(campaign=campaign, state=dict(outputs={}, tasks={}), resolved_plan=lambda p: p)
    monkeypatch.setattr(outputs, 'record', lambda _, p, **kw: dict(identity=p['id']))
    slots = []
    generated = seal(dict(records={'p': dict(details=['full', 'candidate', 'trace'])}))
    for name, op, key in [('gen', 'generate', 'evaluation'), ('score', 'score', 'scores')]:
        value = (generated if name == 'gen' else seal(dict(candidates=gen_ref['sha256'], denominator=1,
                     outcomes={'p': {'equal_budget': dict(success=success, complete=complete, outcome=outcome)}})))
        base = tmp_path / name
        atomic_write(base / 'evidence.json', value)
        ref = reference(base, 'evidence.json')
        if name == 'gen':
            gen_ref = ref
        atomic_write(base / 'result.json', seal(dict(mechanical_status='COMPLETE', operation=op, metrics={key: ref})))
        store.state['tasks'][name] = 'COMPLETE'
        store.state['outputs'][name] = dict(task=name, operation=op, result=reference(tmp_path, base / 'result.json'))
        slots.append(dict(task=name, operation=op, output=key, kind='public_evidence'))
    template = dict(version='progressive-decision-template/v1', source_slot='s', target_slot='t',
                    roster=[dict(task='p', group='g', stratum='s')], seeds=[1], design={}, policy={},
                    remaining={}, kwargs={}, producers=[dict(arm='baseline', seed=1, generation_task='gen', scoring_task='score')])
    atomic_write(tmp_path / 'template.json', seal(dict(version='discovery-channel/v1', role='verifier', kind='public_evidence', data=template)))
    request = dict(operation='publish', artifacts=dict(public=['template.json', *slots]))
    views, _ = outputs.resolve(tmp_path, store, task, request, tmp_path / 'derived-inputs')
    mounted = projections.project(tmp_path, tmp_path / 'projection', 'publish', views, registry=REGISTRY,
                                  authorized=[d.ref for d in views['public']])
    values = [load(p) for p in mounted['public'].glob('*.json')]
    evidence = next(v for v in values if v['data'].get('operation') == 'generate')
    assert evidence['data']['record'] == generated
    assert evidence['data']['artifact']['sha256'] == gen_ref['sha256']
    assert all((tmp_path / d.ref['path']).read_bytes().count(b'\n') == 1 for d in views['public'][1:])
    row = decision_from_outputs(values)[0]['data']['rows'][0]
    assert (row['outcome'], row['complete']) == (outcome, complete)
