"""Trajectory arithmetic and installed evidence replay, with explicit HIP doubles."""
import copy
from pathlib import Path
import pytest

from oeis_learn.generalization.progressive.contracts import seal, digest, reference
from oeis_learn.pipeline_validation import progressive_trajectory as t
from oeis_learn.pipeline_validation import progressive_runtime_timing as runtime
from oeis_learn.pipeline_validation.progressive_workstation_state import write
from tests.unit.test_progressive_training_timing import package_fixture


def fixture(root, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_handoff as h
    package = package_fixture(root)
    cases = [dict(compatibility=digest('case'), package=package)]
    q = dict(images=[digest('cpu'), digest('gpu')], engine_security=dict(engine='podman'))
    monkeypatch.setattr(h, 'original_authority', lambda *a: (q, root))
    monkeypatch.setattr(h, 'validate_device_proofs', lambda *a, **kw: (dict(hardware={'boundary_only':True}), []))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot', lambda root: {})
    write(root/'qualification.json', q); qref = reference(root, root/'qualification.json')
    folder = root/'trajectory'; req = t.plan(root, folder, cases, qref)
    write(folder/'hardware.json', {'boundary_only':True})
    write(folder/'process.json', dict(returncode=0, cleanup_failure=None, failure=None))
    write(folder/'cleanup.json', dict(acknowledged=True, within_bound=True, failure=None))
    items = []
    for i, case in enumerate(req['cases']):
        timing = dict(windows=[dict(tokens=10000, elapsed_us=5000000, cold=False)], loop_us=5000000,
            maximum_shard=1, components=dict(checkpoint_save=dict(calls=2, inclusive_us=2000000)))
        segments = [dict(timing=timing, process=dict(elapsed_us=7000000),
            result=dict(mechanical_status='COMPLETE', metrics=dict(initialization='fresh_random' if j==0 else 'resume'))) for j in range(2)]
        path = folder/'cases'/str(i)/'producer.json'
        write(path, seal(dict(case=case, request=req['identity'], status='PASS', device='actual_HIP',
            package=runtime.case_package(package, case)['identity'], segments=segments,
            measurement_scope='TEST BOUNDARY ONLY; NOT ACTUAL HIP')))
        (path.parent/'diagnostics.zip').write_bytes(b'TEST BOUNDARY ONLY')
        items.append(dict(index=i, status='PASS', producer=reference(folder, path), diagnostics=reference(folder, path.parent/'diagnostics.zip')))
    write(folder/'runtime-cases.json', seal(dict(request=req['identity'], status='PASS', device_block=None, cases=items)))
    return folder, cases, qref


def test_complete_current_scope_replay(tmp_path, monkeypatch):
    folder, cases, q = fixture(tmp_path, monkeypatch)
    report = t.replay(tmp_path, folder, cases=cases, qualification=q)
    assert len(report['cases']) == 13
    assert report['cases'][0]['fixed_us'] == 2000000
    assert report['cases'][0]['checkpoint_us'] == 1000000
    assert runtime.validate_request(__import__('json').loads((folder/'request.json').read_text()))


@pytest.mark.parametrize('change', ['missing', 'cpu', 'request', 'no_restore', 'cleanup', 'hardware', 'support'])
def test_replay_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, change):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    folder, cases, q = fixture(tmp_path, monkeypatch)
    batch = read(folder/'runtime-cases.json')
    if change == 'cleanup': write(folder/'cleanup.json', dict(acknowledged=False, within_bound=True))
    elif change == 'hardware': write(folder/'hardware.json', {'boundary_only':False})
    elif change == 'support': cases[0]['package'] = seal(dict(cases[0]['package'], layout={}))
    elif change == 'missing':
        batch['cases'].pop(); write(folder/'runtime-cases.json', seal(batch))
    else:
        path = folder/batch['cases'][0]['producer']['path']; p = read(path)
        if change == 'cpu': p['device'] = 'cpu'
        elif change == 'request': p['request'] = digest('other')
        else: p['segments'][1]['result']['metrics']['initialization'] = 'fresh_random'
        write(path, seal(p)); batch['cases'][0]['producer'] = reference(folder, path)
        write(folder/'runtime-cases.json', seal(batch))
    with pytest.raises(ValueError): t.replay(tmp_path, folder, cases=cases, qualification=q)


def test_trajectory_counts_probes_once_keeps_fourfold_guard_and_hourly_snapshots(tmp_path):
    p = runtime.case_package(package_fixture(tmp_path), runtime.feasibility_scopes()[0])
    producer = dict(compatibility=digest('c'), measurements=dict(update_us=100, probe_us=999999999, checkpoint_us=1000000))
    rows = [dict(case=c, measured_tokens=10000, measured_steady_us=5000000,
                 fixed_us=2000000, checkpoint_us=1000000) for c in runtime.feasibility_scopes()]
    context = t.cost_context([dict(package=p)], [producer], dict(cases=rows))
    args = dict(delta=1000000, target=1000000, available=4*3600*1000000, arm='static')
    value = t.estimate(p, producer, context, **args)
    assert value['expected_work_us'] == 504000000
    assert value['estimated_work_us'] == 4*508000000
    assert value['maximum_snapshots'] == 6
    slower = copy.deepcopy(producer); slower['measurements']['update_us'] *= 2
    assert t.estimate(p, slower, context, **args)['expected_work_us'] == 1004000000
    context['trajectory']['cases'][-1]['measured_steady_us'] *= 2
    assert t.estimate(p, producer, context, **dict(args, arm='refreshed'))['expected_work_us'] == 1004000000
    # A genuine slow trajectory still exceeds the task cap.
    context['trajectory']['cases'][0]['measured_steady_us'] *= 100
    assert t.estimate(p, producer, context, **args)['estimated_work_us'] > args['available']


def test_legacy_resource_policy_and_estimate_unchanged(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_resources import policy, estimate_training
    p = runtime.case_package(package_fixture(tmp_path), runtime.feasibility_scopes()[0])
    original = policy()
    assert original['version'].endswith('/v3') and policy(trajectory=True)['version'].endswith('/v4')
    m = dict(update_us=5, checkpoint_us=10, probe_us=7)
    r = estimate_training(p, m, delta=1000000, target=1000000, available=4*3600*1000000)
    assert r['estimated_work_us'] == 4*(r['maximum_updates']*5+6*10+2*((r['maximum_updates']+7)//8)*7)
