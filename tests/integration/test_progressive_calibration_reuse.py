"""Canonical prepare/replay/freeze with simulated device boundaries only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from oeis_learn.generalization.progressive.contracts import load, seal, digest, reference
from oeis_learn.pipeline_validation.progressive_workstation_state import write as atomic_write
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
from oeis_learn.pipeline_validation import progressive_calibration_reuse as reuse


def old_fixture(root, monkeypatch):
    from tests.integration.test_progressive_calibration import host_inputs
    from oeis_learn.pipeline_validation import progressive_handoff as h
    from oeis_learn.pipeline_validation.progressive_commands import main
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_research_resources import policy
    from oeis_learn.pipeline_validation.progressive_calibration_storage import POLICY, retire
    host_inputs(root, monkeypatch)
    package = load(root/'learning/training-input.json')
    profile = resolve_profile(package['profile']['base_recipe'], seed=package['profile']['model']['seed'], dropout=0)
    package = seal(dict(package, profile=profile, update_policy=resolve_update_policy(profile, schedule_horizon_tokens=400000000), checkpoint_policy='hourly/v1'))
    atomic_write(root/'learning/training-input.json', package)
    key = compatibility(package)
    channel = load(root/'learning/training-channel.json')
    channel['data'].update(artifact=reference(root, 'learning/training-input.json'), compatibility=key)
    atomic_write(root/'learning/training-channel.json', seal(channel))
    p = load(root/'preparation-plan.json')
    for task in p['payload']['task_plans']:
        task['initializer']['compatibility'] = key; task['caps']['disk_bytes'] = 1 << 30
    for node in p['payload']['graph']['nodes']:
        node['caps']['disk_bytes'] = sum(t['caps']['disk_bytes'] for t in p['payload']['task_plans'] if t['slot'] == node['id'])
    for ref in p['payload']['references']:
        if ref['artifact']['path'] == 'learning/training-channel.json': ref['artifact'] = reference(root, 'learning/training-channel.json')
    q = load(root/'qualification/qualification.json')
    q['covered_capabilities'] = ['model:'+digest({k:v for k,v in profile['model'].items() if k != 'seed'}), 'language:bigint-body/v1']
    publish_q(root/'qualification', q)
    runtime = load(root/'runtime.json'); runtime['payload']['scope']['model'] = q['covered_capabilities'][0][6:]
    runtime['payload']['qualification'] = [reference(root, 'qualification/qualification.json')]
    atomic_write(root/'runtime.json', seal(runtime)); p['payload']['runtime'] = reference(root, 'runtime.json')
    for ref in p['payload']['references']:
        if ref['kind'] == 'qualification': ref['artifact'] = reference(root, 'qualification/qualification.json')
    atomic_write(root/'resource-policy.json', policy())
    p['payload']['references'].append(dict(name='research_resource_policy', role='custodian', kind='policy', artifact=reference(root, 'resource-policy.json')))
    atomic_write(root/'preparation-plan.json', p)
    plan = load(root/'calibration-plan.json'); plan['checkpoint_retention'] = POLICY
    atomic_write(root/'calibration-plan.json', seal(plan))
    b = load(root/'blueprint.json')
    for ref in b['payload']['references']:
        if ref['name'] == 'preparation_plan': ref['artifact'] = reference(root, 'preparation-plan.json')
        if ref['name'] == 'iteration_calibration_plan': ref['artifact'] = reference(root, 'calibration-plan.json')
    b = seal(b); atomic_write(root/'construction/blueprint.json', b)
    monkeypatch.setattr('oeis_learn.generalization.progressive.storage.checkpoint_size', lambda *a, **kw: (32<<20, 32))
    monkeypatch.setattr(h, 'original_authority', lambda root, ref: (load(checked_file(root, ref)), checked_file(root, ref).parent))
    from oeis_learn.generalization.qualification import FACTS
    monkeypatch.setattr('oeis_learn.pipeline_validation.generalization_qualification_io.current_facts',
        lambda *a, **kw: {k:load(Path(kw['producer_root'])/'qualification.json')[k] for k in FACTS})
    def stage(host, workspace, image, output, argv, **kwargs):
        req = load(output/'request.json'); producers = {}; retired = {}
        from oeis_learn.generalization.progressive.shared_support import restore
        for case in req['cases']:
            package = restore(output, case['artifact'])
            producer = seal(dict(version='progressive-iteration-calibration/v1', source=req['source'], image=image,
                device='actual_HIP', status='PASS', compatibility=case['compatibility'],
                maximum_target_tokens=max(len(r['tokens']) for shard in package['shards'] for r in shard),
                effective_batch=16, measurements=dict(update_us=6000000, probe_us=10, checkpoint_us=20),
                measurement_scope='TEST BOUNDARY ONLY; NOT ACTUAL HIP', checkpoint_bytes=32<<20, checkpoint_inodes=32))
            path = output/'cases'/case['compatibility'][7:]/'producer.json'; atomic_write(path, producer)
            atomic_write(path.parent/'checkpoints/generation/manifest.json', {'boundary_only':True})
            retired_ref = retire(path.parent, producer, maximum_bytes=2*req['checkpoint_bytes'])
            producers[case['compatibility']] = reference(output, path)
            retired[case['compatibility']] = reference(output, path.parent/retired_ref['path'])
        atomic_write(output/'producer.json', seal(dict(version='progressive-calibration-run/v1', request=req['identity'],
            device='actual_HIP', status='PASS', producers=producers, retirements=retired, actual_elapsed_us=100)))
        atomic_write(output/'container-inspect.json',[dict(State=dict(Running=False,OOMKilled=False,ExitCode=0))])
        atomic_write(output/'process.json', dict(returncode=0, failure=None, cleanup_failure=None))
        atomic_write(output/'cleanup.json', dict(acknowledged=True, within_bound=True, failure=None))
    monkeypatch.setattr(h, 'stage', stage)
    args = SimpleNamespace(operation='prepare', blueprint='construction/blueprint.json', qualification='qualification', output='prepared')
    assert main(args, root, lambda _: SimpleNamespace(engine=SimpleNamespace(name='docker'))) == 2
    old = load(root/'prepared/prepared.json')
    assert old['payload']['deficits'] and all(d.startswith('training_time:') for d in old['payload']['deficits'])
    source = load(root/'qualification/source.json')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_calibration_equivalence.ORIGINAL_SOURCE', digest(source))
    return b, old


def publish_q(folder, q):
    from oeis_learn.generalization.qualification import FACTS
    p = load(folder/'producer.json')
    p.update(facts={k:q[k] for k in FACTS}, capabilities=q['covered_capabilities'])
    atomic_write(folder/'producer.json', seal(p))
    q['producer']['receipt'] = reference(folder, 'producer.json')
    atomic_write(folder/'qualification.json', seal(q))


def recovery_fixture(root, monkeypatch):
    from oeis_learn.pipeline_validation.progressive_research_resources import policy
    from oeis_learn.pipeline_validation.progressive_runtime_timing import feasibility_scopes
    b, old = old_fixture(root, monkeypatch)
    import shutil
    shutil.copytree(root/'qualification', root/'next-qualification')
    source = load(root/'qualification/source.json')
    source['src/oeis_learn/pipeline_validation/progressive_research_resources.py'] = digest('HOST BOUNDARY ONLY')
    atomic_write(root/'next-qualification/source.json', source)
    q = load(root/'next-qualification/qualification.json')
    q['source_closure'] = reference(root/'next-qualification', 'source.json')
    publish_q(root/'next-qualification', q)
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot', lambda root: source)
    qref = reference(root, 'next-qualification/qualification.json')
    proof = reuse.build_proof(root, root, qref)
    atomic_write(root/'revision/calibration-reuse.json', proof)
    raw = reuse.raw_prepared(root, b)
    runtime = load(root/'runtime.json'); runtime['payload']['scope']['source'] = q['source_closure']['sha256']
    runtime['payload']['qualification'] = [qref]
    atomic_write(root/'next-runtime.json', seal(runtime))
    payload = copy.deepcopy(raw['payload']); payload.pop('blueprint'); payload['runtime'] = reference(root, 'next-runtime.json')
    for r in payload['references']:
        if r['kind'] == 'qualification': r['artifact'] = qref
    payload['references'] = [r for r in payload['references'] if r['name'] != 'research_resource_policy']
    atomic_write(root/'revision/resource-policy.json', policy(trajectory=True))
    # Only installed measurement is doubled. Reference wiring, estimator,
    # original calibration replay, preparation and freeze remain real.
    report = seal(dict(request=reference(root, 'revision/calibration-reuse.json'), cases=[
        dict(case=c, measured_tokens=10000, measured_steady_us=10000, fixed_us=10, checkpoint_us=20) for c in feasibility_scopes()]))
    atomic_write(root/'revision/trajectory-report.json', report)
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_trajectory.replay', lambda *a, **kw: report)
    for name, kind, filename in [('research_resource_policy', 'policy', 'resource-policy.json'),
                                ('calibration_reuse', 'policy', 'calibration-reuse.json'), ('training_trajectory', 'source', 'trajectory-report.json')]:
        payload['references'].append(dict(name=name, role='custodian', kind=kind, artifact=reference(root, root/'revision'/filename)))
    atomic_write(root/'revision/preparation-plan.json', dict(version='progressive-preparation-plan/v1', payload=payload))
    revised = copy.deepcopy(b)
    for ref in revised['payload']['references']:
        if ref['name'] == 'preparation_plan': ref['artifact'] = reference(root, 'revision/preparation-plan.json')
    revised = seal(revised); atomic_write(root/'revision/blueprint.json', revised)
    return b, old, revised, proof


def test_canonical_reuse_preserves_originals_and_freezes_without_device_calibration(tmp_path, monkeypatch, capsys):
    from oeis_learn.pipeline_validation.progressive_commands import main
    from oeis_learn.pipeline_validation import progressive_handoff as h
    from oeis_learn.pipeline_validation.progressive_calibration import validate_prepared
    oldb, old, b, proof = recovery_fixture(tmp_path, monkeypatch)
    originals = {p:p.read_bytes() for p in (tmp_path/'prepared').rglob('*') if p.is_file()}
    cost = checked_file(tmp_path, reuse.named(old, 'iteration_calibration_cost')); original_cost = cost.read_bytes()
    monkeypatch.setattr(h, 'stage', lambda *a, **kw: pytest.fail('calibration must not execute again'))
    args = SimpleNamespace(operation='prepare', blueprint='revision/blueprint.json', qualification='next-qualification', output='next-prepared')
    assert main(args, tmp_path, lambda _: object()) == 0
    prepared = load(tmp_path/'next-prepared/prepared.json')
    assert prepared['payload']['deficits'] == []
    assert reuse.named(prepared, 'iteration_calibration_cost') == reuse.named(old, 'iteration_calibration_cost')
    assert [t['iteration_budget'] for t in prepared['payload']['task_plans']] == [t['iteration_budget'] for t in old['payload']['task_plans']]
    validate_prepared(tmp_path, b, prepared)
    assert main(SimpleNamespace(operation='freeze', blueprint='revision/blueprint.json', prepared='next-prepared',
        qualification='next-qualification', output='next-frozen'), tmp_path, None) == 0
    assert cost.read_bytes() == original_cost and all(p.read_bytes() == v for p,v in originals.items())
    assert main(args,tmp_path,lambda _:pytest.fail('completed metadata must be replayed without device work'))==0
    changed = copy.deepcopy(prepared); changed['payload']['task_plans'][0]['iteration_budget']['update_us'] -= 1
    with pytest.raises(ValueError, match='publication differs'): validate_prepared(tmp_path, b, seal(changed))
    # Rewritten original metadata cannot be laundered by the new blueprint.
    path = checked_file(tmp_path, proof['batch']); path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError): validate_prepared(tmp_path, b, prepared)


@pytest.mark.parametrize('change', ['source', 'hardware', 'runtime', 'request', 'cost', 'allowance', 'retirement', 'input'])
def test_reuse_rejects_changed_authority_or_original_debit(tmp_path, monkeypatch, change):
    _, old, b, proof = recovery_fixture(tmp_path, monkeypatch)
    if change == 'allowance':
        path=next((tmp_path/'run-state/discovery-calibration').glob('*/preparation-allowance.json'));value=load(path)
        value['spent_seconds']=0;atomic_write(path,seal(value))
    elif change == 'retirement':
        batch=load(checked_file(tmp_path,proof['batch']));rref=next(iter(batch['retirements'].values()))
        path=checked_file(tmp_path,proof['batch']).parent/rref['path'];path.write_text('{}')
    elif change == 'input':
        req=load(checked_file(tmp_path,proof['request']));path=checked_file(tmp_path,proof['request']).parent/req['cases'][0]['artifact']['path'];path.write_text('{}')
    elif change == 'cost':
        checked_file(tmp_path, reuse.named(old, 'iteration_calibration_cost')).write_text('')
    elif change == 'request':
        path = checked_file(tmp_path, proof['request']); value = load(path); value['image'] = digest('other'); atomic_write(path, seal(value))
    elif change == 'hardware':
        monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.validate_device_proofs', lambda *a, **kw: ({'hardware':{}}, []))
    else:
        qpath = tmp_path/'next-qualification/qualification.json'; q = load(qpath)
        if change == 'source':
            source = load(qpath.parent/'source.json'); source['uv.lock'] = digest('changed')
            atomic_write(qpath.parent/'source.json', source); q['source_closure'] = reference(qpath.parent, 'source.json')
        else: q['runtime']['driver'] = 'changed'
        publish_q(qpath.parent, q)
    with pytest.raises(ValueError): reuse.assemble(tmp_path, b, reuse.raw_prepared(tmp_path, b), tmp_path/'next-prepared')
