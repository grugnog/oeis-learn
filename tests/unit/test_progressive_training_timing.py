"""Real CPU update/checkpoint paths, with hardware claims kept distinct."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from oeis_learn.pipeline_validation import progressive_training_timing as t
from oeis_learn.pipeline_validation.progressive_workstation_state import read,write,Session
from oeis_learn.generalization.progressive.contracts import seal,digest,reference,load


def package_fixture(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    fixture(tmp_path/'fixture',interrupt=False)
    p=load(tmp_path/'fixture/training-input.json')
    return seal(dict(p,shards=p['shards']+[copy.deepcopy(p['shards'][0])]))


def request_fixture(root,package):
    output=root/'run-state/timing/measure-training/timing'
    write(output/'private/training-input.json',package)
    inputs=seal(dict(source=digest({}),original_inputs=digest('original service diagnostic'),
        package=reference(output,output/'private/training-input.json'),support_summary={'bounds':[]}))
    write(output/'training-timing-inputs.json',inputs)
    req=t.plan(root,output,inputs,digest('image'),root/'run-state/timing')
    return output,inputs,req


def test_real_updates_restore_and_cost_projection(tmp_path):
    package=package_fixture(tmp_path)
    output,inputs,req=request_fixture(tmp_path,package)
    before=(output/'private/training-input.json').read_bytes()
    batches=t.batches(package)
    excluded={p['id'] for p in package['layout']['probes'].values()}
    assert all(r['draw_identity'] not in excluded for _,rows in batches for r in rows)
    result=t.measure(package,output/'measurement',device='cpu',seconds=60)
    assert result['device']=='cpu' and result['status']=='PASS'
    assert len(result['updates'])==18 and len(result['checkpoint_read_us'])==5
    assert all(r['ledger']['updates']==i+1 for i,r in enumerate(result['updates']))
    assert result['updates'][-1]['ledger']['consumed']==sum(r['charged_tokens'] for r in result['updates'])
    assert all(v>0 for v in result['components'].values())
    assert result['checkpoint']['purpose']=='acceptance'
    assert result['checkpoint_files']
    report=t.report(output,inputs)
    assert report['device']=='cpu' and report['expected_total_seconds'] is None
    assert not report['executable'] and not report['scientific_budgets_changed']
    assert [r['tokens'] for r in report['horizons']]==list(t.HORIZONS)
    rows=result['updates'][1:16]
    expected=sum(r['charged_tokens'] for r in rows)*1e6/sum(r['packing_us']+r['update_us'] for r in rows)
    assert report['steady_tokens_per_second']==pytest.approx(expected)
    for row in report['horizons']:
        assert row['update_only_seconds']==pytest.approx(row['tokens']/expected)
        assert row['training_component_projection_seconds']>row['update_only_seconds']
    assert (output/'private/training-input.json').read_bytes()==before
    with pytest.raises(FileExistsError):t.measure(package,output/'measurement',device='cpu')


def test_update_failure_keeps_irreversible_debit_and_partial_report(tmp_path,monkeypatch):
    package=package_fixture(tmp_path);output,inputs,_=request_fixture(tmp_path,package)
    from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate
    def fail(*args,**kwargs):raise RuntimeError('injected update failure')
    monkeypatch.setattr(PolicyUpdate,'update',fail)
    with pytest.raises(RuntimeError,match='injected'):t.measure(package,output/'measurement',device='cpu')
    assert read(output/'measurement/debits/0/started.json')['ledger']['consumed']>0
    assert read(output/'measurement/training-timing-progress.json')['status']=='FAIL'
    assert t.report(output,inputs)['status']=='INCOMPLETE'


def test_requests_and_owner_prevent_changed_or_repeated_scope(tmp_path):
    package=package_fixture(tmp_path);output,inputs,req=request_fixture(tmp_path,package)
    assert t.plan(tmp_path,output,inputs,req['image'],tmp_path/'run-state/timing')==req
    with pytest.raises(ValueError,match='sibling'):t.plan(tmp_path,output,inputs,req['image'],tmp_path/'run-state/other')
    for changes in ({'updates':1000},{'seconds':999999},{'scope':'research'}):
        with pytest.raises(ValueError,match='closed'):t.validate_request(seal(dict(req,**changes)))


def test_stdlib_cli_and_phase_options():
    import subprocess,sys
    script=Path(__file__).resolve().parents[2]/'scripts/discovery_hip_handoff.py'
    args=[sys.executable,'-S',str(script),'--engine','podman','--session','run-state/timing',
          '--dry-run','measure-training','--service-session','run-state/services']
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['sequence']==['verify-original-service-inputs','HIP-image','training-only-update-timing','runtime-review']
    assert subprocess.run(args[:-2],capture_output=True).returncode!=0


def test_workstation_failure_still_packages_report_and_cannot_restart(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_training_timing_workstation import measure
    package=package_fixture(tmp_path);output,inputs,req=request_fixture(tmp_path,package)
    session=Session(tmp_path,'run-state/timing',dict(source=req['source']),'measure-training')
    w=Workflow(session,SimpleNamespace(phase='measure-training',service_session='run-state/service'));calls=[]
    monkeypatch.setattr(t,'gather',lambda *a:inputs)
    monkeypatch.setattr(w,'host',lambda:SimpleNamespace(engine=SimpleNamespace(facts=lambda:{}),policy={}))
    def images(targets):
        assert targets==('gpu',)
        session.journal['steps']['measure-training/image-gpu']=dict(status='COMPLETE')
        return {'gpu':req['image']}
    monkeypatch.setattr(w,'images',images)
    def stage(*args,**kwargs):
        assert kwargs['gpu'] is True;calls.append('HIP boundary')
        raise RuntimeError('injected device failure')
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.stage',stage)
    with session.own():measure(w)
    with session.own():measure(w)
    assert calls==['HIP boundary'] and session.code()==1
    assert read(output/'training-runtime-review.json')['status']=='INCOMPLETE'
    import zipfile
    with zipfile.ZipFile(session.folder/'share/measure-training.zip') as z:
        assert any(n.endswith('training-runtime-review.json') for n in z.namelist())
        assert not any('/private/' in n or '/checkpoints/' in n for n in z.namelist())


def donor_fixture(root):
    from oeis_learn.generalization.progressive.prepared_support import policy
    from oeis_learn.generalization.progressive.models import resolve_profile
    package=package_fixture(root)
    rule=policy(seed=13)
    # Explicit native-producer boundary; all constructors and file bindings below
    # are real. These test rows do not confer native or HIP authority.
    support=seal(dict(version='progressive-prepared-support/v1',policy=rule['identity'],diagnostic=False,
        shards=[[seal(dict(r,native_verified=True)) for r in shard] for shard in package['shards']],
        layout=package['layout'],refresh_updates=package['refresh_updates']))
    materials=root/'run-state/original/materials'
    write(materials/'private/support.json',support)
    summary=seal(dict(status='COMPLETE',support=support['identity'],policy=rule,diagnostic=False,bounds=[]))
    write(materials/'support-summary.json',summary)
    m=seal(dict(support=reference(root,materials/'private/support.json'),
        support_summary=reference(root,materials/'support-summary.json')))
    write(materials/'materials.json',m)
    donor=root/'run-state/services';folder=donor/'measure-services/services'
    inputs=seal(dict(materials=reference(root,materials/'materials.json'),support=m['support'],
        historical_measurements=[dict(case={'profile':resolve_profile(package['profile']['base_recipe'],dropout=0)})]))
    write(folder/'service-inputs.json',inputs)
    req=seal(dict(inputs=inputs['identity']));write(folder/'request.json',req)
    review=seal(dict(service_request=req['identity']));write(folder/'service-budget-review.json',review)
    write(folder/'producer.json',seal({'status':'PASS'}))
    files={'original-inputs':'service-inputs.json','service-plan':'request.json',
           'service-measurements':'producer.json','budget-review':'service-budget-review.json'}
    write(donor/'journal.json',seal(dict(steps={'measure-services/'+k:dict(status='COMPLETE',
        outputs=[reference(root,folder/v)]) for k,v in files.items()})))
    (donor/'session.lock').touch()
    return donor,support


def test_gather_preserves_originals_and_builds_canonical_package(tmp_path,monkeypatch):
    donor,support=donor_fixture(tmp_path)
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:{})
    before={p:p.read_bytes() for p in (tmp_path/'run-state').rglob('*.json')}
    output=tmp_path/'run-state/timing/inputs'
    inputs=t.gather(tmp_path,'run-state/services',output)
    package=load(output/'private/training-input.json')
    assert package['shards']==support['shards'] and package['layout']==support['layout']
    assert package['update_policy']['effective_batch']==16
    assert t.gather(tmp_path,'run-state/services',output)==inputs
    assert all(p.read_bytes()==raw for p,raw in before.items())
    p=donor/'measure-services/services/service-inputs.json'
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='identity'):t.gather(tmp_path,'run-state/services',output)


def test_installed_entry_rejects_foreign_source_before_device_work(tmp_path,monkeypatch):
    package=package_fixture(tmp_path);output,_,req=request_fixture(tmp_path,package)
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.source_snapshot',lambda:{'changed':'source'})
    monkeypatch.setenv('OEIS_QUALIFICATION_IMAGE',req['image'])
    with pytest.raises(ValueError,match='source/image'):t.run_inside(output)
    assert not (output/'hardware.json').exists()
