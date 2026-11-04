import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fresh_pilot_training_probe as m
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference


@pytest.mark.parametrize('failure',[None,'prepare','train-static','proof','initialization','forecast','KeyboardInterrupt'])
def test_probe_canonical_order_failure_packaging_and_no_research(tmp_path,monkeypatch,failure):
    import fresh_pilot_smoke,foundation_local
    from oeis_learn.pipeline_validation import generalization_inputs as inputs,generalization_runtime as runtime,generalization_acceptance as acceptance,generalization_lifecycle as lifecycle
    monkeypatch.setattr(m.api,'ROOT',tmp_path)
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda *a:{})
    (tmp_path/'old').mkdir();qualified=tmp_path/'qualified';write(qualified/'qualification.json',{})
    write(qualified/'feature.json',{})
    import fresh_pilot_resume
    monkeypatch.setattr(fresh_pilot_resume,'disk_preflight',lambda *a:{})
    monkeypatch.setattr(m,'original_job',lambda *a:({},'old-receipt'))
    monkeypatch.setattr(m,'qualified',lambda *a:(qualified,dict(images=['cpu','gpu'])))
    protocol=seal(dict(version='fresh-pilot-training-probe/v1',first_shard_tokens=16000,checkpoint_policy='hourly/v1',research_milestones=3))
    monkeypatch.setattr(fresh_pilot_smoke,'reduced_blueprint',lambda *a,**kw:({},protocol))
    monkeypatch.setattr(inputs,'assemble_study',lambda *a:dict(manifest='study.json'))
    monkeypatch.setattr(runtime,'load',lambda *a:(tmp_path,dict(identity='study'),dict(storage={})))
    monkeypatch.setattr(m,'verify_plan',lambda *a:None)
    monkeypatch.setattr(foundation_local,'Host',lambda *a:None)
    monkeypatch.setattr(lifecycle,'guard',lambda *a:nullcontext())
    calls=[]
    def invoke(argv,logs,seconds):
        operation=argv[argv.index('generalization')+1];calls.append(operation)
        if failure=='KeyboardInterrupt':raise KeyboardInterrupt()
        output=tmp_path/argv[argv.index('--output')+1];stage=output.name
        result=dict(status='PASS')
        if operation=='prepare':result['manifest']='prepared.json'
        if operation=='train':
            windows=[dict(updates=8,tokens=8000,seconds=2,through_updates=n) for n in (12,20,28)]
            training=dict(status='PASS',tokens=32000,charged_tokens=32000,discarded_tokens=0,checkpoint_policy='hourly/v1',pending_tokens=0,updates=32,
                timing=dict(version='generalization-training-timing/v1',windows=windows,invocation_seconds=60,checkpoint_policy='hourly/v1',max_checkpoint_seconds=1.,checkpoint_calls=2,journal_writes=65))
            if failure=='forecast':training['timing']['windows'][0]['seconds']=4
            result['runs']=[dict(root=stage,result=training)]
            (output/'training'/stage).mkdir(parents=True)
        status='FAIL' if failure==stage else 'PASS'
        write(output/'result.json',seal(dict(status=status,result=result)))
        return 0 if status=='PASS' else 1
    monkeypatch.setattr(m.api,'invoke',invoke)
    monkeypatch.setattr(m.api,'latest_checkpoint',lambda learner,*a,**kw:(dict(initialization=str(learner) if failure=='initialization' else 'same'),None))
    monkeypatch.setattr(acceptance,'training_proof',lambda *a:dict(status='FAIL' if failure=='proof' else 'PASS'))
    code=m.main(['--prepared-run','old','--output','probe','--qualified','qualified'])
    record=json.loads((tmp_path/'probe/result.json').read_text())
    expected='PASS' if failure is None else 'BLOCKED' if failure=='forecast' else 'INTERRUPTED' if failure=='KeyboardInterrupt' else 'FAIL'
    assert record['status']==expected and code==m.api.EXIT_CODES[expected],record.get('error')
    assert record['scientific_status']=='UNMEASURED' and not record['automatic_research']
    assert calls==(['preflight'] if failure=='KeyboardInterrupt' else ['preflight','prepare'] if failure=='prepare' else
        ['preflight','prepare','train'] if failure in {'train-static','proof'} else ['preflight','prepare','train','train'])
    with zipfile.ZipFile(tmp_path/'probe/report.zip') as z:
        assert sorted(z.namelist())==['inventory.json','result.json']
        assert json.loads(z.read('result.json'))==record
        assert json.loads(z.read('inventory.json'))['files']==[reference(tmp_path/'probe/result.json',tmp_path/'probe')]


def test_native_plan_rejects_selection_drift_before_training(tmp_path):
    from oeis_learn.decoder.bigint_codec import encode_body
    from fresh_pilot_smoke import batch_charges
    rows=[dict(source='(program (return (int 0)))',tokens=encode_body('(program (return (int 0)))')) for _ in range(256)]
    write(tmp_path/'rows.json',rows);ref=reference(tmp_path/'rows.json',tmp_path)
    costs=batch_charges(rows,256)
    protocol=seal(dict(version='fresh-pilot-training-probe/v1',training_sources=[digest([r['source'] for r in rows])]*2,
        batch_charges=[costs,costs],target_tokens=m.budget([costs,costs]),minimum_updates=24,first_shard_tokens=sum(costs)))
    study=dict(budgets=dict(target_tokens=protocol['target_tokens']))
    runtime=dict(training={a:dict(batch_size=16,shards=[ref]*n) for a,n in [('static',1),('refreshed',2)]})
    m.verify_plan(tmp_path,study,runtime,protocol)
    rows[0]['tokens']=[1];write(tmp_path/'rows.json',rows,exclusive=False)
    changed=reference(tmp_path/'rows.json',tmp_path);runtime['training']['static']['shards']=[changed]
    with pytest.raises(ValueError,match='encoding'):m.verify_plan(tmp_path,study,runtime,protocol)


def test_failure_package_recovers_only_bounded_timing_metadata(tmp_path):
    learner=tmp_path/'research/train-static/training/run/execution/role/private/learner'
    write(learner/'timing.json',dict(version='generalization-training-timing/v1',windows=[]))
    write(learner/'secret.json',dict(weights='NEVER SHARE'))
    record=dict(status='FAIL')
    path=m.package(tmp_path,record)
    with zipfile.ZipFile(path) as z:
        assert b'NEVER SHARE' not in z.read('result.json')
        assert json.loads(z.read('result.json'))['partial_timings']['static']['windows']==[]


@pytest.mark.parametrize('defect',[None,'source','resources','image','seal','payload'])
def test_reused_qualification_rejects_stale_or_changed_authority_before_learning(tmp_path,monkeypatch,defect):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn import container_runtime
    from oeis_learn.pipeline_validation import generalization_setup
    folder=tmp_path/'qualified';write(folder/'source.json',{})
    authority=seal(dict(version='generalization-qualification/v2',source_closure=reference(folder/'source.json',folder),
        resources=dict(memory_bytes=64<<30,cpus=4),images=['sha256:'+'a'*64,'sha256:'+'b'*64]))
    if defect=='resources':authority=seal(dict(authority,resources=dict(memory_bytes=32<<30,cpus=4)))
    if defect=='seal':authority['identity']='bad'
    write(folder/'qualification.json',authority)
    if defect=='payload':write(folder/'source.json',{'changed':True},exclusive=False)
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda *a:{'changed':True} if defect=='source' else {})
    monkeypatch.setattr(container_runtime,'Engine',lambda *a:SimpleNamespace(image=lambda image:'different' if defect=='image' else image))
    monkeypatch.setattr(generalization_setup,'assemble_acceptance',lambda *a,**kw:pytest.fail('reuse must not repeat acceptance'))
    args=SimpleNamespace(qualified='qualified',engine='podman')
    if defect:
        with pytest.raises(ValueError):m.qualified(tmp_path,args,tmp_path/'probe')
    else:assert m.qualified(tmp_path,args,tmp_path/'probe')==(folder,authority)


@pytest.mark.parametrize('defect',[None,'recipe','selected','inventory'])
def test_original_candidates_require_original_inventory_and_recipe(tmp_path,monkeypatch,defect):
    from oeis_learn.pipeline_validation import generalization_runtime
    old=tmp_path/'old';prepared=old/'research/prepare';base=prepared/'preparation/role/private/prepared'
    write(base/'study.json',{})
    refs={}
    for name in ('recipe','descriptors','exclusions'):
        write(base/(name+'.json'),{'original':name})
        write(old/(name+'.json'),{'original':name})
        refs[name]=reference(base/(name+'.json'),base)
    selected=base/'cells/protected/private/selected.json';write(selected,[])
    cfg=dict(recipe='old/recipe.json',descriptors='old/descriptors.json',exclusions='old/exclusions.json',
        cells=[dict(name='protected',role='selection')],initializer=None,seeds=[20261016])
    write(old/'blueprint.json',cfg);write(old/'design.json',seal({}))
    job=seal(dict(version='fresh-pilot-smoke-job/v1',engine='podman',qualified='old/qualified',manifest=str((base/'study.json').relative_to(tmp_path)),
        blueprint=reference(old/'blueprint.json',tmp_path),design=reference(old/'design.json',tmp_path)))
    write(old/'private/smoke-job.json',job)
    inventory=[reference(p,prepared) for p in prepared.rglob('*.json')]
    if defect=='inventory':inventory=[r for r in inventory if not r['path'].endswith('selected.json')]
    receipt=seal(dict(version='generalization-command-result/v2',status='PASS',stage='prepare',original=None,
        inventory=inventory,result=dict(manifest=job['manifest'])))
    write(prepared/'result.json',receipt)
    monkeypatch.setattr(generalization_runtime,'load',lambda *a:(base,dict(data={k:refs[k] for k in ('descriptors','exclusions')}),dict(preparation=None,recipe=refs['recipe'])))
    if defect=='recipe':write(old/'recipe.json',{'substituted':True},exclusive=False)
    if defect=='selected':write(selected,['changed'],exclusive=False)
    if defect:
        with pytest.raises(ValueError):m.original_job(tmp_path,'old','podman')
    else:assert m.original_job(tmp_path,'old','podman')==(job,receipt['identity'])
