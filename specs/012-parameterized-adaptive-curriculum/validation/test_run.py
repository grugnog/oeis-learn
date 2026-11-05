import importlib.util
from pathlib import Path
import pytest


def helper():
    spec=importlib.util.spec_from_file_location('pac_serial',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scope_order_gates_research_and_no_gpu_before_admission():
    h=helper()
    assert h.stage_order()==['admission','qualification','calibration','acceptance']
    order=h.stage_order(research=True)
    assert order.index('admission')<order.index('baseline')<order.index('train-B')
    assert not any('D' in n for n in order)
    assert 'train-D' in h.stage_order(research=True,arms=['B','C','D'])


def test_wrapper_delegates_to_canonical_cli_with_explicit_engine_and_qualified_source():
    h=helper();calls=[]
    result=h.invoke('podman','evaluate','study.json','output',qualification='producer',extra=['--checkpoints','A=descriptor.json'],
        seconds=90000,runner=lambda argv,**kw:calls.append((argv,kw)) or {'status':'PASS'})
    assert result['status']=='PASS'
    argv,options=calls[0]
    assert argv[1:6]==['scripts/foundation_run.py','--engine','podman','generalization','evaluate']
    assert options['seconds']==90000
    with pytest.raises(ValueError):h.canonical_command('auto','train','study','out')
    with pytest.raises(ValueError):h.canonical_command('podman','train','study','out')


def test_campaign_resume_verifies_outputs_and_recovery_never_launches(tmp_path):
    import json
    from oeis_learn.pipeline_validation.identity import write,seal,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    h=helper();write(tmp_path/'inputs.json',{'source':'mechanical'})
    stages=[dict(name=name,operation='prepare' if name=='admission' else 'qualify',manifest='inputs.json',
        seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[reference(tmp_path/'inputs.json',tmp_path)],
        source=digest('source'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):
        calls.append(stage['name']);write(target/'observed.json',{'status':'PASS'})
        return dict(status='PASS',process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),
            producer=digest(stage['name']),artifacts=[reference(target/'observed.json',target)])
    one=h.execute_plan(plan,tmp_path/'one',workspace=tmp_path,runner=runner)
    assert one['status']=='PARTIAL' and calls==h.stage_order()
    calls.clear()
    two=h.execute_plan(plan,tmp_path/'two',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert two['status']=='PARTIAL' and calls==[]
    immutable=(tmp_path/'one/workflow.json').read_bytes()
    h.recover(tmp_path/'one',tmp_path/'recovered')
    assert calls==[] and (tmp_path/'one/workflow.json').read_bytes()==immutable
    (tmp_path/'one/admission/observed.json').write_text('changed')
    calls.clear();h.execute_plan(plan,tmp_path/'three',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert calls==['admission'] # identical regenerated producer permits descendant reuse
    changed=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=plan['inputs'],source=digest('new-source'),images=plan['images'])
    calls.clear();h.execute_plan(changed,tmp_path/'four',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert calls==h.stage_order()


@pytest.mark.parametrize('outcome',['FAIL','PARTIAL','BLOCKED','INCONCLUSIVE','INTERRUPTED'])
def test_failure_suppresses_descendants_and_preserves_primary(tmp_path,outcome):
    from oeis_learn.pipeline_validation.identity import digest
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='docker',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):calls.append(stage['name']);return dict(status=outcome,process=dict(outcome='COMPLETE',returncode=1),cleanup=dict(acknowledged=True,within_bound=True),producer=digest('failed'),artifacts=[])
    result=h.execute_plan(plan,tmp_path/'run',workspace=tmp_path,runner=runner)
    assert calls==['admission'] and result['status']!='PASS'
    assert result['errors'][0]['stage']=='admission' and (tmp_path/'run/report.zip').is_file()


def test_export_failure_after_primary_failure_retains_both_and_last_resort(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.identity import digest
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='docker',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    def failed(stage,target):raise RuntimeError('primary private details')
    def broken(*a):raise OSError('secondary exporter failure')
    monkeypatch.setattr(h,'recover',broken)
    result=h.execute_plan(plan,tmp_path/'failed',workspace=tmp_path,runner=failed)
    assert [e['kind'] for e in result['errors']]==['RuntimeError','OSError']
    assert 'private details' not in str(result) and (tmp_path/'failed/workflow.json').is_file()


def test_interrupted_campaign_resumes_only_unfinished_closure_and_binds_identities(tmp_path,capsys):
    from oeis_learn.pipeline_validation.identity import digest,write,seal
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper();stages=[dict(name=name,operation='prepare',manifest='inputs',seconds=3600,extra=[],requires=[] if i==0 else [h.stage_order()[i-1]]) for i,name in enumerate(h.stage_order())]
    plan=h.prepare_plan(tmp_path,engine='podman',stages=stages,inputs=[],source=digest('src'),images=[digest('cpu'),digest('gpu')])
    calls=[]
    def runner(stage,target):
        calls.append(stage['name'])
        if stage['name']=='calibration':raise KeyboardInterrupt()
        write(target/'result.json',seal(dict(status='PASS',source=plan['source'],images=plan['images'],engine=plan['engine'])))
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'result.json',target)],
            process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True))
    first=h.execute_plan(plan,tmp_path/'first',workspace=tmp_path,runner=runner)
    assert first['status']=='INTERRUPTED' and calls==['admission','qualification','calibration']
    original=(tmp_path/'first/workflow.json').read_bytes()
    def completed(stage,target):
        calls.append(stage['name']);write(target/'result.json',{'status':'PASS'})
        return dict(status='PASS',producer=digest(stage),artifacts=[reference(target/'result.json',target)],
            process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True))
    calls.clear();resumed=h.execute_plan(plan,tmp_path/'resumed',workspace=tmp_path,runner=completed,previous=tmp_path/'first')
    assert calls==['calibration','acceptance'] and (tmp_path/'first/workflow.json').read_bytes()==original
    assert all(e['binding'].startswith('sha256:') for e in resumed['receipts'])
    assert verify_archive(tmp_path/'resumed/report.zip')['software_status']=='PARTIAL'
    assert 'measured progress unavailable' in capsys.readouterr().out


def test_bootstrap_failure_packages_independent_negative_evidence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    h=helper()
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{})
    args=SimpleNamespace(output='blocked',engine='podman',fresh_control=False,bandit_only=False,research=False,seed=7)
    assert h.bootstrap_failure(args,FileNotFoundError('/private/original missing'),workspace=tmp_path)==1
    verdict=verify_archive(tmp_path/'blocked/bootstrap-failure/report.zip')
    assert verdict['software_status']=='BLOCKED' and not verdict['feature_qualified']
    import zipfile
    with zipfile.ZipFile(tmp_path/'blocked/bootstrap-failure/report.zip') as archive:
        assert all(b'/private/original' not in archive.read(n) for n in archive.namelist())


def test_failed_train_resume_requires_verified_full_state_and_exact_config(tmp_path,monkeypatch):
    import json
    from oeis_learn.pipeline_validation import generalization_commands as commands,generalization_runtime as runtime_module
    from oeis_learn.pipeline_validation.identity import digest,write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_execution import bind_compatibility
    h=helper();stage=dict(name='train-C',manifest='@acceptance')
    base=tmp_path/'base';base.mkdir()
    write(base/'bundle.json',dict(identity=digest('bundle')));write(base/'recipe.json',{})
    write(base/'study.json',dict(seed=7));write(base/'runtime.json',dict(prepared=dict(training=reference(base/'bundle.json',base)),recipe=reference(base/'recipe.json',base)))
    receipts={'acceptance':dict(resolved_manifest='base/study.json',resolved_runtime='base/runtime.json')}
    old=tmp_path/'old';root=old/'train-C/execution/training-C/role/private/learner/checkpoints';root.mkdir(parents=True)
    path=root/'generation/manifest.json';path.parent.mkdir()
    config=dict(batch_size=32)
    data=digest(dict(prepared=digest('bundle'),arm='C',batch_size=32))
    contract=digest(dict(config=config,data=data,**bind_compatibility('cuda',{})))
    write(path,dict(contract=contract));ref=reference(path,root);write(root/'generalization-latest.json',ref)
    calls=[]
    def verified(workspace,value):calls.append(value);return root,ref,[]
    monkeypatch.setattr(commands,'_resume_inputs',verified)
    monkeypatch.setattr(runtime_module,'learner_config',lambda *a,**kw:config)
    result=h._model_resume({},stage,old,tmp_path,receipts)
    assert result==str(path.relative_to(tmp_path)) and calls==[result]
    write(path,dict(contract=digest('wrong')),exclusive=False)
    with pytest.raises(ValueError):h._model_resume({},stage,old,tmp_path,receipts)
