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
    assert one['status']=='PASS' and calls==h.stage_order()
    calls.clear()
    two=h.execute_plan(plan,tmp_path/'two',workspace=tmp_path,runner=runner,previous=tmp_path/'one')
    assert two['status']=='PASS' and calls==[]
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
