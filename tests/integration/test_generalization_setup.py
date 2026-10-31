import json
import pytest


def setup(root,monkeypatch):
    from tests.integration.test_generalization_inputs import legacy_inputs,import_uniform
    from oeis_learn.pipeline_validation.generalization_setup import assemble_acceptance
    from oeis_learn.experiments import source_snapshot
    import shutil
    from types import SimpleNamespace
    # Tiny synthetic weights exercise assembly/authority, not actual disk capacity.
    # The production-sized prospective reservations must not depend on pytest tmpfs.
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(total=128<<30,used=64<<30,free=64<<30))
    legacy_inputs(root);import_uniform(root,bundle='run.zip',metadata='metadata.zip',weights='train',output='imported')
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:dict(version='fixture-snapshot'))
    return assemble_acceptance(root,imported='imported',images=['sha256:'+'1'*64,'sha256:'+'2'*64],output='acceptance')


def test_generated_session_prepares_with_actual_isolation_and_pending_dependency(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_session import load_session
    from oeis_learn.pipeline_validation.generalization_runtime import validate_execution,publish_runtime,learner_config
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_study
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.synthesis.reference import evaluate
    result=setup(tmp_path,monkeypatch)
    session,children=load_session(tmp_path,result['session']['manifest'])
    assert session['version']=='generalization-acceptance-session/v2'
    for name,(base,study,runtime,_,_) in children.items():
        validate_execution(base,study,runtime)
        def native(source,indices):
            r=evaluate(source,indices);return dict(outcome=r.outcome,values=list(r.values))
        prep=validate('preparation',json.loads((base/'preparation.json').read_text()))
        destination=tmp_path/'prepared'/name
        got=prepare_study(base,prep,study,output=destination,execute=native)
        assert got['status']=='PASS',got
        s,r=publish_runtime(base,destination,study,runtime);plan=validate_execution(destination,s,r)
        if name=='parameter-targets':assert plan['target_pairs_parameterized']==4
        if name=='refresh':
            from oeis_learn.generalization.contracts import verify_artifact
            from oeis_learn.generalization.budgets import charge_tokens
            rows=[row for ref in r['training']['refreshed']['shards'] for row in json.loads(verify_artifact(destination,ref))]
            assert charge_tokens([x['tokens'] for x in rows[:8]],bos=1,pad=0)<=s['budgets']['target_tokens']
        if name=='replay-resume':
            assert plan['initializer'] is None
            with pytest.raises(ValueError,match='unresolved'):learner_config(s,r,plan['recipe'],'continued_sft',s['seeds'][0])
    from tests.integration.test_generalization_acceptance import helper
    h=helper();h.ROOT=tmp_path
    assert h.main(['--engine','docker','--manifest',result['session']['manifest'],'--output','run-state/new','--dry-run'])==0
    assert not (tmp_path/'run-state/new').exists()


def test_new_selection_resolves_only_its_original_session_dependency(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_session import load_session
    from oeis_learn.pipeline_validation.generalization_runtime import publish_runtime,validate_execution
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_study,reference
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.pipeline_validation.generalization_selection_io import publish_selected
    from oeis_learn.pipeline_validation import generalization_roles, generalization_role, generalization_workflow
    from oeis_learn.generalization.role_contracts import Role
    result=setup(tmp_path,monkeypatch);session,children=load_session(tmp_path,result['session']['manifest'])
    prepared={'cases':{}};resolved={}
    for name,(base,s,r,_,_) in children.items():
        destination=tmp_path/'run/prepare'/name
        def native(src,indices):
            x=evaluate(src,indices);return dict(outcome=x.outcome,values=list(x.values))
        prep=validate('preparation',json.loads((base/'preparation.json').read_text()))
        assert prepare_study(base,prep,s,output=destination,execute=native)['status']=='PASS'
        s,r=publish_runtime(base,destination,s,r);resolved[name]=(destination,s,r)
        prepared['cases'][name]=dict(manifest=reference(destination/'study.json',tmp_path),runtime=reference(destination/'runtime.json',tmp_path))
    base,source,runtime=resolved['retained-eval']
    available=generalization_workflow._checkpoints(base,runtime['evaluation']['checkpoints']);checkpoint=available[0][0]
    candidates=[dict(checkpoint=c,selections=source['source_closure']) for c,_ in available]
    behavior=[dict(id=c['identity'],tokens=c['tokens'],role='selection',structural=({'s':'f'},{'s':False}),parameter=({'p':'g'},{'p':False})) for c,_ in available]
    decision=seal(dict(version='generalization-checkpoint-selection/v1',status='PASS',scientific_status='UNMEASURED',targets=source['selection']['cohort'],candidates=candidates,
        measurements=[dict(checkpoint=b['id'],loss=1.,behavior=b) for b in behavior],best_loss=checkpoint['identity'],selected=checkpoint['identity'],eligible=[c['identity'] for c,_ in available],reason='selected'))
    def invoke(host,image,workspace,name,artifacts,**kw):
        out=workspace/kw['output']/'role';out.mkdir(parents=True)
        return generalization_role.HANDLERS[(Role(name),kw['operation'])](kw['source_roots'],out,kw['settings'],seconds=kw['seconds'])
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    case=tmp_path/'run/retained-eval/case'
    def evaluated(*args,**kw):
        publication=publish_selected(None,tmp_path,case/'private/evaluation/initializer',base,source,runtime,decision,available)
        return dict(status='PASS',initializer=publication,selection=decision)
    monkeypatch.setattr(generalization_workflow,'evaluate',evaluated)
    from oeis_learn.pipeline_validation.generalization_acceptance import run_case
    qualification=digest('original-qualification')
    proof=run_case('retained-eval',None,tmp_path,case,base,source,runtime,{},qualification=qualification,evidence_root=tmp_path/'run')
    assert proof['status']=='PASS',proof
    stage=seal(dict(version='generalization-session-result/v1',session=session['identity'],operation='acceptance',status='PASS',result=proof))
    write(case.parent/'result.json',stage)
    from oeis_learn.pipeline_validation.generalization_bootstrap import resolve
    replay=tmp_path/'run/replay-resume';replay.mkdir()
    from oeis_learn.generalization import contracts
    from oeis_learn.pipeline_validation import generalization_commands
    original_shape=contracts.shape
    validations=[]
    def host_shape(*args,**kw):raise ModuleNotFoundError('schema dependency unavailable on stdlib host')
    def container_validate(*args,**kw):
        validations.append(kw['operation'])
        with monkeypatch.context() as container:
            container.setattr(contracts,'shape',original_shape)
            # This authority fixture uses fake weights and a minimal legacy
            # recipe, not a constructible tensor model. Keep provenance real.
            from oeis_learn.pipeline_validation import generalization_training
            container.setattr(generalization_training,'storage_preflight',lambda *a:{'fixture_only':True})
            return invoke(*args,**kw)
    with monkeypatch.context() as scoped:
        scoped.setattr(contracts,'shape',host_shape)
        scoped.setattr(generalization_commands,'project_and_invoke',container_validate)
        root,s,r,plan=resolve(tmp_path,replay,*resolved['replay-resume'],host=None,session=session,children=children,prepared=prepared,qualification=qualification)
        assert validations==['validate-runtime']
        assert plan['initializer']['checkpoint']==checkpoint['identity']
        scoped.setattr(generalization_commands,'project_and_invoke',lambda *a,**kw:dict(status='FAIL',error='initializer rejected'))
        failed=tmp_path/'run/failed-validation'
        with pytest.raises(ValueError,match='initializer rejected'):
            resolve(tmp_path,failed,*resolved['replay-resume'],host=None,session=session,children=children,prepared=prepared,qualification=qualification)
        assert not (failed/'bootstrap.json').exists()
    assert validate_execution(root,s,r)['initializer']['checkpoint']==checkpoint['identity']
    assert json.loads((replay/'bootstrap.json').read_text())['source_study']==source['identity']
    changed=dict(stage,session=digest('other-session'));write(case.parent/'result.json',seal(changed),exclusive=False)
    with pytest.raises(ValueError,match='dependency'):
        resolve(tmp_path,tmp_path/'run/bad',*resolved['replay-resume'],host=None,session=session,children=children,prepared=prepared,qualification=qualification)


@pytest.mark.parametrize('free_bytes,exit_code', [(64<<30,0),(3<<29,2)])
def test_setup_cli_uses_only_stdlib_and_inspects_both_image_tags(tmp_path,free_bytes,exit_code):
    from tests.integration.test_generalization_inputs import legacy_inputs,import_uniform
    import os,subprocess,sys
    from pathlib import Path
    legacy_inputs(tmp_path);import_uniform(tmp_path,bundle='run.zip',metadata='metadata.zip',weights='train',output='imported')
    code='''import shutil
from types import SimpleNamespace
shutil.disk_usage=lambda _: SimpleNamespace(total=128<<30,used=(128<<30)-FREE_BYTES,free=FREE_BYTES)
from oeis_learn import container_runtime
from oeis_learn.experiments import source_snapshot
from oeis_learn.pipeline_validation.generalization_inputs import main
class Engine:
 def __init__(self,name): assert name=='podman'
 def image(self,tag): return {'cpu-tag':'sha256:'+'1'*64,'gpu-tag':'sha256:'+'2'*64}[tag]
container_runtime.Engine=Engine
source_snapshot.collect_snapshot=lambda root: {'version':'fixture-snapshot'}
raise SystemExit(main(['acceptance','--engine','podman','--imported','imported','--cpu-image','cpu-tag','--gpu-image','gpu-tag','--output','assembled'],workspace='''+repr(str(tmp_path))+'))'
    code=code.replace('FREE_BYTES',str(free_bytes))
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()))
    result=subprocess.run([sys.executable,'-S','-c',code],env=env,text=True,capture_output=True)
    assert result.returncode==exit_code,result.stderr
    if exit_code:
        assert 'insufficient storage reservation' in result.stderr
        assert not (tmp_path/'assembled/generalization-acceptance.json').exists()
        return
    report=json.loads(result.stdout)
    assert report['status']=='ASSEMBLED' and report['qualification']=='NOT_RUN'
    assert report['images']==['sha256:'+'1'*64,'sha256:'+'2'*64]


def test_prepared_session_load_still_rejects_insufficient_observed_space(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_session import load_session
    result=setup(tmp_path,monkeypatch)
    path=tmp_path/result['session']['manifest'];original=path.read_bytes()
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(total=2<<30,used=1<<29,free=3<<29))
    with pytest.raises(ValueError,match='insufficient storage reservation'):
        load_session(tmp_path,result['session']['manifest'])
    assert path.read_bytes()==original
