"""Actual CPU tensors/checkpoints, with explicit device/data boundary adapters.

This exercises the complete handler, not HIP or production data/performance proof.
"""
import copy
import json
from types import SimpleNamespace
import pytest


def test_installed_rehearsal_all_three_real_learners_and_in_block_resume(tmp_path,monkeypatch):
    import torch
    from tests.integration.test_generalization_curriculum import prepared,models
    from oeis_learn.pipeline_validation import generalization_pac_rehearsal as r0,generalization_role as role
    from oeis_learn.pipeline_validation import generalization_curriculum as curriculum,generalization_evaluation as evaluation
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.identity import digest
    batches,bundle=prepared(tmp_path)
    recipe=resolve(dict(version='bigint-bootstrap/v1'));recipe['learning']['learning_rate']=.001
    source=digest(source_snapshot());encoder,decoder=models()
    initializer=curriculum.initial_weights_identity(encoder,decoder)
    study=dict(version='PAC-screen/v1',backend=digest('backend'),identity=digest('CPU fixture study'),seed=20261016,arms=['B','C','D'],target_tokens=25000000,
        source=source,model=digest({k:v for k,v in recipe['model'].items() if k!='seed'}),optimizer=digest(recipe['learning']),
        initializer=initializer,inventory=batches.inventory['identity'])
    from oeis_learn.pipeline_validation import generalization_execution as execution
    monkeypatch.setattr(execution,'verify_pac_runtime',lambda *a:None)  # explicit authority fixture adapter
    monkeypatch.setattr(r0,'require_hip',lambda:None)
    monkeypatch.setattr(r0,'require_device_rng',lambda loaded:None)
    monkeypatch.setattr(r0,'production_batches',lambda root,bundle,arm:prepared_batches(root,bundle,arm))
    from oeis_learn.pipeline_validation.generalization_training import PacPreparedBatches
    def prepared_batches(root,bundle,arm):return PacPreparedBatches(root,bundle,arm=arm,batch_size=2,updates=25000000)
    monkeypatch.setattr(bigint_worker,'models',lambda recipe,device:(*models(),{}))
    def forbidden_semantic(*a,**kw):raise AssertionError('R0 must explicitly omit timed semantic generation')
    monkeypatch.setattr(curriculum,'semantic_probe',forbidden_semantic)
    calls=[];original=role.learner
    def learner(inputs,output,settings,*,seconds):
        calls.append((settings['config']['arm'],settings['stop_after_updates'],settings['resume'] is not None))
        assert settings['config']['batch_size']==32 and settings['config']['target_tokens']==25000000
        assert settings['config']['pac']['purpose']=='acceptance' and 'cadence' not in settings['config']['pac']
        assert settings['config']['pac']['semantic']=='r0-unmeasured/v1'
        adapted=copy.deepcopy(settings);adapted['config'].update(device='cpu',batch_size=2)
        value=original(inputs,output,adapted,seconds=seconds)
        value['device']='cuda'  # transport adapter only; actual tensors are CPU
        return value
    monkeypatch.setattr(role,'learner',learner)
    load=evaluation.load_retained;observe=curriculum.observe_targets
    monkeypatch.setattr(evaluation,'load_retained',lambda *a,**kw:load(*a,**dict(kw,device='cpu')))
    monkeypatch.setattr(curriculum,'observe_targets',lambda *a,**kw:observe(*a,**dict(kw,device='cpu')))
    runtime=dict(version='pac-runtime/v1',purpose='research',envelope=envelope(purpose='research',arms=['B','C','D']),images=[digest('cpu'),digest('gpu')],engine='podman',prepared=dict(training=None,admission=None))
    environment=digest(dict(images=runtime['images'],engine=runtime['engine'],backend=study['backend']))
    # The canonical launcher owns the mount root and runtime scratch before dispatch.
    # Calling rehearse directly with a nonexistent root missed the live R0 failure.
    from oeis_learn.generalization.role_contracts import Role
    output=tmp_path/'rehearsal';output.mkdir()
    scratch=output/'.runtime';scratch.mkdir();(scratch/'launcher.txt').write_text('owned by launcher')
    monkeypatch.setattr(role,'ROLE_OUTPUTS',{Role.LEARNER:SimpleNamespace(destination=str(output))})
    monkeypatch.setattr(role,'ROLE_INPUTS',{Role.LEARNER:[SimpleNamespace(channel='training_targets',destination=str(tmp_path/'data'))]})
    request=dict(operation='pac-rehearse',device='cuda',seconds=r0.rehearsal_envelope(runtime['envelope'])['outer'],
        settings=dict(recipe=recipe,source=source,environment=environment,study=study,bundle=bundle,runtime=runtime))
    assert role.main(['learner',json.dumps(request)])==0
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='PASS',result
    assert calls==[('B',256,False),('C',256,False),('C',129,False),('C',256,True),('D',256,False)]
    assert result['resume']['equal'] and result['resume']['pac_equal']
    assert result['runs']['C']['channels']['bandit']==192 and result['runs']['D']['valid_rewards']==31
    assert result['production_retention_validated'] is False
    assert all(v['training_status']=='PARTIAL' and v['changed'] and v['finite'] for v in result['runs'].values())
    saved=(output/'rehearsal.json').read_bytes()
    terminal=(output/'result.json').read_bytes()
    # A duplicate dispatch must not merge or overwrite a prior diagnostic run.
    with pytest.raises(FileExistsError):role.main(['learner',json.dumps(request)])
    assert (output/'result.json').read_bytes()==terminal
    assert (output/'rehearsal.json').read_bytes()==saved
    assert (scratch/'launcher.txt').read_text()=='owned by launcher'
    assert len(calls)==5


@pytest.mark.parametrize('existing',['absent','launcher','private-directory','private-file','private-symlink'])
def test_rehearsal_claims_only_fresh_private_work(tmp_path,existing):
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import claim_output
    output=tmp_path/'output'
    if existing!='absent':
        output.mkdir();(output/'.runtime').mkdir()
    if existing=='private-directory':
        (output/'private').mkdir();(output/'private/retained').write_bytes(b'keep')
    elif existing=='private-file':(output/'private').write_bytes(b'keep')
    elif existing=='private-symlink':(output/'private').symlink_to(tmp_path,target_is_directory=True)
    if existing.startswith('private-'):
        before=sorted(tmp_path.rglob('*'))
        with pytest.raises(FileExistsError):claim_output(output)
        assert sorted(tmp_path.rglob('*'))==before
        if existing=='private-directory':assert (output/'private/retained').read_bytes()==b'keep'
        elif existing=='private-file':assert (output/'private').read_bytes()==b'keep'
        else:assert (output/'private').is_symlink()
    else:
        assert claim_output(output)==output
        assert (output/'private').is_dir()
        if existing=='launcher':assert (output/'.runtime').is_dir()


def test_rehearsal_encloses_all_children_and_keeps_generous_limits():
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import rehearsal_envelope
    e=envelope(purpose='research',arms=['B','C','D']);r=rehearsal_envelope(e)
    assert r['work']>=5*e['learner']['outer']+4*e['proof']['outer']
    assert r['outer']>=r['work']+r['startup']+r['publication']+r['projection']+r['cleanup']


def test_combined_measured_storage_counts_rehearsal_snapshots(tmp_path,monkeypatch):
    from tests.integration.test_generalization_curriculum import models
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_training import storage_preflight
    monkeypatch.setattr(bigint_worker,'models',lambda *a:(*models(),{}))
    e=envelope(purpose='research',arms=['B','C','D'],rehearsal=True)
    runtime=dict(version='pac-runtime/v1',purpose='research',envelope=e);study=dict(arms=['B','C','D'])
    before=storage_preflight({},study,runtime)
    after=storage_preflight({},study,runtime,extra_generations=16)
    assert after['required_bytes']>before['required_bytes'] and after['rehearsal_generations']==16
    e['storage']['reserved_bytes']=before['required_bytes']
    with pytest.raises(ValueError,match='combined'):storage_preflight({},study,runtime,extra_generations=16)


def test_actual_cpu_cannot_satisfy_live_rehearsal_gate(monkeypatch):
    import torch
    from oeis_learn.pipeline_validation import generalization_execution as execution
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import require_hip
    monkeypatch.setattr(execution,'verify_execution',lambda:None)
    monkeypatch.setattr(torch.version,'hip',None)
    with pytest.raises(RuntimeError,match='actual HIP'):require_hip()


def test_research_cannot_disable_semantic_probes(tmp_path):
    from oeis_learn.pipeline_validation.generalization_training import run_torch
    from tests.integration.test_generalization_runtime import pac_config
    c=pac_config(tmp_path);c['pac'].update(purpose='research',semantic='r0-unmeasured/v1')
    with pytest.raises(ValueError,match='semantic'):
        run_torch(tmp_path/'data',tmp_path/'forbidden',c)
    assert not (tmp_path/'forbidden').exists()
