"""Actual CPU tensors/checkpoints, with explicit device/data boundary adapters.

This exercises the complete handler, not HIP or production data/performance proof.
"""
import copy
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
    monkeypatch.setattr(curriculum,'semantic_probe',lambda *a,**kw:None)
    calls=[];original=role.learner
    def learner(inputs,output,settings,*,seconds):
        calls.append((settings['config']['arm'],settings['stop_after_updates'],settings['resume'] is not None))
        assert settings['config']['batch_size']==32 and settings['config']['target_tokens']==25000000
        assert settings['config']['pac']['purpose']=='acceptance' and 'cadence' not in settings['config']['pac']
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
    result=r0.rehearse({'training_targets':tmp_path/'data'},tmp_path/'rehearsal',
        dict(recipe=recipe,source=source,environment=environment,study=study,bundle=bundle,runtime=runtime),seconds=r0.rehearsal_envelope(runtime['envelope'])['outer'])
    assert calls==[('B',256,False),('C',256,False),('C',129,False),('C',256,True),('D',256,False)]
    assert result['resume']['equal'] and result['resume']['pac_equal']
    assert result['runs']['C']['channels']['bandit']==192 and result['runs']['D']['valid_rewards']==31
    assert result['production_retention_validated'] is False
    assert all(v['training_status']=='PARTIAL' and v['changed'] and v['finite'] for v in result['runs'].values())


def test_rehearsal_encloses_all_children_and_keeps_generous_limits():
    from oeis_learn.pipeline_validation.generalization_pac_setup import envelope
    from oeis_learn.pipeline_validation.generalization_pac_rehearsal import rehearsal_envelope
    e=envelope(purpose='research',arms=['B','C','D']);r=rehearsal_envelope(e)
    assert r['work']>=5*e['learner']['outer']+4*e['proof']['outer']
    assert r['outer']>=r['work']+r['startup']+r['publication']+r['projection']+r['cleanup']
