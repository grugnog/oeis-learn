"""Qualification proof. This test must execute on actual HIP, never CPU fallback."""
import os
import math
from pathlib import Path
import pytest

pytestmark=pytest.mark.gpu


def test_generalization_device_load_and_finite_update(tmp_path):
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from tests.integration.test_bigint_workflow import _fixture
    from oeis_learn.fresh.bigint_worker import models,_batch_loss
    from oeis_learn.fresh.common import file_hash
    from oeis_learn.pipeline_validation.identity import seal,write
    from oeis_learn.pipeline_validation.generalization_execution import verify_execution
    execution_policy=verify_execution()
    _,_,settings,examples=_fixture('cuda',tmp_path)
    encoder,decoder,_=models(settings['recipe'],'cuda')
    saved=tmp_path/'qualification-weights.pt'
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),saved)
    loaded=torch.load(saved,map_location='cpu',weights_only=True)
    encoder.load_state_dict(loaded['encoder']);decoder.load_state_dict(loaded['decoder'])
    parameters=list(encoder.parameters())+list(decoder.parameters())
    before=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=settings['recipe']['learning']['learning_rate'])
    loss=_batch_loss(encoder,decoder,examples['training'][:2],'cuda',optimizer=optimizer)
    assert math.isfinite(loss['loss']) and math.isfinite(loss['gradient_norm'])
    assert all(torch.isfinite(p).all() for p in parameters)
    assert any(p.grad is not None for p in parameters)
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
    # Continue through the installed prospective learner/checkpoint/resume path.
    # This stays inside the existing qualification test and operator invocation.
    resumed=resume_probe(tmp_path/'prospective-resume',settings['recipe'],'cuda')
    # supervised_step checks finite loss; persist the actual loaded checkpoint hash.
    record=seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights=file_hash(saved),
        finite_forward=True,finite_backward=True,parameters_changed=True,execution_policy=execution_policy,**resumed))
    destination=os.environ.get('OEIS_GENERALIZATION_DEVICE_EVIDENCE')
    if destination:write(Path(destination),record)


def resume_probe(root,recipe,device):
    """The installed learner and proof handlers, shared by CPU wiring and HIP gate."""
    import json
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.budgets import charge_tokens
    from oeis_learn.pipeline_validation.generalization_role import learner
    from oeis_learn.pipeline_validation.generalization_acceptance import check_training
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.pipeline_validation.identity import digest
    root.mkdir(parents=True)
    source='(program (return (int 0)))'
    row=dict(id='mechanical-zero',role='training',source=source,tokens=encode_body(source),indices=list(map(str,range(100))),values=['0']*100)
    (root/'rows.json').write_text(json.dumps([row]));shard=reference(root/'rows.json',root)
    tokens=charge_tokens([row['tokens']],bos=1,pad=0)
    config=dict(study=digest('generalization-resume-qualification'),arm='static',seed=7,recipe=recipe,
        shards=[shard],data_mode='static',batch_size=1,target_tokens=2*tokens,milestones=[tokens],seconds=180,device=device,checkpoint_policy='hourly/v1')
    def run(name,*,prior=None,stop=None):
        settings=dict(config=config,resume=prior['checkpoint'] if prior else None)
        if stop is not None:settings['stop_after_updates']=stop
        inputs=dict(training_targets=root,verified_training_replay=root/'paused/private/learner/checkpoints' if prior else root)
        return learner(inputs,root/name,settings,seconds=180)
    full=run('full');paused=run('paused',stop=1);resumed=run('resumed',prior=paused)
    assert full['status']==resumed['status']=='PASS' and paused['status']=='PARTIAL'
    assert full['updates']==resumed['updates']==2 and paused['updates']==1
    assert full['device']==resumed['device']==device
    records=[]
    for name,result in [('full',full),('resumed',resumed)]:
        folder=root/name/'private/learner/checkpoints'
        state=GeneralizationCheckpointStore(folder,result['compatibility'],config['target_tokens']).load(result['checkpoint'])
        if device=='cuda':assert state['states']['rng']['device']
        records.append(dict(directory=str(folder.relative_to(root)),compatibility=result['compatibility'],
            cap=config['target_tokens'],checkpoint=result['checkpoint'],roster=reference(folder/'retained.json',folder)))
    proof=check_training({'verified_training_replay':root},root/'proof',dict(runs=records,compare=True),seconds=60)
    assert proof['resume_equal'] is True and all(r['finite'] and r['changed'] for r in proof['runs'])
    return dict(resume_equal=True,resume_checkpoint=resumed['checkpoint']['sha256'],updates=resumed['updates'])


def test_pac_actual_hip_acceptance_and_full_state_boundaries(tmp_path):
    """Implemented for the consolidated E session; never substitute CPU evidence."""
    import torch,time
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from oeis_learn.pipeline_validation.generalization_execution import verify_execution
    from oeis_learn.pipeline_validation.generalization_acceptance import pac_acceptance_fixture,pac_acceptance
    from oeis_learn.pipeline_validation.identity import digest,write
    from oeis_learn.synthesis.client import SynthesisClient
    verify_execution()
    with SynthesisClient(digest('pac-native-acceptance'),cache_root=tmp_path/'native') as native:
        config=pac_acceptance_fixture(tmp_path/'inputs',device='cuda',
            execute=lambda s,i:native.execute(s,i,deadline=time.monotonic()+4))
    result=pac_acceptance({'training_targets':tmp_path/'inputs'},tmp_path/'actual',dict(config=config),seconds=5400)
    assert result['device']=='hip' and result['research_eligible'] and result['status']=='PASS'
    if os.environ.get('OEIS_PAC_DEVICE_EVIDENCE'):
        write(Path(os.environ['OEIS_PAC_DEVICE_EVIDENCE']),result)
