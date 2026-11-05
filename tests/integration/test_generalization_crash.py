"""Real subprocess death and model-bound PAC recovery; CPU wiring is not HIP proof."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

EXIT_AFTER_OPTIMIZER=86


def worker(directory,phase,device,scale):
    """Abrupt os._exit skips learner finalization, leaving only its real journal."""
    import torch
    from oeis_learn.pipeline_validation import generalization_training as training,generalization_curriculum as curriculum
    from oeis_learn.pipeline_validation.identity import write,seal
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.fresh import bigint_worker
    from oeis_learn.pipeline_validation.generalization_curriculum import initial_weights_identity
    root=Path(directory);config=json.loads((root/'config.json').read_text())
    if config['device']!=device:raise ValueError('worker device mismatch')
    if device=='cuda':
        from oeis_learn.pipeline_validation.generalization_execution import verify_execution
        verify_execution()
        if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('actual HIP required')
    elif scale!='tiny':raise ValueError('CPU is a labeled fixture only')
    if scale=='tiny':
        from tests.integration.test_generalization_curriculum import models
        bigint_worker.models=lambda recipe,device:(*models(),{})
    # Sparse semantic generation is orthogonal to crash accounting. Keep real
    # target-loss/reward/grade observations, gradients and persistence untouched.
    curriculum.semantic_probe=lambda *a,**kw:None
    original_controller=curriculum.PacController;controllers=[]
    class Captured(original_controller):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);controllers.append(self)
            if phase=='recover':
                p=self.policy
                write(root/'restored.json',seal(dict(updates=p.updates,valid_rewards=p.kernel.valid_updates,
                    scaler_seen=p.scaler.seen,acquired=sum(a['acquired'] for a in p.review.anchors.values()),
                    graded=sum(a['last_grade'] is not None for a in p.review.anchors.values()),segment=p.segment,
                    discarded_tokens=p.discarded_tokens,charged_tokens=p.charged_tokens,
                    initial_weights=initial_weights_identity(self.encoder,self.decoder))))
    curriculum.PacController=Captured
    if phase=='crash':
        original=training.torch_update
        def injected(*args,**kwargs):
            step=original(*args,**kwargs);updates=0
            def update(batch):
                nonlocal updates
                value=step(batch);updates+=1
                if updates==19:
                    if device=='cuda':torch.cuda.synchronize()
                    p=controllers[0].policy
                    write(root/'crash-marker.json',seal(dict(actual_optimizer_calls=updates,
                        committed_policy_updates=p.updates,valid_rewards=p.kernel.valid_updates,
                        scaler_seen=p.scaler.seen,graded_anchors=sum(a['last_grade'] is not None for a in p.review.anchors.values()),
                        weights=initial_weights_identity(controllers[0].encoder,controllers[0].decoder),
                        device='hip' if device=='cuda' else 'cpu',exit_code=EXIT_AFTER_OPTIMIZER)))
                    os._exit(EXIT_AFTER_OPTIMIZER)
                return value
            return update
        training.torch_update=injected
        training.run_torch(root/'inputs',root/'crashed',config)
        raise AssertionError('learner ended before injected subprocess death')
    if phase!='recover':raise ValueError('closed crash/recover phase required')
    folder=root/'crashed/checkpoints';pointer=json.loads((folder/'generalization-latest.json').read_text())
    elapsed=json.loads((root/'elapsed.json').read_text())['crashed_seconds']
    result=training.run_torch(root/'inputs',root/'recovered',config,resume=(folder,pointer),resume_elapsed=elapsed,stop_after_updates=16)
    loaded=GeneralizationCheckpointStore(root/'recovered/checkpoints',result['compatibility'],config['target_tokens']).load(result['checkpoint'])
    policy=loaded['states']['scheduler']['pac']['policy']
    write(root/'recovered-proof.json',seal(dict(discarded_tokens=result['discarded_tokens'],charged_tokens=result['charged_tokens'],
        useful_tokens=result['tokens'],pending_tokens=result['pending_tokens'],cap=config['target_tokens'],updates=result['updates'],
        learner_seconds=result['learner_seconds'],device=result['device'],reason=result['reason'],
        segment=policy['segment'],policy_discarded_tokens=policy['discarded_tokens'],
        device_rng_present=bool(loaded['states']['rng']['device']),
        checkpoint=result['checkpoint']['sha256'],initial_weights=result['initial_weights'])))


def crash_probe(root,*,device,scale,execute):
    from oeis_learn.pipeline_validation.generalization_acceptance import pac_acceptance_fixture
    from oeis_learn.pipeline_validation.identity import write,seal,digest
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.pipeline_validation.generalization_curriculum import initial_weights_identity
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    config=pac_acceptance_fixture(root/'inputs',execute=execute,device=device)
    config=copy.deepcopy(config);config['target_tokens']*=4;config['seconds']=1800
    config['pac'].update(observation_seconds=1200,publication_seconds=120)
    write(root/'config.json',config)
    def run(phase):
        argv=[sys.executable,'-c','from tests.integration.test_generalization_crash import worker; import sys; worker(*sys.argv[1:])',str(root),phase,device,scale]
        with (root/(phase+'-stdout.log')).open('wb') as stdout,(root/(phase+'-stderr.log')).open('wb') as stderr:
            return subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=2100,check=False)
    start=time.monotonic();crashed=run('crash');elapsed=time.monotonic()-start
    assert crashed.returncode==EXIT_AFTER_OPTIMIZER, 'expected abrupt optimizer subprocess exit; inspect retained private crash-stderr.log'
    write(root/'elapsed.json',dict(crashed_seconds=elapsed))
    marker=json.loads((root/'crash-marker.json').read_text())
    folder=root/'crashed/checkpoints';pointer=json.loads((folder/'generalization-latest.json').read_text())
    journal=json.loads((folder/'generalization-progress.json').read_text())
    manifest=json.loads((folder/pointer['path']).read_text())
    saved=GeneralizationCheckpointStore(folder,manifest['contract'],config['target_tokens']).load(pointer)
    assert saved['ledger'].updates==0 and marker['actual_optimizer_calls']==19 and marker['committed_policy_updates']==18
    p=saved['states']['scheduler']['pac']['policy']
    assert p['kernel']['valid_updates']==0 and p['scaler']['seen']==0
    assert marker['valid_rewards']>0 and marker['scaler_seen']>0 and marker['graded_anchors']>0
    assert journal['closed'] is False and journal['ledger']['consumed']>saved['ledger'].consumed
    assert not (root/'crashed/training.json').exists(), 'crash must not finalize a successful learner'
    recovered=run('recover')
    assert recovered.returncode==0, 'recovery subprocess failed; inspect retained private recover-stderr.log'
    restored=json.loads((root/'restored.json').read_text());result=json.loads((root/'recovered-proof.json').read_text())
    # The saved initial tensor state must be used, rather than the newer weights,
    # kernel/scaler/graded review state left in the dead process.
    import torch
    from oeis_learn.fresh.bigint_worker import models
    if scale=='tiny':
        from tests.integration.test_generalization_curriculum import models as tiny_models
        encoder,decoder=tiny_models()
    else:encoder,decoder,_=models(config['recipe'],'cpu')
    encoder.load_state_dict(saved['states']['weights']['encoder']);decoder.load_state_dict(saved['states']['weights']['decoder'])
    old_weights=initial_weights_identity(encoder,decoder)
    assert restored['initial_weights']==old_weights!=marker['weights']
    assert restored['updates']==restored['valid_rewards']==restored['scaler_seen']==restored['graded']==restored['acquired']==0
    assert restored['segment']==result['segment']=='recovery-1'
    assert restored['discarded_tokens']==journal['ledger']['consumed']>0
    assert result['discarded_tokens']==result['policy_discarded_tokens']==restored['discarded_tokens']
    assert result['updates']==16 and result['charged_tokens']==result['useful_tokens']+result['pending_tokens']+result['discarded_tokens']<=result['cap']
    assert result['learner_seconds']>=elapsed and result['initial_weights']==old_weights
    if device=='cuda':assert result['device_rng_present']
    return seal(dict(version='pac-crash-proof/v1',status='PASS',device='hip' if device=='cuda' else 'cpu',
        diagnostic_only=True,source=digest(source_snapshot()),execution_policy=POLICY_ID if device=='cuda' else None,
        cadence=config['pac']['cadence'],scale=scale,crash_exit=crashed.returncode,actual_optimizer_calls=19,
        saved_updates=0,newer_valid_rewards=marker['valid_rewards'],newer_graded_anchors=marker['graded_anchors'],
        restored_old_weights=True,restored_old_policy=True,crashed_seconds=elapsed,**{k:result[k] for k in [
            'discarded_tokens','charged_tokens','useful_tokens','cap','updates','learner_seconds','segment','checkpoint','device_rng_present']}))


def test_real_cpu_subprocess_death_and_discarded_recovery(tmp_path):
    from oeis_learn.synthesis.reference import evaluate
    def execute(source,indices):
        r=evaluate(source,indices);return dict(outcome=r.outcome,values=list(r.values))
    result=crash_probe(tmp_path/'cpu',device='cpu',scale='tiny',execute=execute)
    assert result['status']=='PASS' and result['device']=='cpu' and result['diagnostic_only']


@pytest.mark.gpu
def test_pac_actual_hip_subprocess_crash_and_recovery(tmp_path):
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from oeis_learn.pipeline_validation.generalization_execution import verify_execution
    from oeis_learn.pipeline_validation.identity import digest,write
    from oeis_learn.synthesis.client import SynthesisClient
    verify_execution()
    with SynthesisClient(digest('pac-crash-native-fixture'),cache_root=tmp_path/'native') as native:
        result=crash_probe(tmp_path/'hip',device='cuda',scale='production',
            execute=lambda s,i:native.execute(s,i,deadline=time.monotonic()+4))
    target=os.environ.get('OEIS_PAC_CRASH_EVIDENCE')
    if target:write(Path(target),result)
