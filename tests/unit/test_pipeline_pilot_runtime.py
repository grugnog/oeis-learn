"""Pinned CPU/HIP metric and complete-state fixtures; not a learned-success test."""
import copy
import time
import pytest


def test_opt_in_recipe_preserves_old_limits_and_cannot_enable_interventions():
    from oeis_learn.pipeline_validation.pilot_recipe import recipe
    from oeis_learn.fresh.bigint_config import resolve
    old=resolve({'version':'bigint-bootstrap/v1'})
    new=recipe()
    assert resolve(new)==new and old['learning']['max_updates']==3000
    assert old['learning']['batch_size']==8 and new['learning']['batch_size']==16
    assert new['data']['training']==32768 and new['model']['d_model']==768
    for field,value in [('curriculum',True),('replay',True),('batch_size',8)]:
        bad=copy.deepcopy(new);bad.pop('identity');bad['learning'][field]=value
        with pytest.raises(ValueError):resolve(bad)
    bad=copy.deepcopy(old);bad.pop('identity');bad['learning']['max_updates']=1048576
    with pytest.raises(ValueError):resolve(bad)
    for section,field,value in [('model','seed',20261017),('data','seed',20261017),('learning','validation_every',100)]:
        bad=copy.deepcopy(new);bad.pop('identity');bad[section][field]=value
        with pytest.raises(ValueError):resolve(bad)


def test_actual_v3_constructor_and_public_independent_cost_sentinels(tmp_path):
    from oeis_learn.pipeline_validation.pilot_recipe import recipe
    from oeis_learn.fresh.bigint_worker import models,client
    from oeis_learn.pipeline_validation.model_sizes import parameters,PROFILES
    from oeis_learn.pipeline_validation.sizing_execution import execution_context
    from oeis_learn.pipeline_validation.pilot_scoring_cost import sentinels
    from oeis_learn.pipeline_validation.identity import digest
    cfg=recipe()
    with execution_context():
        encoder,decoder,counts=models(cfg,'cpu')
        assert counts['actual']==parameters(PROFILES['m100'])
        assert counts['actual']['total']==104496662
    del encoder,decoder
    with client(dict(recipe=cfg,contract_sha256=digest('public pilot cost fixture')),tmp_path) as synthesis:
        result=sentinels(synthesis,deadline=time.monotonic()+120,policies=cfg['pipeline'])
    assert result['complete'] and result['fixture'] and result['learned_success'] is False
    assert len(result['outcomes'])==5 and result['maximum_selected_program_seconds']>0


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_token_weighting_masks_padding_and_keeps_exact_match_distinct(device):
    import torch,os,math
    if device=='cuda' and not (torch.version.hip and torch.cuda.is_available()):
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from oeis_learn.fresh.pilot_metrics import tensor_metrics
    logits=torch.zeros((2,3,4),device=device)
    targets=torch.tensor([[1,2,3],[2,0,0]],device=device)
    r=tensor_metrics(logits,targets,0)
    assert r['tokens']==4 and r['programs']==2 and r['correct_tokens']==0 and r['exact_programs']==0
    assert abs(r['loss_sum']-4*math.log(4))<1e-5


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_pilot_actual_worker_state_and_metric_resume(tmp_path,monkeypatch,device):
    import torch,os,shutil
    if device=='cuda' and not (torch.version.hip and torch.cuda.is_available()):
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from tests.integration.test_bigint_workflow import _fixture
    from oeis_learn.fresh.bigint_worker import train
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.fresh.common import write,read
    from oeis_learn.fresh.pilot_control import Control,configuration
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.pipeline_validation.sizing_execution import execution_context
    run,inputs,settings,examples=_fixture(device,tmp_path)
    settings['pilot_training_ids']=['training']
    settings['pilot']=configuration(source=digest('s'),work=digest('w'),purpose='preflight',ordinal=0,seconds=0,cadence_tokens=100000)
    write(inputs/'settings.json',settings)
    with execution_context():first=train(inputs,tmp_path/'first')
    assert first['pilot']['complete'] and first['update']==0
    # Exercise actual optimizer/RNG/checkpoint work; only the long wall clock is synthetic.
    class FixtureControl(Control):
        def __init__(self,*args,**kw):
            base=time.monotonic();ticks=iter([0,1,2,901,902,903,904])
            super().__init__(*args,clock=lambda:base+next(ticks),**kw)
    monkeypatch.setattr('oeis_learn.fresh.pilot_control.Control',FixtureControl)
    shutil.copytree(tmp_path/'first/checkpoints',inputs/'checkpoints')
    shutil.copyfile(tmp_path/'first/selected.json',inputs/'selected.json')
    settings['pilot']=configuration(source=digest('s'),work=digest('w'),purpose='preflight',ordinal=1,seconds=900,cadence_tokens=100000)
    write(inputs/'settings.json',settings)
    with execution_context():second=train(inputs,tmp_path/'second')
    assert second['pilot']['complete'] and second['update']==1
    assert second['initial_parameter_digest']==first['final_parameter_digest']
    assert second['initial_parameter_digest']!=second['final_parameter_digest']
    loaded=CheckpointStore(tmp_path/'second/checkpoints',run.contract).load(second['checkpoint'])
    assert loaded['ledger'].snapshot()==settings['ledger']
    assert loaded['states']['selection']['pilot']['updates']==1
    measured=read(tmp_path/'second/pilot-validation.json')
    assert all(x['complete'] and x['normalization']=='nonpadding-token-weighted' for x in measured['splits'].values())
    shutil.rmtree(inputs/'checkpoints')
    shutil.copytree(tmp_path/'second/checkpoints',inputs/'checkpoints')
    with pytest.raises(ValueError,match='prior pilot state|duplicate'):
        with execution_context():train(inputs,tmp_path/'third')
