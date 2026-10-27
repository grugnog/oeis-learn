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


def test_actual_pilot_constructor_and_public_independent_cost_sentinels(tmp_path):
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
    import json
    assert result['complete'] and result['fixture'] and result['learned_success'] is False, json.dumps(result,sort_keys=True)
    assert len(result['required_cases'])==6 and len(result['stress_cases'])==1
    assert all(x['complete'] and x['success'] for key in result['required_cases'] for x in result['outcomes'][key].values()), json.dumps(result,sort_keys=True)
    assert result['maximum_selected_program_seconds']>0


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
def test_pilot_actual_worker_state_and_metric_resume(tmp_path,monkeypatch,device,purpose='preflight'):
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
    settings['pilot']=configuration(source=digest('s'),work=digest('w'),purpose=purpose,ordinal=0,seconds=0,cadence_tokens=100000)
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
    settings['pilot']=configuration(source=digest('s'),work=digest('w'),purpose=purpose,ordinal=1,seconds=900,cadence_tokens=100000)
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

@pytest.mark.parametrize('parent_seconds,expected',[(100,True),(30,False)])
def test_full_clock_receipts_keep_parent_cut_and_scientific_failure_distinct(tmp_path,monkeypatch,parent_seconds,expected):
    from types import SimpleNamespace
    from oeis_learn.fresh import bigint_diagnostics as d
    from oeis_learn.pipeline_validation.pilot_clocks import FullClockEvidence,POLICY
    from oeis_learn.pipeline_validation.identity import digest
    wall=[100.]
    monkeypatch.setattr(d,'time',SimpleNamespace(monotonic=lambda:wall[0]))
    evidence=FullClockEvidence(tmp_path,[{'id':'p'}],source=digest('s'),work=digest('w'))
    recipe=dict(model={'seed':1},search={'beam':None},track={'kind':'generic_only'})
    row=dict(id='p',indices=list(map(str,range(20))),values=list(map(str,range(20))))
    gen=d._portfolio(None,None,row,recipe,None,deadline=100.+parent_seconds,evidence=evidence,
        snapshot=digest('snapshot'),evaluation_policy=POLICY)
    request=next(gen)
    assert request.deadline==100.+min(60,parent_seconds)
    wall[0]=request.deadline
    with pytest.raises(StopIteration) as stop:
        gen.send(dict(source='',complete=False,tokens=[],generation_seconds=min(60,parent_seconds),generation_batch_size=1,failure='deadline'))
    result=stop.value.value
    assert not result['complete'] and not result['arm_evidence']['equal_budget']['complete']
    assert evidence.clocks[('p','correct')]['cost_complete'] is expected
    assert evidence.clocks[('p','correct')]['unstarted']==7
    evidence.finish()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_calibration_loads_real_tensor_weights_without_resume(tmp_path,monkeypatch,device):
    import torch,os
    if device=='cuda' and not (torch.version.hip and torch.cuda.is_available()):
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from oeis_learn.pipeline_validation.pilot_calibration import load_snapshot_models
    from oeis_learn.pipeline_validation.pilot_calibration_import import PRODUCER_SOURCE
    from oeis_learn.pipeline_validation.pilot_clocks import POLICY
    from oeis_learn.pipeline_validation.pilot_recipe import recipe
    from oeis_learn.pipeline_validation.identity import file_hash
    from oeis_learn.fresh import bigint_worker
    encoder=torch.nn.Linear(2,2).to(device);decoder=torch.nn.Linear(2,2).to(device)
    expected=dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())
    torch.save(expected,tmp_path/'weights.pt')
    def constructor(cfg,requested):
        assert requested=='cuda'
        return torch.nn.Linear(2,2).to(device),torch.nn.Linear(2,2).to(device),{}
    monkeypatch.setattr(bigint_worker,'models',constructor)
    cfg=dict(recipe=recipe(),checkpoint={'path':'state/checkpoint.json','sha256':'sha256:'+'a'*64})
    cfg['diagnostic_snapshot']=dict(weights_sha256=file_hash(tmp_path/'weights.pt'),pointer=cfg['checkpoint'],
        original_source=PRODUCER_SOURCE,ordinal=0,evaluation_policy=POLICY)
    e,d,_=load_snapshot_models(tmp_path,cfg)
    assert all(torch.equal(e.state_dict()[k],v) for k,v in expected['encoder'].items())
    assert all(torch.equal(d.state_dict()[k],v) for k,v in expected['decoder'].items())
    (tmp_path/'checkpoints').mkdir()
    with pytest.raises(ValueError,match='nonresumable'):load_snapshot_models(tmp_path,cfg)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_full_clock_pilot_actual_worker_state_and_metric_resume(tmp_path,monkeypatch,device):
    from oeis_learn.pipeline_validation.longrun_contract import PURPOSE
    test_pilot_actual_worker_state_and_metric_resume(tmp_path,monkeypatch,device,purpose=PURPOSE)
