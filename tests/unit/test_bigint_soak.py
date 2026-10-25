"""Pinned runtime integration; host simulations cannot qualify these operations."""
import copy
import pytest


def test_image_mount_parent_is_searchable_by_arbitrary_numeric_uid():
    import stat
    from pathlib import Path
    parent=Path('/work')
    assert parent.is_dir() and not parent.is_symlink()
    assert stat.S_IMODE(parent.stat().st_mode)&0o005==0o005


def test_sized_recipe_preserves_legacy_identity_and_rejects_shape_drift():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation.model_sizes import PROFILES
    old=resolve(dict(version='bigint-bootstrap/v1'))
    new=resolve(dict(version='bigint-bootstrap/v2',model_profile=PROFILES['m100']))
    assert old['model']['d_model']==256 and 'model_profile' not in old
    assert new['model']['d_model']==768 and old['identity']!=new['identity']
    assert resolve(new)==new
    for key,value in [('d_model',512),('heads',8),('seed',1)]:
        bad=copy.deepcopy(new);bad['model'][key]=value;bad.pop('identity')
        with pytest.raises(ValueError):resolve(bad)
    with pytest.raises(ValueError):resolve(dict(version='bigint-bootstrap/v1',model_profile=PROFILES['m100']))


def test_actual_trainer_constructs_registered_full_size():
    import torch
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters
    from oeis_learn.pipeline_validation.sizing_execution import execution_context
    torch.set_num_threads(2)
    recipe=resolve(dict(version='bigint-bootstrap/v2',model_profile=PROFILES['m100']))
    with execution_context():
        encoder,decoder,counts=models(recipe,'cpu')
    assert counts['actual']==parameters(PROFILES['m100'])
    assert sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in decoder.parameters())==104496662


def test_engineering_complete_state_restart_retains_exact_ledger(tmp_path):
    from tests.integration.test_bigint_workflow import _fixture
    from oeis_learn.fresh.bigint_worker import train
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.fresh.common import write
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.pipeline_validation.soak_contract import lease
    h='sha256:'+'e'*64
    run,inputs,settings,examples=_fixture('cpu',tmp_path)
    l=lease(h,h,h)
    cfg=seal(dict(version='pipeline-engineering-worker/v1',source=h,work=h,choice=h,
        mode='window',batch_size=8,window=0,resume=False,lease=l,token_target=1,seconds=8,
        instrumented=False,cadence_tokens=1))
    settings['engineering']=cfg;write(inputs/'settings.json',settings)
    root=tmp_path/'persistent'
    first=train(inputs,root)
    assert first['engineering']['complete'] and first['update']>0
    settings['engineering']=seal(dict(cfg,window=1,resume=True));write(inputs/'settings.json',settings)
    second=train(inputs,root)
    assert second['engineering']['complete'] and second['update']>first['update']
    assert first['final_parameter_digest']==second['initial_parameter_digest']
    state=CheckpointStore(root/'checkpoints',run.contract).load(second['checkpoint'])
    assert state['ledger'].snapshot()==settings['ledger']
    assert state['states']['selection']['engineering']['tokens']>first['engineering']['state']['tokens']
    # A third process cannot replay the second window or obtain another allowance.
    with pytest.raises(ValueError,match='first-window'):
        train(inputs,root)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_engineering_uses_actual_trainer_and_checkpoint(tmp_path,device):
    import torch,os
    if device=='cuda' and not (torch.version.hip and torch.cuda.is_available()):
        if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
        pytest.skip('actual HIP required')
    from tests.integration.test_bigint_workflow import _fixture
    from oeis_learn.fresh.bigint_worker import train
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.fresh.common import write
    from oeis_learn.pipeline_validation.identity import seal
    run,inputs,settings,examples=_fixture(device,tmp_path)
    settings['engineering']=seal(dict(version='pipeline-engineering-worker/v1',source='sha256:'+'1'*64,
        work='sha256:'+'2'*64,choice='sha256:'+'3'*64,mode='batch',batch_size=8,
        window=0,resume=False,lease=None,token_target=sum(len(r['tokens']) for r in examples['training']),
        seconds=60,instrumented=False,cadence_tokens=100000))
    write(inputs/'settings.json',settings)
    result=train(inputs,tmp_path/'first')
    assert result['engineering']['complete'] and result['engineering']['updates']==1
    loaded=CheckpointStore(tmp_path/'first/checkpoints',run.contract).load(result['checkpoint'])
    assert loaded['states']['selection']['engineering']['tokens']==settings['engineering']['token_target']
    assert loaded['ledger'].snapshot()==settings['ledger']
