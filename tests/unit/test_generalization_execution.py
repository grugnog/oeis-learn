"""Prospective numerical policy binding; no hardware execution."""
import os
import pytest


def test_startup_environment_is_detached_and_policy_verification_fails_closed(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_execution as execution
    import torch
    environment=execution.startup_environment()
    assert environment['DISABLE_ADDMM_CUDA_LT']=='1'
    assert environment['TORCH_BLAS_PREFER_HIPBLASLT']=='0'
    environment['DISABLE_ADDMM_CUDA_LT']='0'
    assert execution.startup_environment()['DISABLE_ADDMM_CUDA_LT']=='1'
    for key,value in execution.startup_environment().items():monkeypatch.setenv(key,value)
    calls=[]
    monkeypatch.setattr(torch.backends.cuda,'preferred_blas_library',lambda value=None:calls.append(value) or 'BlasBackend.Cublas')
    assert execution.verify_execution()==execution.POLICY_ID
    assert calls==['cublas',None]
    monkeypatch.setenv('DISABLE_ADDMM_CUDA_LT','0')
    with pytest.raises(RuntimeError,match='startup'):execution.verify_execution()


@pytest.mark.parametrize('engine_name',['docker','podman'])
@pytest.mark.parametrize('role_name',['learner','generator'])
def test_prospective_gpu_roles_receive_policy_at_process_start(tmp_path,monkeypatch,engine_name,role_name):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.container_runtime import Engine
    from oeis_learn.generalization.role_contracts import ROLE_INPUTS,Role
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision,generalization_execution as execution
    adapter=Engine(engine_name,executable=engine_name,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda *_:[])
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    channels={m.channel:m.channel for m in ROLE_INPUTS[Role(role_name)]}
    for name in channels:(tmp_path/name).mkdir()
    observed=[]
    def launch(argv,output,**kw):
        observed.extend(argv);(output.parent/'result.json').write_text('{"status":"PASS"}')
    host=SimpleNamespace(engine=adapter,policy=dict(memory_bytes=2<<30,cpus=4),gpu_lock=lambda:nullcontext())
    roles.invoke(host,'sha256:'+'a'*64,tmp_path,role_name,channels,output='out',operation='fixture',settings={},seconds=30,gpu=True,launcher=launch)
    env=dict(observed[i+1].split('=',1) for i,v in enumerate(observed) if v=='--env')
    assert all(env.get(k)==v for k,v in execution.startup_environment().items())


def test_numerical_policy_changes_cuda_checkpoint_compatibility_only():
    from oeis_learn.pipeline_validation.generalization_execution import bind_compatibility,POLICY_ID
    assert bind_compatibility('cpu',{})=={}
    assert bind_compatibility('cuda',{})=={'execution_policy':POLICY_ID}
    with pytest.raises(ValueError):bind_compatibility('cuda',{'execution_policy':'old'})


def test_old_cuda_state_is_rejected_after_policy_binding(tmp_path):
    import json
    from oeis_learn.pipeline_validation.generalization_execution import bind_compatibility
    from oeis_learn.pipeline_validation.identity import digest
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    old=digest(dict(config='frozen',data='frozen'))
    new=digest(dict(config='frozen',data='frozen',**bind_compatibility('cuda',{})))
    states=dict(weights={},optimizer={},scheduler={},rng=dict(python=[],numpy=[],torch=[],device=[]),
        data=dict(cursor=0,pending_batch=None),replay={},history={},pins={},selection={})
    pointer=GeneralizationCheckpointStore(tmp_path,old,10).save(states,Ledger(10),encode=lambda x:json.dumps(x).encode())
    with pytest.raises(ValueError,match='incompatible'):
        GeneralizationCheckpointStore(tmp_path,new,10).load(pointer,decode=json.loads)


def pac_setup(mode='reference-only',optional_d=False):
    from oeis_learn.generalization.experiments import pac_recipe
    from oeis_learn.generalization.execution import pac_control
    from oeis_learn.pipeline_validation.identity import digest,seal
    from tests.unit.test_generalization_contracts import pac_shell
    h=digest('fixture')
    proof={k:h for k in ('initializer','order','model','optimizer','exposure','diagnostics','backend')}
    control=pac_control(proof,proof,original_study=h,source=h,image=h,checkpoint=h,historical_measurements=None)
    exp=pac_shell();exp.update(control_mode=mode,control=control['identity'])
    config=pac_recipe(control_mode=mode,optional_d=optional_d)
    exp.update(arms=config['arms'],comparison_factors=config['comparison_factors'])
    profiles={a:dict(initializer=h,initializer_kind='random',pool=h,targets=h,model=h,optimizer=h,backend=h,
               order=h,token_cap=25_000_000,sampling=config['sampling'][a]) for a in exp['arms']}
    return seal(exp),control,profiles,proof


def test_pac_freeze_defaults_and_control_proof():
    from oeis_learn.generalization.execution import validate_pac_policy,pac_control
    from oeis_learn.generalization.experiments import pac_recipe
    from oeis_learn.pipeline_validation.identity import seal
    exp,control,profiles,proof=pac_setup()
    assert exp['arms']==['B','C'] and 'D' not in exp['arms']
    validate_pac_policy(exp,control,profiles)
    for key in proof:
        original=dict(proof);original.pop(key)
        bad=pac_control(original,proof,**{k:control[k] for k in ('original_study','source','image','checkpoint','historical_measurements')})
        assert bad['disposition']=='REFERENCE_ONLY' and key in bad['reasons']
        matched=seal(dict(exp,control=bad['identity'],control_mode='matched',comparison_factors=['representation','allocation_and_review']))
        with pytest.raises(ValueError,match='matched'):validate_pac_policy(matched,bad,profiles)
    for change in ({'outcomes_seen':True},{'frozen':False},{'initializer_kind':'trained'},{'arms':['B','C','D']}):
        with pytest.raises(ValueError):validate_pac_policy(seal(dict(exp,**change)),control,profiles)
    assert pac_recipe()['token_cap']==25_000_000


def test_pac_exact_arm_identity_and_optional_arms():
    import copy
    from oeis_learn.generalization.execution import validate_pac_policy
    for mode,d in [('matched',False),('fresh-control',True),('reference-only',True)]:
        exp,control,profiles,_=pac_setup(mode,d)
        validate_pac_policy(exp,control,profiles)
        for field in ('initializer','pool','targets','model','optimizer','backend','token_cap'):
            broken=copy.deepcopy(profiles);broken['C'][field]='different'
            with pytest.raises(ValueError):validate_pac_policy(exp,control,broken)
        broken=copy.deepcopy(profiles);broken['B']['private_path']='/private/weights'
        with pytest.raises(ValueError):validate_pac_policy(exp,control,broken)
