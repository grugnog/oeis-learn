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
