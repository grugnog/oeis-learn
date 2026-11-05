"""Versioned prospective HIP numerical policy; host imports remain stdlib-only."""
import os
from types import MappingProxyType
from .identity import digest

ENVIRONMENT=MappingProxyType(dict(DISABLE_ADDMM_CUDA_LT='1',TORCH_BLAS_PREFER_HIPBLASLT='0',
    ROCBLAS_USE_HIPBLASLT='0',ROCBLAS_DEFAULT_ATOMICS_MODE='0'))
POLICY_ID=digest(dict(version='generalization-hip-execution/v1',precision='fp32',
    blas_preference='cublas',environment=dict(ENVIRONMENT)))


def startup_environment():
    return dict(ENVIRONMENT)


def verify_execution():
    """Check process-start flags and accepted preference before prospective model work.

    Never repair a missing startup flag in-process: addmm can cache it before this
    call. A new container is required after any numerical policy change.
    """
    if any(os.environ.get(k)!=v for k,v in ENVIRONMENT.items()):
        raise RuntimeError('prospective HIP startup numerical policy mismatch; launch a fresh configured process')
    import torch
    torch.backends.cuda.preferred_blas_library('cublas')
    if str(torch.backends.cuda.preferred_blas_library()).split('.')[-1].lower()!='cublas':
        raise RuntimeError('prospective hipBLAS preference was not accepted')
    return POLICY_ID


def bind_compatibility(device,extension):
    result=dict(extension)
    if device=='cuda':
        if 'execution_policy' in result:raise ValueError('execution policy is owned by installed learner')
        result['execution_policy']=POLICY_ID
    return result


PAC_RUNTIME_FIELDS=frozenset({'version','identity','study','purpose','recipe','images','source',
    'control','profiles','preparation','prepared','envelope','feature','engine'})


def pac_binding(runtime):
    return digest({k:v for k,v in runtime.items() if k not in {'identity','study'}})


def bind_pac(experiment,runtime):
    from .identity import seal
    experiment=seal(dict(experiment,execution=pac_binding(runtime)))
    runtime=seal(dict(runtime,study=experiment['identity']))
    verify_pac_runtime(experiment,runtime)
    return experiment,runtime


def verify_pac_runtime(experiment,runtime):
    """A distinct canonical authority; never broaden an old P3 recipe."""
    import re
    from .identity import seal
    if (set(runtime)!=PAC_RUNTIME_FIELDS or runtime['version']!='pac-runtime/v1'
        or seal(runtime)!=runtime or runtime['study']!=experiment.get('identity')
        or experiment.get('execution')!=pac_binding(runtime)):
        raise ValueError('closed PAC execution binding required')
    if experiment.get('version')!='PAC-screen/v1' or seal(experiment)!=experiment:
        raise ValueError('intact PAC experiment required')
    if runtime['purpose'] not in {'acceptance','research'} or runtime['engine'] not in {'docker','podman'}:
        raise ValueError('explicit PAC scope and engine required')
    if len(runtime['images'])!=2 or any(not re.fullmatch(r'sha256:[0-9a-f]{64}',v) for v in runtime['images']):
        raise ValueError('immutable PAC image pair required')
    for name in ('recipe','source'):
        ref=runtime[name]
        if set(ref)!={'path','bytes','sha256'} or not re.fullmatch(r'sha256:[0-9a-f]{64}',ref['sha256']):
            raise ValueError('PAC artifact reference required')
    if runtime['preparation'] is not None and runtime['prepared'] is not None:
        raise ValueError('PAC preparation and resolved state are exclusive')
    return runtime


def require_pac_admission(experiment,runtime,operation):
    """Gate every model operation before creating its host/container."""
    if operation not in {'train','evaluate'}:raise ValueError('unsupported PAC model operation')
    prepared=runtime.get('prepared')
    if not prepared or set(prepared)!={'training','admission'}:
        raise ValueError('complete PAC original-input/native admission required')
    return prepared


def validate_pac_execution(root,experiment,runtime):
    import json
    from oeis_learn.generalization.contracts import verify_artifact
    verify_pac_runtime(experiment,runtime)
    from oeis_learn.generalization.execution import validate_pac_policy
    validate_pac_policy(experiment,runtime['control'],runtime['profiles'])
    recipe=json.loads(verify_artifact(root,runtime['recipe'],max_bytes=1<<20))
    source=json.loads(verify_artifact(root,runtime['source'],max_bytes=16<<20))
    if digest(source)!=experiment['source']:raise ValueError('PAC source closure mismatch')
    if digest({k:v for k,v in recipe['model'].items() if k!='seed'})!=experiment['model'] or digest(recipe['learning'])!=experiment['optimizer']:
        raise ValueError('PAC actual model/optimizer recipe mismatch')
    if runtime['prepared'] is None:
        if runtime['preparation'] is None:raise ValueError('PAC preparation authority required')
        from .generalization_preparation import validate_pac_preparation
        validate_pac_preparation(root,runtime['preparation'],experiment,runtime)
        return dict(study=experiment['identity'],recipe=recipe,initializer=None)
    from .generalization_preparation import verify_pac_prepared
    training=verify_pac_prepared(root,experiment,runtime)
    return dict(study=experiment['identity'],recipe=recipe,initializer=None,training=training)
