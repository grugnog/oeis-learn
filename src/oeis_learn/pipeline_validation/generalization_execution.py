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
