"""Scoped, audited FP32 math policy for sizing only; production flags restored."""
from contextlib import contextmanager
from functools import wraps
from .identity import seal
from .model_sizes import EXECUTION


def execution_observation():
    import torch
    cuda=torch.backends.cuda
    backends=[name for name,getter in [('math',cuda.math_sdp_enabled),('flash',cuda.flash_sdp_enabled),
        ('efficient',cuda.mem_efficient_sdp_enabled),('cudnn',cuda.cudnn_sdp_enabled)] if getter()]
    return seal(dict(version='pipeline-sizing-execution/v1',mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        sdpa_backends=backends,float32_matmul=torch.get_float32_matmul_precision(),
        matmul_tf32=cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),dtype=str(torch.get_default_dtype())))


@contextmanager
def execution_context():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    before=execution_observation()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        # The fused encoder layer can bypass Python MHA and its selected path.
        # Disable that route as well as all fused SDPA backends, for eval and train.
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            actual=execution_observation()
            if actual!=EXECUTION:raise RuntimeError('sizing execution policy not enforced: '+str(actual))
            yield actual
    finally:
        torch.backends.mha.set_fastpath_enabled(before['mha_fastpath'])
        torch.set_float32_matmul_precision(before['float32_matmul'])
        torch.backends.cuda.matmul.allow_tf32=before['matmul_tf32']
        torch.backends.cudnn.allow_tf32=before['cudnn_tf32']
        torch.use_deterministic_algorithms(before['deterministic_algorithms'],warn_only=before['deterministic_warn_only'])


def sizing_execution(function):
    @wraps(function)
    def run(*args,**kwargs):
        with execution_context():return function(*args,**kwargs)
    return run


def record_parity(actual,expected,output,label):
    """Compact failure diagnostics; the caller retains the original assertions."""
    import json
    import torch
    from pathlib import Path
    a=actual.detach().cpu();b=expected.detach().cpu()
    if a.shape!=b.shape:raise ValueError('sizing parity shapes differ')
    difference=(a-b).abs();limits=2e-5+2e-4*b.abs()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    row=dict(label=label,shape=list(a.shape),dtype=str(a.dtype),expected_dtype=str(b.dtype),
        finite=finite,mismatched=int((difference>limits).sum()),elements=a.numel(),max_absolute_error=float(difference.max()) if finite else None,
        atol=2e-5,rtol=2e-4,execution=execution_observation())
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    with (path/'parity.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
