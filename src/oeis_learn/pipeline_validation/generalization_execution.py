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
    pac_envelope_graph(runtime['envelope'],purpose=runtime['purpose'],arms=experiment['arms'])
    if experiment['backend']!=POLICY_ID or experiment['budget_graph']!=pac_budget_projection(runtime['envelope'],arms=experiment['arms']):
        raise ValueError('PAC experiment backend/resource graph differs from execution')
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
    from oeis_learn.generalization.contracts import pac_record
    pac_record('envelope',runtime['envelope'],sealed=False)
    recipe=json.loads(verify_artifact(root,runtime['recipe'],max_bytes=1<<20))
    source=json.loads(verify_artifact(root,runtime['source'],max_bytes=16<<20))
    from oeis_learn.experiments.source_snapshot import source_snapshot
    if digest(source)!=experiment['source'] or source!=source_snapshot():raise ValueError('PAC actual installed source closure mismatch')
    if digest({k:v for k,v in recipe['model'].items() if k!='seed'})!=experiment['model'] or digest(recipe['learning'])!=experiment['optimizer']:
        raise ValueError('PAC actual model/optimizer recipe mismatch')
    from .generalization_runtime import execution_refs
    refs=execution_refs(root,experiment,runtime,preparation=runtime['preparation'] is not None)
    actual_bytes=sum(ref['bytes'] for ref in {r['path']:r for r in refs}.values())
    if actual_bytes>runtime['envelope']['storage']['data_bytes']:raise ValueError('PAC declared data storage is below actual input bytes')
    if runtime['prepared'] is None:
        if runtime['preparation'] is None:raise ValueError('PAC preparation authority required')
        from .generalization_preparation import validate_pac_preparation
        validate_pac_preparation(root,runtime['preparation'],experiment,runtime)
        return dict(study=experiment['identity'],recipe=recipe,initializer=None)
    from .generalization_preparation import verify_pac_prepared
    training=verify_pac_prepared(root,experiment,runtime)
    return dict(study=experiment['identity'],recipe=recipe,initializer=None,training=training)


PAC_STAGES=('validation','preparation','learner','generation','acceptance','calibration','resume','proof','export')


def pac_role_envelope(value):
    from .identity import number
    if not isinstance(value,dict) or set(value)!={'work','startup','publication','projection','cleanup','outer'}:
        raise ValueError('closed PAC role reservation required')
    for x in value.values():number(x,positive=True)
    if value['cleanup']<30 or value['outer']<sum(value[k] for k in value if k!='outer'):
        raise ValueError('PAC role cannot enclose work/publication/startup/projection/cleanup')
    return value


def pac_envelope_graph(envelope,*,purpose,arms=('B','C')):
    """Bind the B deadline arithmetic to every installed operation, without caps."""
    from .identity import number
    from oeis_learn.generalization.budgets import pac_deadline_graph,integer
    if set(envelope)!=set(PAC_STAGES)|{'version','campaign','diagnostics_seconds','batch_size','acceptance_tokens','backend','storage'} or envelope['version']!='pac-envelope/v1':
        raise ValueError('closed complete PAC envelope required')
    if envelope['backend']!=POLICY_ID:raise ValueError('PAC backend process-start identity mismatch')
    number(envelope['diagnostics_seconds'],positive=True);integer(envelope['batch_size'],1);integer(envelope['acceptance_tokens'],1)
    if envelope['batch_size']>128 or envelope['acceptance_tokens']>1000000:raise ValueError('bounded PAC acceptance/batch size')
    storage=envelope['storage']
    if set(storage)!={'data_bytes','data_copies','scratch_bytes','logs_bytes','package_bytes','reserved_bytes'}:raise ValueError('complete peak storage reservations required')
    for value in storage.values():integer(value,1)
    nodes={}
    for stage in PAC_STAGES:
        e=pac_role_envelope(envelope[stage])
        nodes[stage]=dict(requested=e['outer'],effective=e['outer'],work=e['work'],children=[],mode='serial',
            cleanup=e['cleanup'],cleanup_floor=30,finalization=0,overhead=e['startup']+e['publication']+e['projection'],contention=0,schema_max=None)
    if purpose=='research':
        for stage,floor in [('preparation',86400),('learner',43200),('resume',43200),('generation',23400)]:
            if envelope[stage]['work']<floor:raise ValueError('PAC research work floor')
    elif purpose!='acceptance':raise ValueError('PAC purpose')
    if envelope['diagnostics_seconds']>envelope['learner']['work']:raise ValueError('diagnostics exceed learner allowance')
    from oeis_learn.generalization.resources import _pac_arms
    _pac_arms(arms)
    for arm in arms[1:]:
        for stage in ('learner','generation','resume','proof','export'):
            nodes[stage+':'+arm]=dict(nodes[stage])
    nodes['campaign']=dict(requested=envelope['campaign'],effective=envelope['campaign'],work=0,children=list(nodes),
        mode='serial',cleanup=0,cleanup_floor=0,finalization=3600,overhead=0,contention=0,schema_max=None)
    return pac_deadline_graph(nodes,root='campaign')


def pac_budget_projection(envelope,*,arms=('B','C')):
    graph=pac_envelope_graph(envelope,purpose='acceptance',arms=arms)
    names=[name for name in graph['nodes'] if name!='campaign']
    return [dict(id=name,seconds=envelope[name.split(':')[0]]['outer'],children=[],mode='serial',
                 overhead_seconds=sum(envelope[name.split(':')[0]][k] for k in ('startup','publication','projection')),
                 cleanup_seconds=envelope[name.split(':')[0]]['cleanup']) for name in names]+[
        dict(id='campaign',seconds=envelope['campaign'],children=names,mode='serial',overhead_seconds=3600,cleanup_seconds=0)]
