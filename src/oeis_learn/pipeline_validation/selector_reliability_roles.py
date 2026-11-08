"""Visible-only GPU scheduling diagnostic with bounded resource observations."""
import time
from .identity import seal,write
from .generalization_preparation import reference

def benchmark(inputs, output, settings, *, seconds):
    from .generalization_role import generator
    from .pilot_clocks import DRAIN,MATCHED
    from .selector_fit_replay_role import resource_snapshot
    import torch
    from oeis_learn.tracking.pipeline_profile import Recorder
    settings=dict(settings)
    policy=settings.pop('benchmark_policy')
    render_node=settings.pop('render_node',None)
    if (policy not in (DRAIN,MATCHED) or settings.get('bank_seeds') is not None
            or settings.get('device')!='cuda' or set(settings)!={
                'checkpoint','prompts','recipe','study','source','device','bank_seeds'}):
        raise ValueError('closed training-only schedule benchmark required')
    before=resource_snapshot();began=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with Recorder(output/'telemetry',instrumented=True,render_node=render_node) as recorder:
        with recorder.phase('benchmark-workload'):
            result=generator(inputs,output,settings,seconds=seconds,pac=True,
                             evaluation_policy=policy,prompt_limit=16)
    telemetry=seal(dict(version='pac-selector-benchmark-telemetry/v1',
        scientific_status='DIAGNOSTIC_ONLY',freeze_eligible=False,policy=policy,
        wall_seconds=time.monotonic()-began,before=before,after=resource_snapshot(),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='role workload including model load; allocator peaks are not GPU utilization'))
    write(output/'benchmark-telemetry.json',telemetry)
    return seal(dict(result,benchmark_artifacts={
        'records':reference(output/'generation/selections.json',output),
        'clock':reference(output/'generation/attempts/portfolio-clocks.json',output),
        'telemetry':reference(output/'benchmark-telemetry.json',output)}))
