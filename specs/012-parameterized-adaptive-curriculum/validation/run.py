#!/usr/bin/env python3
"""Temporary PAC serial composition; scientific authority stays in installed roles."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.pipeline_validation.identity import seal,digest,write,confined,file_hash


def stage_order(*,research=False,arms=('B','C')):
    if len(set(arms))!=len(arms) or not {'B','C'}<=set(arms)<=set('ABCD'):raise ValueError('predeclared complete arms required')
    stages=['admission','qualification','calibration','acceptance']
    if research:
        stages+=['baseline']
        for arm in arms:stages+=['train-'+arm,'evaluate-'+arm]
        stages+=['analysis']
    return stages


def canonical_command(engine,operation,manifest,output,*,qualification=None,extra=()):
    if engine not in {'docker','podman'} or operation not in {'preflight','prepare','qualify','acceptance','evaluate','train'}:
        raise ValueError('registered explicit canonical command required')
    argv=[sys.executable,'scripts/foundation_run.py','--engine',engine,'generalization',operation,
          '--manifest',manifest,'--output',output]
    if operation not in {'qualify','preflight'}:
        if qualification is None:raise ValueError('original qualification required')
        argv+=['--qualification',qualification]
    return argv+list(extra)


def invoke(engine,operation,manifest,output,*,workspace=ROOT,qualification=None,extra=(),seconds,runner=None):
    from oeis_learn.pipeline_validation.supervision import launch
    argv=canonical_command(engine,operation,manifest,output,qualification=qualification,extra=extra)
    if runner is not None:return runner(argv,seconds=seconds)
    # Existing supervisor owns timeout, process-group cancellation and bounded logs.
    target=confined(workspace,output,exists=False)
    launch(argv,target.parent/(target.name+'-supervision'),seconds=seconds,cleanup_seconds=30,stream_cap=1<<20)
    path=target/'result.json'
    if not path.is_file() or path.stat().st_size>16<<20:raise ValueError('canonical terminal receipt missing')
    result=json.loads(path.read_text())
    if seal(result)!=result or result.get('stage')!=operation:raise ValueError('canonical producer binding changed')
    return result
