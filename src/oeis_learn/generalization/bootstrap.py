"""Acceptance-only dependency resolution; no training or fabricated producers."""
import copy,re
from .execution import verify_binding,verify_resolution


def pending(study):
    if not isinstance(study,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',study):raise ValueError('frozen source study required')
    return dict(kind='acceptance_selection',case='retained-eval',study=study)


def is_pending(initialization):
    return isinstance(initialization,dict) and initialization.get('kind')=='acceptance_selection'


def verify_pending(study,runtime):
    value=study['model']['initialization']
    if not is_pending(value):return False
    if value!=pending(value.get('study')) or runtime['purpose']!='acceptance' or study['progression']['stage']!='P4':
        raise ValueError('deferred selection is acceptance-only P4 authority')
    return True


def verify_transition(original,runtime,resolved,sidecar):
    verify_binding(original,runtime);verify_binding(resolved,sidecar)
    if not verify_pending(original,runtime) or resolved['model']['initialization'].get('kind')!='retained':
        raise ValueError('unresolved acceptance dependency')
    a=copy.deepcopy(original);b=copy.deepcopy(resolved)
    for value in (a,b):
        value.pop('identity');value.pop('execution');value['model'].pop('initialization')
    if a!=b:raise ValueError('initializer transition changed frozen choices')
    verify_resolution(runtime,sidecar)
