"""Full update-boundary state policy; serialization of tensor blobs belongs to C."""
import copy
from .budgets import Ledger
from oeis_learn.pipeline_validation.identity import seal

STATE_FIELDS=frozenset({'weights','optimizer','scheduler','rng','data','replay','history','pins','selection'})


def _complete(states):
    if set(states)!=STATE_FIELDS or any(v is None for v in states.values()): raise ValueError('complete state required')
    if set(states['rng'])!={'python','numpy','torch','device'}: raise ValueError('complete RNG state required')
    if set(states['data'])!={'cursor','pending_batch'} or type(states['data']['cursor']) is not int or states['data']['cursor']<0:
        raise ValueError('data cursor and pending batch required')


def snapshot(states,*,contract,ledger):
    """States contain JSON state or content-addressed serialized tensor references.

    Reserve the token debit before launching work. A crash restores the saved
    pre-update weights plus pending batch and already debited ledger exactly once.
    """
    _complete(states)
    if ledger.pending and states['data']['pending_batch'] is None: raise ValueError('pending debit without batch')
    return seal(dict(version='generalization-learner-state/v1',kind='full_state',contract=contract,
                     states=copy.deepcopy(states),ledger=ledger.snapshot(),boundary='before_or_after_complete_update'))


def restore(record,*,contract,cap,root=None):
    from .contracts import artifact_refs, verify_artifact
    if set(record)!={'version','kind','contract','states','ledger','boundary','identity'} or record['version']!='generalization-learner-state/v1' or record['kind']!='full_state' or record['contract']!=contract or seal(record)!=record:
        raise ValueError('incompatible or incomplete full-state restore')
    if record['boundary'] != 'before_or_after_complete_update':
        raise ValueError('invalid update boundary')
    _complete(record['states'])
    refs=tuple(artifact_refs(record['states']))
    if refs and root is None: raise ValueError('state artifact verification root required')
    for ref in refs: verify_artifact(root,ref)
    ledger=Ledger.restore(record['ledger'],cap=cap)
    if ledger.pending and record['states']['data']['pending_batch'] is None: raise ValueError('lost pending batch')
    return dict(states=copy.deepcopy(record['states']),ledger=ledger)
