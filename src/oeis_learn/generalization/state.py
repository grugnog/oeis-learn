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


def bind_pac_state(curriculum,*,inventory,model_update,pending_batch):
    """Bind pure policy to the same model boundary; tensor I/O remains installed C."""
    from .curriculum_state import CurriculumState
    state=CurriculumState.restore(curriculum,inventory=inventory)
    if type(model_update) is not int or model_update!=state.updates or pending_batch!=state.pending:
        raise ValueError('model/policy/pending batch boundary mismatch')
    return state


def pac_snapshot(states,curriculum,*,contract,inventory,source,environment):
    """One model-bound envelope; no weights-only or policy-only resume authority."""
    _complete(states)
    policy=bind_pac_state(curriculum.snapshot(),inventory=inventory,model_update=states['data']['cursor'],pending_batch=states['data']['pending_batch'])
    return seal(dict(version='pac-learner-state/v1',kind='full_state',contract=contract,inventory=inventory,source=source,
                     environment=environment,states=copy.deepcopy(states),curriculum=policy.snapshot()))


def pac_restore(record,*,contract,inventory,source,environment,root=None):
    from .contracts import artifact_refs,verify_artifact
    fields={'version','kind','contract','inventory','source','environment','states','curriculum','identity'}
    if (set(record)!=fields or record['version']!='pac-learner-state/v1' or record['kind']!='full_state' or seal(record)!=record
        or any(record[k]!=v for k,v in dict(contract=contract,inventory=inventory,source=source,environment=environment).items())):
        raise ValueError('complete PAC identity-bound restore required')
    _complete(record['states'])
    refs=tuple(artifact_refs(record['states']))
    if refs and root is None:raise ValueError('verified artifact root required')
    for ref in refs:verify_artifact(root,ref)
    states=copy.deepcopy(record['states'])
    policy=bind_pac_state(record['curriculum'],inventory=inventory,model_update=states['data']['cursor'],pending_batch=states['data']['pending_batch'])
    return dict(states=states,curriculum=policy)


def pac_recover(record,journal,*,new_segment,contract,inventory,source,environment,root=None):
    """A newer journal supplies costs only, never newer policy on older tensors."""
    from .budgets import integer
    from .curriculum import finite
    result=pac_restore(record,contract=contract,inventory=inventory,source=source,environment=environment,root=root)
    s=result['curriculum']
    if (set(journal)!={'version','base','segment','sequence','charged_tokens','charged_seconds','identity'}
        or journal['version']!='pac-cost-journal/v1' or seal(journal)!=journal or journal['base']!=record['identity']
        or journal['segment']!=s.segment):raise ValueError('compatible cost-only journal required')
    integer(journal['sequence']);integer(journal['charged_tokens'])
    if finite(journal['charged_seconds'])<0:raise ValueError('journal time')
    if not isinstance(new_segment,str) or not new_segment or new_segment in s.segments or len(s.segments)>=4096:raise ValueError('new bounded recovery segment required')
    tokens=max(s.charged_tokens,journal['charged_tokens'])
    if tokens>s.cap:raise ValueError('journal exceeds frozen token cap')
    saved_pending=s.pending['tokens'] if s.pending and s.pending['phase'] in {'reserved','failed'} else 0
    s.discarded_tokens+=tokens-s.charged_tokens+saved_pending
    s.charged_tokens=tokens;s.charged_seconds=max(s.charged_seconds,journal['charged_seconds'])
    s.segment=new_segment;s.segments.append(new_segment)
    # Model-bound pre/block/stream state remains saved. Recompute a saved pending
    # batch under a new event ID and a fresh debit, without rewinding its row stream.
    if s.pending:
        s.reservations+=1
        s.pending.update(event_id=f'{new_segment}:update:{s.reservations}',tokens=None,grade=None,phase='planned')
    s.pauses=[p for p in s.pauses if p!='optimizer_failure_requires_restore']
    result['states']['data']['pending_batch']=copy.deepcopy(s.pending)
    return result
