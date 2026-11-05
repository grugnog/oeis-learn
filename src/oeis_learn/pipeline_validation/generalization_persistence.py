"""Periodic tensor persistence with a separate conservative accounting journal."""
import math
from oeis_learn.generalization.budgets import Ledger
from .identity import seal

INTERVAL=3600
POLICIES={'per-update/v1','hourly/v1'}
FIELDS={'version','contract','checkpoint','ledger','discarded_tokens','elapsed_seconds',
        'invocation_base_seconds','closed','identity'}


def policy(config):
    name=config.get('checkpoint_policy','per-update/v1')
    if name not in POLICIES:raise ValueError('registered checkpoint policy required')
    return name


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative accounting required')
    return value


def journal(*,contract,checkpoint,ledger,discarded_tokens,elapsed_seconds,invocation_base_seconds,closed):
    result=seal(dict(version='generalization-progress-journal/v1',contract=contract,checkpoint=checkpoint,
        ledger=ledger.snapshot(),discarded_tokens=discarded_tokens,elapsed_seconds=elapsed_seconds,
        invocation_base_seconds=invocation_base_seconds,closed=closed))
    validate(result,contract=contract,cap=ledger.cap)
    return result


def validate(record,*,contract,cap):
    if set(record)!=FIELDS or record['version']!='generalization-progress-journal/v1' or record['contract']!=contract:
        raise ValueError('compatible complete progress journal required')
    number(record['elapsed_seconds']);number(record['invocation_base_seconds'])
    if seal(record)!=record or type(record['closed']) is not bool:raise ValueError('journal integrity mismatch')
    ledger=Ledger.restore(record['ledger'],cap=cap)
    lost=record['discarded_tokens']
    if type(lost) is not int or lost<0 or ledger.consumed+lost>cap:raise ValueError('journal exposure exceeds frozen budget')
    if record['invocation_base_seconds']>record['elapsed_seconds']:raise ValueError('journal clock precedes invocation')
    ref=record['checkpoint']
    if not isinstance(ref,dict) or set(ref)!={'path','bytes','sha256'}:raise ValueError('snapshot reference required')
    return ledger


def recover(record,*,contract,checkpoint,ledger,discarded_tokens,learner_seconds,seconds_cap,supervised_seconds=None):
    """Roll back weights, never spent budget. Missing crash timing spends the cap.

    The original supervisor includes startup/exit, so charging its entire measured
    invocation is conservative. A full snapshot ahead of the journal is possible
    if publication was interrupted between the two atomic writes.
    """
    live=validate(record,contract=contract,cap=ledger.cap)
    number(learner_seconds);number(seconds_cap)
    if type(discarded_tokens) is not int or discarded_tokens<0:raise ValueError('saved discarded exposure required')
    if checkpoint!=record['checkpoint'] and (ledger.consumed<live.consumed or ledger.updates<live.updates):
        raise ValueError('journal belongs to a newer or unrelated recovery point')
    if record['closed'] and checkpoint==record['checkpoint'] and ledger!=live:
        raise ValueError('closed journal differs from terminal snapshot')
    total=max(ledger.consumed+discarded_tokens,live.consumed+record['discarded_tokens'])
    if total>ledger.cap:raise ValueError('recovery charge exceeds cap')
    seconds=max(learner_seconds,record['elapsed_seconds'])
    if not record['closed']:
        seconds=max(seconds,seconds_cap if supervised_seconds is None else
                    record['invocation_base_seconds']+number(supervised_seconds))
    return dict(discarded_tokens=total-ledger.consumed,learner_seconds=seconds,
        rolled_back=live.updates>ledger.updates or live.consumed>ledger.consumed)


def supervised_elapsed(process,cleanup):
    """Use measured original time only after both host and container cleanup ack."""
    if process is None or cleanup is None:return None
    if (process.get('outcome') not in {'COMPLETE','FAIL','INTERRUPTED'} or process.get('returncode') is None or
        process.get('cleanup_failure') is not None or cleanup.get('within_bound') is not True):return None
    return number(process['elapsed_seconds'])


def pac_boundary(state,ledger):
    """Verify model/policy clocks before a complete tensor generation is published."""
    record=state['scheduler'].get('pac')
    if record is None:return
    if record['version']=='pac-fixed-controller/v1':
        if record['channels']['fixed']!=ledger.updates:raise ValueError('fixed PAC model/observer clock mismatch')
        return
    from oeis_learn.generalization.curriculum_state import CurriculumState
    p=CurriculumState.restore(record['policy'],inventory=record['policy']['inventory'])
    if p.pending and p.pending['phase']=='failed':raise ValueError('failed optimizer cannot publish PAC tensors')
    discarded=state['scheduler'].get('discarded_tokens',0)
    if p.updates!=ledger.updates or p.updates!=state['data']['cursor'] or p.charged_tokens!=ledger.consumed+discarded:
        raise ValueError('PAC model/policy/token boundary mismatch')
    batch=state['data']['pending_batch']
    if (batch is None)!=(p.pending is None) or batch is not None and [r['id'] for r in batch]!=p.pending['rows']:
        raise ValueError('PAC pending encoded batch differs from policy')


def pac_crash_recovery(state,ledger,*,unclean,discarded_tokens):
    """Keep saved weights/policy; later journal advances only costs and segment IDs."""
    record=state['scheduler'].get('pac')
    if record is None or record['version']=='pac-fixed-controller/v1':return ledger
    from oeis_learn.generalization.curriculum_state import CurriculumState
    from dataclasses import replace
    p=CurriculumState.restore(record['policy'],inventory=record['policy']['inventory'])
    if not unclean and discarded_tokens==p.discarded_tokens:return ledger
    # Reservation may have executed before the crash. Its old debit is discarded;
    # recomputing it must reserve again under a fresh event ID.
    if unclean and ledger.debited:
        reserved=ledger.pending[1]
        discarded_tokens+=reserved
        ledger=replace(ledger,consumed=ledger.consumed-reserved,pending=None,debited=False)
    p.discarded_tokens=discarded_tokens;p.charged_tokens=ledger.consumed+discarded_tokens
    segment='recovery-'+str(len(p.segments))
    if len(p.segments)>=4096:raise ValueError('PAC recovery segment cap')
    p.segment=segment;p.segments.append(segment)
    if p.pending and ledger.pending is None:
        p.reservations+=1;p.pending.update(event_id=f'{segment}:update:{p.reservations}',tokens=None,grade=None,phase='planned')
    p.pauses=[x for x in p.pauses if x!='optimizer_failure_requires_restore']
    record['policy']=p.snapshot();state['scheduler']['discarded_tokens']=discarded_tokens
    return ledger
