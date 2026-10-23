"""Stable logical work, strict owner observation and conservative terminal charging."""
from dataclasses import dataclass
import copy
import math
from oeis_learn.pipeline_validation.identity import digest, seal, number


def logical_job(*, contract, snapshot, mode, arm, prompt_id, attempt_ordinal, method, seed):
    if type(attempt_ordinal) is not int or not 0<=attempt_ordinal<8 or type(seed) is not int:
        raise ValueError('bounded stable attempt/seed required')
    payload=dict(contract=contract,snapshot=snapshot,mode=mode,arm=arm,prompt_id=prompt_id,
        attempt_ordinal=attempt_ordinal,method=method,seed=seed)
    return dict(version='pipeline-job/v1',job_id=digest(payload),**payload)


class JobLedger:
    def __init__(self):
        self.records={};self.rejections=[]

    def reserve(self,job,*,deadline,work):
        number(deadline);number(work)
        key=job['job_id']
        if key in self.records:
            raise ValueError('duplicate logical job cannot acquire a new allowance')
        self.records[key]=dict(job=copy.deepcopy(job),deadline=deadline,reserved=work,
                              consumed=0,state='RESERVED',result=None)

    def publish(self,key,result,*,observed,consumed=None):
        number(observed)
        record=self.records[key]
        if record['state']!='RESERVED':
            self.rejections.append(dict(job=key,reason='duplicate_terminal',observed=observed));return False
        if consumed is not None and not 0<=number(consumed)<=record['reserved']:
            raise ValueError('consumption exceeds reservation')
        record['consumed']=record['reserved'] if consumed is None else consumed
        eligible=observed<record['deadline']
        record.update(state='COMMITTED' if eligible else 'INCOMPLETE',observed=observed,
            result=copy.deepcopy(result) if eligible else None,reason=None if eligible else 'late_result')
        return eligible

    def cancel(self,key,reason='cancelled'):
        record=self.records[key]
        if record['state']=='RESERVED':
            record.update(state='CANCELLED',consumed=record['reserved'],reason=reason)

    def boundary(self,policies):
        if any(r['state']=='RESERVED' for r in self.records.values()):
            raise ValueError('checkpoint boundary has unaccounted work')
        return seal(dict(version='pipeline-boundary/v1',policies=policies,
            frontier=sorted(self.records),consumed=sum(r['consumed'] for r in self.records.values()),active=0,reserved=0))


@dataclass(frozen=True)
class NativeRequest:
    job: dict
    prompt_ordinal: int
    deadline: float
    operation: object



def validate_policy(policy):
    fields={'version','selection','archive','workers','overlap','lookahead','batch_size','bookkeeping','measurement'}
    if not isinstance(policy,dict) or set(policy)!=fields or policy['version']!='pipeline-policy/v1':
        raise ValueError('closed pipeline-policy/v1 required')
    if policy['selection']!='visible-only/v2' or policy['archive']!='structure-pareto/v2':
        raise ValueError('concurrency requires common v2 policy')
    if policy['workers'] not in (1,2,4) or type(policy['workers']) is not int or type(policy['overlap']) is not bool:
        raise ValueError('registered worker/overlap policy required')
    if policy['lookahead'] not in (1,2) or policy['batch_size'] not in (1,8,16):
        raise ValueError('unqualified batch/lookahead; batch32 requires separate capacity admission')
    if policy['bookkeeping'] not in ('baseline','hoisted/v1') or policy['measurement'] not in ('ordinary','synchronized'):
        raise ValueError('unknown decoder/measurement policy')
    return copy.deepcopy(policy)
