"""Finite counters and rolling observations around bigint_worker, not a trainer."""
import time
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,digest,validate,number
from oeis_learn.pipeline_validation.pilot_contract import PROFILE,SPEC
from .common import write


def configuration(*,source,work,purpose,ordinal,seconds,cadence_tokens,interrupted=()):
    return seal(dict(version='pipeline-pilot-worker/v1',source=source,work=work,profile=PROFILE['identity'],
        purpose=purpose,ordinal=ordinal,resume=ordinal>0,seconds=seconds,cadence_tokens=cadence_tokens,interrupted=list(interrupted)))


def parse(cfg):
    validate(cfg,{'version','source','work','profile','purpose','ordinal','resume','seconds','cadence_tokens','interrupted'},'pipeline-pilot-worker/v1')
    from oeis_learn.synthesis.state import require_hash
    for k in ('source','work','profile'):require_hash(cfg[k])
    if cfg['profile']!=PROFILE['identity'] or cfg['purpose'] not in ('preflight','longrun'):raise ValueError('registered pilot purpose/profile required')
    maximum=2 if cfg['purpose']=='preflight' else 5
    if type(cfg['ordinal']) is not int or not 0<=cfg['ordinal']<=maximum or cfg['resume'] is not (cfg['ordinal']>0):
        raise ValueError('fixed distinct pilot roles required')
    if (not isinstance(cfg['interrupted'],list) or any(type(x) is not int or not 0<x<cfg['ordinal'] for x in cfg['interrupted'])
        or cfg['interrupted']!=sorted(set(cfg['interrupted']))):raise ValueError('explicit earlier interrupted ordinals required')
    number(cfg['seconds']);number(cfg['cadence_tokens'],positive=True)
    cap=900 if cfg['purpose']=='preflight' else 43200
    if cfg['seconds']>cap or (cfg['ordinal']==0)!=(cfg['seconds']==0) or type(cfg['cadence_tokens']) is not int:
        raise ValueError('fixed finite pilot work required')
    return cfg


class Control:
    def __init__(self,settings,selection,output,*,clock=time.monotonic,minimum_examples=0):
        self.cfg=parse(settings['pilot']);self.clock=clock;self.output=Path(output);self.started=clock()
        self.deadline=self.started+self.cfg['seconds'];self.batch_size=16
        self.binding=digest({k:self.cfg[k] for k in ('source','work','profile','purpose','cadence_tokens')})
        previous=selection.get('pilot')
        if self.cfg['resume']:
            gaps=range(previous['completed_ordinal']+1,self.cfg['ordinal']) if previous else []
            if (not previous or previous['binding']!=self.binding or previous['completed_ordinal']>=self.cfg['ordinal']
                or any(x not in self.cfg['interrupted'] for x in gaps)):
                raise ValueError('identical complete prior pilot state required; no allowance renewal')
            self.state=previous
        else:
            if previous is not None:raise ValueError('fresh pilot cannot reset prior counters')
            self.state=dict(binding=self.binding,tokens=0,unique_tokens=0,examples=0,updates=0,validations=0,completed_ordinal=-1,
                next_validation=self.cfg['cadence_tokens'],components={},bins=[],role_ordinal=-1,seen=[])
        if self.state['role_ordinal']>=self.cfg['ordinal']:raise ValueError('duplicate pilot role')
        self.state['role_ordinal']=self.cfg['ordinal'];self.start_tokens=self.state['tokens'];self.start_updates=self.state['updates']
        self.state['interrupted_ordinals']=sorted(set(self.state.get('interrupted_ordinals',[]))|set(self.cfg['interrupted']))
        self.start_examples=self.state['examples'];self.initial_validation=True;self.last_loss=None;self.last_metrics=None
        self.seen=set(self.state['seen'])
        selection['pilot']=self.state

    def wants_update(self,completed):
        return self.clock()<self.deadline-5 and completed<SPEC['max_updates'] and self.state['tokens']<SPEC['max_tokens']

    def component(self,name,seconds,**counts):
        number(seconds);r=self.state['components'].setdefault(name,{'seconds':0});r['seconds']+=seconds
        for k,v in counts.items():r[k]=r.get(k,0)+number(v)

    def update(self,rows,seconds):
        n=sum(len(r['tokens']) for r in rows)
        if self.state['tokens']+n>SPEC['max_tokens']:raise ValueError('pilot token allowance exceeded')
        for r in rows:
            if r.get('id') is not None and r['id'] not in self.seen:
                if len(self.seen)>=SPEC['training']:raise ValueError('pilot unique-data allowance exceeded')
                self.seen.add(r['id']);self.state['seen'].append(r['id'])
                self.state['unique_tokens']+=len(r['tokens'])
        self.state['tokens']+=n;self.state['examples']+=len(rows);self.state['updates']+=1
        self.component('training',seconds,tokens=n,examples=len(rows),updates=1)
        ordinal=int((self.clock()-self.started)//60)
        bins=self.state['bins'];key=[self.cfg['ordinal'],ordinal]
        if not bins or bins[-1]['key']!=key:
            if len(bins)>=SPEC['max_training_bins']:raise ValueError('pilot rolling metric capacity exhausted')
            bins.append(dict(key=key,tokens=0,examples=0,updates=0,loss_sum=0.,loss_weight=0,correct_tokens=0,exact_programs=0))
        b=bins[-1];b['tokens']+=n;b['examples']+=len(rows);b['updates']+=1
        if self.last_metrics is not None:
            b['loss_sum']+=self.last_metrics['loss_sum'];b['loss_weight']+=self.last_metrics['tokens']
            b['correct_tokens']+=self.last_metrics['correct_tokens'];b['exact_programs']+=self.last_metrics['exact_programs']

    def cycle_due(self):
        return self.state['tokens']>=self.state['next_validation']

    def cycled(self,*,complete=True):
        if self.state['validations']>=SPEC['max_validations']:raise ValueError('pilot validation capacity exhausted')
        self.state['validations']+=1
        while self.state['next_validation']<=self.state['tokens']:self.state['next_validation']+=self.cfg['cadence_tokens']

    def pulse_due(self):return False

    def checkpoint(self,selection):selection['pilot']=self.state

    def finish(self,*,completed=True):
        completed=completed and (self.cfg['ordinal']==0 or
            self.clock()-self.started>=self.cfg['seconds']-10 and self.state['updates']>self.start_updates)
        if completed:self.state['completed_ordinal']=self.cfg['ordinal']
        return seal(dict(version='pipeline-pilot-worker-result/v1',complete=completed,binding=self.binding,
            ordinal=self.cfg['ordinal'],seconds=self.clock()-self.started,updates=self.state['updates']-self.start_updates,
            tokens=self.state['tokens']-self.start_tokens,examples=self.state['examples']-self.start_examples,state=self.state))
