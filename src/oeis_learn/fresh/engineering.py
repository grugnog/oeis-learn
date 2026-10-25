"""Bounded engineering controls around the existing trainer, never a trainer."""
import json
import time
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,digest,validate,number
from oeis_learn.pipeline_validation.soak_contract import SPEC,remaining
from .common import write

FIELDS={'version','source','work','choice','mode','batch_size','window','resume','lease',
        'token_target','seconds','instrumented','cadence_tokens'}


def parse(value,*,fixture=False):
    validate(value,FIELDS,'pipeline-engineering-worker/v1')
    if value['mode'] not in {'batch','joint','observation','window'}:raise ValueError('registered engineering mode required')
    if type(value['batch_size']) is not int or value['batch_size'] not in (8,16,32):raise ValueError('registered training batch required')
    if type(value['window']) is not int or value['window'] not in (0,1):raise ValueError('two fixed work windows required')
    if type(value['resume']) is not bool or type(value['instrumented']) is not bool:raise ValueError('explicit engineering flags required')
    for k in ('source','work','choice'):
        from oeis_learn.synthesis.state import require_hash
        require_hash(value[k])
    for k in ('seconds','cadence_tokens','token_target'):number(value[k],positive=True)
    if value['seconds']>3600 or value['cadence_tokens']%1 or value['token_target']%1:raise ValueError('finite engineering work required')
    if value['mode']=='window':
        remaining(value['lease'])
        if not fixture and value['seconds']!=3600:raise ValueError('complete3600s work window required')
        if value['resume']!=(value['window']==1):raise ValueError('second window must restore complete state')
    elif value['lease'] is not None or value['resume'] or value['window']:
        raise ValueError('only a window can resume its own engineering state')
    return value


class Control:
    def __init__(self,settings,selection,output,*,clock=time.monotonic):
        self.cfg=parse(settings['engineering'],fixture=bool(settings.get('fixture')))
        self.output=Path(output);self.clock=clock;self.started=clock()
        self.deadline=min(self.started+self.cfg['seconds'],self.cfg['lease']['expires']-60 if self.cfg['lease'] else float('inf'))
        self.binding=digest({k:self.cfg[k] for k in ('source','work','choice','mode','batch_size','lease','cadence_tokens')})
        previous=selection.get('engineering')
        if self.cfg['resume']:
            if not previous or previous['binding']!=self.binding or previous['completed_window']!=0:
                raise ValueError('identical complete first-window state required')
            self.state=previous
        else:
            if previous is not None:raise ValueError('fresh engineering cannot reset prior work')
            self.state=dict(binding=self.binding,tokens=0,examples=0,cycles=0,updates=0,
                completed_window=-1,components={},windows=[],next_validation=self.cfg['cadence_tokens'],pulses=0)
        self.start_tokens=self.state['tokens'];self.start_updates=self.state['updates']
        self.start_cycles=self.state['cycles'];self.last=self.started;self.bucket=None
        self.start_pulses=self.state['pulses']
        self.batch_size=self.cfg['batch_size'];self.events=0
        self.next_pulse=self.started

    def wants_update(self,completed):
        return (self.clock()<self.deadline-5 and completed<SPEC['max_updates']
            and self.state['cycles']<SPEC['max_cycles']
            and (self.cfg['mode']=='window' or self.state['tokens']-self.start_tokens<self.cfg['token_target']))

    def component(self,name,seconds,**counts):
        number(seconds)
        now=self.clock();slot=int((now-self.started)//300)
        if self.bucket is None or self.bucket['ordinal']!=slot:
            self.flush(now)
            self.bucket=dict(leg=self.cfg['window'],ordinal=slot,start=now,stop=now,components={})
        for target in (self.state['components'],self.bucket['components']):
            row=target.setdefault(name,dict(seconds=0))
            row['seconds']+=seconds
            for k,v in counts.items():row[k]=row.get(k,0)+number(v)
        self.bucket['stop']=now

    def update(self,rows,seconds):
        tokens=sum(len(r['tokens']) for r in rows)
        self.state['tokens']+=tokens;self.state['examples']+=len(rows);self.state['updates']+=1
        self.component('training',seconds,tokens=tokens,examples=len(rows),updates=1)

    def cycle_due(self):
        return self.cfg['mode'] in {'window','joint'} and (self.state['tokens']>=self.state['next_validation'] or
            self.state['cycles']==self.start_cycles and self.cfg['mode']=='joint')

    def cycled(self):
        self.state['cycles']+=1
        while self.state['next_validation']<=self.state['tokens']:
            self.state['next_validation']+=self.cfg['cadence_tokens']

    def evidence(self,prompt,mode,attempt,event,payload):
        if self.events>=SPEC['max_jobs']*4:raise ValueError('engineering evidence capacity exhausted')
        # Exact payload hash binds large native admissions without copying them into logs.
        compact={k:v for k,v in payload.items() if k in {'tokens','source_sha256','generation_seconds','complete','reason','seed','fit_seconds','clock'}}
        for k in ('raw','fitting'):
            if k in payload:compact[k]={n:v for n,v in payload[k].items() if n in {'complete','outcome','reason','clock','job_id'}}
        row=dict(cycle=self.state['cycles'],prompt=prompt,mode=mode,attempt=attempt,event=event,payload_sha256=digest(payload),payload=compact)
        path=self.output/'engineering-attempts.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
        self.events+=1

    def pulse_due(self):
        return self.cfg['mode']=='window' and self.clock()>=self.next_pulse and self.clock()+30<self.deadline and self.state['pulses']-self.start_pulses<6

    def pulsed(self):
        self.state['pulses']+=1;self.next_pulse=self.started+(self.state['pulses']-self.start_pulses)*600

    def flush(self,now=None):
        if self.bucket is not None:
            self.bucket['stop']=self.clock() if now is None else now
            self.state['windows'].append(self.bucket);self.bucket=None
            if len(self.state['windows'])>32:raise ValueError('engineering window evidence capacity exhausted')

    def checkpoint(self,selection):
        selection['engineering']=self.state

    def finish(self,*,completed=True):
        self.flush()
        elapsed=self.clock()-self.started
        enough=(self.state['tokens']-self.start_tokens>=self.cfg['token_target'] if self.cfg['mode']!='window'
            else elapsed>=self.cfg['seconds']-10 and self.state['cycles']>self.start_cycles)
        complete=completed and enough
        if complete:self.state['completed_window']=self.cfg['window']
        return seal(dict(version='pipeline-engineering-result/v1',binding=self.binding,mode=self.cfg['mode'],
            complete=complete,batch_size=self.batch_size,window=self.cfg['window'],active_seconds=elapsed,
            tokens=self.state['tokens']-self.start_tokens,updates=self.state['updates']-self.start_updates,
            cycles=self.state['cycles']-self.start_cycles,state=self.state))
