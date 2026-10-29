"""Mandatory bounded attempt ledger; long token streams are hashed, not copied."""
import json
from pathlib import Path
from .identity import seal,digest,write


class Evidence:
    def __init__(self,output,rows):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.events=0;self.bytes=0
        self.jobs={(r['id'],mode,i):'UNSTARTED' for r in rows for mode,n in
            [('correct',8),('shuffled',1),('absent',1),('template',8)] for i in range(n)}

    def __call__(self,prompt,mode,attempt,event,payload):
        key=(prompt,mode,attempt)
        if key not in self.jobs:raise ValueError('unregistered pilot attempt')
        old=self.jobs[key]
        allowed={'UNSTARTED':{'STARTED'},'STARTED':{'GENERATED'},'GENERATED':{'COMMITTED','INCOMPLETE'}}
        if event not in allowed.get(old,set()):raise ValueError('duplicate/out-of-order pilot evidence')
        self.jobs[key]=event;self.events+=1
        if self.events>15984:raise ValueError('registered pilot event capacity exceeded')
        compact={k:payload[k] for k in ('source_sha256','complete','seed','generation_seconds','fit_seconds') if k in payload}
        tokens=payload.get('tokens',[])
        if type(tokens) is int:
            if not 0<=tokens<=1024:raise ValueError('generation token count outside original cap')
            compact['tokens_count']=tokens;compact['tokens_sha256']=None
        else:
            compact['tokens_count']=len(tokens);compact['tokens_sha256']=digest(tokens)
        if 'reason' in payload:compact['reason']=str(payload['reason'])[:256]
        for k in ('raw','fitting'):
            if k in payload:compact[k]={n:v for n,v in payload[k].items() if n in {'complete','outcome','clock','job_id'}}
        row=dict(prompt=prompt,mode=mode,attempt=attempt,event=event,payload_sha256=digest(payload),payload=compact)
        raw=json.dumps(row,sort_keys=True,allow_nan=False)+'\n'
        if len(raw.encode())>2048:raise ValueError('pilot event byte capacity exceeded')
        self.bytes+=len(raw.encode())
        if self.bytes>16<<20:raise ValueError('pilot mandatory event stream exceeded16MiB')
        with (self.output/'attempts.jsonl').open('a') as stream:stream.write(raw)

    def __enter__(self):return self

    def __exit__(self,*exc):self.finish()

    def finish(self):
        from collections import Counter
        result=seal(dict(version='pipeline-pilot-attempt-index/v1',events=self.events,bytes=self.bytes,
            expected=len(self.jobs),states=dict(Counter(self.jobs.values())),
            jobs=[dict(prompt=p,mode=m,attempt=i,state=s) for (p,m,i),s in sorted(self.jobs.items())]))
        write(self.output/'attempt-index.json',result)
        return result
