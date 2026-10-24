"""Append-only public attempt outcomes survive partial generation/native failure."""
import json
from pathlib import Path
from .identity import seal,digest

TERMINAL={'COMMITTED','INCOMPLETE','UNSTARTED'}
EDGES={'PLANNED':{'STARTED','UNSTARTED'},'STARTED':{'GENERATED','INCOMPLETE'},
       'GENERATED':{'COMMITTED','INCOMPLETE'}}


class AttemptEvidence:
    def __init__(self,path):
        self.path=Path(path);self.records={};self.stream=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x')
        return self

    def append(self,row):
        self.stream.write(json.dumps(seal(row),sort_keys=True,allow_nan=False)+'\n')
        self.stream.flush()

    def plan(self,prompts,modes,*,parent):
        if self.records or len(set(prompts))!=len(prompts) or len(prompts)>256:
            raise ValueError('one unique finite prompt roster required')
        for prompt in prompts:
            for mode,count in modes.items():
                if type(count) is not int or not 1<=count<=8:
                    raise ValueError('bounded attempt allocation required')
                for ordinal in range(count):
                    key=(prompt,mode,ordinal)
                    row=dict(version='pipeline-attempt/v1',attempt=digest([parent,*key]),
                        parent=parent,prompt=prompt,mode=mode,ordinal=ordinal,state='PLANNED')
                    self.records[key]=row;self.append(row)

    def event(self,prompt,mode,ordinal,state,observations):
        key=(prompt,mode,ordinal)
        if key not in self.records or state not in EDGES.get(self.records[key]['state'],set()):
            raise ValueError('unknown attempt or immutable/illegal outcome transition')
        row=dict(self.records[key],state=state,observations=observations)
        self.records[key]=row;self.append(row)

    def __exit__(self,*exc):
        try:
            for key,row in self.records.items():
                if row['state'] not in TERMINAL:
                    state='UNSTARTED' if row['state']=='PLANNED' else 'INCOMPLETE'
                    terminal=dict(row,state=state,charging='no work started' if state=='UNSTARTED' else 'original reservation retained',
                        reason='role stopped before terminal acknowledgement')
                    self.records[key]=terminal;self.append(terminal)
        finally:
            self.stream.close()
