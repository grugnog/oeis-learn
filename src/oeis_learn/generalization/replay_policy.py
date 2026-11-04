"""Versioned task-first sampler; admission verification is mandatory on restore."""
from dataclasses import dataclass,replace
from fractions import Fraction
import random
from .replay_admission import AdmittedEntry
from oeis_learn.pipeline_validation.identity import seal,digest

@dataclass(frozen=True)
class ReplayPool:
    seed: int
    entries: tuple[AdmittedEntry,...] = ()
    cursor: int = 0
    evictions: tuple[str,...] = ()
    capacity: int = 4096
    duplicate_scope: str = 'source'

    def __post_init__(self):
        if type(self.seed) is not int or not 0<=self.seed<2**32 or type(self.cursor) is not int or self.cursor<0 or not 1<=self.capacity<=4096 or self.duplicate_scope not in {'source','finite-behavior'}:
            raise ValueError('replay policy bounds')
        counts={}; ids=set()
        for entry in self.entries:
            r=entry.record()
            if seal(r)!=r or r['identity'] in ids: raise ValueError('invalid admitted entry')
            ids.add(r['identity']);counts[r['task']]=counts.get(r['task'],0)+1
        if len(self.entries)>self.capacity or any(n>2 for n in counts.values()): raise ValueError('replay capacity exceeded')

    def add(self,entry):
        if not isinstance(entry,AdmittedEntry): raise ValueError('independent admission required')
        r=entry.record()
        key='final_source' if self.duplicate_scope=='source' else 'verification'
        def duplicate(old):
            a=old.record()
            return a['task']==r['task'] and (a[key]==r[key] if key=='final_source' else a[key]['values']==r[key]['values'])
        if any(duplicate(e) for e in self.entries): return self
        entries=list(self.entries)+( [entry] ); evicted=[]
        def oldest(xs): return min(xs,key=lambda e:(e.record()['order'],e.record()['identity']))
        same=[e for e in entries if e.record()['task']==r['task']]
        if len(same)>2:
            e=oldest(same);entries.remove(e);evicted.append(e.record()['identity'])
        while len(entries)>self.capacity:
            e=oldest(entries);entries.remove(e);evicted.append(e.record()['identity'])
        return replace(self,entries=tuple(entries),evictions=self.evictions+tuple(evicted))

    def task_weights(self):
        tasks={e.record()['task'] for e in self.entries}
        return {t:Fraction(1,len(tasks)) for t in sorted(tasks)}

    def sample(self,batch_size):
        if type(batch_size) is not int or batch_size<1: raise ValueError('batch size')
        groups={}
        for e in self.entries: groups.setdefault(e.record()['task'],[]).append(e)
        samples=[]
        for offset in range(batch_size//4 if groups else 0):
            rng=random.Random(int(digest([self.seed,self.cursor+offset])[7:],16))
            task=rng.choice(sorted(groups))
            samples.append(rng.choice(sorted(groups[task],key=lambda e:e.record()['identity'])))
        return tuple(samples),replace(self,cursor=self.cursor+len(samples))

    def snapshot(self):
        return seal(dict(version='generalization-replay-pool/v1',seed=self.seed,cursor=self.cursor,capacity=self.capacity,
                         duplicate_scope=self.duplicate_scope,entries=[e.record() for e in self.entries],evictions=list(self.evictions)))

    @classmethod
    def restore(cls,record,*,verify):
        import json
        if set(record)!={'version','identity','seed','cursor','capacity','duplicate_scope','entries','evictions'} or record['version']!='generalization-replay-pool/v1' or seal(record)!=record:
            raise ValueError('incompatible replay snapshot')
        entries=[]
        for r in record['entries']:
            e=AdmittedEntry(json.dumps(r,sort_keys=True).encode())
            checked=verify(e)
            if not isinstance(checked,AdmittedEntry) or checked.record()!=r: raise ValueError('replay re-verification failed')
            entries.append(checked)
        return cls(record['seed'],tuple(entries),record['cursor'],tuple(record['evictions']),record['capacity'],record['duplicate_scope'])
