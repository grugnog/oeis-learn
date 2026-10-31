"""Fixed descriptors and counter-keyed numeric draws; exhaustion never recycles."""
from dataclasses import dataclass
import json
import random
from .contracts import digest
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.data.bigint_programs import canonical_family


def claim_source(source,seen):
    """Charge the draw but reject an already materialized refresh source."""
    key=digest(source)
    if key in seen:raise ValueError('duplicate refresh source')
    seen.add(key)

@dataclass(frozen=True,init=False)
class RefreshPlan:
    descriptors: bytes
    seed: int
    capacity: int

    def __init__(self,descriptors,*,seed,capacity):
        if type(seed) is not int or not 0<=seed<2**32 or type(capacity) is not int or not 1<=capacity<=1_000_000 or not descriptors:
            raise ValueError('invalid bounded refresh')
        for d in descriptors:
            if seal(d)!=d or d['version']!='generalization-descriptor/v1' or d['numeric_law']!='independent-uniform-integers[-16,16]/v1' or canonical_family(d['parameter_source'])!=d['family']:
                raise ValueError('changed descriptor/law')
        if len({d['family'] for d in descriptors})!=len(descriptors): raise ValueError('duplicate fixed family')
        object.__setattr__(self,'descriptors',json.dumps(descriptors,sort_keys=True).encode())
        object.__setattr__(self,'seed',seed); object.__setattr__(self,'capacity',capacity)

    @property
    def identity(self): return digest([json.loads(self.descriptors),self.seed,self.capacity])

    def take(self,cursor,count):
        from .targets import ground
        if type(cursor) is not int or type(count) is not int or cursor<0 or count<1 or cursor+count>self.capacity:
            raise ValueError('refresh exhausted; no recycling')
        descriptors=json.loads(self.descriptors); rows=[]
        for ordinal in range(cursor,cursor+count):
            d=descriptors[ordinal%len(descriptors)]
            rng=random.Random(int(digest([self.identity,ordinal])[7:],16))
            assignment={name:str(rng.randint(-16,16)) for name in d['parameter_names']}
            source=ground(d['parameter_source'],assignment)
            rows.append(dict(id=digest([self.identity,ordinal]),descriptor=d['identity'],family=d['family'],stage=d['stage'],
                             variant=digest(assignment),assignment=assignment,parameter_source=d['parameter_source'],source=source,origin='generic_random'))
        return tuple(rows),cursor+count

    def shard(self,cursor,count,*,admit):
        """Every draw is consumed, including exclusions; no hidden retry/recycling."""
        rows,next_cursor=self.take(cursor,count)
        accepted=[]; rejected=[]
        for row in rows:
            reason=admit(row)
            (rejected if reason else accepted).append((row['id'],reason) if reason else row)
        return seal(dict(version='generalization-shard-decision/v1',plan=self.identity,start=cursor,next_cursor=next_cursor,
                         admitted=accepted,rejected=rejected,exhausted=next_cursor==self.capacity))
