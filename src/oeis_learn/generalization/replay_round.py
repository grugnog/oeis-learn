"""One search/distillation round with separately charged paired-control costs."""
from dataclasses import dataclass
from oeis_learn.pipeline_validation.identity import number

@dataclass(frozen=True)
class SearchLedger:
    """Serial reserve/settle accounting: admission bounds never erase actual cost."""
    records: tuple[tuple[str,int,float],...] = ()
    pending: tuple[str,float] | None = None
    overruns: tuple[tuple[str,float,float],...] = ()

    def __post_init__(self):
        tasks=set()
        if len(self.records)>256:raise ValueError('search task cap')
        for task,count,elapsed in self.records:
            if not isinstance(task,str) or not task or task in tasks or type(count) is not int or not 1<=count<=8:
                raise ValueError('invalid search ledger')
            number(elapsed);tasks.add(task)
        if self.pending is not None:
            task,reserved=self.pending
            measured=dict((t,s) for t,_,s in self.records)
            if task not in tasks or number(reserved,positive=True)>min(60-measured.get(task,0),14400-sum(measured.values())):
                raise ValueError('invalid pending search')
        for task,elapsed,reserved in self.overruns:
            if task not in tasks or number(elapsed)<=number(reserved,positive=True):raise ValueError('invalid overrun')

    def available(self,task):
        rows={t:(n,s) for t,n,s in self.records}
        n,s=rows.get(task,(0,0))
        if self.pending is not None or n>=8 or task not in rows and len(rows)>=256:return 0
        return max(0,min(60-s,14400-sum(v[1] for v in rows.values())))

    def reserve(self,task,seconds):
        number(seconds,positive=True)
        if not isinstance(task,str) or not task or seconds>self.available(task):
            raise ValueError('search round exhausted')
        rows={t:(n,s) for t,n,s in self.records};n,s=rows.get(task,(0,0))
        rows[task]=(n+1,s)
        return SearchLedger(tuple((t,*rows[t]) for t in sorted(rows)),(task,seconds),self.overruns)

    def finish(self,seconds):
        number(seconds)
        if self.pending is None:raise ValueError('no reserved search attempt')
        task,reserved=self.pending
        records=tuple((t,n,s+seconds if t==task else s) for t,n,s in self.records)
        overruns=self.overruns+((task,seconds,reserved),) if seconds>reserved else self.overruns
        return SearchLedger(records,None,overruns)

    def charge(self,task,seconds):
        """Compatibility for known in-budget costs; live work uses reserve/finish."""
        return self.reserve(task,seconds).finish(seconds)

    def snapshot(self):
        from oeis_learn.pipeline_validation.identity import seal
        return seal(dict(version='generalization-search-ledger/v2',records=self.records,pending=self.pending,overruns=self.overruns))

    @classmethod
    def restore(cls,record):
        from oeis_learn.pipeline_validation.identity import validate
        validate(record,{'version','records','pending','overruns'},'generalization-search-ledger/v2')
        return cls(tuple(tuple(r) for r in record['records']),
                   tuple(record['pending']) if record['pending'] is not None else None,
                   tuple(tuple(r) for r in record['overruns']))


def round_plan(admissions,*,target_format='literal',rounds=1):
    if type(admissions) is not int or admissions<0 or target_format!='literal' or rounds!=1:
        raise ValueError('unregistered replay interaction or round')
    return dict(status='empty_yield' if admissions==0 else 'ready',tokens_per_arm=4_000_000,
                seconds_per_arm=7200,milestones=(1_000_000,2_000_000,4_000_000),arms=('continued_sft','verified_replay'),
                search_cost_separate=True,efficiency_claim_requires_cost_matched_control=True)
