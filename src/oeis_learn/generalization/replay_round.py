"""One search/distillation round with separately charged paired-control costs."""
from dataclasses import dataclass
from oeis_learn.pipeline_validation.identity import number

@dataclass(frozen=True)
class SearchLedger:
    records: tuple[tuple[str,int,float],...] = ()

    def charge(self,task,seconds):
        number(seconds,positive=True)
        rows={t:(n,s) for t,n,s in self.records}; n,s=rows.get(task,(0,0))
        if n>=8 or s+seconds>60 or sum(v[1] for v in rows.values())+seconds>14400 or task not in rows and len(rows)>=256:
            raise ValueError('search round exhausted')
        rows[task]=(n+1,s+seconds)
        return SearchLedger(tuple((t,*rows[t]) for t in sorted(rows)))


def round_plan(admissions,*,target_format='literal',rounds=1):
    if type(admissions) is not int or admissions<0 or target_format!='literal' or rounds!=1:
        raise ValueError('unregistered replay interaction or round')
    return dict(status='empty_yield' if admissions==0 else 'ready',tokens_per_arm=4_000_000,
                seconds_per_arm=7200,milestones=(1_000_000,2_000_000,4_000_000),arms=('continued_sft','verified_replay'),
                search_cost_separate=True,efficiency_claim_requires_cost_matched_control=True)
