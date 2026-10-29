"""Opt-in original-clock accounting; cost completeness never implies science."""
from collections import Counter
from .identity import seal,number,write
from .pilot_evidence import Evidence
POLICY='portfolio-clock60/v1'
ATTEMPTS=dict(correct=8,shuffled=1,absent=1,template=8)


def clock_record(*,prompt,mode,snapshot,start,scope_deadline,finish,expected,started,terminal):
    for x in (start,scope_deadline,finish):number(x)
    if (mode not in ATTEMPTS or type(expected) is not int or expected!=ATTEMPTS[mode]
        or any(type(x) is not int for x in (started,terminal)) or not 0<=terminal<=started<=expected or finish<start):
        raise ValueError('registered original portfolio clock/counts required')
    original=start+60;effective=min(original,scope_deadline)
    all_done=started==terminal==expected and finish<scope_deadline
    exhausted=finish>=original and scope_deadline>=original and terminal==started
    reason='attempt_roster_terminal' if all_done else 'original_clock_exhausted' if exhausted else 'parent_cut' if scope_deadline<original else 'incomplete'
    return seal(dict(version='pipeline-portfolio-clock/v1',policy=POLICY,prompt=prompt,mode=mode,snapshot=snapshot,
        start=start,original_deadline=original,scope_deadline=scope_deadline,effective_deadline=effective,finish=finish,
        expected=expected,started=started,terminal=terminal,unstarted=expected-started,
        cost_complete=bool(all_done or exhausted),termination=reason))


class FullClockEvidence(Evidence):
    def __init__(self,output,rows,*,source,work):
        super().__init__(output,rows);self.clocks={};self.source=source;self.work=work;self.clock_summary=None

    def portfolio(self,prompt,mode,row):
        key=(prompt,mode)
        if key in self.clocks or (prompt,mode,0) not in self.jobs:raise ValueError('duplicate/unregistered portfolio clock')
        expected=ATTEMPTS[mode]
        states=[self.jobs[(prompt,mode,i)] for i in range(expected)]
        check=clock_record(**{k:row[k] for k in ('prompt','mode','snapshot','start','scope_deadline','finish','expected','started','terminal')})
        if (check!=row or row['prompt']!=prompt or row['mode']!=mode
            or row['started']!=sum(s!='UNSTARTED' for s in states)
            or row['terminal']!=sum(s in ('COMMITTED','INCOMPLETE') for s in states)):
            raise ValueError('clock differs from registered attempt ledger')
        self.clocks[key]=row

    def finish(self):
        attempts=super().finish();expected=len(self.jobs)//18*4
        self.clock_summary=seal(dict(version='pipeline-portfolio-clocks/v1',policy=POLICY,source=self.source,work=self.work,
            expected_portfolios=expected,missing_portfolios=expected-len(self.clocks),
            cost_complete=len(self.clocks)==expected and all(r['cost_complete'] for r in self.clocks.values()),
            terminations=dict(Counter(r['termination'] for r in self.clocks.values())),
            attempts=attempts['identity'],clocks=[self.clocks[k] for k in sorted(self.clocks)]))
        write(self.output/'portfolio-clocks.json',self.clock_summary)
        return attempts
