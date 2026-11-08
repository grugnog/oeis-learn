"""Installed outcome-blind cohort/exclusion policy; no learner or model imports."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

from oeis_learn.pipeline_validation.identity import seal,digest,write

PRIOR='sha256:725d54628ef32a36a1c315143287475700c7f2b6db3e14a5d12b9e58731cf537'
PARENT_PREPARATION='sha256:56955fda4ffba7eda116fbcfc26b174e3b0d381130db1185d1d8149b13073b7f'
SEED=20261104
LAW='independent-uniform-integers[-1024,1024]/v1'


def design(prior):
    if seal(prior)!=prior or prior['identity']!=PRIOR:raise ValueError('reviewed family classification required')
    counts=Counter((r['stratum'],r['family']) for r in prior['roster'].values())
    cells=[dict(stratum=s,family=f,count=n,draws=max(64,n*16)) for (s,f),n in sorted(counts.items())]
    if sum(c['draws'] for c in cells)>8192:raise ValueError('reservoir bound')
    return seal(dict(version='prospective-retained-design/v1',seed=SEED,numeric_law=LAW,
        parent_classification=PRIOR,classification_policy=prior['policy'],families=prior['families'],cells=cells,
        counts=dict(Counter(r['category'] for r in prior['roster'].values())),
        checkpoints=prior['checkpoints'],bank_seeds=list(range(20261101,20261109)),role_seconds=21600,scoring_role_seconds=1200,
        inference=dict(visible_terms=20,total_terms=100,attempts=8,seconds=60,policy='portfolio-clock60-matched-drain/v2'),
        controls=['raw','rescue','equal_budget','shuffled','absent','template','bank'],
        decision=dict(primary='certified structural family-macro equal_budget confirmed success',
            secondary='certified parameter family-macro equal_budget confirmed success',completion_floor=.95,
            scope='single training seed; prospective selection screen; no automatic advancement'),
        assessment='SEALED; custodian exclusion-only access',training_tokens=0))


def draw(protocol, *, seed=None):
    from oeis_learn.data.bigint_programs import BigintSampler
    from oeis_learn.generalization.refresh import RefreshPlan
    if seal(protocol)!=protocol:raise ValueError('frozen design required')
    seed=SEED if seed is None else seed
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('bounded explicit draw seed required')
    sampler=BigintSampler(20261016);catalog={}
    for i in range(4096):
        d=sampler.descriptor(i)
        if d['family'] in protocol['families']:catalog.setdefault(d['family'],seal(dict(d,numeric_law=LAW)))
    if set(catalog)!=set(protocol['families']):raise ValueError('family support missing')
    rows=[]
    for cell in protocol['cells']:
        d=catalog[cell['family']]
        if d['parameter_source']!=protocol['families'][d['family']]['parameter_source']:raise ValueError('family changed')
        plan=RefreshPlan([d],seed=seed+(cell['stratum']=='structural'),capacity=8192)
        batch,_=plan.take(0,cell['draws'])
        rows.extend(dict(r,role='selection',holdout=cell['stratum']) for r in batch)
    return rows


class CandidateExclusions:
    """Exact candidate-side reverse index; original corpus memory stays per cell."""
    def __init__(self,rows):
        self.rows={};self.bad=defaultdict(set)
        self.keys={k:defaultdict(set) for k in ('source','group','numeric_variant','finite_window','family','structural')}
        for r in rows:
            key=r['id']
            if key in self.rows or len(r['values'])!=100:raise ValueError('unique full-horizon candidates required')
            self.rows[key]=r
            for kind,value in self._keys(r):self.keys[kind][value].add(key)
            self.keys['family'][r['family']].add(key)
            if r['holdout']=='structural':self.keys['structural'][r['family']].add(key)

    @staticmethod
    def _keys(r):
        yield 'source',r['source']
        yield 'group',r['group']
        yield 'numeric_variant',(r['family'],str(r['variant']))
        values=tuple(r['values'])
        for i in range(81):yield 'finite_window',values[i:i+20]

    def observe(self,rows,*,role,holdout,deadline=float('inf')):
        if role not in {'training','selection','assessment'} or holdout not in {'parameter','structural'}:
            raise ValueError('exclusion role')
        for r in rows:
            if time.monotonic()>=deadline:raise TimeoutError('exclusion scan deadline')
            if len(r['values'])!=100:raise ValueError('original exclusion horizon')
            for kind,value in self._keys(r):
                for key in self.keys[kind].get(value,()):self.bad[key].add(kind)
            if role!='selection':
                kind='family' if holdout=='structural' else 'structural'
                for key in self.keys[kind].get(r['family'],()):self.bad[key].add('structural_family')

    def rejected(self):return dict(self.bad)


def roster(rows,protocol):
    expected=Counter({(c['stratum'],c['family']):c['count'] for c in protocol['cells']})
    if (Counter((r['holdout'],r['family']) for r in rows)!=expected or
        len({r['source'] for r in rows})!=len(rows) or len({r['id'] for r in rows})!=len(rows)):
        raise ValueError('fresh cohort shortfall or duplicate source/identity')
    return {digest(['selection',r['id'],r['source']]):dict(family=r['family'],stratum=r['holdout'],
            category=protocol['families'][r['family']]['category']) for r in rows}

