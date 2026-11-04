import copy
import json
from pathlib import Path
import sys
import time

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]/'src'))
from oeis_learn.pipeline_validation.identity import seal,digest


def policy():
    import prospective_evaluation_policy as p
    return p


def row(name='a',*,family='family-a',holdout='parameter',start=0,source=None):
    return dict(id=name,family=family,holdout=holdout,source=source or name,group=name,
                variant=name,values=list(map(str,range(start,start+100))))


def test_shifted_window_and_old_same_role_are_excluded():
    p=policy();candidate=row(start=0)
    index=p.CandidateExclusions([candidate])
    index.observe([row('old',start=80)],role='selection',holdout='parameter')
    assert index.rejected()=={'a':{'finite_window'}}


@pytest.mark.parametrize('field,reason',[('source','source'),('group','group'),('variant','numeric_variant')])
def test_exact_identity_exclusions(field,reason):
    p=policy();a=row();b=row('old',start=1000);b[field]=a[field]
    index=p.CandidateExclusions([a]);index.observe([b],role='training',holdout='parameter')
    assert index.rejected()=={'a':{reason}}


@pytest.mark.parametrize('candidate,old,role,excluded',[
    ('parameter','parameter','training',False),('structural','parameter','training',True),
    ('parameter','structural','assessment',True),('structural','structural','selection',False)])
def test_family_separation_both_directions(candidate,old,role,excluded):
    p=policy();index=p.CandidateExclusions([row(holdout=candidate)])
    index.observe([row('old',start=1000)],role=role,holdout=old)
    assert ('a' in index.rejected())==excluded


def test_exact_windows_do_not_collapse_large_integers():
    p=policy();a=row();b=row('old');a['values']=[str(2**200+i) for i in range(100)]
    b['values']=[str(2**200+i+1) for i in range(0,200,2)]
    index=p.CandidateExclusions([a]);index.observe([b],role='training',holdout='parameter')
    assert not index.rejected()


def test_duplicate_candidate_identity_and_expired_deadline_rejected():
    p=policy()
    with pytest.raises(ValueError):p.CandidateExclusions([row(),row()])
    index=p.CandidateExclusions([row()])
    with pytest.raises(TimeoutError):index.observe([row('old')],role='training',holdout='parameter',deadline=0)


def test_protocol_frozen_counts_and_no_outcome_inputs():
    p=policy();prior=json.loads((HERE/'observable-evaluation-protocol.json').read_text())
    result=p.design(prior)
    assert result==p.design(copy.deepcopy(prior))
    assert result['seed']==20261104
    assert sum(c['count'] for c in result['cells'])==256
    assert result['counts']==dict(certified=153,ambiguous=55,unresolved=48)
    assert 'outcomes' not in json.dumps(result)
    bad=copy.deepcopy(prior);bad['roster'].pop(next(iter(bad['roster'])))
    with pytest.raises(ValueError):p.design(seal(bad))


def test_fixed_reservoir_is_unique_and_same_family_allocation():
    p=policy();d=p.design(json.loads((HERE/'observable-evaluation-protocol.json').read_text()))
    rows=p.draw(d)
    assert len(rows)<=8192
    assert len({r['id'] for r in rows})==len(rows)
    assert {r['family'] for r in rows}==set(d['families'])
    assert all(r['origin']=='generic_random' for r in rows)


def test_roster_freeze_rejects_shortfall_and_duplicate_source():
    p=policy();d=dict(cells=[dict(stratum='parameter',family='family-a',count=2)],families={'family-a':{'category':'certified'}})
    with pytest.raises(ValueError):p.roster([row()],d)
    with pytest.raises(ValueError):p.roster([row(),row('b',source='a')],d)
    actual=p.roster([row(),row('b',start=1000)],d)
    assert len(actual)==2 and all(v['category']=='certified' for v in actual.values())


def small_design():
    from oeis_learn.data.bigint_programs import BigintSampler
    from oeis_learn.generalization.refresh import RefreshPlan
    p=policy();prior=json.loads((HERE/'observable-evaluation-protocol.json').read_text());d=p.design(prior)
    # Two ordinary polynomial families; no model outcome affects the fixture.
    cells=[]
    for s in ('parameter','structural'):
        cell=next(c for c in d['cells'] if c['stratum']==s and d['families'][c['family']]['stage']=='A')
        cells.append(dict(cell,count=1,draws=3))
    d.update(cells=cells,families={c['family']:d['families'][c['family']] for c in cells})
    return seal(d)


@pytest.mark.parametrize('failure',[None,'native','shortfall','tampered'])
def test_native_preparation_never_publishes_failed_cohort(tmp_path,monkeypatch,failure):
    p=policy();d=small_design()
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    old=tmp_path/'old.json';old.write_text('[]')
    ref=reference(old,tmp_path)
    if failure=='tampered':old.write_text('[{}]')
    originals=[dict(root=tmp_path,ref=ref,name='old',role='training',holdout='parameter',count=0)]
    if failure=='shortfall':monkeypatch.setattr(p,'draw',lambda _:[])
    def native(source,indices):
        from oeis_learn.synthesis.reference import evaluate
        return dict(outcome='ok',values=['999']*100 if failure=='native' else list(evaluate(source,indices).values))
    out=tmp_path/'new'
    if failure:
        with pytest.raises(ValueError):p.prepare(d,originals,out,execute=native,seconds=60,workers=1)
        assert not (out/'summary.json').exists()
    else:
        result=p.prepare(d,originals,out,execute=native,seconds=60,workers=1)
        assert result['selected']==2 and len(result['roster'])==2
        assert result['status']=='PASS' and result['original_rows']==0


def test_family_macro_and_incomplete_denominators():
    p=policy();mapping={str(i):dict(category='certified',stratum='parameter' if i==3 else 'structural',family='A' if i<2 else 'B') for i in range(4)}
    scores={arm:{str(i):{'equal_budget':dict(success=i<2,complete=i!=2)} for i in range(4)} for arm in ('static','refreshed')}
    result=p.summarize(mapping,scores,['equal_budget'])
    s=result['cells']['certified/structural']['equal_budget']
    assert s['arms']['static']['family_macro']==.5
    assert s['arms']['static']['tasks']==3 and s['arms']['static']['unresolved']==1
    assert result['primary_interpretable'] is False
    scores['refreshed'].pop('0')
    with pytest.raises(ValueError):p.summarize(mapping,scores,['equal_budget'])
