import pytest
from oeis_learn.synthesis.reference import evaluate


def test_mechanical_fixture_is_reproducible_isolated_and_nontrivial():
    from oeis_learn.generalization.acceptance_inputs import mechanical_inputs,diagnostic_candidate
    from oeis_learn.generalization.refresh import RefreshPlan
    from oeis_learn.generalization.exclusions import ExclusionIndex
    from oeis_learn.generalization.targets import pair
    result=mechanical_inputs();assert result==mechanical_inputs()
    rows,_=RefreshPlan(result['descriptors'],seed=result['seed'],capacity=256).take(0,64)
    assert len({r['source'] for r in rows})>=20
    index=ExclusionIndex()
    for cell in result['cohorts']:
        for row in cell['rows']:
            exact=evaluate(row['source'],tuple(map(str,range(100))))
            assert exact.outcome=='ok'
            index=index.add(dict(id=row['id'],role=cell['role'],holdout=cell['holdout'],family=row['family'],source=row['source'],values=list(exact.values),group=row['group'],variant=str(row['variant'])))
    row=rows[0];proof=pair(row['source'],row['parameter_source'],row['assignment'],indices=tuple(map(str,range(100))),sharing='explicit')
    assert proof['disposition']=='supported' and proof['parameter_count']>=2
    candidate=diagnostic_candidate(row['source'])
    assert evaluate(candidate['final_source'],tuple(map(str,range(100)))).values==evaluate(row['source'],tuple(map(str,range(100)))).values
    assert len(candidate['tokens'])<len(proof['literal_tokens'])


def test_diagnostic_cannot_rewrite_arbitrary_or_control_source():
    from oeis_learn.generalization.acceptance_inputs import diagnostic_candidate
    with pytest.raises(ValueError):diagnostic_candidate('(program (return (int 9)))')


def test_research_cannot_use_diagnostic_policy_and_full_audit_still_applies():
    from oeis_learn.generalization.acceptance_inputs import search_policy,DIAGNOSTIC,mechanical_inputs,diagnostic_candidate
    from oeis_learn.generalization.refresh import RefreshPlan
    from oeis_learn.generalization.replay_admission import admit
    assert search_policy('research','model')==8
    assert search_policy('acceptance',DIAGNOSTIC)==1
    with pytest.raises(ValueError):search_policy('research',DIAGNOSTIC)
    fixture=mechanical_inputs();rows,_=RefreshPlan(fixture['descriptors'],seed=fixture['seed'],capacity=256).take(0,1)
    source=rows[0]['source'];indices=list(map(str,range(100)));values=list(evaluate(source,indices).values)
    task=dict(id='mechanical',role='training',ancestry=['mechanical','training'],source=source,indices=indices,values=values)
    candidate=diagnostic_candidate(source)
    def production(s,i):return evaluate(s,i).values
    assert admit(task,**candidate,origin_checkpoint='actual-checkpoint',production=production,production_identity='native-fixture',order=0)
    task['values'][-1]='999999999999'
    with pytest.raises(ValueError,match='disagreement'):
        admit(task,**candidate,origin_checkpoint='actual-checkpoint',production=production,production_identity='native-fixture',order=0)
