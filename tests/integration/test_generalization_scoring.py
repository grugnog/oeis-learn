"""Scorer wiring validates isolated inputs before native execution."""
import json
import pytest
from oeis_learn.pipeline_validation.generalization_preparation import reference


def fixture(tmp_path):
    truth=tmp_path/'truth';truth.mkdir()
    candidates=tmp_path/'candidates';candidates.mkdir()
    row=dict(id='one',role='selection',source='(program (return (int 0)))',
             family='sha256:'+'1'*64,stratum='parameter',
             indices=list(map(str,range(100))),values=['0']*100)
    (truth/'rows.json').write_text(json.dumps([row]))
    (candidates/'rows.json').write_text(json.dumps({'one':{'selections':{}}}))
    return truth,candidates,row


def test_selection_scoring_uses_verified_disjoint_inputs(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_scoring import score_selection
    from oeis_learn.fresh import bigint_diagnostics
    truth,candidates,row=fixture(tmp_path)
    def score(selections,targets,client,*,deadline):
        assert targets=={'one':row} and set(selections)=={'one'}
        return {'splits':{},'outcomes':{}}
    monkeypatch.setattr(bigint_diagnostics,'score',score)
    result=score_selection(truth,reference(truth/'rows.json',truth),candidates,
        reference(candidates/'rows.json',candidates),client=None,output=tmp_path/'out',seconds=10)
    assert result['status']=='PASS' and (tmp_path/'out/scoring.json').is_file()


def test_selection_rejects_assessment_truth_before_scoring(tmp_path):
    from oeis_learn.pipeline_validation.generalization_scoring import score_selection
    truth,candidates,row=fixture(tmp_path);row['role']='assessment'
    (truth/'rows.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='selection'):
        score_selection(truth,reference(truth/'rows.json',truth),candidates,
            reference(candidates/'rows.json',candidates),client=None,output=tmp_path/'out',seconds=10)
    assert not (tmp_path/'out').exists()


def test_role_distinguishes_handler_failure_from_unknown_operation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import generalization_role as role
    from oeis_learn.generalization.role_contracts import Role
    monkeypatch.setattr(role,'ROLE_OUTPUTS',{Role.SELECTION:SimpleNamespace(destination=str(tmp_path))})
    def broken(*args,**kwargs):raise KeyError('missing selected candidate field')
    monkeypatch.setitem(role.HANDLERS,(Role.SELECTION,'broken'),broken)
    runtime=dict(settings={},operation='broken',seconds=10,device='cpu')
    assert role.main(['selection',json.dumps(runtime)])==0
    record=json.loads((tmp_path/'result.json').read_text())
    assert record['status']=='FAIL' and record['reason']=='KeyError'


def test_retained_candidates_loss_and_behavior_use_same_selection_roster(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_scoring as scoring,generalization_evaluation as evaluation
    from oeis_learn.fresh import pilot_metrics
    from oeis_learn.pipeline_validation.identity import seal,digest
    truth,candidate_root,row=fixture(tmp_path)
    rows=[row,dict(row,id='two',stratum='structural')]
    (truth/'rows.json').write_text(json.dumps(rows))
    (candidate_root/'weights.pt').write_bytes(b'weights')
    weights=reference(candidate_root/'weights.pt',candidate_root)
    candidates=[]
    for number in (1,2):
        checkpoint=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=weights,
            state=dict(path='private-state-not-mounted.json',sha256=digest('state'),bytes=20),
            initialization=digest('init'),parent=None,tokens=number,updates=number,pins=['retained'],compatibility=digest('contract')))
        candidates.append(dict(checkpoint=checkpoint,selections=reference(candidate_root/'rows.json',candidate_root)))
    count=[]
    def score(*args,**kwargs):
        count.append(1)
        return dict(result=dict(outcomes={r['id']:{'equal_budget':dict(complete=True,success=len(count)==2)} for r in rows}))
    monkeypatch.setattr(scoring,'score_selection',score)
    monkeypatch.setattr(evaluation,'load_retained',lambda *a,**k:(None,None,{}))
    def measure(encoder,decoder,measured,device,**kw):
        assert len(measured)==2 and all(len(r['values'])==20 for r in measured)
        assert device=='cpu'
        return dict(complete=True,loss=float(len(count)))
    monkeypatch.setattr(pilot_metrics,'measure',measure)
    result=scoring.select_checkpoints(truth,reference(truth/'rows.json',truth),candidate_root,candidates,
        recipe={},client=None,output=tmp_path/'selected',seconds=30)
    assert result['best_loss']==candidates[0]['checkpoint']['identity']
    assert result['selected']==candidates[1]['checkpoint']['identity']
    assert len(result['measurements'])==2
    rows[1]['role']='assessment';(truth/'rows.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='selection-only'):
        scoring.select_checkpoints(truth,reference(truth/'rows.json',truth),candidate_root,candidates,
            recipe={},client=None,output=tmp_path/'forbidden',seconds=30)
    assert not (tmp_path/'forbidden').exists()


def test_assessment_requires_frozen_plan_active_debit_and_exact_candidates(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_scoring as scoring
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.fresh.bigint_state import expose_final,finish_final
    from tests.integration.test_bigint_workflow import _plan
    truth,candidates,row=fixture(tmp_path);row['role']='assessment'
    (truth/'rows.json').write_text(json.dumps([row]))
    targets=reference(truth/'rows.json',truth);selected=reference(candidates/'rows.json',candidates)
    authorization_root=tmp_path/'authorization';authorization_root.mkdir()
    plan=_plan();plan['cohort']=targets['sha256'];plan['denominator']=1
    arm=plan['arms'][0]
    name,exposure=expose_final(tmp_path/'custodian',plan,cohort=plan['cohort'],checkpoint=arm[2],
        recipe=arm[3],portfolio=arm[4],runtime=arm[5],seed=arm[1],debit=10)
    authorization=seal(dict(version='generalization-assessment-authorization/v1',plan=plan,
        exposure=exposure,arm=name,targets=targets,candidates=selected,checkpoint=arm[2]))
    path=authorization_root/'authorization.json';path.write_text(json.dumps(authorization))
    calls=[]
    def score(selections,targets,client,**kwargs):
        calls.append(1);assert targets['one']['role']=='assessment'
        return dict(outcomes={})
    monkeypatch.setattr(scoring,'score_frozen',score)
    result=scoring.score_assessment(truth,authorization_root,candidates,reference(path,authorization_root),
        client=None,output=tmp_path/'result',seconds=10)
    assert result['status']=='PASS' and len(calls)==1
    (candidates/'rows.json').write_text('{}')
    with pytest.raises(ValueError,match='size|hash'):
        scoring.score_assessment(truth,authorization_root,candidates,reference(path,authorization_root),
            client=None,output=tmp_path/'changed',seconds=10)
    authorization['exposure']=finish_final(tmp_path/'custodian',plan,name,'completed')
    path.write_text(json.dumps(seal(authorization)))
    with pytest.raises(ValueError,match='active'):
        scoring.score_assessment(truth,authorization_root,candidates,reference(path,authorization_root),
            client=None,output=tmp_path/'completed',seconds=10)
    assert len(calls)==1
