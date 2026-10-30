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
