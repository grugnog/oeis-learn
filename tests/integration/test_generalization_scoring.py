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
