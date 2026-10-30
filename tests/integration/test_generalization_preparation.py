import json
import pytest
from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell
from oeis_learn.pipeline_validation.generalization_roles import mounts


def source(i):
    return dict(id=str(i),stage='A',source=f'(program (return (add $index (int {i}))))',parameter_source=f'(program (return (add $index (int {i}))))',family='family'+str(i),variant=str(i),group='group'+str(i),origin='generic_random')

def execute(src,indices):
    i=int(src.split('(int ')[1].split(')')[0]);return dict(values=[str(int(n)+i) for n in indices],outcome='ok')

def test_separate_visible_and_scorer_files_and_shortfall(tmp_path):
    result=prepare_cell([source(1),source(2)],role='selection',holdout='parameter',count=2,quotas={},structural=False,output=tmp_path/'good',execute=execute)
    assert result['status']=='PASS'
    visible=json.loads((tmp_path/'good/prompts.json').read_text())
    assert all(set(r)=={'id','indices','values'} and len(r['values'])==20 for r in visible)
    assert len(json.loads((tmp_path/'good/private/targets.json').read_text())[0]['values'])==100
    failed=prepare_cell([source(1)],role='assessment',holdout='structural',count=2,quotas={'loop_active':2},structural=True,output=tmp_path/'short',execute=execute)
    assert failed['status']=='BLOCKED' and not (tmp_path/'short/prompts.json').exists()
    assert not (tmp_path/'short/study.json').exists()
    with pytest.raises(ValueError): prepare_cell([],role='training',holdout='parameter',count=1,quotas={},structural=False,output=tmp_path/'good',execute=execute)


def test_native_disagreement_does_not_publish(tmp_path):
    result=prepare_cell([source(1)],role='training',holdout='parameter',count=1,quotas={},structural=False,output=tmp_path/'bad',execute=lambda s,i:dict(values=['999']*100,outcome='ok'))
    assert result['status']=='BLOCKED'


def test_mounts_are_separate_and_read_only(tmp_path):
    for name in ('prompts','weights','truth'): (tmp_path/name).mkdir()
    result=mounts(tmp_path,'generator',{'visible_prompts':'prompts','frozen_weights':'weights'},output='generated')
    assert all(m['read_only'] for m in result[:-1]) and not result[-1]['read_only']
    with pytest.raises(ValueError): mounts(tmp_path,'generator',{'visible_prompts':'prompts','frozen_weights':'weights','assessment_truth':'truth'},output='generated')
