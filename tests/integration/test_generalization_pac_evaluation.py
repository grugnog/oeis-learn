import json
import pytest
from oeis_learn.pipeline_validation.identity import seal,digest,write
from oeis_learn.pipeline_validation.generalization_preparation import reference


def test_custodian_publishes_only_visible_prefix_and_rejects_changed_truth(tmp_path):
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import publish_views
    row=dict(id='original',role='selection',source='(program (return $index))',family=digest('family'),holdout='structural',
             indices=list(map(str,range(100))),values=list(map(str,range(100))))
    write(tmp_path/'truth.json',[row]);key=digest(['selection',row['id'],row['source']])
    admission=seal(dict(version='pac-admission/v2',structural=reference(tmp_path/'truth.json',tmp_path),roster={key:dict(family=digest('family'),category='certified',stratum='structural')}))
    write(tmp_path/'admission.json',admission);ref=reference(tmp_path/'admission.json',tmp_path)
    result=publish_views({'mechanical_sources':tmp_path},tmp_path/'out',dict(admission=ref),seconds=60)
    prompts=json.loads((tmp_path/'out'/result['prompts']['path']).read_text())
    assert prompts==[dict(id=key,indices=list(map(str,range(20))),values=list(map(str,range(20))))]
    assert 'source' not in prompts[0] and 'family' not in prompts[0]
    (tmp_path/'truth.json').write_text('[]')
    with pytest.raises(ValueError):publish_views({'mechanical_sources':tmp_path},tmp_path/'bad',dict(admission=ref),seconds=60)


def test_checkpoint_request_rejects_unbound_weights_and_unfrozen_arm(tmp_path):
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import checkpoint_request
    with pytest.raises((ValueError,OSError)):checkpoint_request(tmp_path,'D=weights.pt',{'arms':['B','C']},{},None,None)
    with pytest.raises((ValueError,OSError)):checkpoint_request(tmp_path,'B=weights.pt',{'arms':['B','C']},{},None,None)
