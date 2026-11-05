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


def test_actual_scorer_adapter_preserves_no_fit_completion_and_censoring(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from tests.unit.test_generalization_metrics import pac_roster
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import score_views
    from oeis_learn.synthesis import client
    roster={key:dict(row,family=digest(row['family'])) for key,row in pac_roster().items()}
    truth=[dict(id=key,role='selection',stratum='structural',family=row['family'],
        source='(program (return (int 0)))',indices=list(map(str,range(100))),values=['0']*100) for key,row in roster.items()]
    modes=['raw','rescue','equal_budget','shuffled','absent','template','bank']
    selected={key:dict(selections=dict.fromkeys(modes),arm_evidence={m:dict(complete=True,reason='no_admitted_candidate') for m in modes}) for key in roster}
    censored=next(iter(roster));selected[censored]['arm_evidence']['equal_budget']=dict(complete=False,reason='decoder deadline')
    write(tmp_path/'truth.json',truth);write(tmp_path/'roster.json',roster);write(tmp_path/'candidates.json',selected)
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:nullcontext(None))
    result=score_views(dict(selection_supervision=tmp_path,selection_candidates=tmp_path),tmp_path/'result',
        dict(targets=reference(tmp_path/'truth.json',tmp_path),roster=reference(tmp_path/'roster.json',tmp_path),
             candidates=reference(tmp_path/'candidates.json',tmp_path),study=digest('study')),seconds=30)
    assert result['outcomes']['equal_budget'][censored] is None
    assert result['controls']['equal_budget']['certified']['censored']==1
    assert result['controls']['bank']['certified']['censored']==0
    assert all(result['outcomes']['bank'][key] is False for key in roster)
    assert set(result['controls'])==set(modes)


def test_frozen_retained_checkpoint_read_is_stdlib_only(tmp_path):
    import subprocess,sys,os
    from pathlib import Path
    (tmp_path/'weights.pt').write_bytes(b'opaque retained weights')
    write(tmp_path/'state.json',{})
    descriptor=seal(dict(version='generalization-checkpoint/v1',kind='full_state',weights=reference(tmp_path/'weights.pt',tmp_path),
        state=reference(tmp_path/'state.json',tmp_path),initialization=digest('init'),parent=None,tokens=25_000_000,updates=1,pins=['latest'],compatibility=digest('contract')))
    write(tmp_path/'checkpoint.json',descriptor)
    code='''import sys,json
from pathlib import Path
from oeis_learn.pipeline_validation.generalization_pac_evaluation import checkpoint_request
root=Path(sys.argv[1]);d=json.loads((root/'checkpoint.json').read_text())
a,_,_=checkpoint_request(root,'A=checkpoint.json',dict(arms=['B','C'],control_mode='reference-only'),dict(control=dict(checkpoint=d['identity'])),None,None)
assert a=='A'
'''
    repo=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-S','-c',code,str(tmp_path)],cwd=repo,
        env=dict(os.environ,PYTHONPATH=str(repo/'src')),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
