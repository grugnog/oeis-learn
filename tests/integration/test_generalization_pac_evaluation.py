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


@pytest.mark.parametrize('amended',[False,True])
def test_actual_scorer_adapter_preserves_no_fit_completion_and_censoring(tmp_path,monkeypatch,amended):
    from contextlib import nullcontext
    from tests.unit.test_generalization_metrics import pac_roster
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import score_views
    from oeis_learn.synthesis import client
    roster={key:dict(row,family=digest(row['family'])) for key,row in pac_roster().items()}
    if amended:
        from tests.unit.test_generalization_structural_amendment import amended_roster
        roster=amended_roster()
    truth=[dict(id=key,role='selection',stratum='structural',family=row['family'],
        source='(program (return (int 0)))',indices=list(map(str,range(100))),values=['0']*100) for key,row in roster.items()]
    modes=['raw','rescue','equal_budget','shuffled','absent','template','bank']
    selected={key:dict(selections=dict.fromkeys(modes),arm_evidence={m:dict(complete=True,reason='no_admitted_candidate') for m in modes}) for key in roster}
    censored=next(k for k,v in roster.items() if v['category']=='certified');selected[censored]['arm_evidence']['equal_budget']=dict(complete=False,reason='decoder deadline')
    roster_record=roster
    if amended:
        from oeis_learn.generalization import structural
        roster_record=dict(version='pac-evaluation-roster/v2',structural_scope=structural.scope(structural.amended_design(structural.prior_protocol())),tasks=roster)
    write(tmp_path/'truth.json',truth);write(tmp_path/'roster.json',roster_record);write(tmp_path/'candidates.json',selected)
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:nullcontext(None))
    result=score_views(dict(selection_supervision=tmp_path,selection_candidates=tmp_path),tmp_path/'result',
        dict(targets=reference(tmp_path/'truth.json',tmp_path),roster=reference(tmp_path/'roster.json',tmp_path),
             candidates=reference(tmp_path/'candidates.json',tmp_path),study=digest('study')),seconds=30)
    assert result['outcomes']['equal_budget'][censored] is None
    assert result['controls']['equal_budget']['certified']['censored']==1
    assert result['controls']['bank']['certified']['censored']==0
    assert all(result['outcomes']['bank'][key] is False for key in roster)
    assert set(result['controls'])==set(modes)
    if amended:assert result['controls']['bank']['certified']['families']==15 and result['structural_scope']['version']=='pac-structural-scope/v2'


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


def test_amended_custodian_publishes_bound_scope_without_generator_truth(tmp_path):
    from tests.unit.test_generalization_structural_amendment import amended_roster
    from oeis_learn.generalization import structural
    from oeis_learn.pipeline_validation.generalization_pac_evaluation import publish_views
    rows=[];roster={}
    for i,metadata in enumerate(amended_roster().values()):
        source=f'(program (return (int {i})))';row=dict(id=str(i),role='selection',holdout='structural',source=source,
            family=metadata['family'],indices=list(map(str,range(100))),values=[str(i)]*100)
        rows.append(row);roster[digest(['selection',row['id'],source])]=metadata
    write(tmp_path/'truth.json',rows)
    scope=structural.scope(structural.amended_design(structural.prior_protocol()))
    write(tmp_path/'admission.json',seal(dict(version='pac-admission/v3',structural_scope=scope,roster=roster,structural=reference(tmp_path/'truth.json',tmp_path))))
    result=publish_views(dict(mechanical_sources=tmp_path),tmp_path/'views',dict(admission=reference(tmp_path/'admission.json',tmp_path)),seconds=30)
    visible=json.loads((tmp_path/'views'/result['prompts']['path']).read_text())
    assert len(visible)==128 and all(set(r)=={'id','indices','values'} and len(r['values'])==20 for r in visible)
    actual=json.loads((tmp_path/'views'/result['roster']['path']).read_text())
    assert actual['structural_scope']==scope and actual['tasks']==roster
