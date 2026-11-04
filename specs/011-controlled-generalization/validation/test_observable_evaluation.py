import importlib.util
from pathlib import Path
import sys
import time
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('observable_evaluation',HERE/'observable_evaluation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def branch(threshold=20,op='lt',yes='(mul $index (param.get p0))',no='(add $index (param.get p1))'):
    return f'(program (local $b0 bigint) (if ({op} $index (int {threshold})) (then (set $b0 {yes})) (else (set $b0 {no}))) (return (get $b0)))'


def test_polynomial_certificate_and_shared_parameters():
    source='(program (return (add (mul (param.get p0) (mul $index $index)) (mul (param.get p0) $index))))'
    c=m.classify(source,deadline=time.monotonic()+5)
    assert c['category']=='certified'
    assert c['certificate']['paths'][0]['degree_bound']==2
    assert c['certificate']['paths'][0]['visible_indices']==list(range(20))


@pytest.mark.parametrize('threshold,expected',[(18,'certified'),(19,'unresolved'),(20,'ambiguous')])
def test_branch_requires_enough_distinct_observations(threshold,expected):
    c=m.classify(branch(threshold),deadline=time.monotonic()+5)
    assert c['category']==expected


def test_single_late_point_is_ambiguous_but_parameter_free_path_is_safe():
    assert m.classify(branch(46,'eq'),deadline=time.monotonic()+5)['category']=='ambiguous'
    assert m.classify(branch(46,'eq',yes='(int 7)'),deadline=time.monotonic()+5)['category']=='certified'


def test_no_witness_is_not_a_certificate():
    cancelled=branch(20,no='(sub (param.get p1) (param.get p1))')
    c=m.classify(cancelled,deadline=time.monotonic()+5)
    assert c['category']=='unresolved' and c['witness'] is None


def test_loop_and_parameter_control_are_not_certified():
    loop='(program (local $b0 bigint) (local $c0 control) (set $b0 (param.get p0)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (param.get p1)))) (return (get $b0)))'
    assert m.classify(loop,deadline=time.monotonic()+5)['category']=='unresolved'
    controlled=branch().replace('(int 20)','(param.get p2)')
    assert m.classify(controlled,deadline=time.monotonic()+5)['category']=='unresolved'


def test_high_degree_is_not_certified_and_path_specific_local_degree_is_used():
    power='$index'
    for _ in range(5):power=f'(mul {power} {power})'
    c=m.classify(f'(program (return (mul (param.get p0) {power})))',deadline=time.monotonic()+5)
    assert c['category']=='unresolved'
    assert m.classify(branch(3,yes='(mul (param.get p0) (mul $index $index))'),deadline=time.monotonic()+5)['category']=='certified'


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError):m.classify('(program (return (get $unknown)))',deadline=time.monotonic()+5)


def test_projection_preserves_categories_and_censoring_under_swapped_arms():
    rows={'a':{'category':'certified'},'b':{'category':'ambiguous'},'c':{'category':'unresolved'}}
    def result(win,done=True):return {'equal_budget':dict(success=win,complete=done)}
    scores={'static':{'a':result(True),'b':result(False),'c':result(False,False)},
            'refreshed':{'a':result(False),'b':result(False,False),'c':result(True)}}
    before=repr(rows); x=m.project(rows,scores)
    y=m.project(rows,dict(static=scores['refreshed'],refreshed=scores['static']))
    assert repr(rows)==before and x['static']==y['refreshed']
    assert x['static']['unresolved']['evaluation_unresolved']==1
    assert x['refreshed']['ambiguous']['classification_tasks']==1
    scores['static']['extra']=result(True)
    with pytest.raises(ValueError):m.project(rows,scores)


def test_witness_resource_limit_stays_unresolved():
    c=m.classify(branch(20),deadline=time.monotonic()-1)
    assert c['category']=='unresolved' and c['reason']=='analysis_deadline'


def test_projection_rejects_candidate_file_and_selected_source_mismatch():
    records={'a':{'selections':{'equal_budget':None}}}
    report=dict(candidates={'sha256':'good','bytes':12},result={'outcomes':{'a':{'equal_budget':dict(candidate_sha256=None,complete=True,success=False)}}})
    ref=dict(sha256='good',bytes=12)
    assert m.validate_scores(records,report,ref,{'proposals':[]})[1]==0
    with pytest.raises(ValueError):m.validate_scores(records,report,dict(ref,sha256='wrong'),{'proposals':[]})
    report['result']['outcomes']['a']['equal_budget']['candidate_sha256']='wrong'
    with pytest.raises(ValueError):m.validate_scores(records,report,ref,{'proposals':[]})


def test_protocol_family_bindings_reject_cross_arm_disagreement():
    from oeis_learn.pipeline_validation.identity import digest
    tasks=[str(i) for i in range(256)];roster={}
    for task in tasks:
        for arm in ('static','refreshed'):
            roster[digest([arm,'equal_budget',task])]=dict(family=arm,cell=arm+'/equal_budget/parameter')
    with pytest.raises(ValueError,match='paired task/family'):
        m.freeze_protocol({'identity':'x'},{'roster':roster},tasks,{}, {},deadline=time.monotonic()+5)


def test_failure_export_and_overwrite_are_explicit(tmp_path):
    import json
    out=tmp_path/'result'
    with pytest.raises(RuntimeError):m.run(tmp_path/'missing.zip',out,seconds=1)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['selected_scope']=='FAIL' and summary['qualified'] is False
    assert m.verify_input(out.with_suffix('.zip'))['summary_state']=='FAIL'
    before=(out/'summary.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(tmp_path/'missing.zip',out,seconds=1)
    assert (out/'summary.json').read_bytes()==before
