import copy
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation.identity import digest,seal
from oeis_learn.synthesis.admission import VisiblePrompt,source_identity

SOURCE='(program (return (param.get p0)))'
INDICES=list(map(str,range(20)))
PROMPT=dict(id='case',indices=INDICES,values=INDICES)


def fixture():
    clock=dict(admitted=100.,deadline=104.,dispatched=100.01,completed=102.,owner_observed=102.01)
    details=[dict(position=i,seed=987+i,source=SOURCE,source_sha256=source_identity(SOURCE),
        generation=dict(complete=True),fitting=dict(complete=i!=3,outcome='resource_exhausted' if i==3 else 'unsat_in_scope',
        clock=clock),native=dict(outcome='resource_exhausted' if i==3 else 'unsat_in_scope')) for i in range(8)]
    records={'case':dict(details=dict(correct=dict(details=details)),selections=dict(equal_budget=None))}
    targets={'case':['resource_exhausted' if i==3 else 'unsat_in_scope' for i in range(8)]}
    return records,targets


def test_exact_extraction_keeps_all_attempts_and_only_visible_prompt():
    from oeis_learn.generalization.selector_fit_replay import extract
    records,targets=fixture();before=copy.deepcopy(records)
    packet=extract(records,[PROMPT],targets)
    assert len(packet['attempts'])==8 and packet['attempts'][3]['original_outcome']=='resource_exhausted'
    assert [r['seed'] for r in packet['attempts']]==list(range(987,995))
    assert all(r['service_seconds']==4 for r in packet['attempts'])
    assert packet['prompts']==[PROMPT] and records==before
    assert packet['attempts'][0]['prompt_sha256']==VisiblePrompt(tuple(INDICES),tuple(INDICES)).sha256


@pytest.mark.parametrize('mutation', ['source','seed','position','incomplete','outcomes','cap','prompt'])
def test_changed_or_partial_original_attempts_are_rejected(mutation):
    from oeis_learn.generalization.selector_fit_replay import extract
    records,targets=fixture();d=records['case']['details']['correct']['details'][0];prompts=[copy.deepcopy(PROMPT)]
    if mutation=='source':d['source']='different'
    if mutation=='seed':d['seed']=True
    if mutation=='position':d['position']=1
    if mutation=='incomplete':d['generation']['complete']=False
    if mutation=='outcomes':targets['case'][0]='verified_solution'
    if mutation=='cap':d['fitting']['clock']['deadline']=105
    if mutation=='prompt':prompts[0]['values']+=['20']
    with pytest.raises(ValueError):extract(records,prompts,targets)


def test_timeout_selection_excludes_decoder_failures_and_completed_cases():
    from oeis_learn.generalization.selector_fit_replay import timeout_targets
    case=lambda complete,reason,selected=None:dict(equal_budget=dict(complete=complete,reason=reason,
        candidate_sha256=selected,upstream=dict(generation=dict(attempted=8,completed=8,expected=8),
        fitting=dict(outcomes=['unsat_in_scope']*7+['resource_exhausted']))))
    outcomes={'timeout':case(False,'resource_exhausted:timeout'),'decoder':case(False,'grammar_dead_end'),
              'complete':case(True,'continuation_mismatch')}
    assert list(timeout_targets(outcomes))==['timeout']
    with pytest.raises(ValueError):timeout_targets({'chosen':case(False,'resource_exhausted:timeout','selected')})


def test_core_binding_rejects_changed_solver_but_allows_new_diagnostic_module():
    from oeis_learn.generalization.selector_fit_replay import check_core
    old={'src/oeis_learn/synthesis/client.py':'sha','crates/oeis_synthesis_core/src/smt.rs':'rust','Dockerfile':'image'}
    check_core(old,dict(old,**{'src/oeis_learn/pipeline_validation/new_diagnostic.py':'new'}))
    with pytest.raises(ValueError):check_core(old,dict(old,**{'src/oeis_learn/synthesis/client.py':'changed'}))
    with pytest.raises(ValueError):check_core(old,dict(old,**{'src/oeis_learn/synthesis/new_policy.py':'new'}))


def test_replay_preserves_limits_seeds_and_reports_no_scientific_authority(tmp_path):
    from oeis_learn.generalization.selector_fit_replay import extract
    from oeis_learn.pipeline_validation.selector_fit_replay_role import replay
    records,targets=fixture();packet=extract(records,[PROMPT],targets);calls=[]
    class Worker:
        stats={}
        @contextmanager
        def affinity(self,n):yield
    class Client:
        def __init__(self,*a,**k):self.worker=Worker()
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a,**kw):return dict(outcome='ok')
        def runtime_identity(self,*a):return digest('runtime')
        reference_sha256=digest('reference')
        def fit(self,source,prompt,**kw):
            calls.append((source,prompt,kw))
            assert 3.9<kw['deadline']-time.monotonic()<=4
            return dict(outcome='unsat_in_scope',complete=True,admitted=[],native=dict(elapsed_seconds=.01,timing={'search_allowance_seconds':1.2}))
    result=replay(packet,{},tmp_path,concurrency=4,repetitions=2,client_factory=Client)
    assert len(calls)==16 and [c[2]['seed'] for c in calls[:8]]==list(range(987,995))
    assert all(len(c[1].values)==20 for c in calls)
    assert result['freeze_eligible'] is False and result['scientific_status']=='DIAGNOSTIC_ONLY'
    assert len(result['runs'])==2 and all(len(r['attempts'])==8 for r in result['runs'])
    assert result['runs'][0]['cases']['case']['fit_search_complete'] is True


def test_transport_observer_delegates_exact_request_and_deadline():
    from oeis_learn.pipeline_validation.selector_fit_replay_role import ObservedWorker
    response=dict(complete=True,metrics=dict(queue_wait_seconds=.01),outcome='ok')
    request=dict(operation='fit',payload='opaque');deadline=time.monotonic()+1;seen=[]
    worker=SimpleNamespace(stats={},call=lambda r,**kw:(seen.append((r,kw)) or response))
    observed=ObservedWorker(worker);observed.local.calls=[]
    assert observed.call(request,deadline=deadline) is response
    assert seen==[(request,dict(deadline=deadline))]
    assert observed.local.calls[0]['metrics']==response['metrics']
    assert 'payload' not in str(observed.local.calls)


def test_matrix_changes_only_cpu_quota_for_resource_comparison():
    from oeis_learn.pipeline_validation.selector_fit_replay_commands import matrix
    rows=matrix(8)
    assert [r['concurrency'] for r in rows]==[1,4,4]
    assert [r['cpus'] for r in rows]==[4,4,8]
    with pytest.raises(ValueError):matrix(4)


def test_replay_rejects_hidden_values_and_changed_source_before_native_work(tmp_path):
    from oeis_learn.generalization.selector_fit_replay import extract
    from oeis_learn.pipeline_validation.selector_fit_replay_role import replay
    records,targets=fixture();packet=extract(records,[PROMPT],targets)
    packet['prompts'][0]['values']+=list(map(str,range(20,100)))
    packet=seal(packet)
    with pytest.raises(ValueError):replay(packet,{},tmp_path,concurrency=1,repetitions=1,client_factory=lambda *a:pytest.fail('invalid prompt reached native'))
    packet=extract(records,[PROMPT],targets);packet['attempts'][0]['source']='changed';packet=seal(packet)
    with pytest.raises(ValueError):replay(packet,{},tmp_path,concurrency=1,repetitions=1,client_factory=lambda *a:pytest.fail('changed source reached native'))


def test_native_verification_failure_cannot_pass_as_diagnostic_success(tmp_path):
    from oeis_learn.generalization.selector_fit_replay import extract
    from oeis_learn.pipeline_validation.selector_fit_replay_role import replay
    records,targets=fixture();packet=extract(records,[PROMPT],targets)
    class Worker:
        stats={}
        @contextmanager
        def affinity(self,n):yield
    class Client:
        reference_sha256=digest('reference')
        def __init__(self,*a):self.worker=Worker()
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def runtime_identity(self,*a):return digest('runtime')
        def execute(self,*a,**k):return dict(outcome='ok')
        def fit(self,*a,**k):return dict(outcome='verification_failure',complete=False,admitted=[],reason='independent disagreement')
    result=replay(packet,{},tmp_path,concurrency=1,repetitions=1,client_factory=Client)
    assert result['status']=='FAIL' and result['integrity_failures']==8
    assert result['runs'][0]['attempts'][0]['reason']=='independent disagreement'
