"""Local tests of the temporary resume matrix; no HIP execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('probe_resume',HERE/'probe_resume.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_plan_has_no_search_evaluation_or_qualification_suite():
    args=SimpleNamespace(engine='podman',run='run-state/diagnostic-001',output='run-state/matrix')
    plan=helper.plan(args)
    assert [s['name'] for s in plan]==['compatibility','continue-a','continue-b','split-first','split-rest','compare']
    assert [s for s in plan if s['name']=='split-rest'][0]['needs']==['compatibility','split-first']
    assert all('qualify' not in s['argv'] and 'evaluate' not in s['argv'] for s in plan)


def test_training_selects_same_start_for_repeats_and_an_extra_boundary_for_split(tmp_path):
    context=dict(paused=tmp_path/'paused',seed=7,paused_updates=1,final_updates=3)
    def previous(name):return tmp_path/name
    for name in ('continue-a','continue-b','split-first'):
        start,stop=helper.branch(name,context,previous)
        assert start==tmp_path/'paused'
        assert stop==(2 if name=='split-first' else None)
    assert helper.branch('split-rest',context,previous)==(tmp_path/'split-first',None)


def test_comparison_is_exact_and_covers_every_pair():
    import torch
    from oeis_learn.generalization.budgets import Ledger
    def state(x):
        return dict(ledger=Ledger(10),states=dict(weights={'w':torch.tensor([x])},optimizer={},rng={},data={},replay={}))
    records={n:state(1.) for n in ('a','b','split','original','old-resumed')}
    records['split']=state(1.000001)
    result=helper.compare_states(records)
    assert len(result)==10
    assert result['a:b']['equal']
    assert not result['a:split']['equal']
    assert not result['a:split']['components']['weights']['equal']


def test_independent_branches_continue_after_failure(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import execute
    args=SimpleNamespace(engine='podman',run='run-state/diagnostic-001',output='run-state/matrix')
    called=[]
    def runner(stage,root):
        called.append(stage['name']);return {'status':'FAIL' if stage['name']=='continue-a' else 'PASS'}
    execute(helper.plan(args),tmp_path,runner=runner)
    assert 'continue-b' in called and 'split-rest' in called
    assert 'compare' in called  # Comparison preserves whichever pairs are available.


def test_missing_split_checkpoint_blocks_only_its_dependent_branch(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import execute
    args=SimpleNamespace(engine='podman',run='run-state/diagnostic-001',output='run-state/matrix')
    called=[]
    def runner(stage,root):
        called.append(stage['name']);return {'status':'FAIL' if stage['name']=='split-first' else 'PASS'}
    execute(helper.plan(args),tmp_path,runner=runner)
    assert 'continue-b' in called and 'split-rest' not in called and 'compare' in called


def test_matrix_helpers_do_not_invalidate_existing_installed_sources():
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    assert not any('probe_resume' in p for p in collect_snapshot(helper.ROOT))


@pytest.mark.parametrize('name',helper.BRANCHES)
def test_production_training_receives_frozen_inputs_and_expected_stop(tmp_path,monkeypatch,name):
    import json
    from oeis_learn.pipeline_validation import generalization_commands as commands
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    paused=tmp_path/'paused';(paused/'checkpoints').mkdir(parents=True)
    (paused/'checkpoints/state.json').write_text('{}')
    prior=dict(checkpoint={'path':'state.json','sha256':'sha256:'+'a'*64},compatibility='contract')
    (paused/'training.json').write_text(json.dumps(prior))
    pool=tmp_path/'pool';pool.mkdir();(pool/'pool.json').write_text('{}')
    output=tmp_path/'matrix'/name;output.mkdir(parents=True)
    if name=='split-rest':
        stage=output.parent/'split-first';stage.mkdir()
        (stage/'result.json').write_text(json.dumps(dict(status='PASS',learner='paused')))
    recipe=object();study=object();runtime={'replay':{'tasks':{'path':'tasks.json'}}}
    execution={'recipe':recipe}
    monkeypatch.setattr(commands,'_validate',lambda *a,**k:execution)
    calls=[]
    def train(host,workspace,destination,base,observed_study,observed_runtime,plan,**kw):
        assert observed_study is study and observed_runtime is runtime and plan is execution
        assert kw['resume']==str((paused/'checkpoints/state.json').relative_to(tmp_path))
        assert kw['arms']==['verified_replay'] and kw['seeds']==[7]
        stop=2 if name=='split-first' else None
        assert kw['stop_after_updates']==stop
        calls.append(kw)
        return dict(runs=[dict(root='learner',result=dict(status='PARTIAL' if stop else 'PASS',updates=stop or 3,
                    compatibility='contract',device='cuda',checkpoint={'sha256':'sha256:'+'b'*64}))])
    monkeypatch.setattr(commands,'train',train)
    context=dict(base=tmp_path,study=study,runtime=runtime,paused=paused,seed=7,
                 paused_updates=1,final_updates=3,pool=pool)
    result=helper.train_branch(name,context,SimpleNamespace(select_gpu=lambda:None),output)
    assert result['status']=='PASS' and result['start_checkpoint']==prior['checkpoint']['sha256']
    assert len(calls)==1


def test_export_preserves_comparisons_but_excludes_private_models(tmp_path,monkeypatch):
    import json,zipfile
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import save
    from oeis_learn.pipeline_validation.bundle import verify
    root=tmp_path/'matrix';root.mkdir()
    save(root/'diagnostic.json',dict(status='PASS',plan=[]))
    save(root/'compare/comparison.json',dict(status='PASS',repeat_equal=False))
    save(root/'private/secret.json',{'private':'must not escape'})
    archive=helper.export(root);verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert 'comparison.json' in z.namelist()
        assert not any('private' in n for n in z.namelist())
        assert json.loads(z.read('summary.json'))['feature_qualified'] is False


@pytest.mark.parametrize('engine',['docker','podman'])
@pytest.mark.parametrize('start',['paused','fresh'])
def test_comparison_container_mounts_all_endpoints_read_only_without_gpu(tmp_path,monkeypatch,engine,start):
    import json
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import supervision
    monkeypatch.syspath_prepend(str(HERE))
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    from diagnose_hip import save
    adapter=Engine(engine,executable=engine,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda *_:pytest.fail('CPU comparison must not mount a GPU'))
    host=SimpleNamespace(engine=adapter,policy=dict(memory_bytes=16<<30,cpus=4))
    full=tmp_path/'full';full.mkdir();old=tmp_path/'old';old.mkdir()
    output=tmp_path/'matrix/compare';output.mkdir(parents=True)
    for name in ('continue-a','continue-b','split-first','split-rest'):
        folder=tmp_path/name;folder.mkdir()
        save(output.parent/name/'result.json',dict(status='PASS',learner=name))
    captured=[]
    def launch(argv,*a,**kw):
        captured.append(argv)
        save(output/'private/comparison/comparison.json',dict(complete=True,observation='new_continuations_agree'))
    monkeypatch.setattr(supervision,'launch',launch)
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    result=helper.compare(host,dict(start=start,saved={'full':full,'resumed':old,'paused':old},runtime={'images':['cpu','sha256:'+'a'*64]}),output)
    assert result['complete']
    argv=captured[0]
    assert '/helper/probe_resume.py' in argv
    assert ('--container-fresh' if start=='fresh' else '--container') in argv
    mounts=[argv[i+1] for i,arg in enumerate(argv) if arg=='--mount']
    assert len(mounts)==(9 if start=='fresh' else 7) and all('readonly' in m for m in mounts[:-1])
    assert '--network=none' in argv


def test_fresh_plan_propagates_mode_and_branches_from_initializer(tmp_path):
    args=SimpleNamespace(engine='podman',run='old',output='new',start='fresh')
    for stage in helper.plan(args):
        assert stage['argv'][-4:]==['--start','fresh','--policy','observe']
    context=dict(paused=tmp_path/'paused',paused_updates=1,start='fresh')
    for name in ('continue-a','continue-b','split-first'):
        assert helper.branch(name,context,lambda _:None)==(None,1 if name=='split-first' else None)
    assert helper.branch('split-rest',context,lambda _:tmp_path/'new-pause')==(tmp_path/'new-pause',None)


@pytest.mark.parametrize('name',['continue-a','continue-b','split-first'])
def test_fresh_training_uses_initializer_without_resume(tmp_path,monkeypatch,name):
    import json
    from oeis_learn.pipeline_validation import generalization_commands as commands
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    old=tmp_path/'old';old.mkdir()
    (old/'training.json').write_text(json.dumps(dict(compatibility='contract')))
    pool=tmp_path/'pool';pool.mkdir();(pool/'pool.json').write_text('{}')
    runtime={'replay':{'tasks':{'path':'tasks.json'}}}
    initializer={'weights':{'sha256':'sha256:'+'c'*64}}
    execution={'initializer':initializer}
    monkeypatch.setattr(commands,'_validate',lambda *a,**k:execution)
    def train(*a,**kw):
        assert a[6] is execution
        assert kw['resume'] is None
        assert kw['stop_after_updates']==(1 if name=='split-first' else None)
        return dict(runs=[dict(root='learner',result=dict(status='PARTIAL' if name=='split-first' else 'PASS',
            updates=1 if name=='split-first' else 3,compatibility='contract',device='cuda',checkpoint={'sha256':'sha256:'+'d'*64}))])
    monkeypatch.setattr(commands,'train',train)
    context=dict(base=tmp_path,study={},runtime=runtime,paused=old,seed=7,start='fresh',
                 paused_updates=1,final_updates=3,pool=pool)
    result=helper.train_branch(name,context,SimpleNamespace(select_gpu=lambda:None),tmp_path/'matrix'/name)
    assert result['start_checkpoint'] is None and result['initializer']==initializer['weights']['sha256']


def test_export_includes_bounded_checkpoint_provenance(tmp_path,monkeypatch):
    import json,zipfile
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import save
    root=tmp_path/'matrix';root.mkdir()
    save(root/'diagnostic.json',dict(status='PASS',plan=[dict(name='continue-a')]))
    save(root/'continue-a/result.json',dict(status='PASS',start_checkpoint='sha256:'+'a'*64,
        checkpoint='sha256:'+'b'*64,updates=3,learner_status='PASS',initializer=None,
        learner='private/location',secret='must not escape'))
    with zipfile.ZipFile(helper.export(root)) as z:
        provenance=json.loads(z.read('branches.json'))
        assert provenance['continue-a']['start_checkpoint']=='sha256:'+'a'*64
        assert 'learner' not in provenance['continue-a'] and 'secret' not in provenance['continue-a']


def boundary_fixture(tmp_path):
    import torch,json
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.pipeline_validation.identity import seal
    roots={}
    for name in ('a','b','paused','old-paused','original'):
        root=tmp_path/name;root.mkdir();roots[name]=root
        state=dict(weights={'w':torch.tensor([1.])},optimizer={},rng={'python':[1],'numpy':[2],'torch':torch.tensor([3]),'device':[]},data=dict(cursor=0,pending_batch=None),replay={},
                   scheduler={},history={},pins={},selection={})
        store=GeneralizationCheckpointStore(root/'checkpoints','contract',100)
        initial=store.save(state,Ledger(100))
        state['weights']['w']=torch.tensor([3. if name!='old-paused' else 3.0001])
        final=store.save(state,Ledger(100,consumed=10,updates=1))
        (root/'training.json').write_text(json.dumps(dict(checkpoint=final,compatibility='contract')))
        (root/'checkpoints/retained.json').write_text(json.dumps(seal(dict(checkpoints=[dict(pins=['initial'],state=initial)]))))
    return roots


def test_initial_and_pause_comparisons_use_verified_full_states(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    roots=boundary_fixture(tmp_path)
    result=helper.boundary_comparisons(roots)
    assert result['missing']==[] and len(result['comparisons'])==5
    for label,row in result['comparisons'].items():
        assert row['equal']==label.startswith('initial:')
    assert result['comparisons']['pause:paused:old-paused']['components']['rng']['equal']
    assert not result['comparisons']['pause:paused:old-paused']['components']['weights']['equal']


def test_boundary_comparison_discloses_missing_branch_and_rejects_corruption(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(HERE))
    roots=boundary_fixture(tmp_path)
    del roots['paused']
    result=helper.boundary_comparisons(roots)
    assert set(result['missing'])=={'pause:paused:old-paused','initial:a:paused'}
    roster=json.loads((roots['a']/'checkpoints/retained.json').read_text())
    ref=roster['checkpoints'][0]['state']
    (roots['a']/'checkpoints'/ref['path']).write_text('{}')
    with pytest.raises(ValueError):helper.boundary_comparisons(roots)



@pytest.mark.parametrize('policy',['rocblas','no-lt'])
def test_fresh_policy_propagates_and_wraps_training_only(tmp_path,monkeypatch,policy):
    from contextlib import contextmanager
    monkeypatch.syspath_prepend(str(HERE))
    import trace_resume,diagnose_hip
    args=SimpleNamespace(engine='podman',run='old',output='out',start='fresh',policy=policy,internal='continue-a')
    assert all(s['argv'][-2:]==['--policy',policy] for s in helper.plan(args))
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    monkeypatch.setattr(helper,'load_context',lambda _:dict(start='fresh'))
    monkeypatch.setattr(diagnose_hip,'host_for',lambda _:object())
    active=[]
    @contextmanager
    def instrument(requested,*,tracing):
        assert requested==policy and tracing is False
        active.append(True)
        try:yield
        finally:active.pop()
    monkeypatch.setattr(trace_resume,'instrument',instrument)
    def train(name,context,host,output):
        assert active and context['policy']==policy;return dict(status='PASS')
    monkeypatch.setattr(helper,'train_branch',train)
    result=helper.internal(args)
    assert result['requested_policy']==policy and not active
