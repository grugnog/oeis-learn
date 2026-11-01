"""CPU regressions for diagnostic instrumentation, not HIP qualification."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('trace_resume',HERE/'trace_resume.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_plan_has_independent_controls_and_traces():
    args=SimpleNamespace(engine='podman',run='old',matrix='fresh',output='out')
    stages=helper.plan(args)
    assert [s['name'] for s in stages]==['compatibility','old-control','new-control','old-traced','new-traced','compare']
    assert all(s['needs']==['compatibility'] for s in stages[1:])
    assert sum(s['seconds'] for s in stages)==4020


def test_first_divergence_names_phase_and_preserves_absence():
    old=[dict(phase='before',update=1,weights='x'),dict(phase='forward',update=1,logits='a')]
    new=[dict(phase='before',update=1,weights='x'),dict(phase='forward',update=1,logits='b')]
    assert helper.first_difference(old,new)==dict(equal=False,index=1,phase='forward',update=1,fields=['logits'])
    assert helper.first_difference(old,old)['equal']
    assert helper.first_difference(old,[])['phase']=='missing_record'


@pytest.mark.parametrize('layers,deferred',[(False,False),(True,False),(True,True)])
def test_trace_preserves_update_and_rng_and_restores_hooks(tmp_path,monkeypatch,layers,deferred):
    import sys,types,torch
    from oeis_learn.pipeline_validation import generalization_training as training
    module=types.ModuleType('oeis_learn.rl.sft_trainer')
    in_forward=[False]
    def logits(encoder,decoder,*args,**kwargs):
        in_forward[0]=True
        try:return decoder(encoder(torch.ones(1,2)))
        finally:in_forward[0]=False
    cpu=torch.Tensor.cpu
    def checked_cpu(tensor,*args,**kwargs):
        if deferred:assert not in_forward[0], 'deferred capture read back during the original forward'
        return cpu(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',checked_cpu)
    module.teacher_forced_logits=logits
    monkeypatch.setitem(sys.modules,'oeis_learn.rl.sft_trainer',module)
    import oeis_learn.rl
    monkeypatch.setattr(oeis_learn.rl,'sft_trainer',module,raising=False)
    def factory(encoder,decoder,optimizer,*,device,scheduler=None):
        def update(batch):
            optimizer.zero_grad(set_to_none=True)
            loss=module.teacher_forced_logits(encoder,decoder).sum();loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()),1.)
            optimizer.step();return {'loss':float(loss.detach())}
        return update
    monkeypatch.setattr(training,'torch_update',factory)
    def models():
        torch.manual_seed(17)
        a=torch.nn.Linear(2,2);b=torch.nn.Linear(2,1)
        return a,b,torch.optim.SGD(list(a.parameters())+list(b.parameters()),lr=.01)
    a,b,opt=models();factory(a,b,opt,device='cpu')([{'private':'never export'}])
    expected=[p.detach().clone() for p in list(a.parameters())+list(b.parameters())]
    expected_rng=torch.get_rng_state()
    a,b,opt=models();clip=torch.nn.utils.clip_grad_norm_
    undo=helper.install_trace(tmp_path/'trace.jsonl',layers=layers,defer_layers=deferred)
    try:training.torch_update(a,b,opt,device='cpu')([{'private':'never export'}])
    finally:undo()
    assert all(torch.equal(x,y) for x,y in zip(expected,list(a.parameters())+list(b.parameters())))
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert training.torch_update is factory and module.teacher_forced_logits is logits
    assert torch.nn.utils.clip_grad_norm_ is clip
    text=(tmp_path/'trace.jsonl').read_text();assert 'never export' not in text
    rows=[json.loads(line) for line in text.splitlines()]
    assert [r['phase'] for r in rows]==['execution','before','forward','backward','clip','after']
    assert rows[1]['weights']!=rows[-1]['weights']
    if layers:
        assert len(rows[2]['layer_events'])==2
        assert rows[2]['layer_events'][0]['module']=='encoder'
        assert not a._forward_hooks and not b._forward_hooks



def test_instrumentation_bootstrap_is_scoped_and_does_not_install_for_validation(monkeypatch):
    from oeis_learn.pipeline_validation import generalization_roles as roles
    monkeypatch.setattr(roles,'ROLE_PROGRAM','sentinel=True\n')
    monkeypatch.setattr('sys.argv',['-c','preparation',json.dumps(dict(operation='validate-runtime'))])
    with pytest.raises(RuntimeError):
        with helper.instrument():
            scope={};exec(roles.ROLE_PROGRAM,scope)
            assert scope['sentinel'] is True
            assert 'install_trace' in scope
            raise RuntimeError('test unwind')
    assert roles.ROLE_PROGRAM=='sentinel=True\n'


def test_partial_trace_is_preserved_and_complete_trace_is_required(tmp_path):
    path=tmp_path/'trace.jsonl'
    path.write_text(json.dumps(dict(phase='execution',update=0))+'\n'+json.dumps(dict(phase='before',update=1))+'\n')
    assert len(helper.read_trace(path,complete=False))==2
    with pytest.raises(ValueError):helper.read_trace(path)
    path.write_text('x'*(65537))
    with pytest.raises(ValueError):helper.read_trace(path,complete=False)


@pytest.mark.parametrize('name',helper.RUNS)
def test_internal_binds_correct_pause_and_scopes_tracing(tmp_path,monkeypatch,name):
    import sys
    monkeypatch.syspath_prepend(str(HERE))
    import probe_resume,diagnose_hip
    from contextlib import contextmanager
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    context=dict(paused=tmp_path/'old',new_paused=tmp_path/'new',seed=7)
    monkeypatch.setattr(helper,'context',lambda _:context)
    monkeypatch.setattr(diagnose_hip,'host_for',lambda _:object())
    active=[]
    @contextmanager
    def instrument():
        active.append(True)
        try:yield
        finally:active.pop()
    monkeypatch.setattr(helper,'instrument',instrument)
    def train(branch,ctx,host,output):
        assert branch=='continue-a'
        assert ctx['paused']==tmp_path/('new' if name.startswith('new-') else 'old')
        assert bool(active)==name.endswith('-traced')
        raise ValueError('stop before any actual training')
    monkeypatch.setattr(probe_resume,'train_branch',train)
    with pytest.raises(ValueError,match='stop before'):
        helper.internal(SimpleNamespace(run='old',matrix='new',engine='podman',output='out',internal=name))
    assert active==[]


@pytest.mark.parametrize('engine',['docker','podman'])
def test_comparison_uses_read_only_cpu_mounts_and_preserves_partial_traces(tmp_path,monkeypatch,engine):
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import save
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import supervision
    import probe_resume
    monkeypatch.setattr(helper,'ROOT',tmp_path);monkeypatch.setattr(probe_resume,'ROOT',tmp_path)
    adapter=Engine(engine,executable=engine,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda *_:pytest.fail('CPU comparison cannot use GPU'))
    host=SimpleNamespace(engine=adapter,policy=dict(memory_bytes=16<<30,cpus=4))
    roots={n:tmp_path/n for n in (*helper.RUNS,'full','resumed')}
    for root in roots.values():root.mkdir()
    output=tmp_path/'matrix/compare';output.mkdir(parents=True)
    for name in helper.RUNS:
        save(output.parent/name/'result.json',dict(status='PASS',learner=name,trace=[{'phase':'execution','update':0}]))
    commands=[]
    def launch(argv,*a,**kw):
        commands.append(argv)
        save(output/'private/comparison/comparison.json',dict(complete=True))
    monkeypatch.setattr(supervision,'launch',launch);monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    helper.compare(host,dict(saved=roots,runtime={'images':['cpu','sha256:'+'a'*64]}),output)
    argv=commands[0];mounts=[argv[i+1] for i,v in enumerate(argv) if v=='--mount']
    assert len(mounts)==8 and all('readonly' in m for m in mounts[:-1])
    assert '--network=none' in argv and '/helper/trace_resume.py' in argv
    assert set(json.loads((output/'private/helper/traces.json').read_text()))==set(helper.RUNS)


def test_failed_trace_export_survives_missing_final_comparison(tmp_path,monkeypatch):
    import zipfile
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import save
    import probe_resume
    from oeis_learn.pipeline_validation.bundle import verify
    root=tmp_path/'trace';root.mkdir()
    save(root/'diagnostic.json',dict(status='FAIL',plan=[dict(name='old-traced')]))
    save(root/'old-traced/trace.json',[dict(phase='execution',update=0)])
    archive=probe_resume.export(root);manifest=verify(archive)
    assert next(r for r in manifest['members'] if r['path']=='traces/old-traced.json')['mandatory']
    with zipfile.ZipFile(archive) as z:assert 'traces/old-traced.json' in z.namelist()


def test_real_checkpoint_comparison_reports_observer_effect_and_missing_endpoint(tmp_path,monkeypatch):
    import shutil,torch
    monkeypatch.syspath_prepend(str(HERE))
    from test_inspect_resume import fixture
    from diagnose_hip import save
    fixture_root=tmp_path/'fixture';fixture_root.mkdir()
    roots=fixture(fixture_root,monkeypatch)
    saved=tmp_path/'saved';saved.mkdir()
    for name,source in [('old-control','full'),('new-control','paused'),('old-traced','resumed'),
                        ('new-traced','paused'),('original','full'),('old-resumed','resumed')]:
        shutil.copytree(roots[source],saved/name)
    trace_path=tmp_path/'traces.json'
    save(trace_path,{'old-traced':[dict(phase='execution',update=0),dict(phase='forward',update=1,logits='a')],
                     'new-traced':[dict(phase='execution',update=0),dict(phase='forward',update=1,logits='b')]})
    helper.comparison_container(saved,tmp_path/'out',trace_path)
    result=json.loads((tmp_path/'out/comparison.json').read_text())
    assert result['complete'] and len(result['comparisons'])==15
    assert result['control_equal'] and not result['traced_equal']
    assert result['observer_effect']==dict(old_equal=False,new_equal=True)
    assert result['first_trace_difference']['phase']=='forward'
    assert result['feature_qualified'] is False
    traces=json.loads(trace_path.read_text())
    for rows in traces.values():rows[0]['diagnostic_policy']='rocblas'
    save(trace_path,traces)
    helper.comparison_container(saved,tmp_path/'intervention',trace_path)
    intervention=json.loads((tmp_path/'intervention/comparison.json').read_text())
    assert intervention['observer_effect'] is None
    assert intervention['intervention_effect']==dict(old_equal=False,new_equal=True)
    assert intervention['diagnostic_policy']=='rocblas'
    shutil.rmtree(saved/'old-traced')
    helper.comparison_container(saved,tmp_path/'partial',trace_path)
    result=json.loads((tmp_path/'partial/comparison.json').read_text())
    assert not result['complete'] and result['missing']==['old-traced']
    assert result['traced_equal'] is None and result['first_trace_difference']['phase']=='forward'


def test_rocblas_policy_sets_documented_controls_before_selecting_backend(monkeypatch):
    import torch,os
    calls=[]
    def preferred(value=None):
        calls.append((value,os.environ.get('ROCBLAS_USE_HIPBLASLT'),os.environ.get('ROCBLAS_DEFAULT_ATOMICS_MODE')))
        return 'BlasBackend.Cublas'
    monkeypatch.setattr(torch.backends.cuda,'preferred_blas_library',preferred)
    for key in ('ROCBLAS_USE_HIPBLASLT','ROCBLAS_DEFAULT_ATOMICS_MODE'):monkeypatch.delenv(key,raising=False)
    helper.configure_policy('rocblas')
    assert calls==[('cublas','0','0'),(None,'0','0')]
    with pytest.raises(ValueError):helper.configure_policy('unknown')


def test_policy_is_propagated_to_every_stage():
    args=SimpleNamespace(engine='podman',run='old',matrix='fresh',output='out',policy='rocblas')
    assert all(s['argv'][-2:]==['--policy','rocblas'] for s in helper.plan(args))


def test_diagnostic_policy_bootstrap_is_learner_only_and_restores_role_program(monkeypatch):
    import torch
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation import generalization_training as training
    original=training.torch_update
    monkeypatch.setattr(roles,'ROLE_PROGRAM','sentinel=True\n')
    monkeypatch.setattr('sys.argv',['-c','learner',json.dumps(dict(operation='train'))])
    calls=[]
    monkeypatch.setattr(torch.backends.cuda,'preferred_blas_library',lambda value=None:calls.append(value) or 'BlasBackend.Cublas')
    for key in ('ROCBLAS_USE_HIPBLASLT','ROCBLAS_DEFAULT_ATOMICS_MODE'):monkeypatch.delenv(key,raising=False)
    # Avoid importing the learner's optional local native dependencies in this CPU test.
    import sys,types,oeis_learn.rl
    module=types.ModuleType('oeis_learn.rl.sft_trainer')
    monkeypatch.setitem(sys.modules,'oeis_learn.rl.sft_trainer',module)
    monkeypatch.setattr(oeis_learn.rl,'sft_trainer',module,raising=False)
    try:
        with helper.instrument('rocblas'):
            scope={};exec(roles.ROLE_PROGRAM,scope)
            assert scope['sentinel'] is True and calls==['cublas',None]
            assert training.torch_update is not original
    finally:training.torch_update=original
    assert roles.ROLE_PROGRAM=='sentinel=True\n'



def test_policy_bootstrap_can_run_without_installing_any_tensor_observer(monkeypatch):
    import torch,sys
    from oeis_learn.pipeline_validation import generalization_roles as roles
    from oeis_learn.pipeline_validation import generalization_training as training
    original=training.torch_update
    monkeypatch.setattr(roles,'ROLE_PROGRAM','sentinel=True\n')
    monkeypatch.setattr(sys,'argv',['-c','learner',json.dumps(dict(operation='train'))])
    calls=[]
    monkeypatch.setattr(torch.backends.cuda,'preferred_blas_library',lambda value=None:calls.append(value) or 'BlasBackend.Cublas')
    for key in ('ROCBLAS_USE_HIPBLASLT','ROCBLAS_DEFAULT_ATOMICS_MODE'):monkeypatch.delenv(key,raising=False)
    with helper.instrument('rocblas',tracing=False):
        scope={};exec(roles.ROLE_PROGRAM,scope)
        assert scope['sentinel'] and calls==['cublas',None]
        assert training.torch_update is original
        assert 'def install_trace' not in roles.ROLE_PROGRAM



def test_layer_difference_distinguishes_input_difference_from_operation_output():
    a=[dict(module='encoder.linear',kind='Linear',input='a',output='x')]
    b=[dict(module='encoder.linear',kind='Linear',input='a',output='y')]
    result=helper.first_layer_difference(a,b)
    assert result['module']=='encoder.linear' and result['input_equal'] and not result['output_equal']
    b[0]['input']='b'
    assert not helper.first_layer_difference(a,b)['input_equal']
    assert helper.first_layer_difference(a,a)['equal']
    assert helper.first_layer_difference(a,[])['reason']=='missing_event'


@pytest.mark.parametrize('deferred',[False,True])
def test_module_hooks_are_removed_when_forward_raises(tmp_path,monkeypatch,deferred):
    import sys,types,torch,oeis_learn.rl
    from oeis_learn.pipeline_validation import generalization_training as training
    module=types.ModuleType('oeis_learn.rl.sft_trainer')
    def logits(encoder,decoder):
        encoder(torch.ones(1,2));raise RuntimeError('forward failure')
    module.teacher_forced_logits=logits
    monkeypatch.setitem(sys.modules,'oeis_learn.rl.sft_trainer',module)
    monkeypatch.setattr(oeis_learn.rl,'sft_trainer',module,raising=False)
    monkeypatch.setattr(training,'torch_update',lambda a,b,o,**kw:lambda batch:module.teacher_forced_logits(a,b))
    a=torch.nn.Linear(2,2);b=torch.nn.Linear(2,1)
    opt=torch.optim.SGD(list(a.parameters())+list(b.parameters()),lr=.01)
    undo=helper.install_trace(tmp_path/'trace.jsonl',layers=True,defer_layers=deferred)
    try:
        with pytest.raises(RuntimeError,match='forward failure'):training.torch_update(a,b,opt,device='cpu')([])
    finally:undo()
    assert not a._forward_hooks and not b._forward_hooks
    assert module.teacher_forced_logits is logits


def test_layer_trace_stage_propagation_and_bounded_read(tmp_path,monkeypatch):
    args=SimpleNamespace(engine='podman',run='old',matrix='fresh',output='out',layers=True)
    assert all(s['argv'][-1]=='--layers' for s in helper.plan(args))
    path=tmp_path/'oversized';path.write_bytes(b'x'*((2<<20)+1))
    monkeypatch.setattr(Path,'read_text',lambda *a,**kw:pytest.fail('must reject before reading'))
    with pytest.raises(ValueError,match='trace byte cap'):helper.read_trace(path)



def test_deferred_capture_clones_without_cpu_reads_and_obeys_bound(monkeypatch):
    import torch
    from dataclasses import dataclass
    @dataclass
    class Row:
        tensor:object
    x=torch.ones(3)
    monkeypatch.setattr(torch.Tensor,'cpu',lambda *_:pytest.fail('capture must not read to CPU'))
    budget=dict(bytes=0,nodes=0)
    result=helper.capture_value(Row(x),budget)
    x.add_(1)
    assert torch.equal(result['tensor'],torch.ones(3))
    assert budget['bytes']==12
    with pytest.raises(ValueError,match='memory cap'):helper.capture_value(x,dict(bytes=1<<30,nodes=0))


def test_deferred_plan_and_bootstrap_propagate_capture_mode():
    args=SimpleNamespace(engine='podman',run='old',matrix='fresh',output='out',layers=True,defer_layers=True)
    assert all(s['argv'][-2:]==['--layers','--defer-layers'] for s in helper.plan(args))
