"""Explicit local tests for the disposable helper, outside qualification sources."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('diagnose_hip', HERE / 'diagnose_hip.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def args(**changes):
    values = dict(engine='podman', manifest='run-state/inputs/generalization-acceptance.json',
                  previous='run-state/acceptance-015', output='run-state/diagnostic-001',
                  probes=['native', 'resume', 'retained'], cases=list(helper.CASES))
    return SimpleNamespace(**dict(values, **changes))


def test_plan_uses_original_producer_and_preparation_without_broad_suites():
    plan = helper.plan(args(), 'sha256:' + 'a' * 64)
    assert plan[0]['name'] == 'compatibility'
    assert not any(s['operation'] in ('qualify', 'prepare', 'preflight') for s in plan)
    for stage in plan:
        if stage['operation'] == 'acceptance':
            assert '--reuse-run' not in stage['argv']
            assert 'run-state/acceptance-015/qualify' in stage['argv']
            assert 'run-state/acceptance-015/prepare' in stage['argv']
    assert next(s for s in plan if s['name'] == 'replay-resume')['needs'] == ['compatibility', 'retained-eval']


def test_replay_selection_adds_its_real_dependency_once():
    plan = helper.plan(args(cases=['replay-resume'], probes=[]), 'image')
    assert [s['name'] for s in plan] == ['compatibility', 'retained-eval', 'replay-resume']


def test_failure_continues_independent_cases_but_blocks_replay(tmp_path):
    plan = helper.plan(args(probes=[]), 'image')
    called = []
    def runner(stage, root):
        called.append(stage['name'])
        return {'status': 'FAIL' if stage['name'] == 'retained-eval' else 'PASS'}
    record = helper.execute(plan, tmp_path, runner=runner)
    assert 'refresh' in called and 'parameter-targets' in called and 'failure-packaging' in called
    assert 'replay-resume' not in called
    assert record['results'][-1]['status'] == 'BLOCKED'
    assert record['feature_qualified'] is False


def test_incompatible_source_blocks_every_probe_and_case(tmp_path):
    calls = []
    def runner(stage, root):
        calls.append(stage['name']); return {'status': 'FAIL'}
    helper.execute(helper.plan(args(), 'image'), tmp_path, runner=runner)
    assert calls == ['compatibility']


@pytest.mark.parametrize('result', [{'status':'INTERRUPTED'}, {'status':'FAIL','containment':'UNACKNOWLEDGED'}])
def test_unclean_or_interrupted_run_stops_new_work(tmp_path, result):
    calls=[]
    def runner(stage, root):
        calls.append(stage['name'])
        return {'status':'PASS'} if stage['name']=='compatibility' else result
    record=helper.execute(helper.plan(args(), 'image'), tmp_path, runner=runner)
    assert calls==['compatibility','probe-native']
    assert record['status'] != 'PASS'


def test_archive_exports_only_bounded_controls_not_private_state(tmp_path):
    root=tmp_path/'run';root.mkdir()
    (root/'private').mkdir();(root/'private/weights.pt').write_bytes(b'PRIVATE')
    (root/'unreviewed.json').write_text('{"secret":"PRIVATE"}')
    record={'status':'FAIL','feature_qualified':False,'results':[]}
    helper.save(root/'diagnostic.json',record)
    archive=helper.export(root)
    import zipfile
    from oeis_learn.pipeline_validation.bundle import verify
    verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert not any('weights' in n or 'unreviewed' in n for n in z.namelist())
        assert json.loads(z.read('summary.json'))['feature_qualified'] is False


def test_trace_preserves_scheduler_result_and_logs_no_source(capsys):
    import time
    from oeis_learn.fresh import pipeline_scheduler
    from oeis_learn.fresh.pipeline_jobs import NativeRequest, logical_job
    original=pipeline_scheduler.drive
    try:
        helper.install_trace()
        job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id='private',attempt_ordinal=0,method='secret-method',seed=1)
        def task():
            result=yield NativeRequest(job,0,time.monotonic()+10,lambda:'PRIVATE SOURCE')
            return result
        results,_=pipeline_scheduler.drive(None,[task()],sample=lambda *_:None)
        assert results==['PRIVATE SOURCE']
        logs=capsys.readouterr().err
        assert 'native-start' in logs and 'native-end' in logs
        assert 'PRIVATE' not in logs and 'secret-method' not in logs
    finally:pipeline_scheduler.drive=original


def test_helper_does_not_change_frozen_production_inventory():
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot=collect_snapshot(helper.ROOT)
    assert not any('diagnose_hip' in p for p in snapshot)


def test_dry_run_and_help_need_no_container_or_third_party_imports():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-S',str(HERE/'diagnose_hip.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0, result.stderr


@pytest.mark.parametrize('engine', ['docker','podman'])
@pytest.mark.parametrize('probe', ['native','resume'])
def test_real_engine_command_has_only_explicit_directory_mounts(tmp_path,monkeypatch,engine,probe):
    from contextlib import nullcontext
    from oeis_learn.container_runtime import Engine
    from oeis_learn.pipeline_validation import supervision
    adapter=Engine(engine,executable=engine,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda node:['--device=/dev/kfd','--device='+node])
    host=SimpleNamespace(engine=adapter,policy=dict(render_node='/dev/dri/renderD128',memory_bytes=16<<30,cpus=4),
                         select_gpu=lambda:None,gpu_lock=nullcontext)
    commands=[]
    def launch(command,logs,**kw):
        commands.append(command)
        assert kw['seconds']+kw['cleanup_seconds'] < helper.PROBES[probe]
        helper.save(tmp_path/'probe/result.json',{'status':'PASS'})
    monkeypatch.setattr(supervision,'launch',launch)
    monkeypatch.setattr(supervision,'reclaim',lambda *a:None)
    assert helper.launch_probe(host,'sha256:'+'a'*64,probe,tmp_path)['status']=='PASS'
    command=commands[0]
    mounts=[command[i+1] for i,arg in enumerate(command) if arg=='--mount']
    assert len(mounts)==2 and 'readonly' in mounts[0] and 'readonly' not in mounts[1]
    assert '--network=none' in command and '--read-only' in command
    assert ('--device=/dev/kfd' in command)==(probe=='resume')
    assert not any('sock' in m for m in mounts)


def test_native_probe_uses_valid_contract_hash_and_real_admission(tmp_path,monkeypatch):
    # Model/native dependencies aren't installed on the host. Exercise the smoke
    # harness through strict adapters here; actual native behavior remains a HIP-host probe.
    import sys
    from oeis_learn.pipeline_validation import generalization_replay as replay
    from oeis_learn.synthesis import client
    from oeis_learn.synthesis.state import require_hash
    observed=[]
    class Client:
        def __init__(self,contract,**kw):require_hash(contract);observed.append(contract)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    pool=SimpleNamespace(entries=[1],snapshot=lambda:{'pool':'exact'})
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda _:None,set_num_interop_threads=lambda _:None))
    monkeypatch.setattr(client,'SynthesisClient',Client)
    def search(tasks,**kw):
        assert kw['diagnostic'] and kw['seconds']==60 and len(tasks)==1
        assert len(tasks[0]['values'])==100
        return pool,{'status':'PASS'}
    monkeypatch.setattr(replay,'search_round',search)
    monkeypatch.setattr(replay,'restore_pool',lambda record,**kw:pool)
    assert helper.container_probe('native')['restore_equal']
    assert len(observed)==1


def test_unacknowledged_cleanup_is_not_hidden_by_success(tmp_path):
    helper.save(tmp_path/'role/supervision/cleanup.json',dict(acknowledged=False,within_bound=False))
    assert helper.uncontained(tmp_path)


def test_scope_stays_diagnostic_even_when_all_selected_cases_pass(tmp_path):
    record=helper.execute(helper.plan(args(probes=[],cases=['failure-packaging']),'image'),tmp_path,
                          runner=lambda *a:{'status':'PASS'})
    assert record['status']=='PASS' and record['feature_qualified'] is False
    assert record['scientific_status']=='UNMEASURED'


def test_trace_preserves_nonquiescent_failure_and_cancel(monkeypatch,capsys):
    import time
    from oeis_learn.fresh import pipeline_scheduler as scheduler
    from oeis_learn.fresh.pipeline_jobs import NativeRequest,logical_job
    from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
    original=scheduler.drive
    cancelled=[]
    monkeypatch.setattr(scheduler.threading,'Thread',lambda **kw:SimpleNamespace(
        start=lambda:None,join=lambda **kw:None,is_alive=lambda:True))
    job=logical_job(contract='c',snapshot='s',mode='m',arm='a',prompt_id='p',attempt_ordinal=0,method='x',seed=1)
    def task():
        yield NativeRequest(job,0,time.monotonic()+.02,lambda:None,cancel=lambda:cancelled.append(True))
    try:
        helper.install_trace()
        with pytest.raises(NonQuiescentWorker):
            scheduler.drive(None,[task()],sample=lambda *_:None)
        assert cancelled==[True]
        assert 'cancel-end' in capsys.readouterr().err
    finally:scheduler.drive=original


def test_resume_probe_rejects_cpu_fallback(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(
        set_num_threads=lambda _:None,set_num_interop_threads=lambda _:None,
        version=SimpleNamespace(hip=None),cuda=SimpleNamespace(is_available=lambda:False)))
    with pytest.raises(RuntimeError,match='actual HIP required'):
        helper.container_probe('resume')


def test_failed_probe_still_runs_real_case_not_fake_pass(tmp_path):
    called=[]
    def runner(stage,root):
        called.append(stage['name'])
        return dict(status='FAIL' if stage['name']=='probe-native' else 'PASS')
    result=helper.execute(helper.plan(args(probes=['native'],cases=['refresh']),'image'),tmp_path,runner=runner)
    assert called==['compatibility','probe-native','refresh']
    assert result['status']=='FAIL'


@pytest.mark.parametrize('application_status,exit_code,expected', [('PASS',0,'PASS'),('FAIL',0,'FAIL'),('PASS',1,'FAIL')])
def test_actual_subprocess_preserves_process_and_application_outcomes(tmp_path,monkeypatch,application_status,exit_code,expected):
    import sys
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'scripts'))
    target=tmp_path/'probe-native'
    program='import json,sys;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir();(p/"result.json").write_text(json.dumps({"status":sys.argv[2]}));sys.exit(int(sys.argv[3]))'
    stage=dict(name='probe-native',operation='native',seconds=5,needs=[],target=str(target),
               argv=[sys.executable,'-c',program,str(target),application_status,str(exit_code)])
    result=helper.run(stage,tmp_path)
    assert result['status']==expected
    assert result['exit_code']==exit_code
    assert result['containment']=='NO_UNACKNOWLEDGED_CLEANUP'
