"""Failure diagnostics are bounded public copies, never execution authority."""
import json
from pathlib import Path
import zipfile
import pytest

from tests.integration.test_generalization_acceptance import helper


def put(root,name,payload):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return path


def test_recovery_captures_terminal_case_and_packaging_failures_first(tmp_path):
    root=tmp_path/'run';root.mkdir()
    expected={
        'replay-resume/result.json':b'{"status":"FAIL","error":"dependency rejected"}',
        'replay-resume/case/receipt.json':b'{"state":"FAIL","reason":"ValueError"}',
        'retained-eval/result.json':b'{"status":"PASS","result":{"initializer":null}}',
        'retained-eval/case/evaluation.json':b'{"selection":"inconclusive"}',
        'report/packaging/role/result.json':b'{"status":"FAIL","error":"archive rejected"}',
        'report/packaging/role/supervision/stderr.log':b'packaging details',
    }
    for name,payload in expected.items():put(root,name,payload)
    put(root,'replay-resume/private/resolved/study.json',b'PRIVATE')
    ledger=dict(receipts=[dict(stage='replay-resume',exit_code=1)],status='FAIL')
    put(root,'workflow.json',json.dumps(ledger).encode())
    archive=helper().recover(root,ledger)
    with zipfile.ZipFile(archive) as z:
        for name,payload in expected.items():assert z.read('diagnostics/'+name)==payload
        assert all('private' not in n for n in z.namelist())


def test_existing_run_diagnostic_export_is_stdlib_and_read_only(tmp_path):
    import os,subprocess,sys
    root=tmp_path/'run';root.mkdir();output=tmp_path/'export'
    put(root,'replay-resume/result.json',b'{"status":"FAIL","error":"original rejection"}')
    put(root,'summary.json',b'{"selected_scope":"FAIL"}')
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    command=[sys.executable,'-S','-m','oeis_learn.pipeline_validation.generalization_diagnostics',
             '--run',str(root),'--output',str(output)]
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve()))
    result=__import__('subprocess').run(command,env=env,text=True,capture_output=True,timeout=15)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout);archive=Path(report['archive'])
    from oeis_learn.pipeline_validation.bundle import verify
    assert verify(archive)['summary_state']=='PARTIAL'
    with zipfile.ZipFile(archive) as z:
        assert z.read('diagnostics/replay-resume/result.json')==before['replay-resume/result.json']
        assert json.loads(z.read('summary.json'))['feature_qualified'] is False
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    original=archive.read_bytes()
    again=subprocess.run(command,env=env,text=True,capture_output=True,timeout=15)
    assert again.returncode!=0 and archive.read_bytes()==original


def test_recovery_retains_timeout_records_and_original_failure(tmp_path):
    root=tmp_path/'failed';root.mkdir()
    expected={
        'preflight/result.json':b'{"status":"FAIL","reason":"TimeoutError"}',
        'preflight/summary.json':b'{"selected_scope":"FAIL"}',
        'preflight/parameter-targets/validation/role/supervision/process.json':b'{"outcome":"FAIL","returncode":125}',
        'preflight/parameter-targets/validation/role/supervision/cleanup.json':b'{"acknowledged":true}',
        'preflight/parameter-targets/validation/role/supervision/stdout.log':b'',
        'preflight/parameter-targets/validation/role/supervision/stderr.log':b'time="fixture" level=debug msg="startup"\nvalidation: python-start unix=123\nTimeout (0:00:30)!\n',
    }
    for name,payload in expected.items():put(root,name,payload)
    put(root,'preflight/parameter-targets/validation/private/input/truth.json',b'PRIVATE')
    put(root,'preflight/parameter-targets/validation/role/checkpoints/weights.pt',b'WEIGHTS')
    ledger=dict(receipts=[dict(stage='preflight',exit_code=1)],status='FAIL')
    put(root,'workflow.json',json.dumps(ledger).encode())
    archive=helper().recover(root,ledger)
    from oeis_learn.pipeline_validation.bundle import verify
    assert verify(archive)['summary_state']=='FAIL'
    with zipfile.ZipFile(archive) as z:
        for name,payload in expected.items():assert z.read('diagnostics/'+name)==payload
        assert all('private' not in n and 'checkpoints' not in n for n in z.namelist())
        inventory=json.loads(z.read('diagnostics/index.json'))
        assert inventory['qualification_evidence'] is False
        assert {r['source'] for r in inventory['files'] if r['status']=='COPIED'}>=set(expected)


def test_diagnostics_bound_reads_preserve_json_and_reject_links(tmp_path):
    from oeis_learn.pipeline_validation.generalization_diagnostics import collect,JSON_CAP,LOG_CAP
    root=tmp_path/'run';root.mkdir();dest=tmp_path/'public'
    put(root,'preflight/result.json',b'x'*(JSON_CAP+1))
    log='preflight/parameter-targets/validation/role/supervision/stderr.log'
    put(root,log,b'opening'+b'x'*(LOG_CAP*2)+b'last error')
    outside=put(tmp_path,'secret.json',b'SECRET')
    (root/'preflight/summary.json').symlink_to(outside)
    (root/'report').symlink_to(tmp_path,target_is_directory=True)
    inventory=collect(root,dest)
    rows={r['source']:r for r in inventory['files']}
    assert rows['preflight/result.json']['status']=='OMITTED'
    assert rows['preflight/summary.json']['status']=='OMITTED'
    assert rows[log]['status']=='TRUNCATED'
    data=(dest/log).read_bytes()
    assert len(data)<=LOG_CAP and data.startswith(b'opening') and data.endswith(b'last error')
    assert not (dest/'preflight/result.json').exists()
    assert b'SECRET' not in b''.join(p.read_bytes() for p in dest.rglob('*') if p.is_file())
    with pytest.raises(FileExistsError):collect(root,dest)


def test_diagnostics_total_cap_is_explicit(tmp_path):
    from oeis_learn.pipeline_validation import generalization_diagnostics as d
    root=tmp_path/'run';root.mkdir()
    put(root,'preflight/result.json',b'{}')
    put(root,'preflight/summary.json',b'{}')
    result=d.collect(root,tmp_path/'public',cap=2)
    assert result['exported_bytes']==2
    assert any(r.get('reason')=='total byte cap' for r in result['files'])


@pytest.mark.parametrize('stage,suite', [('cpu.semantic','bigint-cpu'),('hip.preflight','bigint-gpu')])
def test_recovery_includes_nested_qualification_failures(tmp_path,stage,suite):
    root=tmp_path/'run';root.mkdir()
    base=f'qualify/pipeline/stages/{stage}/{suite}-regressions/'
    expected={base+'results.xml':b'<testsuite><testcase name="failure"><failure>details</failure></testcase></testsuite>',
              base+'stdout.log':b'FAILED test_example - actual failure',
              base+'stderr.log':b'error details',base+'process.json':b'{"returncode":1}',
              base+'cleanup.json':b'{"acknowledged":true}',base+'receipt.json':b'{"status":"FAIL"}',
              f'qualify/pipeline/stages/{stage}/environment/stdout.log':b'environment observation'}
    for name,payload in expected.items():put(root,name,payload)
    put(root,base+'private/results.xml',b'PRIVATE')
    put(root,base+'pytest/unreviewed/stdout.log',b'PRIVATE')
    ledger=dict(receipts=[dict(stage='qualify',exit_code=1)],status='FAIL')
    put(root,'workflow.json',json.dumps(ledger).encode())
    archive=helper().recover(root,ledger)
    from oeis_learn.pipeline_validation.bundle import verify
    assert verify(archive)['summary_state']=='FAIL'
    with zipfile.ZipFile(archive) as z:
        for name,payload in expected.items():assert z.read('diagnostics/'+name)==payload
        assert all('private' not in n and 'pytest' not in n for n in z.namelist())


def test_oversized_junit_is_omitted_whole(tmp_path):
    from oeis_learn.pipeline_validation.generalization_diagnostics import collect,XML_CAP
    root=tmp_path/'run';root.mkdir()
    name='qualify/pipeline/stages/cpu.semantic/bigint-cpu-regressions/results.xml'
    put(root,name,b'x'*(XML_CAP+1))
    result=collect(root,tmp_path/'public')
    row=next(r for r in result['files'] if r['source']==name)
    assert row['status']=='OMITTED' and row['reason']=='XML byte cap'
    assert not (tmp_path/'public'/name).exists()


@pytest.mark.parametrize('mismatch', [None,'source','guest'])
def test_image_inspection_progress_preserves_integrity_checks(tmp_path,monkeypatch,mismatch):
    import subprocess,sys
    from types import SimpleNamespace
    from oeis_learn import container_validation
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda root:{'fixture':True})
    captured={}
    class Engine:
        def command(self,**kw):captured.update(kw);return ['fixture-inspection']
    host=SimpleNamespace(engine=Engine(),policy=dict(workspace=str(tmp_path),memory_bytes=2<<30,cpus=1))
    monkeypatch.setattr(supervision,'launch',lambda *a,**k:None)
    monkeypatch.setattr(supervision,'reclaim',lambda *a,**k:None)
    output=tmp_path/'inspection'
    container_validation.inspect_current_image(host,'cpu-image',output,seconds=120)
    assert captured['gpu'] is False
    assert captured['startup_diagnostics'] is True
    argv=captured['argv'];code=argv[argv.index('-c')+1]
    code=code.replace('/work/input/expected.json',str(output/'inputs/expected.json'))
    prefix="""import sys,types
snapshot=types.ModuleType('oeis_learn.experiments.source_snapshot')
snapshot.source_snapshot=lambda: SNAPSHOT
sys.modules[snapshot.__name__]=snapshot
config=types.ModuleType('oeis_learn.fresh.bigint_config')
config.packaged_artifacts=lambda: {'guest_sha256':'original'}
sys.modules[config.__name__]=config
common=types.ModuleType('oeis_learn.fresh.common')
common.file_hash=lambda _: GUEST
sys.modules[common.__name__]=common
""".replace('SNAPSHOT',repr({'fixture':False if mismatch=='source' else True})).replace('GUEST',repr('changed' if mismatch=='guest' else 'original'))
    result=subprocess.run([sys.executable,'-u','-c',prefix+code],text=True,capture_output=True,timeout=10)
    assert 'inspection: python-start' in result.stderr
    assert 'inspection: source-snapshot' in result.stderr
    markers=[line for line in result.stderr.splitlines() if line.startswith('inspection:')]
    stamps=[float(line.split('unix=')[1]) for line in markers]
    assert stamps == sorted(stamps) and all(stamp>0 for stamp in stamps)
    if mismatch:
        assert result.returncode!=0
        assert ('current installed package/source mismatch' if mismatch=='source' else 'changed packaged guest') in result.stderr
        assert 'inspection: complete' not in result.stderr
    else:
        assert result.returncode==0,result.stderr
        assert 'inspection: packaged-artifacts' in result.stderr and 'inspection: complete' in result.stderr
        assert json.loads(result.stdout)['source']=={'fixture':True}


@pytest.mark.parametrize('outcome', ['pass','fail','hang'])
def test_validation_bootstrap_survives_failure_and_exposes_stalls(tmp_path,monkeypatch,outcome):
    import sys
    from types import SimpleNamespace
    from oeis_learn.generalization.role_contracts import Role,ROLE_INPUTS
    from oeis_learn.pipeline_validation import generalization_roles as roles,supervision
    captured={}
    class Engine:
        def command(self,**kw):captured.update(kw);return ['fixture']
    host=SimpleNamespace(engine=Engine(),policy=dict(memory_bytes=2<<30,cpus=1))
    channels={m.channel:m.channel for m in ROLE_INPUTS[Role.PREPARATION]}
    for path in channels.values():(tmp_path/path).mkdir()
    def launch(command,output,**kw):
        assert kw['seconds']==120 and kw['stream_cap']==1<<20
        (output.parent/'result.json').write_text('{"status":"PASS"}')
    monkeypatch.setattr(supervision,'reclaim',lambda *a,**kw:None)
    roles.invoke(host,'cpu',tmp_path,Role.PREPARATION,channels,output='role',
                 operation='validate-runtime',settings={},seconds=120,launcher=launch)
    assert captured['gpu'] is False and captured['startup_diagnostics'] is True
    argv=captured['argv'];code=argv[argv.index('-c')+1]
    # Accelerate only the diagnostic timer in this local subprocess. The actual
    # production wrapper keeps 30-second dumps and the unchanged external budget.
    code=code.replace('30,repeat=True','0.1,repeat=True')
    prefix="""import runpy,sys,time,json
def fixture(name,run_name):
    assert name=='oeis_learn.pipeline_validation.generalization_role'
    assert run_name=='__main__'
    assert sys.argv[1]=='preparation'
    assert json.loads(sys.argv[2])['operation']=='validate-runtime'
    if OUTCOME=='fail':raise RuntimeError('fixture rejection')
    if OUTCOME=='hang':time.sleep(10)
runpy.run_module=fixture
""".replace('OUTCOME',repr(outcome))
    logs=tmp_path/'subprocess'
    command=[sys.executable,'-u','-c',prefix+code,*argv[argv.index('-c')+2:]]
    if outcome=='hang':
        with pytest.raises(TimeoutError):supervision.launch(command,logs,seconds=.5,stream_cap=4096)
    elif outcome=='fail':
        with pytest.raises(RuntimeError):supervision.launch(command,logs,seconds=5,stream_cap=4096)
    else:supervision.launch(command,logs,seconds=5,stream_cap=4096)
    stderr=(logs/'stderr.log').read_text()
    assert 'validation: python-start unix=' in stderr
    if outcome=='hang':assert 'Timeout (' in stderr and 'fixture' in stderr
    if outcome=='fail':assert 'fixture rejection' in stderr
    process=json.loads((logs/'process.json').read_text())
    assert (process['outcome']=='COMPLETE') == (outcome=='pass')


@pytest.mark.parametrize('failure', [None,'runtime','storage'])
def test_validation_milestones_locate_last_reached_phase(tmp_path,monkeypatch,capsys,failure):
    from oeis_learn.pipeline_validation import generalization_role as role
    from oeis_learn.pipeline_validation import generalization_runtime as runtime
    from oeis_learn.pipeline_validation import generalization_training as training
    def validate(*args):
        if failure=='runtime':raise ValueError('runtime rejected')
        return {'recipe':{}}
    def storage(*args):
        if failure=='storage':raise ValueError('storage rejected')
        return {'reserved':123}
    monkeypatch.setattr(runtime,'validate_execution',validate)
    monkeypatch.setattr(training,'storage_preflight',storage)
    args=({'mechanical_sources':tmp_path},tmp_path,{'study':{},'runtime':{}})
    if failure:
        with pytest.raises(ValueError,match=failure+' rejected'):role.validate_runtime(*args,seconds=120)
    else:
        result=role.validate_runtime(*args,seconds=120)
        assert result=={'status':'PASS','plan':{'recipe':{},'storage':{'reserved':123}}}
    lines=capsys.readouterr().err.splitlines()
    phases=[line.split()[1] for line in lines]
    assert phases==(['runtime-plan'] if failure=='runtime' else
                    ['runtime-plan','storage-preflight'] if failure=='storage' else
                    ['runtime-plan','storage-preflight','complete'])
    assert all(float(line.split('unix=')[1])>0 for line in lines)
