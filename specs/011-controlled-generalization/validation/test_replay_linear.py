import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('replay_linear',HERE/'replay_linear.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_schedule_and_cross_process_classification():
    assert len(helper.schedule())==9
    rows=[]
    for policy,trial in helper.schedule():
        rows.append(dict(policy=policy,trial=trial,status='PASS',result=dict(variants={'linear':dict(output_sha256=str(trial) if policy=='observe' else 'same',repeat_equal=True)})))
    result=helper.summarize(rows)
    assert result['complete'] and not result['policies']['observe']['linear']['cross_process_equal']
    assert result['policies']['rocblas']['linear']['cross_process_equal']
    assert not helper.summarize(rows[:-1])['complete']


def test_saved_operand_discovery_checks_stage_and_payload_hash(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.identity import file_hash
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    run=tmp_path/'run';stage=run/'old-traced';role=stage/'role';learner=role/'private/learner';learner.mkdir(parents=True)
    operand=role/'linear-operands.pt';operand.write_bytes(b'trusted private placeholder')
    (run/'diagnostic.json').write_text(json.dumps(dict(metadata={'images':['cpu','gpu']})))
    result=dict(status='PASS',learner='run/old-traced/role/private/learner',trace=[dict(phase='forward',linear_operand_sha256=file_hash(operand))])
    (stage/'result.json').write_text(json.dumps(result))
    assert helper.saved_operand(run)[0]==operand
    operand.write_bytes(b'corrupt')
    with pytest.raises(ValueError):helper.saved_operand(run)


@pytest.mark.parametrize('engine',['docker','podman'])
def test_actual_command_is_isolated_gpu_with_directory_mounts(tmp_path,monkeypatch,engine):
    from oeis_learn.container_runtime import Engine
    adapter=Engine(engine,executable=engine,details={})
    monkeypatch.setattr(adapter,'user_args',lambda:[])
    monkeypatch.setattr(adapter,'device_args',lambda *_:['--device','/dev/dri/renderD128'])
    dirs=[tmp_path/n for n in ('helper','inputs','output')]
    for p in dirs:p.mkdir()
    host=SimpleNamespace(engine=adapter,policy=dict(memory_bytes=16<<30,cpus=4,render_node='/dev/dri/renderD128'))
    argv=helper.command(host,'sha256:'+'a'*64,*dirs,'test','no-lt','sha256:'+'b'*64)
    mounts=[argv[i+1] for i,v in enumerate(argv) if v=='--mount']
    assert len(mounts)==3 and all('readonly' in m for m in mounts[:-1])
    assert '--network=none' in argv and '--read-only' in argv
    assert '/dev/dri/renderD128' in argv and not any('sock' in m for m in mounts)


def test_export_excludes_private_operands_even_on_failure(tmp_path,monkeypatch):
    import zipfile
    monkeypatch.syspath_prepend(str(HERE))
    from diagnose_hip import save
    from oeis_learn.pipeline_validation.bundle import verify
    root=tmp_path/'replay';root.mkdir()
    save(root/'summary.json',dict(selected_scope='FAIL',feature_qualified=False))
    save(root/'results.json',dict(rows=[]))
    private=root/'private/inputs';private.mkdir(parents=True)
    (private/'linear-operands.pt').write_bytes(b'private tensor contents')
    archive=helper.export(root);verify(archive)
    with zipfile.ZipFile(archive) as z:
        assert not any('private' in n or n.endswith('.pt') for n in z.namelist())
        assert json.loads(z.read('summary.json'))['selected_scope']=='FAIL'


def test_run_failure_preserves_cleanup_and_flags_containment(tmp_path,monkeypatch):
    from contextlib import nullcontext
    monkeypatch.syspath_prepend(str(HERE))
    from oeis_learn.pipeline_validation import supervision,generalization_lifecycle
    import diagnose_hip
    monkeypatch.setattr(helper,'command',lambda *a:[])
    monkeypatch.setattr(generalization_lifecycle,'guard',lambda *_:nullcontext())
    monkeypatch.setattr(supervision,'launch',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('process failure')))
    reclaimed=[]
    monkeypatch.setattr(supervision,'reclaim',lambda *a:reclaimed.append(True))
    monkeypatch.setattr(diagnose_hip,'uncontained',lambda _:True)
    host=SimpleNamespace(gpu_lock=lambda:nullcontext(),engine=object())
    result=helper.run_one(host,'image',tmp_path,tmp_path,tmp_path/'out','observe','hash')
    assert result['status']=='FAIL' and result['containment']=='UNACKNOWLEDGED' and reclaimed==[True]
