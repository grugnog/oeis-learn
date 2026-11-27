"""Source boundaries, owner fencing and the stdlib recovery handoff."""
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal, reference, load
from oeis_learn.pipeline_validation.progressive_workstation_state import write, Session
from oeis_learn.pipeline_validation import progressive_calibration_equivalence as eq


@pytest.mark.parametrize('show_empty_supported',[False,True])
def test_protected_fingerprints_keep_empty_fields_on_both_dump_apis(monkeypatch,show_empty_supported):
    import inspect
    original_dump=eq.ast.dump
    options={'show_empty':True} if 'show_empty' in inspect.signature(original_dump).parameters else {}
    calls=[]
    def old_dump(node,*,include_attributes=False):
        calls.append('legacy')
        return original_dump(node,include_attributes=include_attributes,**options)
    def new_dump(node,*,include_attributes=False,show_empty=False):
        calls.append(show_empty)
        # Python 3.13's new default is unsuitable for the original 3.12 pins.
        assert show_empty is True
        return original_dump(node,include_attributes=include_attributes,**options)
    monkeypatch.setattr(eq.ast,'dump',new_dump if show_empty_supported else old_dump)
    old={eq.PREFIX+p:digest('old') for p in eq.PROTECTED}
    current={p:digest('host verifier changed') for p in old}
    monkeypatch.setattr(eq,'ORIGINAL_SOURCE',digest(old))
    assert eq.validate(old,current)==sorted(current)
    assert calls==[True if show_empty_supported else 'legacy']*3


def available_stdlib_interpreters():
    # The running interpreter is mandatory. Extra installed versions supplement
    # it; absent optional binaries are not selected into zero-skip qualification.
    return sorted({sys.executable,*(p for name in ('python3.12','python3.13') if (p:=shutil.which(name)))})


def test_stdlib_selection_always_includes_current_python_without_optional_binaries(monkeypatch):
    monkeypatch.setattr(shutil,'which',lambda name:None)
    assert available_stdlib_interpreters()==[sys.executable]


@pytest.mark.parametrize('executable',available_stdlib_interpreters())
def test_actual_stdlib_interpreters_preserve_pins_and_reject_code_changes(tmp_path,executable):
    code='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from oeis_learn.pipeline_validation import progressive_calibration_equivalence as eq
from oeis_learn.generalization.progressive.contracts import digest
original={eq.PREFIX+p:digest('original') for p in eq.PROTECTED}
eq.ORIGINAL_SOURCE=digest(original)
current={p:digest('host-only change') for p in original}
assert eq.validate(original,current)==sorted(current)
relative='generalization/progressive/iteration.py'
base=Path(sys.argv[2]); path=base/relative;path.parent.mkdir(parents=True)
path.write_text((Path(eq.__file__).resolve().parents[1]/relative).read_text().replace('return sum(phases)','return 0'))
original={eq.PREFIX+relative:digest('original')};eq.ORIGINAL_SOURCE=digest(original)
try:eq.validate(original,{eq.PREFIX+relative:digest('changed')},package_root=base)
except ValueError as exc:assert 'protected calibration execution changed' in str(exc)
else:raise AssertionError('modified iteration guard accepted')
assert not {'torch','numpy','optuna','jsonschema'} & set(sys.modules)
print(sys.version.split()[0], 'all original pins match; changed guard rejected')
'''
    result=subprocess.run([executable,'-S','-c',code,str(Path(__file__).resolve().parents[2]/'src'),str(tmp_path)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'changed guard rejected' in result.stdout


def test_reviewed_source_boundary_keeps_tensor_native_and_lock_bytes(monkeypatch):
    old={'src/oeis_learn/pipeline_validation/progressive_training.py':digest('original'), 'uv.lock':digest('lock')}
    monkeypatch.setattr(eq,'ORIGINAL_SOURCE',digest(old))
    current=dict(old, **{'src/oeis_learn/pipeline_validation/progressive_research_resources.py':digest('planner')})
    assert eq.validate(old,current)==['src/oeis_learn/pipeline_validation/progressive_research_resources.py']
    assert 'src/oeis_learn/generalization/progressive/freeze.py' in eq.validate(old,dict(current,**{
        'src/oeis_learn/generalization/progressive/freeze.py':digest('host storage admission changed')}))
    assert 'scripts/discovery_repair_reservation.py' in eq.validate(old,dict(current,**{
        'scripts/discovery_repair_reservation.py':digest('external metadata repair only')}))
    for path in ('uv.lock','src/oeis_learn/pipeline_validation/progressive_training.py',
                 'crates/oeis_wasm_evaluator/src/lib.rs','src/oeis_learn/decoder/new_kernel.py','scripts/unreviewed.py'):
        with pytest.raises(ValueError,match='execution source'):
            eq.validate(old,dict(current,**{path:digest('changed')}))
    with pytest.raises(ValueError,match='reviewed original'):eq.validate({},current)


@pytest.mark.parametrize('relative',list(eq.PROTECTED))
def test_mixed_host_device_modules_reject_protected_function_edits(tmp_path,monkeypatch,relative):
    original={eq.PREFIX+relative:digest('original')}
    monkeypatch.setattr(eq,'ORIGINAL_SOURCE',digest(original))
    package=Path(eq.__file__).resolve().parents[1]
    path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((package/relative).read_bytes())
    current={eq.PREFIX+relative:digest('new host verifier')}
    assert eq.validate(original,current,package_root=tmp_path)
    # Alter executable protected code rather than a comment/line number.
    text=path.read_text()
    target=('def _run_inside(' if relative.endswith('progressive_calibration.py') else
            'def margin(' if relative.endswith('iteration.py') else 'def case_package(')
    text=text.replace(target,target.replace('def ','def corrupted_'))
    path.write_text(text)
    with pytest.raises(ValueError,match='protected'):eq.validate(original,current,package_root=tmp_path)


def test_fenced_original_is_exact_selected_donor(tmp_path):
    from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original
    from oeis_learn.pipeline_validation.progressive_workstation_state import Blocked
    donor=tmp_path/'run-state/old/prepare-research';write(donor/'construction/blueprint.json',{})
    current=tmp_path/'run-state/new'
    previous=seal(dict(version='progressive-preparation-data-reuse/v1',recipient='run-state/old'))
    proof=seal(dict(version='progressive-preparation-owner-proof/v5',predecessor=previous,
        original_costs_retained=True,scientific_allowance_transferred=False,blueprint=reference(tmp_path,donor/'construction/blueprint.json')))
    write(current/'preparation-owner-proof.json',proof)
    write(tmp_path/'history/previous.json',previous);write(tmp_path/'history/proof.json',proof)
    write(current/'preparation-owner.json',seal(dict(version='progressive-preparation-owner/v2',
        receipt=seal(dict(previous,recipient='run-state/new')),previous=reference(tmp_path,'history/previous.json'),
        proof=reference(tmp_path,'history/proof.json'))))
    w=SimpleNamespace(root=tmp_path,s=SimpleNamespace(folder=current),args=SimpleNamespace(reuse_calibration_session='run-state/old'))
    assert fenced_original(w)==donor
    w.args.reuse_calibration_session='run-state/other'
    with pytest.raises(Blocked):fenced_original(w)
    w.args.reuse_calibration_session='run-state/old'
    retained=(tmp_path/'history/previous.json').read_bytes()
    write(tmp_path/'history/previous.json',seal(dict(previous,recipient='run-state/unrelated')))
    with pytest.raises(ValueError):fenced_original(w)
    (tmp_path/'history/previous.json').write_bytes(retained)
    write(current/'preparation-owner-proof.json',seal(dict(proof,original_costs_retained=False)))
    with pytest.raises(ValueError):fenced_original(w)


@pytest.mark.parametrize('with_headroom',[False,True])
@pytest.mark.parametrize('fail',[False,True])
def test_wrapper_recovery_measure_revision_resume_and_failure_zip(tmp_path,monkeypatch,fail,with_headroom):
    from tests.integration.test_progressive_calibration_reuse import recovery_fixture
    from oeis_learn.pipeline_validation import progressive_calibration_reuse_workstation as wmodule
    from oeis_learn.pipeline_validation import progressive_trajectory as trajectory
    from oeis_learn.pipeline_validation.progressive_runtime_timing import case_package
    from oeis_learn.pipeline_validation.progressive_calibration_reuse import raw_prepared
    original_replay=trajectory.replay
    _,_,revised,_=recovery_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(trajectory,'replay',original_replay)
    session=Session(tmp_path,'run-state/new',{},'prepare-research');base=session.folder/'prepare-research'
    shutil.copytree(tmp_path/'next-qualification',base/'qualification')
    payload=raw_prepared(tmp_path,revised)['payload'];payload.pop('blueprint')
    current_runtime=load(tmp_path/payload['runtime']['path']);current_runtime['payload']['qualification']=[reference(tmp_path,base/'qualification/qualification.json')]
    write(base/'construction/runtime.json',seal(current_runtime));payload['runtime']=reference(tmp_path,base/'construction/runtime.json')
    for r in payload['references']:
        if r['kind']=='qualification':r['artifact']=reference(tmp_path,base/'qualification/qualification.json')
    payload['references']=[r for r in payload['references'] if r['name'] not in {'calibration_reuse','training_trajectory','research_resource_policy'}]
    if with_headroom:
        from tests.unit.test_progressive_research_roster import fixture
        from oeis_learn.generalization.progressive.research_roster import construct
        from oeis_learn.generalization.progressive.research_budget import reservation
        from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
        from oeis_learn.pipeline_validation.progressive_preparation_preflight import reservation as extra
        _,panels,screen=fixture()
        data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
        outer=reservation(construct(scientific_round(data,expanded=True),panels,screen))
        write(base/'roster/research-reservation.json',outer)
        write(base/'construction/research-reservation.json',outer)
        payload['references'].append(dict(name='research-reservation-v2',role='custodian',kind='policy',artifact=reference(tmp_path,base/'construction/research-reservation.json')))
        write(base/'preparation-headroom.json',extra(tmp_path,reference(tmp_path,base/'roster/research-reservation.json')))
    write(base/'construction/preparation-plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    b=copy.deepcopy(revised)
    for ref in b['payload']['references']:
        if ref['name']=='preparation_plan':ref['artifact']=reference(tmp_path,base/'construction/preparation-plan.json')
    write(base/'construction/blueprint.json',seal(b))
    session.journal['steps']['prepare-research/construction']=dict(status='COMPLETE')
    w=SimpleNamespace(root=tmp_path,base=base,s=session,args=SimpleNamespace(reuse_calibration_session='old'),host=lambda:object())
    monkeypatch.setattr(wmodule,'fenced_original',lambda w:tmp_path)
    calls=[]
    def stage(host,root,image,output,argv,**kw):
        calls.append('trajectory'); assert kw['gpu'] and 35000<kw['seconds']<=36300
        write(output/'container-inspect.json',[dict(State=dict(Running=False,OOMKilled=False,ExitCode=0))])
        write(output/'process.json',dict(returncode=1 if fail else 0,failure=None,cleanup_failure=None))
        write(output/'cleanup.json',dict(acknowledged=True,within_bound=True,failure=None))
        if fail:
            (output/'stderr.log').write_text('original injected device failure')
            raise RuntimeError('explicit device boundary failure')
        req=load(output/'request.json');package=load(output/req['package']['path']);rows=[]
        write(output/'hardware.json',{'boundary_only':True})
        for i,case in enumerate(req['cases']):
            timing=dict(windows=[dict(tokens=10000,elapsed_us=10000,cold=False)],loop_us=10000,
                maximum_shard=1,components=dict(checkpoint_save=dict(calls=2,inclusive_us=100)))
            segments=[dict(timing=timing,process=dict(elapsed_us=11000),result=dict(mechanical_status='COMPLETE',
                metrics=dict(initialization='fresh_random' if j==0 else 'resume'))) for j in range(2)]
            path=output/'cases'/str(i)/'producer.json'
            write(path,seal(dict(request=req['identity'],case=case,package=case_package(package,case)['identity'],
                segments=segments,status='PASS',device='actual_HIP',measurement_scope='TEST BOUNDARY ONLY')))
            (path.parent/'diagnostics.zip').write_bytes(b'fixture')
            rows.append(dict(index=i,status='PASS',producer=reference(output,path),diagnostics=reference(output,path.parent/'diagnostics.zip')))
        write(output/'runtime-cases.json',seal(dict(request=req['identity'],status='PASS',device_block=None,cases=rows)))
    monkeypatch.setattr('oeis_learn.pipeline_validation.progressive_handoff.stage',stage)
    with session.own():path=wmodule.prepare(w)
    if fail:
        assert session.journal['steps']['prepare-research/calibration-reuse-plan']['status']=='BLOCKED'
        assert not (tmp_path/path).exists()
    else:
        assert (tmp_path/path).is_file()
        assert session.journal['steps']['prepare-research/calibration-reuse-plan']['status']=='COMPLETE'
        if with_headroom:
            from oeis_learn.pipeline_validation.progressive_preparation_preflight import verified
            assert verified(tmp_path,raw_prepared(tmp_path,load(tmp_path/path))['payload']['references'])
            assert load(base/'training-trajectory/trajectory-reservation.json')['reserved_qualification_seconds']==load(base/'preparation-headroom.json')['reserved_qualification_seconds']
    with session.own():assert wmodule.prepare(w)==path
    assert calls==['trajectory']
    with zipfile.ZipFile(session.folder/'share/prepare-research.zip') as z:
        assert not any('/private/' in n for n in z.namelist())
        assert any(n.endswith('calibration-reuse.json') for n in z.namelist())
        assert any(n.endswith('trajectory-reservation.json') for n in z.namelist())
        if fail:assert b'original injected' in z.read('prepare-research/training-trajectory/stderr.log')
        else:assert any(n.endswith('trajectory-report.json') for n in z.namelist())


def test_stdlib_cli_recovery_flags():
    script=Path(__file__).resolve().parents[2]/'scripts/discovery_hip_handoff.py'
    args=[sys.executable,'-S',str(script),'--engine','podman','--session','run-state/new','prepare-research',
        '--registration-session','run-state/registered','--reuse-preparation-data','run-state/data',
        '--reuse-calibration-session','run-state/old','--expanded-runtime-budgets','--run-after-prepare','--dry-run']
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '13 current-source' in result.stdout and 'run-or-resume' in result.stdout
    assert 'reuse-exact-calibration' in result.stdout and 'CPU-native-census' not in result.stdout
    args.remove('--expanded-runtime-budgets')
    assert subprocess.run(args,capture_output=True).returncode!=0


@pytest.mark.parametrize('with_progress',[False,True])
@pytest.mark.parametrize('fault',[None,'science','frozen','running','history','command','cleanup','result','construction','stdout','truncated-log'])
def test_completed_calibration_owner_transfer_is_time_only_and_fenced(tmp_path,monkeypatch,fault,with_progress):
    from tests.unit.test_progressive_preparation_reuse import construction_owner
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof,claim_owner
    from oeis_learn.pipeline_validation import progressive_completed_calibration_recovery as recovery
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    folder,receipt=construction_owner(tmp_path,monkeypatch);base=folder/'prepare-research'
    q=load(base/'qualification/qualification.json')
    source=load(checked_file(base/'qualification',q['source_closure']))
    monkeypatch.setattr(eq,'ORIGINAL_SOURCE',digest(source))
    monkeypatch.setattr('oeis_learn.experiments.source_snapshot.collect_snapshot',lambda root:source)
    blueprint=seal(dict(fixture='construction'))
    prepared=seal(dict(payload=dict(deficits=['training_time:first'])))
    write(base/'construction/blueprint.json',blueprint);write(base/'prepared/prepared.json',prepared)
    write(base/'prepared/calibration/producer.json',seal(dict(boundary_only=True)))
    write(base/'prepared/calibration/request.json',seal(dict(boundary_only=True)))
    # The integration suite replays the original complete calibration. Here that
    # boundary is doubled so every owner/journal/termination condition is varied.
    monkeypatch.setattr(recovery,'originals',lambda *a:(blueprint,prepared,{},[]))
    result=seal(dict(operation='prepare',mechanical_status='PARTIAL',reason='BOUND_OUTCOME_BLIND_PREPARATION',
        producer_error=None,scientific_status='UNMEASURED',missing_evidence=prepared['payload']['deficits'],
        metrics=dict(prepared=reference(base/'prepared',base/'prepared/prepared.json'))))
    write(base/'prepared/result.json',result)
    write(base/'prepared/owner.json',seal(dict(version='discovery-owner/v1',output=str((base/'prepared').relative_to(tmp_path)))))
    write(base/'prepared/operation-binding.json',seal(dict(operation='prepare',blueprint=blueprint['identity'],qualification=reference(tmp_path,base/'qualification/qualification.json'))))
    write(base/'construction/construction-summary.json',seal(dict(status='COMPLETE',executable=False,assessment_exposed=False,exact_calibration_cases=0,blueprint=reference(tmp_path,base/'construction/blueprint.json'))))
    session=load(folder/'session.json');session['binding'].update(engine='podman',memory_gib=64,cpus=8)
    write(folder/'session.json',seal(session))
    journal=load(folder/'journal.json');journal['steps']['prepare-research/construction']=dict(status='COMPLETE',
        outputs=[reference(tmp_path,base/'construction'/n) for n in ('blueprint.json','construction-summary.json')])
    journal['steps']['prepare-research/prepare']=dict(status='BLOCKED')
    logs=base/'logs/prepare/0000'
    write(logs/'command.json',dict(argv=['python3','-S',str(tmp_path/'scripts/foundation_run.py'),'--engine','podman',
        '--memory-gib','64','--cpus','8','--host-free-memory-gib','24','discovery','prepare',
        '--blueprint',str((base/'construction/blueprint.json').relative_to(tmp_path)),
        '--qualification',str((base/'qualification').relative_to(tmp_path)),'--output',str((base/'prepared').relative_to(tmp_path))]))
    write(logs/'process.json',dict(returncode=2,cleanup_failure=None,failure=None,outcome='COMPLETE'))
    write(logs/'stdout.log',result)
    if with_progress:
        path=logs/'stdout.log'
        path.write_text('  running 30.0s; remaining 259170.0s; logs /workspace/calibration\n'+path.read_text())
    if fault=='science':journal['steps']['run-research/start']=dict(status='COMPLETE')
    if fault=='frozen':(base/'frozen').mkdir()
    if fault=='running':journal['steps']['prepare-research/prepare']['status']='RUNNING'
    if fault=='history':journal['steps']['prepare-research/prepare']['history']=[dict(status='FAILED')]
    if fault=='construction':write(base/'construction/blueprint.json',{})
    if fault=='command':write(logs/'command.json',dict(argv=['python','unsafe']))
    if fault=='cleanup':write(logs/'process.json',dict(returncode=2,cleanup_failure='still running'))
    if fault=='result':write(base/'prepared/result.json',seal(dict(result,producer_error={'message':'unrelated failure'})))
    if fault=='stdout':write(logs/'stdout.log',dict(result,mechanical_status='INVALID'))
    if fault=='truncated-log':
        process=load(logs/'process.json');process['omitted_bytes']={'stdout':1};write(logs/'process.json',process)
    write(folder/'journal.json',seal(journal))
    if fault:
        with pytest.raises((ValueError,OSError)):failure_proof(tmp_path,receipt)
    else:
        original=(folder/'journal.json').read_bytes()
        proof=failure_proof(tmp_path,receipt)
        assert proof['version'].endswith('/v5') and proof['original_costs_retained']
        claim=tmp_path/'run-state/ownership.json';write(claim,receipt)
        recipient=seal(dict(receipt,recipient='run-state/replacement',current_source=digest('new host source')))
        claim_owner(tmp_path,claim,recipient)
        assert (folder/'journal.json').read_bytes()==original
        assert load(claim)['receipt']==recipient
        with pytest.raises(ValueError,match='superseded'):claim_owner(tmp_path,claim,receipt)
        # Round012 acquired the donor then failed CPU qualification. A fresh
        # recipient must follow that original ownership chain back to round009.
        from tests.unit.test_progressive_preparation_reuse import failed_owner
        from oeis_learn.pipeline_validation.progressive_calibration_reuse_workstation import fenced_original
        failed_owner(tmp_path,tmp_path/recipient['recipient'],recipient)
        latest=seal(dict(recipient,recipient='run-state/next',current_source=digest('timeout fix')))
        claim_owner(tmp_path,claim,latest)
        w=SimpleNamespace(root=tmp_path,s=SimpleNamespace(folder=tmp_path/latest['recipient']),
            args=SimpleNamespace(reuse_calibration_session=str(folder.relative_to(tmp_path))))
        assert fenced_original(w)==base
        # Two successive CPU qualification failures match rounds 012 and 013.
        failed_owner(tmp_path,tmp_path/latest['recipient'],latest)
        newer=seal(dict(latest,recipient='run-state/final',current_source=digest('skip fix')))
        claim_owner(tmp_path,claim,newer)
        w.s.folder=tmp_path/newer['recipient']
        assert fenced_original(w)==base
        from oeis_learn.pipeline_validation.progressive_preparation_preflight import prepare as preflight
        w.base=w.s.folder/'prepare-research'
        write(w.base/'roster/research-reservation.json',seal(dict(
            version='progressive-research-reservation/v2',status='READY',
            setup_seconds=dict(base_qualification=7200,qualify=3700,acceptance=7300),
            ancillary_seconds=dict(qualification_acceptance=20000),total_seconds=30000,
            limits_seconds=dict(qualification_acceptance=86400),selected_maximum_seconds=100000)))
        headroom,files=preflight(w)
        assert headroom['additional_seconds']==11800 and all(p.is_file() for p in files)
        assert load(w.base/'execution-preflight.json')['calibration_donor']==str(base.relative_to(tmp_path))
        w.args.reuse_calibration_session=recipient['recipient']
        with pytest.raises(ValueError):fenced_original(w)
        w.args.reuse_calibration_session=str(folder.relative_to(tmp_path))
        middle=tmp_path/recipient['recipient']
        journal=load(middle/'journal.json');journal['steps']['run-research/start']=dict(status='COMPLETE')
        write(middle/'journal.json',seal(journal))
        with pytest.raises(ValueError):fenced_original(w)


@pytest.mark.parametrize('body',[
    '', '  running 30.0s; remaining 90.0s; logs /workspace/calibration\n',
    '{"ok": true}\n{"ok": true}', '{"ok": true}\ntrailing output',
    '{"ok": true, "ok": false}', '{"ok": NaN}', '{"ok": Infinity}',
    '{"ok": true', 'unexpected failure\n{"ok": true}',
])
def test_recovery_stdout_rejects_missing_ambiguous_or_invalid_results(tmp_path,body):
    from oeis_learn.pipeline_validation.progressive_process_result import stdout_result
    path=tmp_path/'stdout.log';path.write_text(body)
    with pytest.raises(ValueError,match='canonical result stdout'):
        stdout_result(path)


def test_recovery_reads_real_supervisor_progress_before_terminal_json(tmp_path,capsys):
    import json
    from oeis_learn.pipeline_validation.supervision import launch
    from oeis_learn.pipeline_validation.progressive_process_result import stdout_result
    expected={'operation':'prepare','mechanical_status':'PARTIAL','missing_evidence':['training_time:first']}
    # The canonical host supervisor writes heartbeats to its own stdout, before
    # progressive_commands prints its single terminal result. Exercise that path.
    launch([sys.executable,'-S','-c','pass'],tmp_path/'child',seconds=10,heartbeat=0)
    print(json.dumps(expected,sort_keys=True))
    output=capsys.readouterr().out
    assert output.startswith('  running ')
    path=tmp_path/'stdout.log';path.write_text(output)
    assert stdout_result(path)==expected
    assert path.read_text()==output
    with pytest.raises(ValueError,match='bounded canonical result stdout'):
        stdout_result(path,maximum=10)


def test_research_workstation_passes_cpu_headroom_to_canonical_qualification(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_research_qualification as research
    from oeis_learn.pipeline_validation import generalization_qualification_io as io
    seen=[]
    def produce(host,root,output,**kwargs):
        seen.append(kwargs)
        return {'status':'PASS'}  # Host boundary only, not a hardware receipt.
    monkeypatch.setattr(io,'produce',produce)
    def step(name,action,**kwargs):
        return action()[0] if name=='base-qualification' else None
    w=SimpleNamespace(root=tmp_path,base=tmp_path/'prepare-research',
        s=SimpleNamespace(step=step,binding={'source':digest('current')}),
        host=lambda:object(),images=lambda:{'qualification':'cpu','gpu':'hip'},
        installed=lambda *a,**kw:None,run_command=lambda *a,**kw:None)
    research.qualify(w,{}, {})
    assert seen==[dict(images=['cpu','hip'],contract=digest('current'),cpu_seconds=10800)]
