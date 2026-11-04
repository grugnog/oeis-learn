import importlib.util
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('fresh_pilot',HERE/'fresh_pilot.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_outer_reservations_include_validation_cleanup_and_packaging():
    assert sum(m.RESERVATIONS.values())+m.REPORT_SECONDS<=m.CAMPAIGN_SECONDS
    assert m.RESERVATIONS['prepare']>=14400+3600
    assert m.RESERVATIONS['train-static']>=21600+3600
    assert m.RESERVATIONS['generate-static']>21600
    with pytest.raises(TimeoutError):m.allowance(m.CAMPAIGN_SECONDS-m.REPORT_SECONDS,1)
    assert m.allowance(0,21600)==21600


def test_cli_and_role_resources_match_qualification():
    command=m.canonical('podman','sha256:'+'1'*64,'train','out',
        ['--manifest','study.json','--arms','static','--seeds','20261016','--qualification','qualified'])
    assert command[command.index('--memory-gib')+1]=='64'
    assert command[command.index('--cpus')+1]=='4'
    assert command.index('--memory-gib')<command.index('generalization')
    assert '--resume' not in command


def test_latest_checkpoint_requires_original_roster_and_real_bytes(tmp_path):
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    cp=tmp_path/'checkpoints';cp.mkdir();(cp/'blob.pt').write_bytes(b'actual test bytes')
    descriptor=seal(dict(weights=reference(cp/'blob.pt',cp),tokens=20,updates=1,initialization=digest('init')))
    roster=seal(dict(version='generalization-retained/v1',study=digest('study'),arm='static',seed=20261016,
        checkpoints=[descriptor],latest=descriptor['identity']))
    write(cp/'retained.json',roster)
    result=dict(tokens=20,updates=1,retained=roster['identity'])
    assert m.latest_checkpoint(tmp_path,result,study=digest('study'),arm='static')[0]==descriptor
    (cp/'blob.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError):m.latest_checkpoint(tmp_path,result,study=digest('study'),arm='static')


def test_dry_run_is_stdlib_only_and_never_touches_engine(tmp_path):
    import subprocess
    done=subprocess.run([sys.executable,'-S',str(HERE/'fresh_pilot.py'),'run','--engine','podman',
        '--cpu-image','cpu-tag','--gpu-image','gpu-tag','--output','run-state/not-created','--dry-run'],
        capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert str(m.CAMPAIGN_SECONDS) in done.stdout and 'NOT_RUN' in done.stdout
    assert not (m.ROOT/'run-state/not-created').exists()


@pytest.mark.parametrize('resuming,failure',[
    (resuming,failure) for resuming in (False,True,'rebind')
    for failure in (None,'feasibility','preflight','prepare','smoke','train-static','train-refreshed',
        'generate-static','score-static','generate-refreshed','score-refreshed','partial-training','missing-verifier','initialization','projection')
    if not resuming or failure not in ({'feasibility','prepare'} if resuming=='rebind' else {'feasibility','preflight','prepare'})])
def test_serial_command_boundaries_and_failure_packaging(tmp_path,monkeypatch,failure,resuming):
    """Mock expensive processes, retain real controller/lineage/packaging checks."""
    import json
    from types import SimpleNamespace
    from contextlib import nullcontext
    from oeis_learn.pipeline_validation.identity import write,seal,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation import generalization_inputs, generalization_runtime, generalization_lifecycle
    from oeis_learn.pipeline_validation.bundle import pack,verify
    from oeis_learn.experiments import source_snapshot
    from test_fresh_pilot_report import fixture
    study,rows=fixture();base=tmp_path/'resolved';base.mkdir()
    write(base/'targets.json',rows);write(base/'prompts.json',[dict(id=r['id']) for r in rows])
    write(base/'cohort.json',dict(role='selection',private_payloads=[reference(base/'targets.json',base)],public_payloads=[reference(base/'prompts.json',base)]))
    study['data']=dict(manifests=[reference(base/'cohort.json',base)])
    write(base/'recipe.json',{})
    runtime=dict(storage={},recipe=reference(base/'recipe.json',base))
    qualified=tmp_path/'qualified';qualified.mkdir();write(qualified/'source.json',{})
    write(qualified/'qualification.json',dict(source_closure=reference(qualified/'source.json',qualified),
        resources=dict(memory_bytes=64<<30,cpus=4),images=['sha256:'+'1'*64,'sha256:'+'2'*64]))
    imported=tmp_path/'imported';imported.mkdir()
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda p:{})
    design=seal(dict(version='synthetic-design',numeric_law='synthetic-law',descriptor_screen='synthetic-screen'))
    import fresh_pilot_feasibility
    monkeypatch.setattr(fresh_pilot_feasibility,'verify_prepared',lambda *a:None)
    def blueprint(root,design,**kw):
        write(root/kw['output']/'design.json',design);write(root/kw['output']/'blueprint.json',{});return {}
    monkeypatch.setattr(m,'blueprint',blueprint)
    monkeypatch.setattr(generalization_inputs,'assemble_study',lambda *a:dict(manifest='fake-study.json'))
    monkeypatch.setattr(generalization_runtime,'load',lambda *a:(base,study,runtime))
    monkeypatch.setattr(generalization_lifecycle,'guard',lambda *a:nullcontext())
    calls=[];saved={}
    import fresh_pilot_storage
    def project(workspace,manifest,output):
        if failure=='projection':raise ValueError('execution projection exceeds peak shard reservation')
        destination=workspace/output;destination.mkdir(parents=True)
        return str((destination/'study.json').relative_to(workspace)),seal(dict(test_only=True,storage=dict(peak_shard_bytes=0)))
    monkeypatch.setattr(fresh_pilot_storage,'project',project)
    def public(output,**kwargs):
        from fresh_pilot_report import REQUIRED
        saved.update(kwargs);assert kwargs['plan']['required_stages']==list(REQUIRED)
        assert any(name=='fresh/recipe.json' and path.read_bytes()==(base/'recipe.json').read_bytes() for path,name in kwargs['extra'])
        status=kwargs['stages'][-1]['status']
        write(output/'fresh/study.json',study)
        return dict(software_status=status)
    monkeypatch.setattr(m,'assemble_public',public)
    def invoke(argv,logs,*,seconds):
        name=logs.name;calls.append(name);output=tmp_path/argv[argv.index('--output')+1]
        assert seconds>0
        if name=='report':
            write(output/'summary.json',dict(selected_scope=saved['stages'][-1]['status']))
            pack(output,mandatory=['summary.json']);return m.EXIT_CODES[saved['stages'][-1]['status']]
        if name=='verify':
            if failure=='missing-verifier':return 1
            archive=tmp_path/argv[argv.index('--archive')+1];verify(archive)
            status=saved['stages'][-1]['status'];ref=reference(archive,tmp_path)
            write(output/'summary.json',seal(dict(selected_scope=status,archive=ref,integrity='PASS')))
            write(output/'result.json',seal(dict(version='generalization-verification-result/v1',status='COMPLETE',
                archive=ref,summary=reference(output/'summary.json',output),exit_code=m.EXIT_CODES[status])))
            return m.EXIT_CODES[status]
        if name==failure:
            write(output/'result.json',seal(dict(status='FAIL')));return 1
        result=dict(status='PASS')
        if name=='feasibility':
            write(output/'private/design.json',design)
            write(output/'feasibility.json',seal(dict(protocol=design['identity'],status='FEASIBLE')))
            result['feasibility']=reference(output/'feasibility.json',output)
        if name=='prepare':result=dict(status='PASS',result=dict(manifest='resolved-study.json'))
        if name.startswith('train-'):
            assert output.parent==tmp_path/'pilot/execution'
            assert argv[argv.index('--manifest')+1]=='pilot/execution/private/prepared/study.json'
            arm=name.removeprefix('train-');learner=output/'training/run/execution/role/private/learner'
            cp=learner/'checkpoints';cp.mkdir(parents=True);(cp/'blob.pt').write_bytes(b'synthetic weights')
            checkpoint=seal(dict(weights=reference(cp/'blob.pt',cp),tokens=20,updates=1,
                initialization=digest(arm if failure=='initialization' else 'common')))
            roster=seal(dict(version='generalization-retained/v1',study=study['identity'],arm=arm,seed=20261016,
                checkpoints=[checkpoint],latest=checkpoint['identity']))
            write(cp/'retained.json',roster)
            status='PARTIAL' if failure=='partial-training' else 'PASS'
            training=seal(dict(status=status,tokens=20,updates=1,retained=roster['identity']))
            write(learner/'training.json',training)
            result=dict(status=status,result=dict(runs=[dict(root='run/execution/role/private/learner',result=training)]))
        if name.startswith(('generate-','score-')):
            job=json.loads((tmp_path/argv[argv.index('--job')+1]).read_text())
            if name.startswith('generate-'):
                assert set(job['artifacts'])=={'visible_prompts','frozen_weights'}
                assert job['settings']['bank_seeds']==list(range(20261101,20261109))
                write(output/'execution/role/generation/selections.json',{})
            else:
                assert set(job['artifacts'])=={'selection_supervision','selection_candidates'}
                write(output/'execution/role/selection/scoring.json',dict(result=dict(outcomes={})))
        write(output/'result.json',seal(result));return m.EXIT_CODES[result['status']]
    monkeypatch.setattr(m,'invoke',invoke)
    args=SimpleNamespace(output='pilot',engine='podman',imported='imported',qualified='qualified')
    import fresh_pilot_throughput_host
    args.throughput='throughput';write(tmp_path/'throughput/result.json',seal(dict(status='PASS')))
    monkeypatch.setattr(fresh_pilot_throughput_host,'require_for_run',lambda a:(dict(identity=digest('throughput'),binding=dict(image='sha256:'+'1'*64)),dict(protocol=design['identity'])))
    import fresh_pilot_resume
    monkeypatch.setattr(fresh_pilot_resume,'disk_preflight',lambda *a:dict(free_bytes=256<<30,required_free_bytes=224<<30))
    if resuming:
        previous=tmp_path/'previous'
        write(previous/'research/design/design.json',design);write(previous/'research/design/blueprint.json',{})
        write(previous/'research/protocol-summary.json',{})
        write(previous/'feasibility/feasibility.json',{})
        prior_stages=[dict(stage=name,status='PASS',exit_code=0,elapsed_seconds=1,command=['original'],originals=[])
            for name in ('feasibility','preflight','prepare')]
        parent=seal(dict(status='FAIL',stages=prior_stages,acceptance=dict(exit_code=0)))
        write(previous/'workflow.json',parent)
        context=dict(old=previous,qualified=qualified,images=['sha256:'+'1'*64,'sha256:'+'2'*64],
            manifest='resolved-study.json',design=design,source_check={},workflow=parent)
        args.resume_prepared='previous'
        monkeypatch.setattr(fresh_pilot_resume,'prepared_inputs',lambda *a:context)
        monkeypatch.setattr(fresh_pilot_resume,'continuation_record',lambda *a:seal(dict(parent=parent['identity'])))
        monkeypatch.setattr(fresh_pilot_resume,'throughput_proof',lambda *a:(tmp_path/'throughput',dict(identity=digest('throughput')),dict(protocol=design['identity'])))
        original_parent=(previous/'workflow.json').read_bytes()
        if resuming=='rebind':
            import fresh_pilot_rebind
            args.rebind_prepared='previous';args.training_probe='probe';args.resume_prepared=None
            monkeypatch.setattr(fresh_pilot_rebind,'paths',lambda *a:None)
            def transition(root,args,target):
                manifest,view=project(root,'resolved-study.json','pilot/execution/private/prepared')
                blueprint(root,design,output='pilot/research/design')
                for name in ('rebind.json','training-probe.json','parent-preparation-summary.json','execution-view.json','research/protocol-summary.json'):
                    write(target/name,seal(dict(fixture=True)))
                return dict(context,execution_manifest=manifest,view=view,design_root=target/'research/design',
                    probe=dict(identity=digest('hourly-probe')),reused={s['stage']:dict(s,execution='REUSED') for s in prior_stages})
            monkeypatch.setattr(fresh_pilot_rebind,'transition',transition)
    code=m.run(args)
    if resuming is True:
        assert not set(calls)&{'feasibility','preflight','prepare'}
        assert calls[0]=='smoke' if failure!='projection' else not calls
        assert (previous/'workflow.json').read_bytes()==original_parent
        assert not (tmp_path/'pilot/research/design').exists()
    workflow=json.loads((tmp_path/'pilot/workflow.json').read_text());archive=tmp_path/workflow['archive']
    assert archive.is_file();verify(archive)
    if resuming is True:
        assert workflow['resume'] and all(s['execution']=='REUSED' for s in workflow['stages'][:3])
        if failure=='smoke':
            import zipfile
            with zipfile.ZipFile(archive) as z:assert 'parent-workflow.json' in z.namelist() and 'resume.json' in z.namelist()
    if resuming=='rebind':
        assert not set(calls)&{'feasibility','prepare'}
        assert calls[0]=='preflight' if failure!='projection' else not calls
        assert (previous/'workflow.json').read_bytes()==original_parent
        if failure not in {'projection','preflight'}:
            assert [s['stage'] for s in workflow['stages'][:4]]==['feasibility','preflight','prepare','smoke']
            assert [s.get('execution') for s in workflow['stages'][:3]]==['REUSED',None,'REUSED']
        if failure in {'preflight','smoke'}:
            import zipfile
            with zipfile.ZipFile(archive) as z:assert {'rebind.json','training-probe.json'}<=set(z.namelist())
    if failure is None:assert code==0 and workflow['status']=='PASS'
    elif failure=='partial-training':assert code==3 and 'score-refreshed' in calls
    else:assert code==1 and workflow['status']=='FAIL'
    if failure in {'feasibility','preflight','prepare','smoke','projection'}:assert not any(n.startswith('train-') for n in calls)
    if failure=='projection':assert 'smoke' not in calls
    if failure=='feasibility':assert calls==['feasibility']
    if failure=='train-static':assert 'train-refreshed' not in calls
    if failure=='initialization':assert not any(n.startswith('generate-') for n in calls)
    if failure=='missing-verifier':assert archive.name=='recovery.zip'
    if saved:assert saved['stages'][-1]['stage']=='controller'


@pytest.mark.parametrize('changed',[False,True])
def test_role_uses_canonical_qualification_and_exact_frozen_generation(tmp_path,monkeypatch,changed):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.identity import seal,write,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation import generalization_commands, generalization_runtime, generalization_roles
    import foundation_local
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(foundation_local,'Host',lambda args:SimpleNamespace(args=args))
    images=['sha256:'+'1'*64,'sha256:'+'2'*64]
    write(tmp_path/'qualified/qualification.json',dict(images=images))
    write(tmp_path/'recipe.json',{'synthetic':True})
    protocol=seal(dict(bank_seeds=m.BANK_SEEDS));write(tmp_path/'protocol.json',protocol)
    study=dict(identity=digest('study'),hypothesis='frozen protocol '+protocol['identity'])
    runtime=dict(purpose='research',images=images,recipe=reference(tmp_path/'recipe.json',tmp_path),storage={})
    monkeypatch.setattr(generalization_runtime,'load',lambda *a:(tmp_path,study,runtime))
    used=[]
    def qualification(host,workspace,output,base,original,sidecar,args):
        assert original==study and sidecar==runtime and args.qualification=='qualified';used.append('qualification')
    monkeypatch.setattr(generalization_commands,'_qualification',qualification)
    def invoke(host,image,root,role,artifacts,**kw):
        assert image==images[1] and role=='generator' and kw['gpu'] is True
        assert kw['seconds']==21600 and set(kw['source_roots'])=={'visible_prompts','frozen_weights'}
        assert host.args.memory_gib==64 and host.args.cpus==4
        used.append('generation');return dict(status='PASS')
    monkeypatch.setattr(generalization_roles,'project_and_invoke',invoke)
    (tmp_path/'visible').mkdir();(tmp_path/'weights').mkdir()
    weights=dict(path='model.pt',bytes=1,sha256=digest('weights'))
    checkpoint=seal(dict(weights=weights))
    job=seal(dict(version='fresh-pilot-role/v1',engine='podman',qualified='qualified',manifest='manifest.json',
        study=study['identity'],study_root='.',protocol=reference(tmp_path/'protocol.json',tmp_path),checkpoint=checkpoint,
        role='generator',operation='evaluate',artifacts={'visible_prompts':[],'frozen_weights':[weights]},
        source_roots={'visible_prompts':'visible','frozen_weights':'weights'},settings=dict(study=study['identity'],
            source=checkpoint['identity'],checkpoint=weights,bank_seeds=[] if changed else m.BANK_SEEDS,
            device='cuda',recipe={'synthetic':True})))
    write(tmp_path/'job.json',job)
    assert m.role('job.json','role-output')==(1 if changed else 0)
    assert used==([] if changed else ['qualification','generation'])


def test_existing_run_diagnostics_never_launch_and_leave_originals_intact(tmp_path,monkeypatch):
    import json,zipfile
    from oeis_learn.pipeline_validation.identity import seal,write
    from oeis_learn.pipeline_validation.bundle import verify
    monkeypatch.setattr(m,'ROOT',tmp_path)
    def forbidden(*a,**kw):raise AssertionError('diagnose must not execute work')
    monkeypatch.setattr(m,'invoke',forbidden)
    write(tmp_path/'original/workflow.json',seal(dict(status='FAIL',stages=[])))
    before=(tmp_path/'original/workflow.json').read_bytes()
    base=tmp_path/'original/research/prepare/preparation/role/private/prepared'
    write(base/'preparation-result.json',seal(dict(status='BLOCKED',cell='refresh-01',reason='QUOTA_SHORTFALL')))
    assert m.main(['diagnose','--run','original','--output','exported'])==0
    assert (tmp_path/'original/workflow.json').read_bytes()==before
    verify(tmp_path/'exported.zip')
    with zipfile.ZipFile(tmp_path/'exported.zip') as z:
        summary=json.loads(z.read('diagnostics/preparation-summary.json'))
        assert summary['terminal']['cell']=='refresh-01'
    assert m.main(['diagnose','--run','original','--output','original/export'])==2
