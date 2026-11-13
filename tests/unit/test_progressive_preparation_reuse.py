"""Original data reuse never transfers a historical HIP or campaign authority."""
import pytest
from oeis_learn.generalization.progressive.contracts import reference,seal,digest
from oeis_learn.pipeline_validation.progressive_workstation_state import write
from oeis_learn.pipeline_validation.progressive_preparation_reuse import adopt


@pytest.fixture
def donor(tmp_path,monkeypatch):
    folder=tmp_path/'run-state/discovery/original';base=folder/'prepare-research'
    source={'src/oeis_learn/pipeline_validation/progressive_research_census.py':digest('unchanged')}
    write(base/'qualification/source.json',source)
    write(base/'qualification/qualification.json',dict(test_boundary=True))
    write(folder/'session.json',seal(dict(binding=dict(source=digest(source)))))
    (folder/'session.lock').touch()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    values={'registered-data':('registered/registered-data.json',data),
        'native-census':('census/census.json',seal(dict(diagnostic=False,training_started=False,assessment_exposed=False))),
        'materials':('materials/materials.json',seal(dict(status='COMPLETE',diagnostic=False))),
        'sensitivity':('sensitivity/sensitivity.json',seal(dict(status='COMPLETE')))}
    steps={}
    for name,(rel,value) in values.items():
        write(base/rel,value);steps['prepare-research/'+name]=dict(status='COMPLETE',outputs=[reference(tmp_path,base/rel)])
    write(folder/'journal.json',seal(dict(steps=steps)))
    # Named original-Q and native-material test boundaries. No hardware claimed.
    from oeis_learn.pipeline_validation import progressive_handoff,progressive_research_bindings
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(progressive_handoff,'original_authority',lambda *a:(dict(source_closure=reference(base/'qualification','source.json')),base/'qualification'))
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:source.copy())
    monkeypatch.setattr(progressive_research_bindings,'inputs',lambda root,d,design,m:None)
    return tmp_path,folder,source,steps


def test_completed_data_reuses_original_refs_with_single_recipient(donor):
    root,folder,source,steps=donor;recipient=root/'run-state/discovery/new'
    originals={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    values,outputs,receipt=adopt(root,str(folder.relative_to(root)),recipient)
    assert receipt['hardware_authority_transferred'] is False and receipt['original_costs_retained']
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert adopt(root,str(folder.relative_to(root)),recipient)[2]==receipt
    with pytest.raises(ValueError):adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/sibling')


def test_owner_transfer_recovers_after_pointer_publication_before_local_copy(donor,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_preparation_owner as owner
    from oeis_learn.pipeline_validation.progressive_workstation_state import read,Session
    root,folder,source,_=donor;first=root/'run-state/discovery/first';second=root/'run-state/discovery/second'
    original=adopt(root,str(folder.relative_to(root)),first)[2];failed_owner(root,first,original)
    retain=owner.retain
    def interrupted(path,value):
        if path==second/'preparation-owner.json':raise OSError('local publication interrupted')
        return retain(path,value)
    monkeypatch.setattr(owner,'retain',interrupted)
    with pytest.raises(OSError,match='interrupted'):adopt(root,str(folder.relative_to(root)),second)
    claim=next((root/'run-state/discovery-preparation-revisions').glob('runtime-*.json'))
    committed=claim.read_bytes()
    monkeypatch.setattr(owner,'retain',retain)
    assert adopt(root,str(folder.relative_to(root)),second)[2]==read(claim)['receipt']
    assert claim.read_bytes()==committed
    session=Session(root,str(second.relative_to(root)),dict(source=digest(source)),'prepare-research')
    with session.own():pass
    import zipfile
    with zipfile.ZipFile(second/'share/prepare-research.zip') as z:
        assert {'preparation-owner.json','preparation-owner-proof.json'}<=set(z.namelist())


def test_chained_cpu_failures_preserve_all_owners_and_never_revive_a_predecessor(donor):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    root,folder,source,_=donor
    receipts=[]
    for name in ('first','second','third'):
        target=root/'run-state/discovery'/name
        receipt=adopt(root,str(folder.relative_to(root)),target)[2];receipts.append(receipt)
        failed_owner(root,target,receipt)
    claim=next((root/'run-state/discovery-preparation-revisions').glob('runtime-*.json'))
    current=read(claim)
    assert current['receipt']==receipts[2]
    previous=read(root/current['previous']['path'])
    assert previous['receipt']==receipts[1]
    assert read(root/previous['previous']['path'])==receipts[0]
    for name in ('first','second'):
        with pytest.raises(ValueError,match='superseded'):
            adopt(root,str(folder.relative_to(root)),root/'run-state/discovery'/name)


@pytest.mark.parametrize('change',['source','run','construction','incomplete','tamper'])
def test_changed_or_scored_donors_cannot_be_reused(donor,change):
    root,folder,source,steps=donor
    if change=='source':source['src/oeis_learn/pipeline_validation/progressive_research_census.py']=digest('changed')
    if change=='run':steps['run-research/invocation-0000']=dict(status='COMPLETE',outputs=[])
    if change=='construction':(folder/'prepare-research/construction').mkdir()
    if change=='incomplete':steps['prepare-research/native-census']['status']='BLOCKED'
    if change=='tamper':write(folder/'prepare-research/materials/materials.json',seal(dict(status='changed')))
    write(folder/'journal.json',seal(dict(steps=steps)))
    with pytest.raises(ValueError):adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/new')
    assert not (root/'run-state/discovery-preparation-revisions').exists()


def test_reused_workstation_publishes_paths_consumed_by_real_subprocess_handoff(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_research_workstation as work
    from oeis_learn.pipeline_validation import progressive_preparation_reuse as reuse
    from oeis_learn.pipeline_validation import progressive_research_launch as launch
    from oeis_learn.pipeline_validation import progressive_research_bridge as bridge
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,read
    from oeis_learn.generalization.progressive.contracts import load
    session=Session(tmp_path,'run-state/discovery/new',dict(source=digest('current')),'prepare-research')
    base=session.folder/'prepare-research';donor=tmp_path/'run-state/discovery/old'
    write(donor/'session.json',seal(dict(binding=dict(registration_session='registration'))))
    data=seal(dict(version='progressive-registered-data/v1',source=digest('original'),design=seal(dict(seeds=[17,19]))))
    values={'registered-data':data,'native-census':seal(dict(deficits=[])),
        'materials':seal(dict(status='COMPLETE',deficits=[])), 'sensitivity':seal(dict(status='COMPLETE'))}
    paths={}
    for name,v in values.items():
        path=donor/(name+'.json');write(path,v);paths[name]=[path]
    monkeypatch.setattr(reuse,'adopt',lambda *a:(values,paths,seal(dict(original_source=digest('original')))))
    monkeypatch.setattr(bridge,'request',lambda data:seal(dict(seed=17,token_target=25000000,seconds=21600)))
    monkeypatch.setattr(bridge,'rehearsal',lambda *a:(dict(status='HISTORICAL_ONLY'),[]))
    visited=[]
    def execute_roster(argv,*a,**kw):
        assert '--roster-only' in argv
        for flag,expected in (('--materials',values['materials']),('--data',data)):
            assert load(tmp_path/argv[argv.index(flag)+1])==expected
        design=load(tmp_path/argv[argv.index('--design')+1]);assert design['version'].endswith('/v3')
        for rel in ('roster-summary.json','private/role-roster.json','research-reservation.json'):
            write(base/'roster'/rel,seal(dict(TEST_BOUNDARY=True,status='COMPLETE')))
        visited.append('roster')
    monkeypatch.setattr(supervision,'launch',execute_roster)
    def prepare(w,data,design,materials):
        assert load(base/'materials/materials.json')==values['materials']
        assert load(base/'sensitivity/sensitivity.json')==values['sensitivity']
        assert design['version'].endswith('/v3');visited.append('qualification-boundary')
    monkeypatch.setattr(launch,'prepare',prepare)
    monkeypatch.setattr(launch,'readiness',lambda *a:(dict(status='TEST_BOUNDARY_NOT_HIP'),[]))
    def images(*a,**kw):
        session.journal['steps']['prepare-research/image-qualification']=dict(status='COMPLETE',outputs=[])
        return dict(qualification=digest('cpu'))
    w=SimpleNamespace(root=tmp_path,base=base,s=session,images=images,
        args=SimpleNamespace(reuse_preparation_data='run-state/discovery/old',registration_session='registration',
            expanded_runtime_budgets=True,rehearsal_session='historical'),
        host=lambda:SimpleNamespace(engine=SimpleNamespace(facts=lambda:{}),policy={}))
    with session.own():work.prepare(w)
    assert visited==['roster','qualification-boundary']
    assert session.journal['steps']['prepare-research/materials']['status']=='COMPLETE'
    assert session.journal['steps']['prepare-research/sensitivity']['status']=='COMPLETE'
    assert session.journal['steps']['prepare-research/native-census']['outputs']==[reference(tmp_path,paths['native-census'][0])]


def failed_owner(root, folder, receipt):
    """Small original-style CPU failure; no synthetic HIP success."""
    write(folder/'session.json',seal(dict(binding=dict(source=receipt['current_source']))))
    (folder/'session.lock').touch()
    write(folder/'journal.json',seal(dict(steps={
        'prepare-research/base-qualification':dict(status='FAILED'),
        'prepare-research/qualification-profiles':dict(status='COMPLETE'),
        'prepare-research/matrix':dict(status='BLOCKED')},errors=[])))
    base=folder/'prepare-research'
    write(base/'matrix/profiles.json',dict(profiles='metadata only'))
    stages={}
    for name,state in {'cpu.semantic':'FAIL','hip.preflight':'BLOCKED',
                       'image.gpu':'PASS','image.qualification':'PASS'}.items():
        stage=base/'base-qualification/pipeline/stages'/name
        files={}
        if state!='BLOCKED':
            for path,value in [('process.json',dict(returncode=1 if state=='FAIL' else 0,cleanup_failure=None)),
                               ('cleanup.json',dict(acknowledged=True,within_bound=True,failure=None))]:
                write(stage/path,value);files[path]=reference(root,stage/path)['sha256']
        stages[name]=seal(dict(stage=name,state=state,files=files,binding=dict(contract=receipt['current_source'])))
        write(stage/'receipt.json',stages[name])
    write(base/'base-qualification/pipeline/summary.json',dict(stages=stages))


def test_failed_cpu_owner_transfers_once_with_history_and_old_client_fence(donor):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    root,folder,source,_=donor;first=root/'run-state/discovery/first';second=root/'run-state/discovery/second'
    original=adopt(root,str(folder.relative_to(root)),first)[2]
    failed_owner(root,first,original)
    originals={p:p.read_bytes() for p in first.rglob('*') if p.is_file()}
    source['tests/contract/test_progressive_bundle.py']=digest('fixed schema test')
    receipt=adopt(root,str(folder.relative_to(root)),second)[2]
    claim=next((root/'run-state/discovery-preparation-revisions').glob('runtime-*.json'))
    owner=read(claim)
    assert owner['receipt']==receipt
    assert read(root/owner['previous']['path'])==original
    proof=read(root/owner['proof']['path'])
    assert proof['predecessor']==original and proof['original_costs_retained']
    assert proof['scientific_allowance_transferred'] is False
    assert all(p.read_bytes()==value for p,value in originals.items())
    assert adopt(root,str(folder.relative_to(root)),second)[2]==receipt
    assert read(second/'preparation-owner-proof.json')==proof
    with pytest.raises(ValueError,match='superseded'):adopt(root,str(folder.relative_to(root)),first)
    # Even an old checkout's immutable publication cannot revive its old owner.
    with pytest.raises(ValueError,match='publication differs'):retain(claim,original)
    with pytest.raises(ValueError):adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/sibling')


@pytest.mark.parametrize('change',['running','history','training','qualification','artifact','cleanup','process','tamper','source','lock'])
def test_owner_transfer_rejects_started_work_unreclaimed_processes_and_changed_proof(donor,change):
    import fcntl
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    root,folder,source,_=donor;first=root/'run-state/discovery/first'
    original=adopt(root,str(folder.relative_to(root)),first)[2];failed_owner(root,first,original)
    journal=read(first/'journal.json')
    if change=='running':journal['steps']['prepare-research/environment']=dict(status='RUNNING')
    if change=='history':journal['steps']['prepare-research/matrix']['history']=[dict(status='FAILED')]
    if change=='training':journal['steps']['run-research/start']=dict(status='BLOCKED')
    if change=='qualification':journal['steps']['prepare-research/matrix']['status']='COMPLETE'
    write(first/'journal.json',seal(journal))
    base=first/'prepare-research';stage=base/'base-qualification/pipeline/stages/cpu.semantic'
    if change=='artifact':(base/'construction').mkdir()
    if change in {'cleanup','process','tamper'}:
        name='process.json' if change=='process' else 'cleanup.json'
        value=__import__('json').loads((stage/name).read_bytes())
        value.update(returncode=None) if change=='process' else value.update(acknowledged=False)
        write(stage/name,value)
        if change!='tamper':
            row=read(stage/'receipt.json');row['files'][name]=reference(root,stage/name)['sha256']
            row=seal(row);write(stage/'receipt.json',row)
            summary=__import__('json').loads((base/'base-qualification/pipeline/summary.json').read_bytes())
            summary['stages']['cpu.semantic']=row;write(base/'base-qualification/pipeline/summary.json',summary)
    if change=='source':write(first/'session.json',seal(dict(binding=dict(source=digest('different')))))
    claim=next((root/'run-state/discovery-preparation-revisions').glob('runtime-*.json'));before=claim.read_bytes()
    with (first/'session.lock').open('a') as lock:
        if change=='lock':fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises((ValueError,BlockingIOError)):
            adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/second')
    assert claim.read_bytes()==before


def feasibility_owner(root,monkeypatch):
    from tests.unit.test_progressive_service_calibration import donor_fixture
    from tests.unit.test_progressive_research_feasibility import producer
    from tests.unit.test_progressive_research_roster import fixture as roster_fixture
    from oeis_learn.generalization.progressive.research_roster import construct
    from oeis_learn.pipeline_validation import progressive_research_feasibility as feasibility
    from oeis_learn.generalization.progressive.contracts import load
    folder,support,source=donor_fixture(root,monkeypatch);base=folder/'prepare-research';timing=base/'evaluation-timing'
    req=load(timing/'request.json');roster=construct(*roster_fixture())
    write(base/'roster/private/role-roster.json',roster)
    materials=load(base/'materials/materials.json');write(base/'materials/materials.json',seal(dict(materials,assessment_exposed=False)))
    rows=[]
    for i,case in enumerate(req['cases']):
        ref=producer(timing,req,i);path=timing/ref['path'];p=load(path)
        write(path,seal(dict(p,scoring_us=1000000000,actual_elapsed_us=1000001000)))
        rows.append(dict(index=i,case=case['identity'],status='PASS',producer=reference(timing,path)))
    write(timing/'evaluation-cases.json',seal(dict(version='progressive-research-evaluation-run/v1',request=req['identity'],
        status='PASS',cases=rows,actual_elapsed_us=14000014000)))
    assert feasibility.estimates(timing,req,roster)['status']=='BLOCKED'
    journal=load(folder/'journal.json')
    for name,row in journal['steps'].items():
        if row.get('outputs'):row['outputs']=[reference(root,root/r['path']) for r in row['outputs']]
    journal['steps']['prepare-research/base-qualification']=dict(status='COMPLETE',outputs=[])
    journal['steps']['prepare-research/evaluation-feasibility']=dict(status='BLOCKED')
    journal['steps']['prepare-research/evaluation-timing']=dict(status='COMPLETE',outputs=[
        reference(root,timing/'evaluation-cases.json'),*[reference(root,timing/r['producer']['path']) for r in rows]])
    write(folder/'journal.json',seal(journal))
    write(timing/'process.json',dict(returncode=0,cleanup_failure=None))
    write(timing/'cleanup.json',dict(acknowledged=True,within_bound=True,failure=None))
    (folder/'session.lock').touch()
    receipt=seal(dict(version='progressive-preparation-data-reuse/v1',recipient=str(folder.relative_to(root)),current_source=req['source']))
    return folder,receipt


def test_scoring_deficit_replacement_preserves_diagnostics_and_requires_requalification(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof,claim_owner
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    folder,receipt=feasibility_owner(tmp_path,monkeypatch)
    original=(folder/'journal.json').read_bytes()
    proof=failure_proof(tmp_path,receipt)
    assert proof['historical_qualification_only'] and proof['current_source_requalification_required']
    assert proof['scientific_allowance_transferred'] is False
    claim=tmp_path/'run-state/discovery-preparation-revisions/test.json';write(claim,receipt)
    new=seal(dict(receipt,recipient='run-state/discovery/next',current_source=digest('corrected planning')))
    claim_owner(tmp_path,claim,new)
    assert read(claim)['receipt']==new and (folder/'journal.json').read_bytes()==original
    with pytest.raises(ValueError,match='superseded'):claim_owner(tmp_path,claim,receipt)


@pytest.mark.parametrize('change',['run','construction','attempt','cleanup','process','support','report','new-request'])
def test_scoring_deficit_replacement_cannot_reset_science_or_accept_incomplete_proof(tmp_path,monkeypatch,change):
    from oeis_learn.pipeline_validation.progressive_preparation_owner import failure_proof
    from oeis_learn.generalization.progressive.contracts import load
    folder,receipt=feasibility_owner(tmp_path,monkeypatch);base=folder/'prepare-research';timing=base/'evaluation-timing'
    journal=load(folder/'journal.json')
    if change=='run':journal['steps']['run-research/start']=dict(status='COMPLETE')
    if change=='attempt':journal['steps']['prepare-research/prepare']=dict(status='BLOCKED',history=[dict(status='FAILED')])
    if change=='new-request':
        path=timing/'request.json';req=load(path)
        write(path,seal(dict(req,version='progressive-research-evaluation-request/v2',scoring_services={'path':'boundary'})))
        journal['steps']['prepare-research/evaluation-plan']['outputs']=[reference(tmp_path,path)]
    write(folder/'journal.json',seal(journal))
    if change=='construction':(base/'construction').mkdir()
    if change=='cleanup':write(timing/'cleanup.json',dict(acknowledged=False,within_bound=True))
    if change=='process':write(timing/'process.json',dict(returncode=None))
    if change=='support':
        support=load(base/'materials/private/support.json');support['diagnostic']=True
        write(base/'materials/private/support.json',seal(support))
    if change=='report':
        report=load(timing/'evaluation-feasibility.json');write(timing/'evaluation-feasibility.json',seal(dict(report,deficits=[])))
    with pytest.raises((ValueError,OSError)):failure_proof(tmp_path,receipt)
