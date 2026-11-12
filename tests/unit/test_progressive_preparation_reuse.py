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
