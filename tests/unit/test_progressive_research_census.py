"""Independent census/reuse boundaries; reference doubles never qualify native/HIP."""
import json
from collections import Counter
from types import SimpleNamespace
import zipfile

import pytest

from oeis_learn.generalization.progressive.contracts import digest, reference, seal
from oeis_learn.pipeline_validation.progressive_workstation_state import write, read, Session
from oeis_learn.pipeline_validation import progressive_research_bridge as bridge
from oeis_learn.pipeline_validation import progressive_research_census as census
from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex


class ReferenceDouble:
    def execute(self, source, indices, *, deadline):
        from oeis_learn.synthesis.reference import evaluate
        r=evaluate(source,indices,deadline=deadline)
        return dict(outcome=r.outcome,values=list(r.values))
    def runtime_identity(self,*args):
        return digest('REFERENCE_DOUBLE_NOT_NATIVE_EVIDENCE')


def inputs(root):
    from oeis_learn.fresh.bigint_config import resolve
    rows=[dict(id='OEIS:A000001',indices=list(map(str,range(100))),
               values=[str(999999+i**3) for i in range(100)])]
    write(root/'candidates.json',rows)
    write(root/'recipe.json',resolve(dict(version='bigint-bootstrap/v1')))
    with ExposureIndex(root/'exposure.sqlite'):
        pass
    data=seal(dict(version='progressive-registered-data/v1',
        candidates=reference(root,'candidates.json'),recipe=reference(root,'recipe.json'),
        exposure=reference(root,'exposure.sqlite')))
    request=bridge.request(data)
    return seal(dict(request,diagnostic=True,training_draws=16,evaluation_draws=24,
                     seconds=60,shard_rows=4,required_per_stratum=2,minimum_groups=1))


def test_research_limits_cannot_be_silently_changed():
    req=bridge.request(seal(dict(version='progressive-registered-data/v1',candidates={},recipe={},exposure={})))
    census.validate_request(req)
    with pytest.raises(ValueError,match='limits'):
        census.validate_request(seal(dict(req,training_draws=8)))
    with pytest.raises(ValueError,match='purpose'):
        census.validate_request(seal(dict(req,diagnostic=1)))


def test_strata_depend_on_real_training_structures_and_subtrees():
    from oeis_learn.generalization.progressive.generator import structure_id
    source='(program (return (add (mul $index $index) (int 1))))'
    f=census.features(source)
    support=dict(structures={structure_id(source)},nodes=f['nodes'],operators=f['operators'])
    def classify(s):
        return census.classify(dict(source=s,structure=structure_id(s)),support)
    assert classify(source.replace('(int 1)','(int 9)'))=='parameter'
    assert classify('(program (return (add $index (int 7))))')=='composition'
    assert classify('(program (return (select (eq $index (int 0)) (int 1) $index)))')=='unseen_structure'


def test_shift_related_records_never_cross_reserved_panels():
    rows=[]
    for i in range(20):
        rows.append(dict(id=str(i),stratum='parameter',values=[str(i*100000+n*n) for n in range(100)]))
    rows.append(dict(rows[0],id='shift',values=[str(n*n) for n in range(1,101)]))
    panels,counts,deficits=census.reserve_panels(rows,seed=3,required=128,minimum_groups=16,
        panels=['development','confirmation-diversity','assessment'])
    owners={r['id']:p for p,rs in panels.items() for r in rs}
    assert owners['0']==owners['shift']
    assert deficits and all(c['task_deficit']>0 for c in counts.values())
    assert panels==census.reserve_panels(rows,seed=3,required=128,minimum_groups=16,
        panels=['development','confirmation-diversity','assessment'])[0]


def test_actual_census_retains_private_outputs_and_quantifies_supply(tmp_path):
    req=inputs(tmp_path)
    result=census.run(tmp_path,tmp_path/'out',req,client=ReferenceDouble())
    assert result['diagnostic'] and not result['research_ready']
    assert result['counts']['TRAINING']>0
    assert result['support']['required_shards']>result['support']['available_full_shards']
    assert 'refresh_cadence_requires_prospective_revision' in result['deficits']
    assert result['support']['proposed_refresh_updates']==8
    assert 'values' not in json.dumps(result) and '(program' not in json.dumps(result)
    for r in result['artifacts']:
        assert r['path'].startswith('private/')
    assert census.run(tmp_path,tmp_path/'out',req,client=ReferenceDouble())==result


def test_native_failure_continues_independent_cases_without_success(tmp_path):
    req=inputs(tmp_path)
    class Once(ReferenceDouble):
        calls=0
        def execute(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:
                raise RuntimeError('first case failed')
            return super().execute(*args,**kwargs)
    client=Once()
    result=census.run(tmp_path,tmp_path/'out',req,client=client)
    assert client.calls>1 and result['counts']['FAILED']==1
    assert result['status']=='INCOMPLETE' and not result['research_ready']
    assert result['failures'][0]['reason']=='RuntimeError: first case failed'
    assert result['counts']['oeis_available']==1


def test_scope_reconciliation_keeps_old_rehearsal_historical(tmp_path):
    folder=tmp_path/'run-state/old'
    write(folder/'result.json',{'status':'COMPLETE'})
    write(folder/'session.json',seal(dict(binding={'source':digest('old')})))
    write(folder/'journal.json',seal(dict(steps={
        'rehearse/'+k:dict(status='COMPLETE',outputs=[reference(tmp_path,folder/'result.json')])
        for k in ('qualify','acceptance')})))
    result,_=bridge.rehearsal(tmp_path,'run-state/old',tmp_path/'out',digest('new'))
    assert result['status']=='HISTORICAL_SOURCE_DIFFERS'
    assert not result['hardware_authority_transferred'] and not result['rerun_requested']
    (folder/'result.json').write_text('changed')
    with pytest.raises(ValueError,match='identity'):
        bridge.rehearsal(tmp_path,'run-state/old',tmp_path/'second',digest('new'))


def test_census_packaging_never_exports_private_reservoirs(tmp_path):
    s=Session(tmp_path,'run-state/new',{},'prepare-research')
    with s.own():
        write(s.folder/'prepare-research/census/census.json',{'status':'INCOMPLETE'})
        write(s.folder/'prepare-research/census/private/panel-reservations.json',{'secret':'PRIVATE'})
    with zipfile.ZipFile(s.folder/'share/prepare-research.zip') as z:
        assert 'prepare-research/census/census.json' in z.namelist()
        assert not any('PRIVATE' in z.read(n).decode() for n in z.namelist())


def test_selected_cpu_build_does_not_request_gpu_and_uses_current_phase(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_workstation as w, supervision
    s=Session(tmp_path,'run-state/new',{},'prepare-research')
    calls=[]
    def launch(argv,*a,**kw):
        calls.append(argv[-2]);write(argv[-3],{'image':'cpu'})
    monkeypatch.setattr(supervision,'launch',launch)
    with s.own():
        s.step('environment',lambda:({},[]))
        assert w.Workflow(s,SimpleNamespace(phase='prepare-research',engine='podman')).images(targets=('qualification',))=={'qualification':'cpu'}
    assert calls==['qualification']


def donor(root,monkeypatch):
    from tests.unit.test_progressive_research_inputs import originals, source_snapshot
    from oeis_learn.pipeline_validation import progressive_research_inputs as inputs_module
    from oeis_learn.pipeline_validation.progressive_research_source import project
    from oeis_learn.pipeline_validation.progressive_research_design import construct
    manifest=originals(root);write(root/'manifest.json',manifest)
    source=source_snapshot(root)
    old_source=digest('reviewed-fixture-data-producer')
    monkeypatch.setattr(inputs_module,'EXPOSURE_COMPATIBLE_SOURCES',{old_source})
    folder=root/'run-state/registration'
    s=Session(root,'run-state/registration',dict(source=old_source,reference_campaign='prior',
        input_manifest={'path':'manifest.json','sha256':reference(root,'manifest.json')['sha256']}),'register-inputs')
    base=folder/'register-inputs'
    with s.own():
        write(base/'source-candidates.json',{'status':'inventory'})
        s.step('source-inventory',lambda:({},[base/'source-candidates.json']))
        projection=project(root,source,base/'source')
        s.step('project-source',lambda:(projection,[base/'source/source-projection.json',
            base/'source/private/oeis-candidates.json',*[root/r['path'] for r in [projection['snapshot'],*projection['originals']]]]))
        registration=inputs_module.audit(root,manifest,base/'registration')
        s.step('register-originals',lambda:(registration,[base/'registration/registration.json',
            *[root/r['workspace_artifact']['path'] for r in registration['checks'] if r.get('status')=='PASS']]))
        construct(root,base/'design',manifest,registration)
        s.step('prospective-design',lambda:('proposal',list((base/'design').glob('*.json'))))
        inputs_module.import_exposure(root,manifest,registration,base/'exposure')
        s.step('import-exposure',lambda:(str((base/'exposure').relative_to(root)),
            [base/'exposure/exposure-import.json',base/'exposure/private/exposure.sqlite']))
        status=seal(dict(status='INPUTS_REGISTERED',research_ready=False))
        write(base/'registration-status.json',status)
        s.step('registration-status',lambda:(status,[base/'registration-status.json']))
    return s


def test_adopts_real_registered_data_without_import_and_rejects_changed_original(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_research_inputs as inputs_module
    old=donor(tmp_path,monkeypatch)
    before=(old.folder/'register-inputs/exposure/private/exposure.sqlite').read_bytes()
    monkeypatch.setattr(inputs_module,'import_exposure',lambda *a,**k:pytest.fail('re-import'))
    data,files=bridge.adopt(tmp_path,'run-state/registration',tmp_path/'new')
    assert data['status']=='COMPLETE' and not data['hardware_authority_transferred']
    assert (tmp_path/data['exposure']['path']).read_bytes()==before
    assert data['candidates']['path'].endswith('/source/private/oeis-candidates.json')
    (tmp_path/'B/tensor.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):
        bridge.adopt(tmp_path,'run-state/registration',tmp_path/'bad')


def test_preparation_has_no_rehearsal_dependency_and_continues_sensitivity(tmp_path,monkeypatch):
    import runpy
    from oeis_learn.pipeline_validation import progressive_workstation as w, supervision
    donor(tmp_path,monkeypatch)
    events=[]
    monkeypatch.setattr(w.Workflow,'host',lambda self:SimpleNamespace(
        engine=SimpleNamespace(facts=lambda:{'fixture':True}),policy={'memory_bytes':64 << 30}))
    def launch(argv,*args,**kw):
        events.append('CPU-image');assert argv[-2]=='qualification';write(argv[-3],{'image':'cpu'})
    monkeypatch.setattr(supervision,'launch',launch)
    def stage(host,image,output,argv,**kw):
        assert not kw['gpu']
        if output.name=='census':
            events.append('census')
            assert len(kw['readonly_inputs'])==3
            raise RuntimeError('original native census failure')
        assert output.name=='sensitivity'
        events.append('sensitivity');assert kw['readonly_inputs']==[]
        write(output/'sensitivity.json',{'kind':'DIAGNOSTIC_ONLY'})
    monkeypatch.setattr(runpy,'run_path',lambda path:{'container_stage':stage})
    s=Session(tmp_path,'run-state/prep',{'source':digest('new')},'prepare-research')
    args=SimpleNamespace(phase='prepare-research',engine='podman',registration_session='run-state/registration',
                         rehearsal_session=None)
    with s.own():
        w.Workflow(s,args).prepare_research()
    assert events==['CPU-image','census','sensitivity']
    assert s.code()==1
    ready=read(s.folder/'prepare-research/readiness.json')
    assert 'native_census_not_completed' in ready['deficits'] and not ready['research_ready']
    with zipfile.ZipFile(s.folder/'share/prepare-research.zip') as z:
        assert 'prepare-research/readiness.json' in z.namelist()
        assert 'original native census failure' in z.read('journal.json').decode()
        assert not any('/private/' in n for n in z.namelist())
    # Reinvocation preserves charged failed stages and reuses completed data.
    events.clear()
    with s.own():
        w.Workflow(s,args).prepare_research()
    assert events==['census']  # only the failed owned phase retries within its original allowance


def test_actual_native_census(fitting_client,tmp_path):
    """Mandatory in the CPU qualification image; never a reference fallback."""
    request=inputs(tmp_path)
    write(tmp_path/'request.json',request)
    result=census.run(tmp_path,tmp_path/'out',request,client=fitting_client)
    assert result['counts']['TRAINING']>0
    assert result['native']==fitting_client.runtime_identity('bigint-body/v1')
    assert not result['counts'].get('FAILED')
    assert not result['qualification'] and result['diagnostic']
    assert result['counts']['TRAINING']+result['counts'].get('REJECTED',0)+result['counts'].get('EVALUATION',0)==40
    from oeis_learn.pipeline_validation.progressive_research_materials import build
    from oeis_learn.generalization.progressive.prepared_support import policy
    materials=build(tmp_path,'out/census.json','request.json','nested/materials',
        policy(seed=request['seed'],shard_rows=request['shard_rows']))
    assert materials['diagnostic'] and materials['status']=='INCOMPLETE'
    assert materials['native']==result['native'] and not materials['original_generation_repeated']



def test_sensitivity_isolates_failures_preserves_original_seeds_and_denominators(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_research_sensitivity as collected
    from tests.unit.test_progressive_sensitivity import scenario
    design={'seed':13,'repeats':2,'scenarios':[scenario(),scenario(effect_ppm=0)]}
    visited=[]
    def verdict(d,ordinal,repeat):
        visited.append((ordinal,repeat));assert d==design
        if (ordinal,repeat)==(0,0): raise RuntimeError('first bootstrap failed')
        return 'INCONCLUSIVE'
    monkeypatch.setattr(collected,'repeat_verdict',verdict)
    r=collected.run(design,tmp_path)
    assert visited==[(0,0),(0,1),(1,0),(1,1)]
    assert r['status']=='INCOMPLETE' and r['scenarios'][0]['missed_gain_rate'] is None
    assert r['scenarios'][1]['inconclusive_rate']==1
    cases=read(tmp_path/'case-outcomes.json')
    assert cases['cases'][0]['reason']=='RuntimeError: first bootstrap failed'
    assert cases['cases'][-1]['status']=='COMPLETE'
    with pytest.raises(ValueError,match='budget reset'):
        collected.run(design,tmp_path)


def test_collected_sensitivity_matches_original_statistics(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_sensitivity import run
    from oeis_learn.generalization.progressive.sensitivity import run as original
    from tests.unit.test_progressive_sensitivity import scenario
    design={'seed':13,'repeats':1,'scenarios':[scenario(effect_ppm=0),scenario(claim='efficiency')]}
    actual=run(design,tmp_path)
    assert actual['status']=='COMPLETE'
    for a,b in zip(actual['scenarios'],original(design)['scenarios']):
        assert {k:v for k,v in a.items() if k!='complete'}==b


def test_expired_sensitivity_never_loses_requested_cases(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_research_sensitivity as c
    from tests.unit.test_progressive_sensitivity import scenario
    monkeypatch.setattr(c,'repeat_verdict',lambda *a:pytest.fail('ran past deadline'))
    r=c.run({'seed':13,'repeats':2,'scenarios':[scenario()]},tmp_path,seconds=0)
    assert r['case_statuses']=={'UNATTEMPTED':2}
    assert r['status']=='INCOMPLETE' and r['scenarios'][0]['missed_gain_rate'] is None


@pytest.mark.parametrize('accepted',[False,True])
def test_automatic_training_needs_completed_preparation_and_acceptance(tmp_path,monkeypatch,accepted):
    from argparse import Namespace
    from oeis_learn.pipeline_validation import progressive_workstation as w
    s=Session(tmp_path,'run-state/prep',{'source':digest('source')},'prepare-research')
    ran=[]
    def research(workflow):
        ran.append((workflow.args.phase,workflow.s.phase))
        workflow.s.step('invocation-0000',lambda:('canonical-run',[]),dependencies=('prepare-research/acceptance',))
    monkeypatch.setattr(w.Workflow,'research',research)
    with s.own():
        if accepted:
            path=s.folder/'prepare-research/acceptance/result.json';write(path,{'complete':True})
            s.step('acceptance',lambda:('original acceptance',[path]))
        else:
            s.step('acceptance',lambda:None,dependencies=('prepare-research/freeze',))
        w.Workflow(s,Namespace(phase='prepare-research')).run_after_prepare()
    assert ran==([('run-research','run-research')] if accepted else [])
    assert (s.folder/'share/prepare-research.zip').is_file()
    assert (s.folder/'share/run-research.zip').is_file()==accepted


def test_automatic_training_rejects_changed_acceptance_bytes(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import progressive_workstation as w
    s=Session(tmp_path,'run-state/prep',{},'prepare-research')
    monkeypatch.setattr(w.Workflow,'research',lambda *a:pytest.fail('training launched'))
    with s.own():
        path=s.folder/'receipt.json';write(path,{'original':1})
        s.step('acceptance',lambda:('original',[path]))
        write(path,{'original':2})
        with pytest.raises(ValueError,match='changed'):
            w.Workflow(s,SimpleNamespace(phase='prepare-research')).run_after_prepare()


def test_census_recovers_publication_without_repeating_completed_native_draws(tmp_path,monkeypatch):
    req=inputs(tmp_path)
    class Counting(ReferenceDouble):
        calls=0
        def execute(self,*a,**kw):self.calls+=1;return super().execute(*a,**kw)
    client=Counting();original=census.write;fail=[True]
    def publish(path,value):
        if path.name=='census.json' and fail:
            fail.pop();raise OSError('publication boundary interrupted')
        return original(path,value)
    monkeypatch.setattr(census,'write',publish)
    with pytest.raises(OSError,match='publication'):census.run(tmp_path,tmp_path/'deep/new/census',req,client=client)
    calls=client.calls
    result=census.run(tmp_path,tmp_path/'deep/new/census',req,client=client)
    assert client.calls==calls and result['counts']['TRAINING']>0
    assert census.run(tmp_path,tmp_path/'deep/new/census',req,client=client)==result
    assert client.calls==calls
    with pytest.raises(ValueError):census.run(tmp_path,tmp_path/'deep/new/census',seal(dict(req,seed=req['seed']+1)),client=client)


def test_unknown_census_interval_is_charged_without_more_native_work(tmp_path):
    from oeis_learn.pipeline_validation.preparation_recovery import Allowance
    req=inputs(tmp_path);output=tmp_path/'out';output.mkdir()
    binding=dict(request=req['identity'],native=ReferenceDouble().runtime_identity())
    state=dict(version='progressive-operational-allowance/v1',binding=binding,seconds=req['seconds'],
        spent_seconds=0,active=dict(ordinal=0,reserved_seconds=req['seconds']),attempts=[])
    write(output/'census-allowance.json',seal(state))
    with pytest.raises(ValueError,match='exhausted'):
        census.run(tmp_path,output,req,client=ReferenceDouble())
    assert not (output/'private/census.sqlite').exists()


def test_round028_family_supply_closes_transitive_history_and_all_shifted_windows():
    from oeis_learn.generalization.progressive.program_search_panels import fresh_groups

    def row(name, start, **attributes):
        return dict(id=name, indices=[str(i) for i in range(100)],
                    values=[str(start+i*i) for i in range(100)], stratum="oeis", **attributes)
    original = row("old", 10000, family="historical-family")
    bridge = row("bridge", 20000, group="historical-family")
    shifted = row("shifted", 30000)
    shifted["values"][79:99] = bridge["values"][54:74]
    free = row("free", 40000)
    groups, counts = fresh_groups([shifted, free, bridge], [original], indexes=())
    assert [[r["id"] for r in rows] for rows in groups.values()] == [["free"]]
    assert counts["excluded"] == 2 and counts["eligible"] == 1
    assert fresh_groups([bridge, shifted, free], [original], indexes=())[0] == groups


def test_round028_calibration_is_disjoint_and_missing_descriptors_stay_unknown():
    from oeis_learn.generalization.progressive.program_search_panels import calibrate_bands, difficulty_band
    from oeis_learn.generalization.progressive.complexity import measure
    from oeis_learn.generalization.progressive.contracts import digest
    records = []
    for i in range(6):
        descriptor = dict(language="bigint-body/v1", operations=["add"], nodes=2**(i+1), depth=2, work=100+i)
        records.append(dict(family=digest(i), descriptor=descriptor, measured=3, exact=3-i//2,
                            denominator=3))
    bands = calibrate_bands(records)
    assert [difficulty_band(r["descriptor"], bands) for r in records] == ["easy","easy","middle","middle","hard","hard"]
    assert difficulty_band(None, bands) == "unknown"
    assert difficulty_band(dict(records[0]["descriptor"], work=None), bands) == "unknown"
    with pytest.raises(ValueError):
        calibrate_bands([records[0], records[0]])
    assert measure(None)["work"] is None


def test_round028_allocates_balanced_families_once_and_reports_shortages():
    from oeis_learn.generalization.progressive.program_search_panels import allocate_roles, RESERVATIONS
    from oeis_learn.generalization.progressive.contracts import digest
    groups = {}
    # One family per band per role and stratum at the small fixture scale.
    for stratum in ("parameter", "composition", "unseen_structure", "oeis"):
        for i in range(27):
            f = digest([stratum, i])
            groups[f] = [dict(id=f"{stratum}-{i}-{j}", stratum=stratum, band=("easy","middle","hard")[i%3])
                         for j in range(2)]
    selection = allocate_roles(groups, seed=17, per_stratum=6)
    assert not selection["deficits"] and set(selection["panels"]) == set(RESERVATIONS)
    ownership = {}
    for panel, rows in selection["panels"].items():
        for member in rows:
            assert ownership.setdefault(member["group"], panel) == panel
        primary = [r for r in rows if r["stratum"] == "composition"]
        if primary:
            assert Counter(r["band"] for r in primary) == {"easy":2, "middle":2, "hard":2}
    assert allocation_ids(selection) == allocation_ids(allocate_roles(dict(reversed(list(groups.items()))),
                                                                       seed=17, per_stratum=6))
    thin = allocate_roles({k:v for k,v in groups.items() if v[0]["stratum"] != "oeis"}, seed=17, per_stratum=6)
    assert thin["deficits"]["development/oeis"]["missing_cases"] == 6
    assert thin["deficits"]["replication/oeis"]["missing_cases"] == 6


def allocation_ids(selection):
    return {p:[r["id"] for r in rows] for p,rows in selection["panels"].items()}


def test_round028_preserves_only_declared_parameter_transfer_across_all_old_roles(tmp_path):
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.generalization.progressive.program_search_panels import fresh_groups
    def variant(k,stratum,flipped=False):
        source = f"(program (return (sub {'(int '+str(k)+') $index' if flipped else '$index (int '+str(k)+')'})))"
        return dict(id=str(k), source=source, structure=structure_id(source), stratum=stratum,
                    indices=list(map(str,range(100))), values=[str(k+i*i) for i in range(100)])
    old = variant(1,"parameter")
    permitted = variant(1000,"parameter")
    exact = dict(old,id="same-old-source",stratum="parameter")
    unrelated = dict(variant(9000,"parameter"),family="held-family")
    with ExposureIndex(tmp_path/"history.sqlite") as index:
        index.add(old,origin="old training",role="training")
    with ExposureIndex(tmp_path/"history.sqlite",readonly=True) as index:
        groups,counts = fresh_groups([permitted,exact,unrelated],[dict(old,declared_groups=["held-family"])],indexes=[index])
        assert [r["id"] for rs in groups.values() for r in rs] == [permitted["id"]]
        assert counts["excluded"] == 2
    # Different AST structures share the normalized operator/proper-subtree inventory.
    old_protected = variant(40000,"unseen_structure",flipped=True)
    assert old_protected["structure"] != permitted["structure"]
    for role in ("development","replication","confirmation","assessment","retention","calibration"):
        assert not fresh_groups([permitted],[dict(old_protected,role=role)],indexes=[])[0]


def test_round028_calibration_is_held_from_each_role_and_replication_is_common_to_both_seeds():
    from oeis_learn.generalization.progressive.graph import generator_look
    from oeis_learn.generalization.progressive.program_search_panels import calibration_families, descriptor, freeze_panels
    groups = {}
    for stratum in ("parameter","composition","unseen_structure","oeis"):
        for i in range(104):
            f = digest([stratum,i])
            groups[f] = [dict(id=f"{stratum}-{i}-{n}",stratum=stratum,operations=["add"],native_verified=True,
                difficulty=dict(depth=2,nodes=4,work=100*(1+i%4))) for n in range(2)]
    samples = calibration_families(groups,17,count=16)
    assert len(samples) == 16
    observed = [dict(family=f,descriptor=descriptor(r),exact=0,measured=2,denominator=3) for f,r in samples.items()]
    reserved = freeze_panels(groups,samples,observed,seed=17)
    assert not reserved["deficits"]
    assert all(r["component"] not in samples for rs in reserved["panels"].values() for r in rs)
    assert {r["band"] for rs in reserved["panels"].values() for r in rs if r["stratum"] == "oeis"} == {"unknown"}
    panels = {p:[dict(task=r["id"],stratum=r["stratum"],group=r["group"]) for r in rs]
              for p,rs in reserved["panels"].items() if p not in {"confirmation","assessment"}}
    replay = [generator_look(panels,registration=digest("prospective"),pair=i,arm="ordinary",target=4_000_000,
                             split="replication") for i in range(2)]
    assert replay[0]["members"] == replay[1]["members"] and replay[0]["seed"] != replay[1]["seed"]
    assert len(replay[0]["members"]) == 160
    with pytest.raises(ValueError,match="calibration"):
        freeze_panels(groups,samples,observed[:-1],seed=17)
