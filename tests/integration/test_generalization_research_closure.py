"""Producer-shaped campaign evidence; no synthetic hardware qualification."""
import importlib.util
import json
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import seal,digest,write
from oeis_learn.pipeline_validation.generalization_preparation import reference

CONTROLS=('raw','rescue','equal_budget','shuffled','absent','template','bank')


def adapter():
    path=Path('specs/012-parameterized-adaptive-curriculum/validation/analysis.py')
    spec=importlib.util.spec_from_file_location('research_closure_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def campaign(tmp_path,complete=68):
    from tests.unit.test_generalization_execution import pac_setup
    from tests.unit.test_generalization_metrics import pac_roster
    from tests.integration.test_generalization_recipes import pac_envelope
    from oeis_learn.pipeline_validation.generalization_execution import bind_pac,pac_budget_projection,POLICY_ID
    from oeis_learn.generalization.metrics import pac_structural_metrics
    from oeis_learn.generalization.retention import empty_history,append_panel
    exp,control,profiles,_=pac_setup();profiles={a:dict(p,backend=POLICY_ID) for a,p in profiles.items()}
    exp=seal(dict(exp,backend=POLICY_ID,budget_graph=pac_budget_projection(pac_envelope())))
    workspace=tmp_path/'workspace';root=workspace/'campaign';authority=workspace/'authority'
    images=[digest('cpu'),digest('gpu')];roster=pac_roster()
    write(authority/'admission.json',seal(dict(status='PASS',roster=roster)))
    ref=dict(path='fixture',sha256=digest('fixture'),bytes=0)
    runtime=dict(version='pac-runtime/v1',study='',purpose='research',recipe=ref,source=ref,images=images,control=control,profiles=profiles,
        preparation=None,prepared=dict(training=ref,admission=reference(authority/'admission.json',authority)),envelope=pac_envelope(),feature=None,engine='podman')
    exp,runtime=bind_pac(exp,runtime);write(authority/'study.json',exp);write(authority/'runtime.json',runtime)
    from oeis_learn.generalization.structural import original_cells
    write(root/'admission/execution/execution/role/private/assembly/resolved/admission.json',seal(dict(status='PASS',source=exp['source'],images=images,roster=roster,originals=original_cells())))
    names=['admission','qualification','calibration','acceptance','baseline','train-B','evaluate-B','train-C','evaluate-C','analysis']
    stages=[];measurements={}
    for n in names:
        folder=root/n;folder.mkdir(parents=True,exist_ok=True)
        row=dict(stage=n,status='PASS',producer=digest(n),binding=digest(['binding',n]),process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),
            elapsed_seconds=17 if n=='train-C' else 11,reused=False,directory=n,artifacts=[])
        if n=='acceptance':row.update(resolved_manifest='authority/study.json',resolved_runtime='authority/runtime.json',resolved_refs=[reference(authority/k,workspace) for k in ('study.json','runtime.json')])
        if n=='baseline' or n.startswith('evaluate-'):
            arm='A' if n=='baseline' else n[-1];view='A-reference' if n=='baseline' and exp['control_mode']=='fresh-control' else arm
            certified=[k for k,v in roster.items() if v['category']=='certified']
            outcomes={c:{k:(None if k in certified[(complete.get(arm,68) if isinstance(complete,dict) else complete):] else (i%3==0 if c=='raw' else i%2==0)) for i,k in enumerate(roster)} for c in CONTROLS}
            measurements[arm]=outcomes
            scoring=seal(dict(version='pac-structural-scoring/v1',status='PASS',study=exp['identity'],roster=roster,outcomes=outcomes,
                controls={c:pac_structural_metrics(roster,v) for c,v in outcomes.items()},scored_cases={},parameter='DEFERRED',assessment='NOT_EXECUTED'))
            base=folder/'execution/evaluation'/view
            write(base/'scoring/role/scoring.json',scoring);write(base/'generation/role/generation/selections.json',{})
        elif n.startswith('train-'):
            arm=n[-1];hist=empty_history();family=digest('training-family')
            for u in (0,10):append_panel(hist,dict(update=u,tokens=u*10,examples=u*2,seconds=u*.5),{family:dict(loss=dict(loss_sum=8.-u/2,tokens=8),semantic=dict(semantic_attempts=8,semantic_completed=8,semantic_success=u//2))})
            allocation=dict(status='RECORDED',tokens=dict(allowed=25000000,useful=100,reserved=0,discarded=0,unspent=24999900),updates=10,
                allocation={family:{c:dict(updates='10' if c=='bandit' else '0',tokens=100 if c=='bandit' else 0,examples=20 if c=='bandit' else 0) for c in ('fixed','bandit','review','fallback')}},
                channels=dict(bandit=10,review=0,fallback=0),review=dict(acquisitions=2,grades=3,successes=1,censored=0,incidental=0),rewards=dict(valid=1,censored=0),observations=4,
                observation_seconds=9 if arm=='C' else 3,costs=dict(preparation_seconds=1,checkpoint_seconds=2,update_seconds=5,publication_seconds=None,invocation_seconds=20),
                monitoring=dict(history=hist,loss_panels=2))
            write(folder/'execution/result.json',seal(dict(status='PASS',result=dict(runs=[dict(arm=arm,result=dict(allocation=allocation,updates=10,learner_seconds=20))]))))
        row['artifacts']=[reference(p,folder) for p in sorted(folder.rglob('*')) if p.is_file()];stages.append(row)
    plan=seal(dict(source=exp['source'],images=images,helper=digest('helper'),research=True,arms=exp['arms'],stages=[dict(name=n,extra=['context',str(exp['seed'])] if n=='admission' else []) for n in names]))
    write(root/'plan.json',plan);write(root/'workflow.json',seal(dict(plan=plan['identity'],status='PARTIAL',errors=[],receipts=stages)))
    return workspace,root,exp,runtime,measurements


def test_campaign_analysis_exports_every_control_and_public_diagnostic(tmp_path):
    from oeis_learn.generalization.pac_analysis import recompute
    from oeis_learn.pipeline_validation.generalization_pac_export import collect
    workspace,root,exp,runtime,measurements=campaign(tmp_path)
    a=adapter();report=a.analyze_campaign(root,output=root/'analysis',workspace=workspace)
    assert report['controls']['B']['raw']!=report['controls']['B']['equal_budget']
    assert set(report['controls']['C'])==set(CONTROLS)
    inputs=json.loads((root/'analysis/research-inputs.json').read_text())
    assert inputs['outcomes']==measurements
    assert report['details']['C']['observation_seconds']==9 and report['details']['B']['observation_seconds']==3
    assert report['details']['C']['costs']['publication_seconds'] is None
    assert report['details']['C']['history']['bins'][0]['first']['coordinates']['update']==0
    assert report['details']['C']['allocation']
    assert report['stage_costs']['train-C']['elapsed_seconds']==17
    assert report['costs']['arms']['B']['examples']==20
    derived=recompute(inputs)
    assert all(report[k]==v for k,v in derived.items() if k!='version')
    workflow=json.loads((root/'workflow.json').read_text());workflow['receipts'][-1]['artifacts']=[reference(p,root/'analysis') for p in (root/'analysis').iterdir()]
    write(root/'workflow.json',seal(workflow),exclusive=False)
    record=collect(root,tmp_path/'public',workspace=workspace)
    names={r['path'] for r in record['inventory']}
    assert {'analysis/analysis.json','analysis/research-inputs.json'}<=names
    assert 'values' not in json.dumps(inputs) and 'original_source' not in json.dumps(inputs)


def test_unavailable_training_measurements_remain_explicit(tmp_path):
    from oeis_learn.generalization.pac_analysis import project_training,recompute
    workspace,root,_,_,_=campaign(tmp_path)
    adapter().analyze_campaign(root,output=root/'analysis',workspace=workspace)
    inputs=json.loads((root/'analysis/research-inputs.json').read_text())
    assert project_training(dict(updates=10)) is None
    inputs['diagnostics']['C']=None
    inputs['stages']['train-C']['elapsed_seconds']=None
    report=recompute(seal(inputs))
    assert report['details']['C'] is None and report['costs']['arms']['C'] is None
    assert report['stage_cost_totals']['unavailable']==1


@pytest.mark.parametrize('claim',['INCOMPLETE','EXPLORATORY_SINGLE_SEED'])
def test_acceptance_scope_cannot_claim_scientific_interpretation(tmp_path,claim):
    from tests.integration.test_generalization_archive import pac_record
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    record=seal(dict(pac_record(),scientific_status=claim))
    with pytest.raises(ValueError,match='scientific|acceptance'):verify_pac_evidence(record,root=tmp_path,finalized=False)


def test_missing_research_payload_cannot_claim_exploratory(tmp_path):
    from tests.integration.test_generalization_archive import pac_record
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    value=pac_record();value['scope']['research']=True
    value.update(required=['admission','qualification','calibration','acceptance','baseline','train-B','evaluate-B','train-C','evaluate-C','analysis'],scientific_status='INCOMPLETE')
    verify_pac_evidence(seal(value),root=tmp_path,finalized=False)
    with pytest.raises(ValueError,match='scientific|research'):verify_pac_evidence(seal(dict(value,scientific_status='EXPLORATORY_SINGLE_SEED')),root=tmp_path,finalized=False)


def analyzed_public(tmp_path,complete=68):
    from oeis_learn.pipeline_validation.generalization_pac_export import collect
    workspace,root,_,_,_=campaign(tmp_path,complete)
    report=adapter().analyze_campaign(root,output=root/'analysis',workspace=workspace)
    workflow=json.loads((root/'workflow.json').read_text())
    workflow['receipts'][-1].update(producer=digest(report),artifacts=[reference(p,root/'analysis') for p in (root/'analysis').iterdir()])
    write(root/'workflow.json',seal(workflow),exclusive=False)
    public=tmp_path/'public';record=collect(root,public,workspace=workspace)
    return public,record


@pytest.mark.parametrize('complete',[64,65])
def test_scientific_gate_is_recomputed_from_bound_68_task_denominator(tmp_path,complete):
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    from oeis_learn.pipeline_validation.generalization_finalization import publish_pac
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    public,record=analyzed_public(tmp_path,complete)
    expected='EXPLORATORY_SINGLE_SEED' if complete>=65 else 'INCOMPLETE'
    assert verify_pac_evidence(record,root=public,finalized=False)['scientific_status']==expected
    archive=publish_pac(record,tmp_path/'export',payload_root=public)
    assert verify_archive(archive)['scientific_status']==expected
    import subprocess,sys,os
    process=subprocess.run([sys.executable,'-S','-c',
        'import sys;from oeis_learn.pipeline_validation.generalization_archive import verify_archive;assert verify_archive(sys.argv[1])["scientific_status"]==sys.argv[2]',str(archive),expected],
        env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())),capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr
    wrong='INCOMPLETE' if complete>=65 else 'EXPLORATORY_SINGLE_SEED'
    with pytest.raises(ValueError,match='scientific'):verify_pac_evidence(seal(dict(record,scientific_status=wrong)),root=public,finalized=False)


def rebind_public(public,record,inputs,report):
    write(public/'analysis/research-inputs.json',seal(inputs),exclusive=False)
    report['inputs']=reference(public/'analysis/research-inputs.json',public/'analysis')
    report=seal(report);write(public/'analysis/analysis.json',report,exclusive=False)
    for s in record['stages']:
        if s['stage']=='analysis':s['producer']=report['identity']
    record['inventory']=[reference(public/r['path'],public) for r in record['inventory']]
    return seal(record)


@pytest.mark.parametrize('change',['roster','control','analysis','control_metric','cost','exposure','history','missing_payload','source','seed','arms','scope','stage_binding','typed_metric'])
def test_freshly_resealed_scientific_counterexamples_are_rejected(tmp_path,change):
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    from oeis_learn.generalization.pac_analysis import recompute
    public,record=analyzed_public(tmp_path)
    inputs=json.loads((public/'analysis/research-inputs.json').read_text());report=json.loads((public/'analysis/analysis.json').read_text())
    if change=='roster':
        key=next(iter(inputs['roster']));other=next(k for k,v in inputs['roster'].items() if v['family']!=inputs['roster'][key]['family'] and v['category']==inputs['roster'][key]['category'])
        inputs['roster'][key],inputs['roster'][other]=inputs['roster'][other],inputs['roster'][key]
    elif change=='control':
        inputs['control']=seal(dict(inputs['control'],checkpoint=digest('another control')))
        inputs['experiment']=seal(dict(inputs['experiment'],control=inputs['control']['identity']))
    elif change=='analysis':report['comparisons']['C-B']['interpretable']=False
    elif change=='control_metric':report['controls']['B']['raw']['certified']['successes']+=1
    elif change=='typed_metric':report['comparisons']['C-B']['interpretable']=1
    elif change=='cost':report['stage_cost_totals']['measured_seconds']+=1
    elif change=='exposure':report['details']['C']['tokens']['useful']+=1
    elif change=='history':report['details']['C']['retention']['panels']+=1
    elif change=='missing_payload':
        record['inventory']=[r for r in record['inventory'] if not r['path'].startswith('analysis/')]
        with pytest.raises(ValueError):verify_pac_evidence(seal(record),root=public,finalized=False)
        return
    elif change=='source':record['source']=digest('another source')
    elif change=='seed':record['scope']['seed']+=1
    elif change=='arms':
        record['scope']['arms'].append('D');record['required'][-1:-1]=['train-D','evaluate-D']
    elif change=='scope':
        record['scope']['research']=False;record['required']=record['required'][:4];record['stages']=record['stages'][:4]
    elif change=='stage_binding':record['stages'][3]['binding']=digest('another acceptance')
    if change in {'roster','control'}:
        report.update(recompute(seal(inputs)));report['version']='pac-analysis/v2'
    record=rebind_public(public,record,inputs,report)
    with pytest.raises(ValueError):verify_pac_evidence(record,root=public,finalized=False)


@pytest.mark.parametrize('location',['control','profile','history'])
def test_public_research_inputs_reject_nested_private_fields(tmp_path,location):
    from oeis_learn.generalization.pac_analysis import recompute
    workspace,root,_,_,_=campaign(tmp_path)
    adapter().analyze_campaign(root,output=root/'analysis',workspace=workspace)
    inputs=json.loads((root/'analysis/research-inputs.json').read_text())
    if location=='control':
        inputs['control']['expected']['hidden_terms']=['PRIVATE']
        inputs['control']=seal(inputs['control']);inputs['experiment']=seal(dict(inputs['experiment'],control=inputs['control']['identity']))
    elif location=='profile':inputs['profiles']['C']['order']='PRIVATE'
    else:inputs['diagnostics']['B']['history']['bins'][0]['first']['families'][digest('training-family')]['hidden_terms']=['PRIVATE']
    with pytest.raises(ValueError):recompute(seal(inputs))


def test_reference_A_completion_does_not_gate_uncompared_C_B_interpretation(tmp_path):
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    public,record=analyzed_public(tmp_path,dict(A=64,B=65,C=65))
    assert verify_pac_evidence(record,root=public,finalized=False)['scientific_status']=='EXPLORATORY_SINGLE_SEED'
