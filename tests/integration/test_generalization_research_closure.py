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
            outcomes={c:{k:(None if k in certified[complete:] else (i%3==0 if c=='raw' else i%2==0)) for i,k in enumerate(roster)} for c in CONTROLS}
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
    record=collect(root,tmp_path/'public')
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
