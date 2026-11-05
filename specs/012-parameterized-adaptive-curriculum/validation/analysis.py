"""Versioned 128-task read-only adapter; historical 256-task guards stay intact."""
import importlib.util
import json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,write
from oeis_learn.generalization.reconciliation import pac_comparison_report

CATEGORIES=('shared_success','arm_only_success','continuation_failure','complete_no_fit','decode_timeout','simple_control_win')


def prior():
    path=Path(__file__).resolve().parents[2]/'011-controlled-generalization/validation/post_pilot_analysis.py'
    spec=importlib.util.spec_from_file_location('pac_prior_case_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def analyze(roster,outcomes,*,experiment,control,profiles,costs=None,records=None,scored=None):
    result=pac_comparison_report(roster,outcomes,experiment=experiment,control=control,profiles=profiles,costs=costs)
    old=prior();scores={}
    for arm,values in sorted(outcomes.items()):
        scores[arm]={key:dict(equal_budget=dict(success=values.get(key) is True,complete=values.get(key) is not None,reason='recorded' if key in values else 'missing')) for key in roster}
    result['legacy_count_adapter']={arm:old.count_outcomes(rows) for arm,rows in scores.items()}
    buckets={name:[] for name in CATEGORIES}
    for arm,values in sorted(outcomes.items()):
        other='C' if arm=='B' else 'B'
        for key in sorted(roster):
            if key not in values:continue
            value=values[key];raw=(scored or {}).get(arm,{}).get(key,{})
            score=raw.get('equal_budget',{})
            if value is True:category='shared_success' if outcomes.get(other,{}).get(key) is True else 'arm_only_success'
            elif value is None:category='decode_timeout'
            elif score.get('candidate_sha256'):category='continuation_failure'
            else:category='complete_no_fit'
            case=dict(arm=arm,task=key,category=category,success=value,reason=score.get('reason','unavailable'),
                original_source=None,parameterized_source=None,fitted_source=None,assignment=None,
                limitation='finite recorded outcome; source/solver detail unavailable')
            buckets[category].append(case)
            if raw.get('template',{}).get('success') is True and value is not True:
                buckets['simple_control_win'].append(dict(case,category='simple_control_win'))
    chosen=[];seen=set()
    for category in CATEGORIES:
        for case in buckets[category][:2]:chosen.append(case);seen.add((case['arm'],case['task'],case['category']))
    for category in CATEGORIES:
        for case in buckets[category]:
            if len(chosen)>=12:break
            if (case['arm'],case['task'],case['category']) not in seen:chosen.append(case);seen.add((case['arm'],case['task'],case['category']))
    result.update(version='pac-analysis/v1',cases=chosen,missing_case_categories=[c for c in CATEGORIES if not buckets[c]],
        case_scope='generated candidate sources only; protected teacher/hidden values omitted',
        uncertainty='task-paired descriptive bounds; no inference across training seeds')
    return result


def analyze_campaign(root,*,output):
    from oeis_learn.generalization.contracts import verify_artifact
    root=Path(root);output=Path(output)
    def read(path):
        if path.is_symlink() or path.stat().st_size>128<<20:raise ValueError('bounded regular analysis input')
        return json.loads(path.read_text())
    workflow=read(root/'workflow.json');plan=read(root/'plan.json')
    if seal(workflow)!=workflow or seal(plan)!=plan:raise ValueError('immutable campaign identities required')
    entries={r['stage']:r for r in workflow['receipts']}
    accepted=entries['acceptance'];workspace=Path(__file__).resolve().parents[3]
    manifest=workspace/accepted['resolved_manifest'];runtime_file=workspace/accepted['resolved_runtime']
    study=read(manifest);runtime=read(runtime_file)
    outcomes={};records={};scored={};roster=None;costs={}
    def verified(name,path):
        entry=entries[name];base=root/entry['directory'];relative=str(path.relative_to(base))
        ref=next((r for r in entry['artifacts'] if r['path']==relative),None)
        if ref is None:raise ValueError('analysis input absent from original producer inventory')
        return json.loads(verify_artifact(base,ref,max_bytes=128<<20))
    for name in ('baseline',*('evaluate-'+a for a in plan['arms'])):
        if name not in entries:continue
        arm='A' if name=='baseline' else name.split('-')[1]
        folder=root/name/'execution/evaluation'/arm
        scoring=verified(name,folder/'scoring/role/scoring.json')
        if scoring['study']!=study['identity'] or seal(scoring)!=scoring:raise ValueError('scorer study/seal mismatch')
        if roster is None:roster=scoring['roster']
        if roster!=scoring['roster']:raise ValueError('paired structural population changed')
        outcomes[arm]=scoring['outcomes']['equal_budget'];scored[arm]=scoring['scored_cases']
        records[arm]=verified(name,folder/'generation/role/generation/selections.json')
    for arm in set(plan['arms'])|{'A'}:
        name='train-'+arm
        if name not in entries:costs[arm]=None;continue
        result=verified(name,root/name/'execution/result.json')['result']['runs'][0]['result']
        allocation=result['allocation'];tokens=allocation['tokens']
        costs[arm]=dict(allowed_tokens=tokens['allowed'],useful_tokens=tokens['useful'],reserved_tokens=tokens['reserved'],discarded_tokens=tokens['discarded'],unspent_tokens=tokens['unspent'],
            examples=sum(v['examples'] for f in allocation['allocation'].values() for v in f.values()),updates=result['updates'],seconds=result['learner_seconds'])
    if roster is None:raise ValueError('no completed structural scoring available')
    result=analyze(roster,outcomes,experiment=study,control=runtime['control'],profiles=runtime['profiles'],costs=costs,records=records,scored=scored)
    write(output/'analysis.json',seal(result));return result
