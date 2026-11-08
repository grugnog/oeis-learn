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


def analyze(roster,outcomes,*,experiment,control,profiles,costs=None,records=None,scored=None,structural_scope=None):
    result=pac_comparison_report(roster,outcomes,experiment=experiment,control=control,profiles=profiles,costs=costs,portable=True,structural_scope=structural_scope)
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
    for case in chosen:enrich_case(case,(records or {}).get(case['arm'],{}).get(case['task']),old)
    result.update(reconstruction=dict(roster=roster,outcomes=outcomes,costs=costs,structural_scope=structural_scope),version='pac-analysis/v1',cases=chosen,missing_case_categories=[c for c in CATEGORIES if not buckets[c]],
        case_scope='generated candidate sources only; protected teacher/hidden values omitted',
        uncertainty='task-paired descriptive bounds; no inference across training seeds')
    return result


def analyze_campaign(root,*,output,workspace=None):
    from oeis_learn.generalization.contracts import verify_artifact
    root=Path(root);output=Path(output)
    def read(path):
        if path.is_symlink() or path.stat().st_size>128<<20:raise ValueError('bounded regular analysis input')
        return json.loads(path.read_text())
    workflow=read(root/'workflow.json');plan=read(root/'plan.json')
    if seal(workflow)!=workflow or seal(plan)!=plan:raise ValueError('immutable campaign identities required')
    entries={r['stage']:r for r in workflow['receipts']}
    accepted=entries['acceptance'];workspace=Path(workspace) if workspace is not None else Path(__file__).resolve().parents[3]
    manifest=workspace/accepted['resolved_manifest'];runtime_file=workspace/accepted['resolved_runtime']
    for path in (manifest,runtime_file):
        ref=next((r for r in accepted['resolved_refs'] if workspace/r['path']==path),None)
        if ref is None:raise ValueError('analysis authority absent from accepted closure')
        verify_artifact(workspace,ref)
    study=read(manifest);runtime=read(runtime_file)
    from oeis_learn.pipeline_validation.generalization_execution import verify_pac_runtime
    verify_pac_runtime(study,runtime)
    if runtime['purpose']!='research' or plan['source']!=study['source'] or plan['arms']!=study['arms'] or plan['images']!=runtime['images']:raise ValueError('frozen campaign authority differs')
    admission=json.loads(verify_artifact(manifest.parent,runtime['prepared']['admission']))
    if seal(admission)!=admission or admission.get('status')!='PASS':raise ValueError('intact native admission required')
    expected_scope=admission.get('structural_scope')
    from oeis_learn.generalization.pac_analysis import CONTROLS,project_training,recompute
    outcomes={};all_outcomes={};references={};scoring_hashes={};records={};scored={};roster=None;structural_scope=None;costs={}
    diagnostics={a:None for a in set(plan['arms'])|{'A'}};training_hashes=dict.fromkeys(diagnostics)
    def verified(name,path):
        entry=entries[name];base=root/entry['directory'];relative=str(path.relative_to(base))
        ref=next((r for r in entry['artifacts'] if r['path']==relative),None)
        if ref is None:raise ValueError('analysis input absent from original producer inventory')
        return json.loads(verify_artifact(base,ref,max_bytes=128<<20))
    for name in ('baseline',*('evaluate-'+a for a in plan['arms'])):
        if name not in entries:continue
        arm='A' if name=='baseline' else name.split('-')[1]
        folder=root/name/'execution/evaluation'/('A-reference' if name=='baseline' and study['control_mode']=='fresh-control' else arm)
        scoring=verified(name,folder/'scoring/role/scoring.json')
        if scoring.get('structural_scope')!=expected_scope:raise ValueError('scoring population differs from admitted scope')
        if scoring['study']!=study['identity'] or seal(scoring)!=scoring:raise ValueError('scorer study/seal mismatch')
        if scoring['roster']!=admission['roster']:raise ValueError('scoring roster differs from admitted population')
        if roster is None:roster=scoring['roster'];structural_scope=scoring.get('structural_scope')
        if scoring.get('structural_scope')!=structural_scope:raise ValueError('paired structural scope changed')
        if roster!=scoring['roster']:raise ValueError('paired structural population changed')
        if set(scoring['outcomes'])!=set(CONTROLS):raise ValueError('all seven scored controls required')
        if name=='baseline' and study['control_mode']=='fresh-control':
            references['A-reference']=dict(outcomes=scoring['outcomes'],scoring=scoring['identity'])
            continue
        all_outcomes[arm]=scoring['outcomes'];scoring_hashes[arm]=scoring['identity']
        outcomes[arm]=scoring['outcomes']['equal_budget'];scored[arm]=scoring['scored_cases']
        records[arm]=verified(name,folder/'generation/role/generation/selections.json')
    for arm in set(plan['arms'])|{'A'}:
        name='train-'+arm
        if name not in entries:costs[arm]=None;continue
        producer=verified(name,root/name/'execution/result.json')
        if seal(producer)!=producer:raise ValueError('training producer seal changed')
        result=producer['result']['runs'][0]['result']
        diagnostics[arm]=project_training(result);training_hashes[arm]=producer['identity']
    if roster is None:raise ValueError('no completed structural scoring available')
    result=analyze(roster,outcomes,experiment=study,control=runtime['control'],profiles=runtime['profiles'],costs=None,records=records,scored=scored,structural_scope=structural_scope)
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    inputs=seal(dict(version='pac-research-inputs/v1',experiment=study,control=runtime['control'],profiles=runtime['profiles'],roster=roster,structural_scope=structural_scope,
        outcomes=all_outcomes,references=references,diagnostics=diagnostics,scoring=scoring_hashes,training=training_hashes,
        stages={n:{k:r.get(k) for k in ('producer','binding','elapsed_seconds')} for n,r in entries.items() if n!='analysis'}))
    result.update(recompute(inputs));result.pop('reconstruction',None);result['version']='pac-analysis/v2'
    output.mkdir(parents=True,exist_ok=True);write(output/'research-inputs.json',inputs)
    result['inputs']=reference(output/'research-inputs.json',output)
    write(output/'analysis.json',seal(result));(output/'cases.md').write_text(case_markdown(result));return result


def enrich_case(case,record,old):
    if record is None:return
    selected=record.get('selections',{}).get('equal_budget')
    if selected:
        old.check_envelope(selected)
        case.update(original_source=selected['original_source'],parameterized_source=selected['transformed_source'],
            fitted_source=selected['selected']['source'],assignment=dict(selected['assignment']),
            limitation='visible20 fit with independently recorded100-term outcome; not a unique generating law')
    else:
        attempts=record.get('details',{}).get('correct',{}).get('details',[])
        if attempts and isinstance(attempts[0].get('source'),str):case['original_source']=attempts[0]['source'][:65536]
        case['limitation']=('censored decoding/fitting/execution; structural correctness unmeasured' if case['category']=='decode_timeout'
            else 'no admitted visible fit; wrong structure versus solver limitation unresolved')
    if case['category']=='simple_control_win':
        control=record.get('selections',{}).get('template')
        if control:old.check_envelope(control);case['simple_control_source']=control['selected']['source']


def case_markdown(result):
    lines=['# Scoped structural case review','',
        'Population: '+str(result.get('structural_scope') or 'historical 27-family scope')+'.',
        'Single-seed exploratory finite agreement. Missing categories: '+', '.join(result['missing_case_categories'])+'.','']
    for i,c in enumerate(result['cases'],1):
        lines += [f"## {i}. {c['arm']} / {c['category']}",'',f"Task `{c['task']}`; finite outcome `{c['success']}`; reason `{c['reason']}`.",'',c['limitation'],'']
        for key in ('original_source','parameterized_source','fitted_source','simple_control_source'):
            lines += [key.replace('_',' ').capitalize()+':','```text',c.get(key) or 'Unavailable','```','']
        lines += ['Fitted assignment: `'+json.dumps(c.get('assignment'),sort_keys=True)+'`.','']
    return '\n'.join(lines)+'\n'
