"""Stdlib workstation driver: probes, an explicit freeze, then one fresh comparison."""
import json
import math
import time
import traceback
import zipfile
from pathlib import Path

from .identity import confined, digest, seal, write, file_hash
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.selector_coverage import POLICY
from oeis_learn.generalization.selector_validation import ARMS, CONTROLS, fresh_design, paired_metrics


def register(commands):
    parser=commands.add_parser('pac-selector-validation',description=__doc__)
    phases=parser.add_subparsers(dest='selector_phase',required=True)
    probe=phases.add_parser('probes',help='validate both selectors on existing training-only probes; no retraining')
    probe.add_argument('--campaign',default='run-state/generalization/pac-012-bcd-r1-002')
    probe.add_argument('--output',required=True)
    probe.add_argument('--preflight-only',action='store_true',help='verify exact local inputs without engine/build/GPU work')
    freeze=phases.add_parser('freeze',help='freeze a separate structural comparison after completed probe validation')
    freeze.add_argument('--probes',required=True)
    freeze.add_argument('--output',required=True)
    structural=phases.add_parser('structural',help='execute one frozen fresh structural comparison')
    structural.add_argument('--plan',required=True)


def read(path, *, sealed=False, cap=64<<20):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Required workstation artifact is missing: '+str(path))
    if path.stat().st_size>cap:raise ValueError('JSON artifact exceeds cap: '+str(path))
    value=json.loads(path.read_text())
    if sealed and seal(value)!=value:raise ValueError('Changed artifact seal: '+str(path))
    return value


def check_ref(root, ref, *, cap=32<<30):
    """Stream weights/exclusion hashes on the stdlib host, without loading tensors."""
    path=confined(root,ref['path'])
    if not path.is_file() or path.stat().st_size!=ref['bytes'] or ref['bytes']>cap or file_hash(path)!=ref['sha256']:
        raise ValueError('Artifact hash/size mismatch: '+str(path))
    return path


def discover(workspace, campaign):
    """Use original inventories and fixed paths, never newest-file heuristics."""
    campaign=confined(workspace,campaign)
    plan=read(campaign/'plan.json',sealed=True);workflow=read(campaign/'workflow.json',sealed=True)
    if workflow['plan']!=plan['identity'] or not plan['research'] or not set(ARMS)<=set(plan['arms']):
        raise ValueError('Original B/D research campaign required')
    entries={r['stage']:r for r in workflow['receipts']}
    if len(entries)!=len(workflow['receipts']):raise ValueError('Duplicate original stage')
    def original(stage, suffix):
        refs=[r for r in entries[stage]['artifacts'] if r['path']==suffix]
        if len(refs)!=1:raise ValueError('Original artifact missing from workflow: '+stage+'/'+suffix)
        return check_ref(campaign/stage,refs[0])
    base='execution/execution/role/private/assembly/resolved/'
    admission_path=original('admission',base+'admission.json')
    admission=read(admission_path,sealed=True)
    if admission['status']!='PASS' or admission['purpose']!='research' or admission['source']!=plan['source']:
        raise ValueError('Original research admission required')
    recipe=reference(original('admission',base+'recipe.json'),workspace)
    inventory=reference(original('admission',base+'training/private/inventory.json'),workspace)
    probes=reference(original('admission',base+'training/private/probes.json'),workspace)
    observed=reference(check_ref(admission_path.parent,admission['structural']),workspace)
    # Bind original exclusion files now, but defer their protected row reads and
    # expensive streaming hashes until the separately frozen structural phase.
    originals=admission['originals']
    from oeis_learn.generalization.structural import original_cells
    if sorted(({k:v for k,v in x.items() if k!='ref'} for x in originals),key=lambda x:x['name'])!=sorted(original_cells(),key=lambda x:x['name']):
        raise ValueError('All sixteen original exclusion references required')
    if any(x['ref'] not in plan['inputs'] for x in originals):raise ValueError('Original exclusions differ from campaign plan')
    checkpoints={}
    for arm in ARMS:
        result_path=original('train-'+arm,'execution/result.json')
        result=read(result_path,sealed=True)
        if result['status']!='PASS':raise ValueError('Completed training producer required for '+arm)
        path='training-'+arm+'/role/private/learner/checkpoints/retained.json'
        matches=[r for r in result['inventory'] if r['path']==path]
        if len(matches)!=1:raise ValueError('Original retained roster missing for '+arm)
        retained_path=check_ref(result_path.parent,matches[0]);retained=read(retained_path,sealed=True)
        runs=result['result']['runs']
        if len(runs)!=1 or runs[0]['arm']!=arm or runs[0]['result']['retained']!=retained['identity']:
            raise ValueError('Retained roster differs from original training result')
        if retained['study']!=result['study'] or retained['arm']!=arm or retained['seed']!=20261016:
            raise ValueError('Retained checkpoint provenance mismatch for '+arm)
        choices=[c for c in retained['checkpoints'] if c['identity']==retained['latest']]
        if len(choices)!=1:raise ValueError('Unique final checkpoint required for '+arm)
        descriptor=choices[0]
        if (seal(descriptor)!=descriptor or descriptor['kind']!='full_state' or 'latest' not in descriptor['pins']
                or not 0<descriptor['tokens']<=25000000):raise ValueError('Original final full-state checkpoint required')
        check_ref(retained_path.parent,descriptor['weights'])
        checkpoints[arm]=dict(descriptor=descriptor,root=str(retained_path.parent.relative_to(workspace)),
                             roster=reference(retained_path,workspace),producer=reference(result_path,workspace))
    return dict(campaign=str(campaign.relative_to(workspace)),original_plan=reference(campaign/'plan.json',workspace),
        workflow=reference(campaign/'workflow.json',workspace),recipe=recipe,inventory=inventory,probes=probes,
        observed=observed,originals=originals,checkpoints=checkpoints)


def envelope(work):
    value=dict(work=work,startup=1800,publication=1800,projection=1800,cleanup=30)
    return dict(value,outer=sum(value.values()))


def budgets(count):
    # Serial worst-case bounds enclose all seven controls, both selector scorings,
    # native/reference execution and every possible two-second branch observation.
    return dict(publication=envelope(86400),generation=envelope(3*(count*5*60+60)+7200+count*7*64*2),
                scoring=envelope(2*count*7*60+7200))


def source(workspace):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    return digest(collect_snapshot(workspace))


def _invoke(host, workspace, plan, output, role, operation, artifacts, roots, settings, kind, *, gpu=False):
    from .generalization_roles import project_and_invoke
    limits=plan['budgets'][kind]
    print(operation+': starting; output '+str(output),flush=True)
    result=project_and_invoke(host,plan['images'][int(gpu)],workspace,role,artifacts,
        source_roots=roots,output=str(output.relative_to(workspace)),operation=operation,settings=settings,
        seconds=limits['outer'],envelope=limits,gpu=gpu)
    print(operation+': '+result['status'],flush=True)
    if result['status'] not in {'PASS','PARTIAL'}:
        raise ValueError(operation+' failed: '+json.dumps(result,sort_keys=True)+'; original result '+str(output/'role/result.json'))
    return result


def compare(host, workspace, plan, output, publication, views):
    inputs=plan['inputs'];results={};generation={}
    recipe=json.loads(verify_artifact(workspace,inputs['recipe'],max_bytes=1<<20))
    for arm in ARMS:
        selected=inputs['checkpoints'][arm];descriptor=selected['descriptor'];root=confined(workspace,selected['root'])
        generated=_invoke(host,workspace,plan,output/arm/'generation','generator','selector-generate',
            {'visible_prompts':[publication['prompts']],'frozen_weights':[descriptor['weights']]},
            {'visible_prompts':views,'frozen_weights':root},dict(checkpoint=descriptor['weights'],prompts=publication['prompts'],
            recipe=recipe,study=plan['identity'],source=descriptor['identity'],device='cuda',bank_seeds=list(range(20261101,20261109))),
            'generation',gpu=True)
        generation[arm]=generated
        gen=output/arm/'generation/role/generation'
        candidates={k:reference(gen/v,gen) for k,v in [('original','selections.json'),
                    ('alternative','alternative-selections.json'),('observations','selector-observations.json')]}
        scored=_invoke(host,workspace,plan,output/arm/'scoring','selection','selector-score',
            {'selection_supervision':[publication['targets'],publication['roster']],'selection_candidates':list(candidates.values())},
            {'selection_supervision':views,'selection_candidates':gen},
            dict(targets=publication['targets'],roster=publication['roster'],**candidates,phase=plan['phase'],study=plan['identity']), 'scoring')
        results[arm]=scored
    return report(workspace,plan,output,results,generation)


def report(workspace,plan,output,results,generation):
    category='training' if plan['phase']=='probes' else 'certified'
    n=256 if plan['phase']=='probes' else 68;floor=math.ceil(.95*n)
    eligible=all(results[a]['trace_incomplete']==0 and generation[a]['status']=='PASS' and
        all(results[a]['metrics'][category]['equal_budget'][v]['complete']>=floor for v in ('original','alternative')) for a in ARMS)
    roster=results['B']['roster']
    if results['D']['roster']!=roster:raise ValueError('B/D comparison roster mismatch')
    cross={}
    for group in sorted({v['category'] for v in roster.values()}):
        subset={k:v for k,v in roster.items() if v['category']==group}
        cross[group]={v:{c:paired_metrics(subset,
            {k:results['B']['outcomes'][v][k][c] for k in subset},
            {k:results['D']['outcomes'][v][k][c] for k in subset}) for c in CONTROLS} for v in ('original','alternative')}
    result=seal(dict(version='pac-selector-validation-result/v1',status='PASS',phase=plan['phase'],plan=plan['identity'],
        scientific_status='COMPLETE_DESCRIPTIVE' if eligible else 'INCOMPLETE',freeze_eligible=eligible,
        completion_floor=floor,selector=POLICY['identity'],assessment='NOT_EXECUTED',training_tokens=0,
        metrics={a:results[a]['metrics'] for a in ARMS},changed={a:results[a]['changed'] for a in ARMS},
        trace_incomplete={a:results[a]['trace_incomplete'] for a in ARMS},B_to_D=cross,
        selector_seconds={a:results[a]['selector_seconds'] for a in ARMS},
        scoring={a:reference(output/a/'scoring/role/scoring.json',workspace) for a in ARMS},generation=generation))
    write(output/'result.json',result)
    lines=['# PAC selector '+plan['phase']+' validation','',result['scientific_status'],
           '', 'Frozen checkpoints; zero training tokens; assessment not executed.',
           '', '| Arm | Original success / complete | Coverage success / complete | Recoveries / regressions (joint complete) |',
           '|---|---|---|---|']
    for arm in ARMS:
        m=results[arm]['metrics'][category]['equal_budget'];a,b,j=m['original'],m['alternative'],m['joint']
        lines.append(f"| {arm} | {a['success']} / {a['complete']} | {b['success']} / {b['complete']} | {j['recoveries']} / {j['regressions']} ({j['count']}) |")
    lines.extend(['',f'Denominator {n}; completion floor {floor}. Full control, category and B/D paired results are in result.json.',
                  'Family-macro bounds account for censoring; they are not confidence intervals. One training seed only.'])
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(output/'report.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json','summary.md'):archive.write(output/name,name)
        archive.write(output/'plan.json','plan.json')
    print(json.dumps(dict(status='PASS',scientific_status=result['scientific_status'],freeze_eligible=eligible,report=str(output/'report.zip')),sort_keys=True))
    return result


def probes(args,workspace,host_factory):
    inputs=discover(workspace,args.campaign)
    output=confined(workspace,args.output,exists=False)
    if output.exists():raise ValueError('Fresh probe output required: '+str(output))
    output.mkdir(parents=True)
    args._selector_output=output
    write(output/'inputs.json',seal(inputs))
    print('Verified original B/D final checkpoints and training probe references.',flush=True)
    if args.preflight_only:
        print(json.dumps(dict(status='PASS',scope='INPUT_PREFLIGHT_ONLY',output=str(output))));return 0
    from .engines import Images
    from .generalization_qualification_io import produce
    host=host_factory(args);images=Images(cap=86400)
    pair=[images.ensure(host,target,output/'build'/target,seconds=43200) for target in ('qualification','gpu')]
    plan=seal(dict(version='pac-selector-validation-plan/v1',phase='probes',inputs=inputs,source=source(workspace),
        engine=host.engine.name,images=pair,selector=POLICY,budgets=budgets(256),training_tokens=0,
        controls=list(CONTROLS),arms=list(ARMS),comparison='shared generated candidate pools; separate selector observation cost',
        assessment='NOT_EXECUTED',qualification=str((output/'qualification').relative_to(workspace))))
    write(output/'plan.json',plan)
    qualified=produce(host,workspace,output/'qualification',images=pair,contract=plan['identity'],pac_seconds=7200)
    if qualified['status']!='PASS':raise ValueError('New source qualification failed: '+json.dumps(qualified)+'; inspect '+str(output/'qualification'))
    publication=_invoke(host,workspace,plan,output/'publication','preparation','selector-probes',
        {'mechanical_sources':[inputs['probes'],inputs['inventory']],'isolation_truth':[]},
        {'mechanical_sources':workspace},dict(probes=inputs['probes'],inventory=inputs['inventory']),'publication')
    compare(host,workspace,plan,output,publication,output/'publication/role')
    return 0


def freeze(args,workspace):
    prior=confined(workspace,args.probes);result=read(prior/'result.json',sealed=True)
    probe=read(prior/'plan.json',sealed=True)
    if (result['version']!='pac-selector-validation-result/v1' or result['phase']!='probes'
            or result['plan']!=probe['identity'] or not result['freeze_eligible'] or result['status']!='PASS'
            or probe['selector']!=POLICY or probe['source']!=source(workspace)):
        raise ValueError('Completed compatible training probe validation required before structural freeze')
    # Check all original score producers, not merely a user-edited eligibility flag.
    scores={a:read(check_ref(workspace,result['scoring'][a]),sealed=True) for a in ARMS}
    for value in scores.values():
        if value['study']!=probe['identity'] or value['trace_incomplete'] or any(value['metrics']['training']['equal_budget'][v]['complete']<244 for v in ('original','alternative')):
            raise ValueError('Probe scoring has incomplete selector/primary validation')
    output=confined(workspace,args.output,exists=False)
    if output.exists():raise ValueError('Fresh freeze directory required')
    plan=seal(dict(probe,phase='structural',protocol=fresh_design(),budgets=budgets(128),
        probe_result=reference(prior/'result.json',workspace),probe_plan=reference(prior/'plan.json',workspace),
        execution=str((output/'execution').relative_to(workspace)),
        decision='certified family-macro; >=65/68 complete per compared arm/selector; all seven controls; single-seed descriptive; no automatic adoption'))
    output.mkdir(parents=True);write(output/'plan.json',plan)
    args._selector_output=output
    print(json.dumps(dict(status='FROZEN',plan=str(output/'plan.json'),identity=plan['identity'],seed=plan['protocol']['seed'])))
    return 0


def structural(args,workspace,host_factory):
    path=confined(workspace,args.plan);plan=read(path,sealed=True)
    if (plan['phase']!='structural' or plan['protocol']!=fresh_design() or plan['selector']!=POLICY
            or plan['source']!=source(workspace) or plan['budgets']!=budgets(128)):
        raise ValueError('Unchanged separately frozen structural plan required')
    check_ref(workspace,plan['probe_result']);check_ref(workspace,plan['probe_plan'])
    inputs=plan['inputs']
    for ref in [inputs['recipe'],inputs['probes'],inputs['observed'],*[x['ref'] for x in inputs['originals']]]:check_ref(workspace,ref)
    for selected in inputs['checkpoints'].values():check_ref(confined(workspace,selected['root']),selected['descriptor']['weights'])
    output=confined(workspace,plan['execution'],exists=False)
    if output.exists():raise ValueError('Frozen structural plan already started; original result stays immutable: '+str(output))
    host=host_factory(args)
    if host.engine.name!=plan['engine']:raise ValueError('Frozen engine changed')
    from .generalization_qualification_io import consume_original
    output.mkdir(parents=True);write(output/'plan.json',plan)
    args._selector_output=output
    consume_original(host,workspace,confined(workspace,plan['qualification']),output/'qualification',
        required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=['weight-load','finite-update','native-exact'],pac_seconds=7200)
    publication=_invoke(host,workspace,plan,output/'publication','preparation','selector-structural',
        {'mechanical_sources':[],'isolation_truth':[inputs['observed'],inputs['probes'],*[x['ref'] for x in inputs['originals']]]},
        {'isolation_truth':workspace},dict(protocol=plan['protocol'],originals=inputs['originals'],
            observed=inputs['observed'],probes=inputs['probes'],study=plan['identity']),'publication')
    compare(host,workspace,plan,output,publication,output/'publication/role')
    return 0


def main(args,workspace,host_factory):
    workspace=Path(workspace)
    try:
        if args.selector_phase=='probes':return probes(args,workspace,host_factory)
        if args.selector_phase=='freeze':return freeze(args,workspace)
        return structural(args,workspace,host_factory)
    except (ValueError,KeyError,OSError,RuntimeError,TimeoutError) as exc:
        failure=seal(dict(status='FAIL',error=type(exc).__name__,message=str(exc)))
        if hasattr(args,'_selector_output'):
            target=args._selector_output
            write(target/'failure.json',failure)
            (target/'traceback.txt').write_text(traceback.format_exc())
            with zipfile.ZipFile(target/'failure.zip','x',zipfile.ZIP_DEFLATED) as archive:
                for name in ('failure.json','traceback.txt'):archive.write(target/name,name)
            print('Failure details: '+str(target/'failure.zip'),flush=True)
        print(json.dumps(failure),flush=True)
        traceback.print_exc()
        return 1
