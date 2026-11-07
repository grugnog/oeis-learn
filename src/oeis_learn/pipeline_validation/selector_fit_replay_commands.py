"""Portable fixed-input CPU diagnostics, separate from the frozen probe experiment."""
import copy
import json
import statistics
import zipfile
from pathlib import Path

from .identity import seal,write,digest,confined
from .generalization_preparation import reference
from .selector_validation_commands import read,check_ref,source,envelope
from oeis_learn.generalization.selector_fit_replay import timeout_targets,check_core


def inputs(workspace,probes):
    root=confined(workspace,probes)
    plan=read(root/'plan.json',sealed=True);result=read(root/'result.json',sealed=True)
    if (plan['phase']!='probes' or result['phase']!='probes' or result['status']!='PASS'
            or result['plan']!=plan['identity']):raise ValueError('original completed probe run required')
    scored=read(check_ref(workspace,result['scoring']['D']),sealed=True)
    if scored['study']!=plan['identity']:raise ValueError('original D scoring producer mismatch')
    for variant in ('original','alternative'):
        for key,cases in scored['outcomes'][variant].items():
            for control,case in cases.items():
                if {k:case[k] for k in ('complete','success','candidate_sha256')}!=result['task_outcomes']['D'][variant][key][control]:
                    raise ValueError('original report/scoring outcome mismatch')
    targets=timeout_targets(scored['outcomes']['original'])
    generation=result['generation']['D'];ref=generation['selector_comparison']['artifacts']['selections.json']
    genroot=root/'D/generation/role/generation'
    generation_ref=reference(check_ref(genroot,ref),workspace)
    if generation['study']!=plan['identity']:raise ValueError('original generation binding mismatch')
    qualification=confined(workspace,plan['qualification'])
    authority=read(qualification/'qualification.json',sealed=True)
    old_source=read(check_ref(qualification,authority['source_closure']))
    if digest(old_source)!=plan['source']:raise ValueError('original qualification source differs from probe source')
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    check_core(old_source,collect_snapshot(workspace))
    resources=authority['resources']
    if resources['cpus']!=4 or resources['memory_bytes']!=16<<30 or resources['native_threads']!=1:
        raise ValueError('this diagnostic matrix requires the recorded four-CPU/16-GiB baseline')
    recipe=read(check_ref(workspace,plan['inputs']['recipe']))
    if recipe['pipeline']['workers']!=4:raise ValueError('original four-worker recipe required')
    for key in ('probes','inventory'):check_ref(workspace,plan['inputs'][key])
    return dict(original_plan=reference(root/'plan.json',workspace),original_result=reference(root/'result.json',workspace),
        original_scoring=result['scoring']['D'],original_source=reference(check_ref(qualification,authority['source_closure']),workspace),
        recipe_ref=plan['inputs']['recipe'],recipe=recipe,targets=targets,generation=generation_ref,
        probes=plan['inputs']['probes'],inventory=plan['inputs']['inventory'])


def matrix(compare_cpus):
    if type(compare_cpus) is not int or not 5<=compare_cpus<=32:raise ValueError('comparison CPU quota must be 5..32')
    return [dict(name='isolated-original-quota',cpus=4,concurrency=1),
            dict(name='concurrent-original-quota',cpus=4,concurrency=4),
            dict(name='concurrent-larger-quota',cpus=compare_cpus,concurrency=4)]


def main(args,workspace,host_factory):
    from .engines import Images
    from .generalization_roles import project_and_invoke
    from oeis_learn.container_validation import _stages,inspect_current_image
    workspace=Path(workspace);data=inputs(workspace,args.probes);configs=matrix(args.compare_cpus)
    if type(args.repetitions) is not int or not 1<=args.repetitions<=5:raise ValueError('one to five repetitions required')
    output=confined(workspace,args.output,exists=False)
    if output.exists():raise ValueError('fresh fit diagnostic output required')
    baseline_args=copy.copy(args);baseline_args.cpus=4;baseline_args.memory_gib=16
    baseline=host_factory(baseline_args)
    hosts={4:baseline}
    # Check the larger quota before builds or measurements. Memory/workers stay fixed.
    bigger_args=copy.copy(baseline_args);bigger_args.cpus=args.compare_cpus
    hosts[args.compare_cpus]=host_factory(bigger_args)
    if any(h.engine.name!=baseline.engine.name for h in hosts.values()):raise ValueError('same engine required')
    output.mkdir(parents=True);args._selector_output=output
    print('fit-diagnostics: verified '+str(len(data['targets']))+' timeout cases; no weights or GPU required',flush=True)
    image=Images(cap=43200).ensure(baseline,'qualification',output/'build',seconds=43200)
    plan=seal(dict(version='pac-fit-diagnostic-plan/v1',source=source(workspace),image=image,engine=baseline.engine.name,
        inputs=data,matrix=configs,repetitions=args.repetitions,memory_bytes=16<<30,workers=4,
        native_threads=1,cache='fresh per repetition; four recorded warmups before each repetition',
        scientific_status='DIAGNOSTIC_ONLY',freeze_eligible=False,
        limits='unchanged resource profile and per-attempt original service allowance; existing client solver reserve',
        schedule='fixed case/attempt order; waves of concurrency; worker slot = attempt index modulo four'))
    # The recipe is public configuration; sources and prompts are not in this plan.
    write(output/'plan.json',plan)
    inspect_current_image(baseline,image,output/'inspection',seconds=900)
    baseline.image=image
    _,checks=_stages(baseline,'bigint-cpu',output/'cpu-native-checks',external_seconds=3600)
    if not checks or any(v['status']!='PASS' for v in checks.values()):raise ValueError('current CPU/native checks failed')
    limits=envelope(7200)
    def invoke(host,path,role,operation,artifacts,roots,settings):
        print(operation+': '+str(path),flush=True)
        value=project_and_invoke(host,image,workspace,role,artifacts,source_roots=roots,
            output=str(path.relative_to(workspace)),operation=operation,settings=settings,
            seconds=limits['outer'],envelope=limits)
        if value['status']!='PASS':raise ValueError(operation+' failed: '+json.dumps(value)+'; inspect '+str(path/'role/result.json'))
        return value
    prepared=invoke(baseline,output/'preparation','preparation','prepare-fit-replay',
        {'mechanical_sources':[data['generation'],data['probes'],data['inventory']],'isolation_truth':[]},
        {'mechanical_sources':workspace},dict(generation=data['generation'],probes=data['probes'],inventory=data['inventory'],targets=data['targets']))
    views=output/'preparation/role';runs=[]
    # Complete the isolated baseline before any resource comparison. No original
    # score, checkpoint, probe result or freeze authorization is rewritten.
    for config in configs:
        result=invoke(hosts[config['cpus']],output/config['name'],'fitter','replay-native-fits',
            {'visible_prompts':[prepared['prompts']],'generated_candidates':[prepared['candidates']]},
            {'visible_prompts':views,'generated_candidates':views},dict(prompts=prepared['prompts'],candidates=prepared['candidates'],
                recipe=data['recipe'],concurrency=config['concurrency'],repetitions=args.repetitions))
        if result['packet']!=prepared['packet']:raise ValueError('resource comparison changed the replay packet')
        runs.append(dict(config=config,result=result))
    report=seal(dict(version='pac-fit-diagnostic-report/v1',status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,plan=plan['identity'],cases=prepared['cases'],attempts=prepared['attempts'],runs=runs,
        interpretation='fit completion and latency under controlled resources; no full-horizon scoring or repair of probe gate'))
    write(output/'result.json',report)
    lines=['# Fixed-candidate native fit diagnostics','',
        'Diagnostic only. Original probe result and structural freeze gate remain unchanged.','',
        '| Configuration | CPUs | Concurrent fits | Median fit workload seconds | Completed fits per repeat | Fully completed case searches per repeat |',
        '|---|---:|---:|---:|---|---|']
    for item in runs:
        config=item['config'];r=item['result']['runs']
        lines.append('| '+config['name']+' | '+str(config['cpus'])+' | '+str(config['concurrency'])+' | '+
            str(round(statistics.median(v['wall_seconds'] for v in r),3))+' | '+
            ', '.join(str(sum(a['complete'] for a in v['attempts'])) for v in r)+' | '+
            ', '.join(str(sum(c['fit_search_complete'] for c in v['cases'].values())) for v in r)+' |')
    lines+=['','Each fit keeps its original source, visible prefix, seed and service allowance.',
        'Queue/transport wall time includes more than solver compute; native timing, restarts and sampled RSS are reported separately.',
        'The original mixed GPU/solver schedule is not reconstructed. Repeated cold caches and explicit warmups are controlled diagnostic conditions.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(output/'report.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json','summary.md','plan.json'):archive.write(output/name,name)
    print(json.dumps(dict(status='PASS',scientific_status='DIAGNOSTIC_ONLY',freeze_eligible=False,report=str(output/'report.zip'))))
    return 0
