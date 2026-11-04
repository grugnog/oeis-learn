#!/usr/bin/env python3
"""Temporary fresh paired pilot: canonical preparation/learning and isolated scoring."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(HERE)]
from oeis_learn.pipeline_validation.identity import confined,seal,write,file_hash
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.interfaces import EXIT_CODES
from fresh_pilot_inputs import research_inputs,blueprint,BANK_SEEDS
from fresh_pilot_report import measurement_plan,terminal_status,assemble_public,diagnostics

CAMPAIGN_SECONDS=96*3600
REPORT_SECONDS=3600
RESERVATIONS=dict(feasibility=22000,acceptance=23700,preflight=7200,prepare=28*3600,smoke=21600,
    **{'train-static':28800,'train-refreshed':28800,'generate-static':23400,
       'score-static':23400,'generate-refreshed':23400,'score-refreshed':23400})


def read(path,cap=16<<20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>cap:raise ValueError('bounded regular JSON required')
    return json.loads(path.read_bytes())


def allowance(elapsed,requested):
    remaining=CAMPAIGN_SECONDS-REPORT_SECONDS-elapsed
    if requested<=0 or requested>remaining:raise TimeoutError('campaign reservation exhausted; preserve reporting')
    return requested


def canonical(engine,image,operation,output,extra=()):
    return [sys.executable,str(ROOT/'scripts/foundation_run.py'),'--engine',engine,'--image',image,
            '--memory-gib','64','--cpus','4','generalization',operation,'--output',str(output),*map(str,extra)]


def latest_checkpoint(learner,result,*,study,arm):
    root=Path(learner)/'checkpoints';roster=read(root/'retained.json')
    if (seal(roster)!=roster or roster['version']!='generalization-retained/v1' or
        roster['identity']!=result['retained'] or roster['study']!=study or roster['arm']!=arm or roster['seed']!=20261016):
        raise ValueError('original paired learner roster required')
    found=[c for c in roster['checkpoints'] if c['identity']==roster['latest']]
    if len(found)!=1:raise ValueError('unique latest checkpoint required')
    checkpoint=found[0]
    if seal(checkpoint)!=checkpoint or checkpoint['tokens']!=result['tokens'] or checkpoint['updates']!=result['updates']:
        raise ValueError('checkpoint exposure differs from original learner result')
    verify_artifact(root,checkpoint['weights'],max_bytes=32<<30)
    return checkpoint,root


def invoke(argv,logs,*,seconds):
    """External bound covers host validation, tensor copies and role cleanup too."""
    from oeis_learn.pipeline_validation.supervision import launch
    try:
        launch(argv,logs,seconds=seconds,stream_cap=1<<20,cleanup_seconds=30)
        return 0
    except KeyboardInterrupt:return 130
    except (OSError,RuntimeError,TimeoutError):
        p=logs/'process.json'
        code=read(p).get('returncode') if p.is_file() else None
        return code if code in (1,2,3,130) else 1


def verified_archive(root,archive,verification,*,code,verified,expected):
    """An exit code alone cannot distinguish a negative scope from verifier failure."""
    terminal=read(verification/'result.json');summary=read(verification/'summary.json')
    ref=reference(archive,root)
    if (seal(terminal)!=terminal or seal(summary)!=summary or
        terminal.get('version')!='generalization-verification-result/v1' or terminal.get('status')!='COMPLETE' or
        terminal.get('exit_code')!=verified or terminal.get('archive')!=ref or summary.get('archive')!=ref or
        terminal.get('summary')!=reference(verification/'summary.json',verification) or
        summary.get('integrity')!='PASS' or summary.get('selected_scope')!=expected or
        verified!=code or EXIT_CODES.get(expected)!=verified):
        raise ValueError('canonical archive verification/binding failed')
    return summary


def role(job_path,output):
    """Child process with fixed existing role operations; never mounts the checkout."""
    from foundation_local import parser,Host
    from oeis_learn.pipeline_validation.generalization_commands import _qualification
    from oeis_learn.pipeline_validation.generalization_roles import project_and_invoke
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope
    job=read(confined(ROOT,job_path));target=confined(ROOT,output,exists=False)
    fields={'version','engine','qualified','manifest','study','study_root','protocol','role','operation',
        'artifacts','source_roots','settings','checkpoint','identity'}
    if set(job)!=fields or seal(job)!=job or job.get('version')!='fresh-pilot-role/v1':raise ValueError('intact frozen role job required')
    if (job['role'],job['operation']) not in {('generator','evaluate'),('selection','score')}:
        raise ValueError('unregistered pilot role')
    if target.exists():raise ValueError('fresh role wrapper output required')
    target.mkdir(parents=True)
    try:
        authority=read(confined(ROOT,job['qualified'])/'qualification.json')
        args=parser().parse_args(canonical(job['engine'],authority['images'][0],'preflight','unused',
            ['--manifest',job['manifest']])[2:])
        host=Host(args)
        from oeis_learn.pipeline_validation.generalization_runtime import load
        base,study,runtime=load(ROOT,job['manifest'])
        if study['identity']!=job['study'] or runtime['purpose']!='research':raise ValueError('job differs from frozen research study')
        protocol=json.loads(verify_artifact(ROOT,job['protocol'],max_bytes=16<<20))
        if seal(protocol)!=protocol or protocol['identity'] not in study['hypothesis']:
            raise ValueError('study does not bind this pilot protocol')
        checkpoint=job['checkpoint'];settings=job['settings']
        if seal(checkpoint)!=checkpoint or settings['study']!=study['identity']:
            raise ValueError('job checkpoint/study binding differs')
        if job['role']=='generator' and (settings['source']!=checkpoint['identity'] or
            settings['checkpoint']!=checkpoint['weights'] or settings['bank_seeds']!=protocol['bank_seeds'] or
            settings['device']!='cuda' or settings['recipe']!=json.loads(verify_artifact(base,runtime['recipe']))):
            raise ValueError('generation differs from frozen checkpoint/protocol')
        from types import SimpleNamespace
        _qualification(host,ROOT,target,base,study,runtime,SimpleNamespace(qualification=job['qualified']))
        roots={k:confined(ROOT,v) for k,v in job['source_roots'].items()}
        image=runtime['images'][1 if job['role']=='generator' else 0]
        with scope(confined(ROOT,job['study_root']),runtime['storage']):
            result=project_and_invoke(host,image,ROOT,job['role'],job['artifacts'],source_roots=roots,
                output=str((target/'execution').relative_to(ROOT)),operation=job['operation'],settings=job['settings'],
                seconds=protocol.get('role_seconds',21600),gpu=job['role']=='generator')
    except (Exception,KeyboardInterrupt) as exc:
        result=dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__,error=str(exc)[:2048])
    write(target/'result.json',seal(result))
    return EXIT_CODES.get(result.get('status'),1)


def evaluate_arm(engine,qualified,research,manifest,study,protocol_ref,arm,checkpoint,
                 weights_root,visible,truth,prompt_ref,truth_ref,recipe,command):
    """Shared full/smoke wiring of the unchanged isolated generator and scorer."""
    def relative(p):return str(Path(p).relative_to(ROOT))
    generation=research/('generate-'+arm)
    common=dict(version='fresh-pilot-role/v1',engine=engine,qualified=relative(qualified),
        manifest=manifest,study=study['identity'],study_root=relative(research),protocol=protocol_ref,checkpoint=checkpoint)
    job=seal(dict(common,role='generator',operation='evaluate',
        artifacts={'visible_prompts':[prompt_ref],'frozen_weights':[checkpoint['weights']]},
        source_roots={'visible_prompts':relative(visible),'frozen_weights':relative(weights_root)},
        settings=dict(checkpoint=checkpoint['weights'],prompts=prompt_ref,recipe=recipe,study=study['identity'],
            source=checkpoint['identity'],device='cuda',bank_seeds=BANK_SEEDS)))
    job_path=research/'private/jobs'/('generate-'+arm+'.json');write(job_path,job)
    _,status=command('generate-'+arm,[sys.executable,str(Path(__file__).resolve()),'role','--job',relative(job_path),
        '--output',relative(generation)],generation)
    if status not in {'PASS','PARTIAL'}:raise RuntimeError('generation failed')
    generated=generation/'execution/role/generation';candidate_ref=reference(generated/'selections.json',generated)
    scoring=research/('score-'+arm)
    job=seal(dict(common,role='selection',operation='score',
        artifacts={'selection_supervision':[truth_ref],'selection_candidates':[candidate_ref]},
        source_roots={'selection_supervision':relative(truth),'selection_candidates':relative(generated)},
        settings=dict(targets=truth_ref,candidates=candidate_ref,study=study['identity'])))
    job_path=research/'private/jobs'/('score-'+arm+'.json');write(job_path,job)
    _,status=command('score-'+arm,[sys.executable,str(Path(__file__).resolve()),'role','--job',relative(job_path),
        '--output',relative(scoring)],scoring)
    if status!='PASS':raise RuntimeError('selection scoring failed')
    scored=read(scoring/'execution/role/selection/scoring.json',128<<20)
    extra=[]
    for name in ('evaluation.json','proposal-bank.json','selections.json'):
        if (generated/name).is_file():extra.append((generated/name,'fresh/generation-'+arm+'/'+name))
    if (generated/'attempts').is_dir():
        for p in sorted((generated/'attempts').rglob('*')):
            if p.is_file() and p.suffix in {'.json','.jsonl'}:extra.append((p,'fresh/generation-'+arm+'/attempts/'+str(p.relative_to(generated/'attempts'))))
    return scored,extra


def recovery(root,workflow):
    """A verified diagnostic ZIP when no research report can be produced."""
    from oeis_learn.pipeline_validation.bundle import pack,verify
    from oeis_learn.fresh.common import write as replace
    folder=root/'recovery';folder.mkdir(exist_ok=False)
    replace(root/'workflow.json',seal(workflow))
    write(folder/'workflow.json',seal(workflow))
    write(folder/'summary.json',dict(selected_scope=workflow['status'],scientific_status='UNMEASURED',
        feature_qualified=False,scope='diagnostic recovery only; no completed research claim'))
    diagnostic_files=diagnostics(root,folder/'diagnostics')
    mandatory=['workflow.json','summary.json',*['diagnostics/'+p.name for p,_ in diagnostic_files]]
    for name in ('resume.json','parent-workflow.json','parent-preparation-summary.json','execution-view.json'):
        if (root/name).is_file():
            write(folder/name,read(root/name));mandatory.append(name)
    archive=pack(folder,mandatory=mandatory,optional_cap=8<<20)
    verify(archive)
    return archive


def diagnose(run_root,output):
    """Export existing control evidence without containers, preparation or learning."""
    from oeis_learn.pipeline_validation.bundle import pack,verify
    root=confined(ROOT,run_root);destination=confined(ROOT,output,exists=False)
    if destination.exists() or destination.with_suffix('.zip').exists() or root==destination or root in destination.parents or destination in root.parents:
        raise ValueError('fresh diagnostic output disjoint from original run required')
    workflow=read(root/'workflow.json')
    if seal(workflow)!=workflow:raise ValueError('original sealed workflow required')
    destination.mkdir(parents=True)
    write(destination/'workflow.json',workflow)
    write(destination/'summary.json',dict(selected_scope=workflow['status'],scientific_status='UNMEASURED',
        qualification_evidence=False,scope='read-only diagnostic export; no new execution'))
    files=diagnostics(root,destination/'diagnostics')
    archive=pack(destination,mandatory=['workflow.json','summary.json',*['diagnostics/'+p.name for p,_ in files]])
    verify(archive)
    print(json.dumps(dict(status='DIAGNOSTIC_EXPORTED',archive=str(archive.relative_to(ROOT))),indent=2))
    return 0


def run(args):
    from oeis_learn.fresh.common import write as replace
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_study
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.pipeline_validation.generalization_workflow import _merge
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.container_runtime import Engine
    from fresh_pilot_throughput_host import require_for_run
    from fresh_pilot_resume import prepared_inputs,throughput_proof,disk_preflight
    resumed=None
    if getattr(args,'resume_prepared',None):
        resumed=prepared_inputs(ROOT,args.resume_prepared,args.output,args.engine)
        throughput_path,throughput_receipt,throughput_report=throughput_proof(sys.modules[__name__],resumed,args.engine)
    else:
        throughput_receipt,throughput_report=require_for_run(args)
        disk_preflight(ROOT)
    root=confined(ROOT,args.output,exists=False)
    if root.exists() or root.with_suffix('.zip').exists():raise ValueError('fresh pilot output required')
    root.mkdir(parents=True);started=time.monotonic();stages=[]
    workflow=dict(version='fresh-pilot-workflow/v1',status='PARTIAL',stages=stages,
        research_authorized=True,campaign_seconds=CAMPAIGN_SECONDS,assessment='SEALED',
        throughput=throughput_receipt['identity'])
    study=None;plan=None;targets=None;qualified=None;scored={};selected={};extra=[];archive=None
    def relative(p):return str(Path(p).relative_to(ROOT))
    def record():replace(root/'workflow.json',seal(workflow))
    def command(name,argv,output):
        seconds=allowance(time.monotonic()-started,RESERVATIONS[name]);began=time.monotonic()
        code=invoke(argv,root/'private/logs'/name,seconds=seconds)
        original=read(output/'result.json') if (output/'result.json').is_file() else None
        if original is not None and seal(original)!=original:original=None
        status=terminal_status(code,original)
        stage=dict(stage=name,status=status,exit_code=code,elapsed_seconds=time.monotonic()-began,
            command=argv,originals=[str(output/'result.json')] if original is not None else [])
        stages.append(stage);record()
        print(name+': '+status,flush=True)
        if status=='INTERRUPTED':raise KeyboardInterrupt()
        return original,status
    record()
    try:
        if resumed is not None:
            from fresh_pilot_resume import continuation_record
            from fresh_pilot_report import preparation_diagnostics
            parent=resumed['old'];qualified=resumed['qualified'];images=resumed['images']
            manifest=resumed['manifest'];design=resumed['design'];source_check=resumed['source_check']
            research=root/'research';research.mkdir()
            design_root=parent/'research/design'
            continuation=continuation_record(ROOT,resumed,throughput_receipt)
            write(root/'resume.json',continuation)
            write(root/'parent-workflow.json',resumed['workflow'])
            write(root/'parent-preparation-summary.json',preparation_diagnostics(parent))
            workflow['resume']=reference(root/'resume.json',root)
            workflow['acceptance']=dict(resumed['workflow']['acceptance'],reused_from=str(parent.relative_to(ROOT)))
            stages.extend(dict(stage,execution='REUSED',parent_workflow=resumed['workflow']['identity'])
                for stage in resumed['workflow']['stages'][:3])
            extra.extend([(root/'resume.json','fresh/resume.json'),
                (root/'parent-workflow.json','fresh/parent-workflow.json'),
                (root/'parent-preparation-summary.json','fresh/parent-preparation-summary.json'),
                (parent/'feasibility/feasibility.json','fresh/feasibility.json'),
                (parent/'research/protocol-summary.json','fresh/protocol.json'),
                (throughput_path/'result.json','fresh/throughput-receipt.json')])
            record()
            print('Verified original feasibility, acceptance and preparation; continuing at smoke',flush=True)
        else:
            feasibility=root/'feasibility'
            result,status=command('feasibility',[sys.executable,str(Path(__file__).resolve()),'feasibility',
                '--output',relative(feasibility)],feasibility)
            if status!='PASS':raise RuntimeError('full source feasibility failed; no hardware launched')
            design=read(feasibility/'private/design.json')
            source_check=json.loads(verify_artifact(feasibility,result['feasibility']))
            if seal(source_check)!=source_check or seal(design)!=design or source_check['protocol']!=design['identity'] or source_check['status']!='FEASIBLE':
                raise ValueError('feasibility/protocol binding differs')
            if throughput_report['protocol']!=design['identity']:raise ValueError('throughput protocol differs from frozen design')
            extra.append((feasibility/'feasibility.json','fresh/feasibility.json'))
            extra.append((confined(ROOT,args.throughput)/'result.json','fresh/throughput-receipt.json'))
            imported=confined(ROOT,args.imported)
            if args.qualified:
                qualified=confined(ROOT,args.qualified)
            else:
                if not args.cpu_image or not args.gpu_image:raise ValueError('prebuilt CPU and GPU image tags required')
                from oeis_learn.pipeline_validation.generalization_setup import assemble_acceptance
                import generalization_hip_serial as serial
                engine=Engine(args.engine);images=[engine.image(args.cpu_image),engine.image(args.gpu_image)]
                setup=assemble_acceptance(ROOT,imported=args.imported,images=images,output=relative(root/'acceptance-inputs'))
                acceptance=root/'acceptance'
                frozen=serial.plan(args.engine,setup['session']['manifest'],relative(acceptance),image=images[0])
                for item in frozen:
                    index=item['argv'].index('generalization')
                    item['argv'][index:index]=['--memory-gib','64','--cpus','4']
                allowance(time.monotonic()-started,RESERVATIONS['acceptance'])
                from oeis_learn.generalization.resources import CAPS
                with scope(acceptance,CAPS,extra_roots=[root/'acceptance-inputs']):code=serial.execute(frozen,acceptance)
                workflow['acceptance']=dict(exit_code=code,archive=relative(acceptance.with_suffix('.zip')));record()
                if code:raise RuntimeError('current-source acceptance did not pass; research not started')
                qualified=acceptance/'report/qualified'
            authority=read(qualified/'qualification.json')
            if json.loads(verify_artifact(qualified,authority['source_closure']))!=collect_snapshot(ROOT):
                raise ValueError('qualification source closure changed')
            if authority['resources']['memory_bytes']!=64<<30 or authority['resources']['cpus']!=4:
                raise ValueError('pilot requires matching qualification at64GiB/four CPUs')
            images=authority['images']
            if images[0]!=throughput_receipt['binding']['image']:raise ValueError('CPU image changed after throughput check')
            research=root/'research';research.mkdir()
            write(research/'source.json',collect_snapshot(ROOT))
            write(research/'protocol-summary.json',seal(dict(version='fresh-pilot-public-protocol/v1',
                protocol=design['identity'],seed=20261016,training_shards=12,rows_per_shard=32768,
                rows_per_family=1024,training_families=32,selection_rows=256,assessment='SEALED',
                numeric_law=design['numeric_law'],descriptor_screen=design['descriptor_screen'],
                reference_workers=4,source_feasibility=source_check['identity'],target_tokens_per_arm=25_000_000,learner_seconds=21600,generation_seconds=21600,
                scoring_seconds=21600,length_boundary=16,bank_seeds=BANK_SEEDS,
                controls=['raw','rescue','equal_budget','shuffled','absent','template','bank'],
                interpretation='single-seed exploratory; matched exposure and95% primary completion required; no automatic advancement')))
            extra.append((research/'protocol-summary.json','fresh/protocol.json'))
            cfg=blueprint(ROOT,design,recipe=relative(imported/'provenance/recipe.json'),source=relative(research/'source.json'),
                images=images,feature=relative(qualified/'feature.json'),output=relative(research/'design'))
            assembly=assemble_study(ROOT,cfg,relative(research/'inputs'))
            manifest=assembly['manifest']
            command_args=['--manifest',manifest,'--qualification',relative(qualified)]
            preflight=research/'preflight'
            _,status=command('preflight',canonical(args.engine,images[0],'preflight',relative(preflight),['--manifest',manifest]),preflight)
            if status!='PASS':raise RuntimeError('research preflight failed')
            prepared=research/'prepare'
            result,status=command('prepare',canonical(args.engine,images[0],'prepare',relative(prepared),command_args),prepared)
            if status!='PASS':raise RuntimeError('research preparation incomplete; no learner launched')
            manifest=result['result']['manifest']
            from fresh_pilot_feasibility import verify_prepared
            verify_prepared(source_check,ROOT/manifest.rsplit('/',1)[0])
            design_root=research/'design'
        # Preparation audit files remain immutable in their original root. The
        # active study contains exactly the canonical execution closure, whose
        # peak includes one temporary serial role projection.
        from fresh_pilot_storage import project
        execution=root/'execution'
        execution_manifest,view=project(ROOT,manifest,relative(execution/'private/prepared'))
        write(root/'execution-view.json',view)
        extra.append((root/'execution-view.json','fresh/execution-view.json'))
        print('execution-storage-plan: '+json.dumps(view['storage'],sort_keys=True),flush=True)
        smoke=root/'smoke'
        job=seal(dict(version='fresh-pilot-smoke-job/v1',engine=args.engine,qualified=relative(qualified),
            manifest=manifest,design=reference(design_root/'design.json',ROOT),
            blueprint=reference(design_root/'blueprint.json',ROOT)))
        job_path=root/'private/smoke-job.json';write(job_path,job)
        _,status=command('smoke',[sys.executable,str(Path(__file__).resolve()),'smoke',
            '--job',relative(job_path),'--output',relative(smoke)],smoke)
        if status!='PASS':raise RuntimeError('paired smoke failed; full learners not launched')
        extra.append((smoke/'result.json','fresh/smoke.json'))
        from oeis_learn.pipeline_validation.generalization_runtime import load
        research=execution;manifest=execution_manifest
        base,study,runtime=load(ROOT,manifest)
        verify_artifact(base,runtime['recipe'],max_bytes=1<<20)
        extra.append((confined(base,runtime['recipe']['path']),'fresh/recipe.json'))
        truth_refs=[];prompt_refs=[]
        for ref in study['data']['manifests']:
            cohort=json.loads(verify_artifact(base,ref,max_bytes=16<<20))
            if cohort['role']=='selection':
                truth_refs.extend(cohort['private_payloads']);prompt_refs.extend(cohort['public_payloads'])
        visible=research/'private/evaluation/visible';truth=research/'private/evaluation/scorer'
        visible.mkdir(parents=True);truth.mkdir(parents=True)
        prompt_ref=_merge(base,prompt_refs,visible/'prompts.json');truth_ref=_merge(base,truth_refs,truth/'targets.json')
        targets=read(truth/'targets.json')
        plan=measurement_plan(study,targets);write(research/'measurement-plan.json',plan)
        protocol_ref=reference(design_root/'design.json',ROOT)
        command_args=['--manifest',manifest,'--qualification',relative(qualified)]
        with scope(research,runtime['storage']),guard(research/'private/disk'):
            for arm in ('static','refreshed'):
                output=research/('train-'+arm)
                result,status=command('train-'+arm,canonical(args.engine,images[0],'train',relative(output),
                    [*command_args,'--arms',arm,'--seeds','20261016']),output)
                if status not in {'PASS','PARTIAL'}:raise RuntimeError('learner failed; preserve original failure')
                runs=result['result'].get('runs',[])
                if len(runs)!=1:raise ValueError('one original learner record required')
                learner=confined(output/'training',runs[0]['root']);training=runs[0]['result']
                checkpoint,weights_root=latest_checkpoint(learner,training,study=study['identity'],arm=arm)
                selected[arm]=dict(checkpoint=checkpoint,training_status=status,tokens=training['tokens'],updates=training['updates'])
                extra.append((learner/'training.json','fresh/training-'+arm+'.json'))
                extra.append((weights_root/'retained.json','fresh/retained-'+arm+'.json'))
                selected[arm]['weights_root']=relative(weights_root)
            if selected['static']['checkpoint']['initialization']!=selected['refreshed']['checkpoint']['initialization']:
                raise ValueError('paired random initial weights differ')
            recipe=json.loads(verify_artifact(base,runtime['recipe']))
            for arm in ('static','refreshed'):
                scored[arm],artifacts=evaluate_arm(args.engine,qualified,research,manifest,study,protocol_ref,
                    arm,selected[arm]['checkpoint'],confined(ROOT,selected[arm]['weights_root']),
                    visible,truth,prompt_ref,truth_ref,recipe,command)
                extra.extend(artifacts)
        workflow['status']='PASS' if all(s['status']=='PASS' for s in stages) else 'PARTIAL'
    except (Exception,KeyboardInterrupt) as exc:
        workflow['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
        workflow['error']=type(exc).__name__+': '+str(exc)[:2048]
        print(workflow['error'],file=sys.stderr,flush=True)
    finally:
        stages.append(dict(stage='controller',status=workflow['status'],exit_code=EXIT_CODES.get(workflow['status'],1),
            elapsed_seconds=time.monotonic()-started,command=['fresh-pilot-controller'],originals=[str(root/'workflow.json')]))
        record()
        if plan is not None:
            try:
                public=root/'public'
                extra.append((root/'workflow.json','fresh/workflow.json'))
                extra.extend(diagnostics(root,root/'diagnostics'))
                for name in ('fresh_pilot.py','fresh_pilot_inputs.py','fresh_pilot_report.py','fresh_pilot_feasibility.py','fresh_pilot_smoke.py','fresh_pilot_resume.py','fresh_pilot_storage.py','fresh_pilot_checkpoint.py'):
                    extra.append((HERE/name,'fresh/helper-'+name+'.txt'))
                evidence=assemble_public(public,study=study,plan=plan,targets=targets,scored=scored,stages=stages,
                    qualified=qualified,selection=dict(policy='frozen latest available, partial exposure descriptive only',
                        primary_complete=all(selected.get(a,{}).get('training_status')=='PASS' for a in ('static','refreshed')),
                        checkpoints=selected,assessment='SEALED'),extra=extra)
                report=root/'report'
                code=invoke(canonical(args.engine,images[0],'report',relative(report),
                    ['--manifest',relative(public/'fresh/study.json'),'--inputs',relative(public)]),root/'private/logs/report',seconds=1800)
                if code not in set(EXIT_CODES.values()) or not report.with_suffix('.zip').is_file():raise RuntimeError('canonical report failed')
                verification=root/'verified'
                verified=invoke(canonical(args.engine,images[0],'verify-bundle',relative(verification),
                    ['--archive',relative(report.with_suffix('.zip'))]),root/'private/logs/verify',seconds=600)
                summary=verified_archive(ROOT,report.with_suffix('.zip'),verification,code=code,verified=verified,
                    expected=evidence['software_status'])
                archive=report.with_suffix('.zip');workflow['verified_scope']=summary['selected_scope']
            except (Exception,KeyboardInterrupt) as exc:
                workflow['status']='FAIL';workflow['report_error']=type(exc).__name__+': '+str(exc)[:2048]
        if archive is None:archive=recovery(root,workflow)
        workflow['archive']=relative(archive);workflow['elapsed_seconds']=time.monotonic()-started;record()
        print(json.dumps(dict(status=workflow['status'],archive=relative(archive),scientific_status='INCONCLUSIVE' if plan else 'UNMEASURED'),indent=2),flush=True)
    return EXIT_CODES.get(workflow['status'],1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    child=commands.add_parser('role');child.add_argument('--job',required=True);child.add_argument('--output',required=True)
    smoke=commands.add_parser('smoke');smoke.add_argument('--job',required=True);smoke.add_argument('--output',required=True)
    feasible=commands.add_parser('feasibility');feasible.add_argument('--output',required=True)
    diagnostic=commands.add_parser('diagnose');diagnostic.add_argument('--run',required=True);diagnostic.add_argument('--output',required=True)
    probe=commands.add_parser('throughput');probe.add_argument('--engine',choices=('docker','podman'),required=True)
    probe.add_argument('--cpu-image',required=True);probe.add_argument('--output',required=True)
    runner=commands.add_parser('run');runner.add_argument('--engine',choices=('docker','podman'),required=True)
    runner.add_argument('--imported',default='run-state/generalization/uniform-import')
    runner.add_argument('--resume-prepared',help='original failed pre-learning run with completed preparation; keep it immutable')
    runner.add_argument('--throughput',help='original compatible PASS CPU/native throughput output')
    runner.add_argument('--cpu-image');runner.add_argument('--gpu-image');runner.add_argument('--qualified')
    runner.add_argument('--output',required=True);runner.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='throughput':
            from fresh_pilot_throughput_host import run as throughput_run
            return throughput_run(args.engine,args.cpu_image,args.output)
        if args.command=='feasibility':
            from fresh_pilot_feasibility import check
            target=confined(ROOT,args.output,exists=False)
            result=check(research_inputs(),target,workers=4)
            # The command succeeded in establishing feasibility, not qualification.
            (target/'result.json').rename(target/'feasibility.json')
            write(target/'result.json',seal(dict(status='PASS' if result['status']=='FEASIBLE' else 'BLOCKED',
                scope='source feasibility only',native_validation='NOT_RUN',hardware_validation='NOT_RUN',
                feasibility=reference(target/'feasibility.json',target))))
            return 0 if result['status']=='FEASIBLE' else 3
        if args.command=='smoke':
            from fresh_pilot_smoke import run as smoke_run
            return smoke_run(sys.modules[__name__],args.job,args.output)
        if args.command=='role':return role(args.job,args.output)
        if args.command=='diagnose':return diagnose(args.run,args.output)
        if args.dry_run:
            print(json.dumps(dict(status='NOT_RUN',reservations=RESERVATIONS,campaign_seconds=CAMPAIGN_SECONDS,
                report_seconds=REPORT_SECONDS,memory_gib=64,cpus=4,required_free_gib=224,
                resume_prepared=args.resume_prepared,assessment='SEALED',historical='DEFERRED',
                gates=('verify original feasibility/acceptance/preparation and throughput; reuse inputs; rerun smoke; full research'
                    if args.resume_prepared else 'compatible recent CPU/native throughput proof, full12-shard CPU feasibility, current-source acceptance, native preparation, paired smoke, full research'),
                training='fresh static/refreshed, seed20261016,25M target-token cap/arm,6 learner hours/arm',
                evaluation='latest available checkpoint; partial exposures descriptive; matched controls plus one shared bank/checkpoint'),indent=2));return 0
        return run(args)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
