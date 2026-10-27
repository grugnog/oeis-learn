"""Static host pilot/preflight stages. All neural/native work stays in pinned roles."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time
from .identity import seal,digest,file_hash,write,confined
from .pilot_contract import SPEC,PROFILE,ARMS,SPLITS
from .runner import CapabilityError,ResourceBreach


def link(source,target):
    if source.is_symlink() or not source.is_file():raise ValueError('regular owned pilot file required')
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError as exc:
        import errno
        if exc.errno!=errno.EXDEV:raise
        shutil.copyfile(source,target)


def checkpoint_view(source,target):
    for p in sorted(source.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in pilot state')
        if p.is_file():link(p,target/p.relative_to(source))


def learning_report(prepared,prompts,snapshots,*,purpose):
    from .pilot_analysis import coverage,paired,extrapolate,compute_estimates
    families={r['id']:r['family'] for r in prompts};strata={r['id']:r.get('strata',{}) for r in prompts};series={};count=3 if purpose=='preflight' else 6
    points=[]
    for ordinal in range(count):
        actual=next((s for s in snapshots if s['ordinal']==ordinal),None)
        points.append(actual or dict(ordinal=ordinal,tokens=None,updates=None,state='UNSTARTED',outcomes={}))
    for split,ids in prepared['roster'].items():
        for arm in ARMS:
            observations=[];baseline=None;trend=[]
            for point in points:
                rows=[]
                for key in ids:
                    x=point.get('outcomes',{}).get(key,{}).get(arm,{})
                    rows.append(dict(id=key,family=families[key],strata=strata[key],success=x.get('success',False),complete=x.get('complete',False)))
                result=coverage(ids,rows)
                stratified={}
                for dimension in sorted({k for r in rows for k in r['strata']} - {'split'}):
                    for label in sorted({r['strata'][dimension] for r in rows if dimension in r['strata']}):
                        subset=[r for r in rows if r['strata'].get(dimension)==label]
                        stratified[dimension+':'+label]=coverage([r['id'] for r in subset],subset)
                controls={}
                if arm=='equal_budget':
                    for control in ('shuffled','absent','template'):
                        comparison=[]
                        for key in ids:
                            x=point.get('outcomes',{}).get(key,{}).get(control,{})
                            comparison.append(dict(id=key,family=families[key],success=x.get('success',False),complete=x.get('complete',False)))
                        controls[control]=paired(comparison,rows)
                if baseline is None:baseline=rows
                observations.append(dict(ordinal=point['ordinal'],tokens=point['tokens'],updates=point['updates'],
                    learner_role_seconds=point.get('learner_role_seconds'),inclusive_wall_seconds=point.get('inclusive_wall_seconds'),
                    examples=point.get('examples'),unique_examples=point.get('unique_examples'),unique_training_tokens=point.get('unique_training_tokens'),
                    effective_example_epochs=point.get('examples',0)/SPEC['training'] if point.get('examples') is not None else None,
                    measurement=result,stratified=stratified,change_from_initialization=paired(baseline,rows),
                    correct_minus_controls=controls,outcomes=rows))
                if point['tokens'] is not None:trend.append(dict(tokens=point['tokens'],accuracy=result['accuracy'],coverage=result['coverage'],
                    complete_roster=digest(sorted(r['id'] for r in rows if r['complete'])),learner_role_seconds=point.get('learner_role_seconds')))
            projection=extrapolate(trend)
            if split in ('parameter_development','structural_development'):
                losses=[s.get('teacher_forcing',{}).get('splits',{}).get(split,{}).get('loss') for s in snapshots]
                values=[x for x in losses if x is not None]
                if len(values)>1 and values[-1]>min(values[:-1]):
                    projection=dict(status='UNKNOWN',reason='development teacher-forced loss worsened; overfitting warning',guaranteed_capacity_upper_bound=None)
            series[split+':'+arm]=dict(points=observations,extrapolation=projection,further_compute=compute_estimates(trend,projection,len(ids)))
    gaps=[]
    for point in sorted(snapshots,key=lambda x:x['ordinal']):
        splits=point.get('teacher_forcing',{}).get('splits',{});training=splits.get('training_panel',{})
        values={}
        for split in ('parameter_development','structural_development'):
            dev=splits.get(split,{})
            values[split]=dict(loss_gap=dev['loss']-training['loss'],token_accuracy_gap=dev['token_accuracy']-training['token_accuracy']) if (
                training.get('complete') and dev.get('complete') and training.get('loss') is not None and dev.get('loss') is not None) else None
        gaps.append(dict(ordinal=point['ordinal'],tokens=point['tokens'],gaps=values))
    return seal(dict(version='pipeline-pilot-learning-report/v1',profile=prepared.get('profile',PROFILE['identity']),purpose=purpose,
        metric='independently_verified_continuation100',series=series,
        actual_snapshots=[{k:v for k,v in s.items() if k!='outcomes'} for s in snapshots],
        teacher_forcing_gaps=gaps,training_panel_interpretation='Fixed32 examples; rolling bins describe the full sampled training stream.',
        control_attempts=dict(correct=8,shuffled=1,absent=1,template=8),control_interpretation='Shuffled/absent have one attempt; equal-budget correct/template have eight. Do not attribute the whole gap to conditioning alone.',
        no_short_test_learning_threshold=True,final_access='SEALED',
        limitations=['One seed; no seed-variance estimate.','Exposed development, not fresh finals.',
            'Conditional scenarios are not statistical confidence bounds or capacity ceilings.'],
        automatic_run_authorized=False))


class PilotStages:
    def __init__(self,adapters):
        self.a=adapters;self.prepared=None;self.manifest=None;self.state=None;self.trainer=None
        self.snapshots=[];self.calibrations=[];self.purpose='preflight';self.deadline=None;self.interrupted=[];self.archives=None
        self.learner_seconds=0.;self.setup_seconds=0.;self.compatibility=None
        self.started=time.monotonic()

    def invoke(self,root,label,mode,files,*,seconds,ordinal=0,worker_seconds=0):
        began=time.monotonic();a=self.a;host=a.host(a.args.engine)
        if self.deadline is not None and time.monotonic()+seconds+5>self.deadline:
            raise TimeoutError('remaining original pilot parent cannot fit next frozen role')
        from .resources import observe,resolve,reserve
        from .soak import peak
        measured=observe();measured['limits'].append(host.policy['memory_bytes'])
        policy=resolve(**measured,profile='workstation90')
        gpu=mode in ('pilot-train','pilot-generate');reservation=reserve(policy,peak(16) if gpu else 32<<30)
        if reservation['reserved_bytes']>.9*policy['ceiling_bytes']:raise CapabilityError('pilot requires ten percent unreserved memory')
        if host.policy['cpus']<6:raise CapabilityError('two Torch and four native equivalents required')
        source=root/'private'/('input-'+label);source.mkdir(parents=True)
        for name,path in files.items():
            if name in ('checkpoints','reference-cache'):checkpoint_view(path,source/name)
            else:link(path,source/name)
        target=root/label
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        target_image=a.images.images[(a.args.engine,'gpu' if gpu else 'qualification')]
        role_seconds=seconds-(time.monotonic()-began)-10
        if role_seconds<=20:raise TimeoutError('pilot role setup consumed its reservation')
        config=dict(version='pipeline-role/v1',source=digest(collect_snapshot(a.root)),contract=a.contract['identity'],mode=mode,
            options=dict(seconds=role_seconds-5,memory_ceiling=policy['ceiling_bytes'],render_node=host.policy.get('render_node'),
                purpose=self.purpose,ordinal=ordinal,worker_seconds=worker_seconds,image=target_image,interrupted=list(self.interrupted)))
        write(source/'expected.json',config)
        write(target/'resource-reservation.json',dict(policy=policy,reservation=reservation))
        from .benchmarks import legacy
        try:
            with host.gpu_lock() if gpu else nullcontext():
                legacy(a.root).container_stage(host,target_image,source,target,mode=mode,seconds=role_seconds,gpu=gpu,
                    argv=['python','-m','oeis_learn.pipeline_validation.role',mode])
            result=json.loads((target/'summary.json').read_text())
            if mode!='pilot-prepare' and result.get('source')!=config['source']:raise ResourceBreach('pilot role source binding changed')
            if self.manifest and mode!='pilot-prepare' and result.get('work')!=self.manifest['identity']:raise ResourceBreach('pilot work binding changed')
            if gpu:
                from .model_sizes import EXECUTION
                if result.get('execution')!=EXECUTION:raise ResourceBreach('pilot numerical execution policy changed')
            if result.get('fatal'):raise ResourceBreach(result['error'])
            if result.get('status')=='FAIL':raise RuntimeError(result['error'])
            if not result.get('resources',{}).get('qualified'):raise ResourceBreach('missing qualified pilot resource audit')
            if result['resources']['role_peak_upper_bytes']>reservation['peak_bytes']:
                raise ResourceBreach('pilot live allocation exceeded frozen reservation')
            return result,target
        finally:
            # These are this invocation's link views, not the operator's source inputs.
            shutil.rmtree(source)

    def train(self,root,ordinal,*,seconds,worker_seconds):
        began=time.monotonic()
        files={n:self.prepared/n for n in ('prepared.json','recipe.json','examples.json')}
        if self.compatibility is not None:files['compatibility.json']=self.compatibility
        if self.state:
            files['checkpoints']=self.state/'checkpoints';files['selected.json']=self.state/'selected.json'
        result,target=self.invoke(root,'train-'+str(ordinal),'pilot-train',files,seconds=seconds,
            ordinal=ordinal,worker_seconds=worker_seconds)
        if result['status']!='COMPLETE':raise RuntimeError('pilot learner role incomplete; original allowance not renewed')
        trainer=result['trainer']
        result['learner_role_seconds']=time.monotonic()-began if ordinal else 0.
        if ordinal:self.learner_seconds+=result['learner_role_seconds']
        else:self.setup_seconds+=time.monotonic()-began
        if self.trainer and self.trainer['final_parameter_digest']!=trainer['initial_parameter_digest']:
            raise ResourceBreach('pilot resumed weights differ from previous full checkpoint')
        previous=self.state;self.state=target/'private/trainer';self.trainer=trainer
        # Keep a diagnostic weights-only milestone; it cannot authorize learning resume.
        pointer=trainer['checkpoint'];checkpoint=json.loads((self.state/'checkpoints'/pointer['path']).read_text())
        snapshot=root/'private/snapshots'/str(ordinal);snapshot.mkdir(parents=True)
        link(self.state/'checkpoints'/checkpoint['files']['weights']['path'],snapshot/'weights.pt')
        write(snapshot/'checkpoint.json',dict(pointer=pointer,record=checkpoint,authority='weights-only diagnostic; nonresumable'))
        if previous is not None:shutil.rmtree(previous)
        return result,target

    def evaluate(self,root,ordinal,*,generate_seconds=6000,score_seconds=1500):
        from oeis_learn.fresh.common import write as replace_json
        settings=json.loads((self.state/'settings-for-evaluation.json').read_text())
        settings['checkpoint']=self.trainer['checkpoint']
        descriptor=root/'private'/('evaluation-settings-'+str(ordinal)+'.json');replace_json(descriptor,settings)
        started=time.monotonic()
        generated,gen=self.invoke(root,'generate-'+str(ordinal),'pilot-generate',
            {'settings.json':descriptor,'checkpoints':self.state/'checkpoints','prompts.json':self.prepared/'prompts.json'},
            seconds=generate_seconds,ordinal=ordinal)
        generation_seconds=time.monotonic()-started
        settings['device']='cpu';settings['pilot_selections_sha256']=file_hash(gen/'selections.json')
        replace_json(descriptor,settings)
        began=time.monotonic()
        scored,score=self.invoke(root,'score-'+str(ordinal),'pilot-score',
            {'settings.json':descriptor,'selections.json':gen/'selections.json','truth.json':self.prepared/'private/truth.json'},
            seconds=score_seconds,ordinal=ordinal)
        scoring_seconds=time.monotonic()-began
        attempts=json.loads((gen/'attempt-index.json').read_text())
        cost_complete=attempts['states'].get('COMMITTED',0)+attempts['states'].get('INCOMPLETE',0)==attempts['expected']
        row=seal(dict(complete=cost_complete,scientific_complete=scored['complete'],seconds=generation_seconds+scoring_seconds,
            generation_seconds=generation_seconds,scoring_seconds=scoring_seconds,
            projected_all_selected_scoring_seconds=max(scoring_seconds,1776*scored['selected_program_cost_seconds']),rows=scored['rows'],
            source=scored['source'],work=scored['work'],checkpoint=self.trainer['checkpoint']))
        self.calibrations.append(row)
        point=dict(ordinal=ordinal,tokens=self.trainer['pilot']['state']['tokens'],updates=self.trainer['update'],
            examples=self.trainer['pilot']['state']['examples'],unique_examples=self.trainer['pilot']['state']['unique_examples'],
            unique_training_tokens=self.trainer['pilot']['state']['unique_tokens'],
            repeated_training_tokens=self.trainer['pilot']['state']['tokens']-self.trainer['pilot']['state']['unique_tokens'],
            repeated_examples=self.trainer['pilot']['state']['examples']-self.trainer['pilot']['state']['unique_examples'],
            effective_token_epochs=self.trainer['pilot']['state']['tokens']/self.manifest['audit']['tokens']['training'],
            inclusive_wall_seconds=time.monotonic()-self.started,
            learner_role_seconds=None if self.interrupted else self.learner_seconds,
            initialization_role_seconds=self.setup_seconds,evaluation_role_seconds=generation_seconds+scoring_seconds,
            checkpoint=self.trainer['checkpoint'],parameter_digest=self.trainer['final_parameter_digest'],
            teacher_forcing=json.loads((self.state/'pilot-validation.json').read_text()),
            outcomes=json.loads((score/'outcomes.json').read_text()),state='COMPLETE')
        from .pilot_runtime import bounded_write
        bounded_write(root/('snapshot-'+str(ordinal)+'.json'),point,6<<20);self.snapshots.append(point)
        for path in (gen/'private',score/'private'):
            if path.exists():shutil.rmtree(path)
        return point

    def prepare(self,stage,output):
        a=self.a
        if not a.args.inputs:raise CapabilityError('pilot requires approved existing --inputs for exposed OEIS development')
        parent=confined(a.root,a.args.inputs)
        manifest=json.loads((parent/'prepared.json').read_text())
        from oeis_learn.synthesis.contracts import identity
        if (manifest.get('version')!='bigint-inputs/v1' or manifest.get('identity')!=identity({k:v for k,v in manifest.items() if k!='identity'})
            or manifest['cohorts']['development']['count']!=8):raise ValueError('compatible eight-row development lineage required')
        # Copy only development views; final files are neither read nor mounted.
        source=output/'private/approved-development';source.mkdir(parents=True)
        files={}
        for name,relative in [('oeis-prompts.json','private/oeis-development-prompts.json'),('oeis-truth.json','private/oeis-development.json')]:
            path=confined(parent,relative);ref=manifest['files'][relative]
            if file_hash(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:raise ValueError('approved development input changed')
            shutil.copyfile(path,source/name);files[name]=source/name
        lineage=seal(dict(version='pipeline-pilot-exposed-development/v1',prepared=manifest['identity'],
            files={name:{'sha256':file_hash(path),'bytes':path.stat().st_size} for name,path in files.items()},
            count=8,exposure='Previously exposed development; accepted engineering soak and input preparation. Not a fresh final cohort.'))
        write(source/'oeis-lineage.json',lineage);files['oeis-lineage.json']=source/'oeis-lineage.json'
        from .pilot_cache_v2 import binding
        from .pilot_cache import binding as legacy_binding
        from oeis_learn.data.bigint_programs import BigintSampler
        from oeis_learn.synthesis.contracts import ResourceProfile
        sampler=BigintSampler(SPEC['seed']);limits=ResourceProfile()
        expected=binding(sampler,limits);legacy_expected=legacy_binding(sampler,limits);seeds=[]
        for previous in a.args.reuse:
            prior=confined(a.root,previous)/'stages/pilot.prepare/prepare'
            index=prior/'private/reference-cache/index.json';proof=prior/'preparation.json'
            if index.is_file() and proof.is_file():
                ref=json.loads(proof.read_text()).get('reference_cache',{})
                if ref.get('binding') in (expected,legacy_expected):
                    if file_hash(index)!=ref['index_sha256']:raise ValueError('retained reference cache index changed')
                    producer=json.loads(proof.read_text())
                    receipt=json.loads((prior.parent/'receipt.json').read_text())
                    seeds.append((ref['count'],index.parent,producer,receipt))
        if seeds:
            count,cache_root,producer,receipt=max(seeds,key=lambda x:x[0])
            files['reference-cache']=cache_root
            # Preserve the complete public producer proof; only its compact
            # lineage goes into the CPU role. No producer PASS is imported.
            retained=dict(preparation=producer,receipt=receipt)
            write(output/'reference-reuse.json',retained)
            origin=producer.get('reference_provenance',{}).get('origin') or dict(
                source=producer['source'],receipt=receipt['identity'],
                elapsed_seconds=receipt.get('observations',{}).get('elapsed_seconds'),
                cost_status=producer.get('preparation_cost_status','ENCLOSING_ROLE_ONLY'),
                provenance=producer.get('provenance','Independent reference in production preparation; see retained public proof.'))
            provenance=seal(dict(version='pipeline-pilot-reference-provenance/v2',binding=expected,input_binding=producer['reference_cache']['binding'],
                index_sha256=file_hash(cache_root/'index.json'),producer=digest(retained),
                receipt=receipt['identity'],state=receipt['state'],origin=origin,authority='RAW_REFERENCE_ONLY'))
            write(source/'reference-provenance.json',provenance)
            files['reference-provenance.json']=source/'reference-provenance.json'
        for engine in a.args.pipeline_engines:
            if a.receipts.get('cpu.'+engine+'.semantic',{}).get('state')=='PASS':
                for path in (output.parent/('cpu.'+engine+'.semantic')).rglob('pytest'):
                    if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
        result,target=self.invoke(output,'prepare','pilot-prepare',files,seconds=stage.seconds-30)
        self.prepared=target/'private/prepared'
        from .pilot_data import check
        self.manifest=check(self.prepared,hidden=True)
        write(output/'prepared-index.json',self.manifest)
        return result

    def report(self,output):
        from .pilot_cost import forecast
        from .pilot_contract import decision
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        a=self.a;source=digest(collect_snapshot(a.root));work=self.manifest['identity'] if self.manifest else digest('unprepared')
        if self.archives is None:
            self.archives=[]
            if len(self.calibrations)==3:
                from .supervision import launch
                import sys
                code='import sys;sys.path.insert(0,'+repr(str(a.root/'src'))+');from oeis_learn.pipeline_validation.pilot_archive import main;main()'
                try:
                    launch([sys.executable,'-S','-c',code,'--output',str(output/'private/archive-probes'),'--source',source],output/'archive-supervision',seconds=540,stream_cap=8<<20)
                    self.archives=json.loads((output/'private/archive-probes/observations.json').read_text())
                except (OSError,RuntimeError,TimeoutError) as exc:write(output/'archive-calibration-failure.json',{'error':str(exc),'complete':False})
            write(output/'archive-calibration.json',self.archives)
        cost=forecast(self.calibrations,source=source,work=work,archives=self.archives);write(output/'cost.json',cost)
        from .pilot_contract import qualification_requirements
        required=sorted(qualification_requirements(a.args.engine))
        primary=all(a.receipts.get(k,{}).get('state') in ('PASS','REUSED') for k in required)
        from .resources import execution_identity
        host=a.hosts.get(a.args.engine)
        qualification=seal(dict(version='pipeline-pilot-qualification/v1',source=source,profile=PROFILE['identity'],work=work,
            image={k:a.images.images.get((a.args.engine,k)) for k in ('gpu','qualification')},engine=a.args.engine,
            resources=execution_identity(host.policy['pipeline_resources']) if host else None,cpus=host.policy['cpus'] if host else None,required=required,
            receipts={k:{field:r[field] for field in ('state','identity') if field in r} for k in required for r in [a.receipts.get(k,{})]},
            state='PASS' if primary else 'BLOCKED'))
        write(output/'qualification.json',qualification)
        if self.manifest:
            report=learning_report(self.manifest,json.loads((self.prepared/'prompts.json').read_text()),self.snapshots,purpose=self.purpose)
            from .pilot_runtime import bounded_write
            bounded_write(output/'learning.json',report,16<<20)
            ref=lambda p:dict(path=str(p.relative_to(a.root)),sha256=file_hash(p))
            candidate=decision(source=source,prepared=ref(self.prepared/'prepared.json'),
                qualification=ref(output/'qualification.json'),cost=ref(output/'cost.json'))
            write(output/'long-run-draft.json',candidate)
        result=seal(dict(version='pipeline-pilot-preflight-report/v1',source=source,work=work,
            primary_plumbing='PASS' if primary else 'BLOCKED',cost=cost['status'],
            snapshots=len(self.snapshots),learning_success_threshold=None,
            readiness='REVIEW_REQUIRED' if primary and cost['within_budget'] else 'BLOCKED',
            original_adoption='INCONCLUSIVE_RETAIN_BASELINE',portability='PENDING' if any(r.get('state') not in ('PASS','REUSED') for k,r in a.receipts.items() if 'docker' in k) else 'NOT_QUALIFIED_BY_PILOT',
            long_run_admitted=False,final_access='SEALED',automatic_run_authorized=False))
        write(output/'readiness.json',result);return result

    def finalize(self,output,invocation,receipts):
        self.a.receipts=dict(receipts)
        previous=invocation/'stages/pilot.report/readiness.json'
        if previous.exists():
            result=json.loads(previous.read_text())
            write(output/'report-reference.json',dict(path=str(previous),sha256=file_hash(previous)))
        else:result=self.report(output)
        return dict(result,engineering=result['primary_plumbing'],research='NOT_EXECUTED')

    def __call__(self,stage,output):
        if stage.id=='pilot.prepare':return self.prepare(stage,output)
        if stage.id=='pilot.report':return self.report(output)
        if stage.id=='pilot.session':
            self.deadline=time.monotonic()+stage.seconds
            self.train(output,0,seconds=600,worker_seconds=0);self.evaluate(output,0)
            for ordinal in (1,2):
                self.train(output,ordinal,seconds=1020,worker_seconds=900);self.evaluate(output,ordinal)
            write(output/'session.json',seal(dict(version='pipeline-pilot-session/v1',snapshots=3,
                purpose='preflight',work=self.manifest['identity'],source=self.manifest['source'],
                actual_updates=self.trainer['update'],actual_tokens=self.trainer['pilot']['state']['tokens'],
                no_learning_threshold=True)))
            return dict(status='COMPLETE',snapshots=3)
        raise ValueError('unknown static pilot stage')


def study_main(args,root,host_factory,record,source):
    from .pilot_contract import parse_decision
    from .runner import execute
    from .registry import Stage
    if record.get('version')=='pipeline-pilot-study/v2':
        from .longrun_contract import parse_decision
    result=parse_decision(record,root=root,source=source)
    if args.scope_report:
        print(json.dumps(result,indent=2));return 0
    if result['status']!='ADMITTED':
        return execute([Stage('pilot.admission',6,available=False,reason=', '.join(result['missing']))],
            confined(root,args.output,exists=False),adapter=lambda *_:None,binding={'source':source,'study':record['identity']},
            cap=172800,reserve=1800,contract=record)
    from .pilot_study import run
    return run(args,root,host_factory,record)
