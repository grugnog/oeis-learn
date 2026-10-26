"""Explicitly admitted static pilot, with immutable budgets and own-state recovery."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from .identity import reference,confined,digest,seal,write,file_hash
from .pilot_contract import SPEC,PROFILE
from .pilot import PilotStages,learning_report
from .registry import Stage
from .runner import execute,ResourceBreach


def stages(cost,ledger=None):
    result=[Stage('pilot.current-images',6,seconds=1200,adapter='pilot-current')]
    hours=SPEC['snapshot_hours']
    for i,hour in enumerate(hours):
        previous='pilot.current-images' if i==0 else 'pilot.train-'+str(i-1)
        seconds=600 if i==0 else int((hour-hours[i-1])*3600)
        result.append(Stage('pilot.train-'+str(i),6,(previous,),seconds=seconds,adapter='pilot-study-train'))
        result.append(Stage('pilot.evaluate-'+str(i),6,('pilot.train-'+str(i),),
            seconds=cost['generator_role_seconds']+cost['scorer_role_seconds'],adapter='pilot-study-evaluate'))
    if ledger is not None:
        from dataclasses import replace
        result=[replace(s,seconds=1) if s.id in ledger.value['slots'] else s for s in result]
    return result


class Study:
    def __init__(self,pilot,ledger,record,cost,root):
        self.p=pilot;self.ledger=ledger;self.record=record;self.cost=cost;self.root=root;self.receipts={}
        self.p.snapshots=[r['result']['snapshot'] for k,r in ledger.value['slots'].items()
            if k.startswith('pilot.evaluate-') and r['state']=='COMPLETE']

    def previous_state(self):
        candidates=[]
        for key,row in self.ledger.value['slots'].items():
            if key.startswith('pilot.train-'):
                ordinal=int(key.rsplit('-',1)[1]);folder=Path(row['output'])/'private/trainer'
                if (folder/'checkpoints/latest.json').is_file():candidates.append((ordinal,row,folder))
        if not candidates:return
        ordinal,row,folder=max(candidates,key=lambda r:r[0])
        self.p.state=folder
        if row['state']=='COMPLETE':
            result=row['result'];self.p.trainer=result['trainer']
        else:
            self.ledger.recover(row['key'])
            descriptor=folder/'settings-for-evaluation.json'
            if not descriptor.exists():descriptor=folder.parent/'worker-input/settings.json'
            # CPU-only audit of every payload before any later GPU work.
            recovery_root=self.root/'recovery';recovery_root.mkdir(parents=True,exist_ok=True)
            key='recover-'+str(ordinal)
            if key in self.ledger.value['slots']:
                previous=self.ledger.value['slots'][key]
                if previous['state']!='COMPLETE':raise ValueError('recovery itself was interrupted; no automatic retry')
                self.p.trainer=previous['result']['trainer']
            else:
                self.ledger.reserve(key,seconds=300,category='setup',output=recovery_root/'recover')
                try:
                    result,target=self.p.invoke(recovery_root,'recover','pilot-recover',
                        {'settings.json':descriptor,'checkpoints':folder/'checkpoints'},seconds=300,ordinal=ordinal)
                    self.p.trainer=result['trainer'];self.ledger.finish(key,'COMPLETE',result=result)
                except BaseException as exc:
                    self.ledger.finish(key,'INTERRUPTED',result={'fatal':isinstance(exc,ResourceBreach),'error':str(exc)})
                    raise

    def __call__(self,stage,output):
        p=self.p;key=stage.id
        if stage.adapter=='pilot-current':
            invocation=len(self.ledger.value['invocations'])-1
            runtime_key='current-images-'+str(invocation)
            self.ledger.reserve(runtime_key,seconds=stage.seconds,category='qualification',output=output)
            try:
                from oeis_learn.container_validation import inspect_current_image
                host=p.a.host(p.a.args.engine)
                for kind in ('qualification','gpu'):
                    inspect_current_image(host,p.a.images.images[(p.a.args.engine,kind)],output/kind,seconds=300)
                self.ledger.finish(runtime_key,'COMPLETE',result={'current_source':self.record['source']})
                self.previous_state()
            except BaseException as exc:
                if self.ledger.value['slots'][runtime_key]['state']=='ACTIVE':
                    self.ledger.finish(runtime_key,'INTERRUPTED',result={'fatal':isinstance(exc,ResourceBreach),'error':str(exc)})
                raise
            return {'current_source':self.record['source'],'actual_package_inspection':True}
        ordinal=int(key.rsplit('-',1)[1]);old=self.ledger.value['slots'].get(key)
        if old is not None:
            if old['state']=='ACTIVE':old=self.ledger.recover(key)
            if old['state']=='COMPLETE':
                if stage.adapter=='pilot-study-evaluate':
                    point=old['result']['snapshot']
                    if not any(x['ordinal']==ordinal for x in p.snapshots):p.snapshots.append(point)
                # Original producer paths and times remain explicit.
                write(output/'reused.json',old)
                return dict(reused_from=old['output'],original_slot=key,original_seconds=old['seconds'])
            self.ledger.recover(key)
            if stage.adapter=='pilot-study-train':
                if ordinal==0:raise ValueError('interrupted fresh baseline cannot be silently replaced')
                p.interrupted=sorted(set(p.interrupted)|{ordinal})
                # Safety recovery can PASS; its lost work is explicitly incomplete.
                return dict(recovery_boundary='retained complete state audited',work_complete=False,interrupted=True)
            return dict(stage_state='PARTIAL',work_complete=False,reason='original evaluation interrupted; allowance remains charged')
        if stage.adapter=='pilot-study-train':
            category='setup' if ordinal==0 else 'learner';target=output/('train-'+str(ordinal))
        else:category='evaluation';target=output
        self.ledger.reserve(key,seconds=stage.seconds,category=category,output=target)
        try:
            if stage.adapter=='pilot-study-train':
                p.interrupted=sorted(int(k.rsplit('-',1)[1]) for k,r in self.ledger.value['slots'].items()
                    if k.startswith('pilot.train-') and r['state']=='INTERRUPTED')
                result,target=p.train(output,ordinal,seconds=stage.seconds-10,worker_seconds=0 if ordinal==0 else stage.seconds-150)
                stored=dict(trainer=result['trainer'],source=result['source'],work=result['work'],output=str(target))
            else:
                train_slot=self.ledger.value['slots']['pilot.train-'+str(ordinal)]
                if train_slot['state']!='COMPLETE':
                    self.ledger.finish(key,'INTERRUPTED',result={'reason':'training milestone incomplete'})
                    return dict(stage_state='PARTIAL',work_complete=False,reason='no complete milestone; no hidden-best checkpoint substitution')
                # A prior failed evaluation must never score a newer checkpoint.
                if p.trainer['checkpoint']!=train_slot['result']['trainer']['checkpoint']:
                    raise ValueError('evaluation cannot substitute a different actual checkpoint')
                point=p.evaluate(output,ordinal,generate_seconds=self.cost['generator_role_seconds'],score_seconds=self.cost['scorer_role_seconds'])
                stored=dict(snapshot=point)
            self.ledger.finish(key,'COMPLETE',result=stored)
            return dict(ordinal=ordinal,work_complete=True,actual_source=self.record['source'])
        except BaseException as exc:
            if self.ledger.value['slots'][key]['state']=='ACTIVE':
                self.ledger.finish(key,'INTERRUPTED',result={'fatal':isinstance(exc,ResourceBreach),'error':type(exc).__name__+': '+str(exc)})
            raise

    def finalize(self,output,invocation,receipts):
        p=self.p
        report=learning_report(p.manifest,json.loads((p.prepared/'prompts.json').read_text()),p.snapshots,purpose='longrun')
        write(output/'learning.json',report);write(output/'debits.json',self.ledger.value)
        complete=len(p.snapshots)==6 and all(r['state']=='COMPLETE' for k,r in self.ledger.value['slots'].items() if k.startswith('pilot.'))
        measured_complete=all(x['measurement']['coverage']>=.95 for v in report['series'].values() for x in v['points'])
        result=seal(dict(version='pipeline-pilot-study-report/v1',study=self.record['identity'],source=self.record['source'],
            research='COMPLETE' if complete and measured_complete else 'INCONCLUSIVE',
            learning='MEASURED_DEVELOPMENT_REVIEW_REQUIRED',snapshots=len(p.snapshots),
            original_adoption='INCONCLUSIVE_RETAIN_BASELINE',historical_T096='NEGATIVE 7730/14400 unchanged',
            long_run_admission='ADMITTED_EXACT_PILOT_ONLY',remaining_seconds=self.ledger.remaining(),
            final_access='SEALED',automatic_run_authorized=False))
        write(output/'report.json',result);return result


def run(args,root,host_factory,record):
    """Called only after complete public authority validation in pilot.study_main."""
    from oeis_learn.fresh.common import lock
    from .pilot_ledger import PilotLedger
    from .pilot_data import check
    from .resources import observe,resolve,execution_identity
    from oeis_learn.container_runtime import Engine
    from .disk import DiskWatchdog
    q=json.loads(reference(root,record['qualification']).read_text());cost=json.loads(reference(root,record['cost']).read_text())
    prepared=reference(root,record['prepared']).parent
    metadata=check(prepared)
    if metadata['identity']!=cost['work'] or metadata['source']!=record['source']:raise ValueError('qualified prepared work/source mismatch')
    if args.engine not in ('auto',q['engine']) or args.image not in (None,q['image']['gpu']):raise ValueError('qualified engine/image mismatch')
    observed=observe();observed['limits'].append(Engine(q['engine']).facts()['memory_bytes'])
    policy=resolve(**observed,profile='workstation90',lower=int(args.memory_gib*(1<<30)) if args.memory_gib else None)
    if execution_identity(policy)!=q['resources']:raise ValueError('qualified pilot resource envelope changed')
    options=copy.copy(args);options.engine=q['engine'];options.image=q['image']['gpu']
    options.memory_gib=policy['ceiling_bytes']/(1<<30)
    if args.cpus is not None and args.cpus!=q['cpus']:raise ValueError('qualified CPU quota changed')
    options.cpus=q['cpus'];options.pipeline_engines=[q['engine']]
    if options.cpus<6:raise ValueError('qualified CPU budget unavailable')
    host=host_factory(options);host.select_gpu()
    budgets=dict(global_seconds=172800,build_per_engine_seconds=0,phase_seconds={},archive_seconds=900,cleanup_seconds=900,
        stream_bytes=8<<20,transient_bytes=16<<30,trace_bytes=128<<20,zip_bytes=512<<20,mandatory_reserve_bytes=16<<20,
        free_disk_reserve_bytes=5<<30,bundle_members=4097)
    host.policy.update(pipeline_resources=policy,pipeline_budgets=budgets,memory_bytes=policy['ceiling_bytes'],cpu_memory_bytes=policy['ceiling_bytes'])
    wrapper=seal(dict(version='pipeline-pilot-execution/v1',scope='learning-pilot-study',budgets=budgets,decision=record))
    owner=SimpleNamespace(args=options,root=Path(root),contract=wrapper,hosts={q['engine']:host},
        host=lambda name:host,images=SimpleNamespace(images={(q['engine'],k):v for k,v in q['image'].items()}))
    pilot=PilotStages(owner);pilot.purpose='longrun';pilot.prepared=prepared;pilot.manifest=metadata
    output=confined(root,args.output,exists=False)
    ancestor=output.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if ancestor.stat().st_dev!=cost['archives'][0]['filesystem_device']:raise ValueError('pilot export filesystem differs from measured cost scope')
    base=Path(root)/'run-state/pipeline-pilots';held=base/record['identity'].split(':')[1];held.mkdir(parents=True,exist_ok=True)
    with lock(held/'controller.lock'):
        ledger=PilotLedger(base,record);maximum=ledger.remaining()
        if args.max_seconds is not None:
            if not 0<args.max_seconds<=172800:raise ValueError('pilot ceiling may only be lowered')
            maximum=min(maximum,args.max_seconds)
        for path in ledger.value['invocations']:confined(root,path)
        if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh pilot recovery output required')
        ledger.invocation(output);pilot.deadline=ledger.value['expires']
        adapter=Study(pilot,ledger,record,cost,output)
        # The outer watch includes linked old state/data across recovery invocations.
        with DiskWatchdog(held,held/'durable-audit',cap=32<<30,extra_roots=[prepared,*map(Path,ledger.value['invocations'])],free_reserve=5<<30):
            return execute(stages(cost,ledger),output,adapter=adapter,binding={'source':record['source'],'study':record['identity']},
                cap=maximum,reserve=1800,contract=wrapper)
