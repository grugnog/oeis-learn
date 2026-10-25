"""One host-owned full-scale engineering collection; no Torch on the host."""
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import time
from .identity import seal,digest,file_hash,write,number
from .soak_contract import SPEC,data_contract,select_batch,lease,remaining
from .model_sizes import PROFILES,EXECUTION,peak_estimate,parameters


def prepare_inputs(recipe,examples,output,**binding):
    if recipe['track']['kind']!='generic_only' or recipe['learning']['curriculum'] or recipe['learning']['replay']:
        raise ValueError('existing strict generic strategy required for hardware soak')
    # Retain the entire prepared visible view; remove unrelated source metadata.
    fields={'id','indices','values','tokens','family','stage','horizon_indices','source','oracle_source','strata'}
    visible={k:[{n:v for n,v in row.items() if n in fields} for row in examples[k]]
        for k in ('training','parameter_development','structural_development')}
    data=data_contract(visible);recipe=copy.deepcopy(recipe);recipe.pop('identity',None)
    recipe.update(version='bigint-bootstrap/v2',model_profile=PROFILES['m100'])
    recipe['model'].update(d_model=768,heads=12,encoder_layers=6,decoder_layers=6,ff=3072)
    recipe['learning']['cost_preflight']=None
    from .benchmarks import BASE_POLICY
    recipe['pipeline']=dict(BASE_POLICY,workers=4,overlap=True,batch_size=16,bookkeeping='hoisted/v1')
    m=seal(dict(version='pipeline-soak-input/v1',source=binding['source'],binding=binding,
        data=data,recipe=digest(recipe),spec=SPEC,execution=EXECUTION))
    target=Path(output)/'input';target.mkdir(parents=True)
    write(target/'work.json',dict(manifest=m,recipe=recipe,examples=visible))
    write(Path(output)/'manifest.json',m)
    return m


def artifact_bound(batch):
    # Three full checkpoint generations (params+AdamW), serialization and conservative cache/input/evidence.
    n=parameters(PROFILES['m100'])['total']
    return 3*12*n+(1<<30)+(512<<20)+(512<<20)


def peak(batch):
    p=PROFILES['m100'];fixed=(8<<30)+32*parameters(p)['total']
    return fixed+(peak_estimate(p)-fixed)*batch//8


class SoakStages:
    def __init__(self,adapters):
        self.a=adapters;self.prepared=None;self.work_hash=None;self.refroot=None;self.refs={};self.choice=None;self.envelope=None

    def invoke(self,stage,output,*,kind,batch=8,instrumented=False,seconds=None,resume=False,window=0,parent=None,role_root=None):
        a=self.a
        from .benchmarks import legacy
        from .runner import CapabilityError,ResourceBreach
        from .resources import observe,resolve,reserve
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        if file_hash(self.prepared/'input/work.json')!=self.work_hash:raise ValueError('prepared work changed')
        inputs=output/'input';inputs.mkdir(parents=True,exist_ok=True)
        for name,source in [('work.json',self.prepared/'input/work.json'),*[(n,self.refroot/r) for n,r in self.refs.items()]]:
            if name!='work.json' and file_hash(source)!=self.ref_hashes[name]:raise ValueError('frozen reference/initialization changed')
            if not (inputs/name).exists():os.link(source,inputs/name)
        host=a.host(a.args.engine);facts=observe();facts['limits'].append(host.policy['memory_bytes'])
        capacity_batch=32 if kind=='joint' else batch
        try:
            policy=resolve(**facts,profile=a.contract['resource_policy']);reservation=reserve(policy,peak(capacity_batch))
            if reservation['reserved_bytes']>.9*policy['ceiling_bytes']:raise ValueError('ten percent unreserved capacity required')
        except ValueError as exc:raise CapabilityError(str(exc)) from exc
        if artifact_bound(batch)>a.contract['budgets']['transient_bytes']-(512<<20):raise CapabilityError('checkpoint/evidence reservation exceeds transient cap')
        host=copy.copy(host);host.policy=dict(host.policy,memory_bytes=policy['ceiling_bytes'],cpu_memory_bytes=policy['ceiling_bytes'],cpus=min(host.policy['cpus'],policy['cpus']))
        envelope=dict(cpus=host.policy['cpus'],ceiling_bytes=host.policy['memory_bytes'],render_node=host.policy['render_node'])
        if self.envelope is None:self.envelope=envelope
        elif self.envelope!=envelope:raise CapabilityError('frozen whole-role CPU/memory/device envelope changed')
        if host.policy['cpus']<6:raise CapabilityError('two Torch plus four native equivalents required')
        target=role_root or output/'role';target.mkdir(parents=True,exist_ok=True)
        config=dict(version='pipeline-role/v1',source=digest(collect_snapshot(a.root)),contract=a.contract['identity'],mode='soak-trial',
            options=dict(kind=kind,batch_size=batch,seconds=seconds or (stage.seconds-30),instrumented=instrumented,
                memory_ceiling=host.policy['memory_bytes'],render_node=host.policy['render_node'],
                choice=self.choice['identity'] if self.choice else self.work['identity'],resume=resume,window=window,lease=parent))
        write(inputs/'expected.json',config,exclusive=False)
        write(target/'resource-reservation.json',dict(policy=policy,reservation=reservation,artifact_upper_bytes=artifact_bound(batch)))
        from contextlib import nullcontext
        with host.gpu_lock():
            legacy(a.root).container_stage(host,a.images.images[(a.args.engine,'gpu')],inputs,target,mode='soak-trial',
                seconds=seconds or stage.seconds,gpu=True,argv=['python','-m','oeis_learn.pipeline_validation.role','soak-trial'])
        result=json.loads((target/'summary.json').read_text())
        if result.get('fatal'):raise ResourceBreach(result.get('error','soak resource breach'))
        if result.get('status')=='FAIL':raise RuntimeError(result.get('error','soak role failure'))
        if (result.get('execution')!=EXECUTION or result.get('resources',{}).get('qualified') is not True
            or result.get('source')!=config['source'] or result.get('work')!=self.work['identity']
            or not result.get('device',{}).get('hip')):raise ValueError('actual source/device/resource/execution evidence required')
        if kind!='window' and (target/'private').exists():shutil.rmtree(target/'private',ignore_errors=False)
        return result

    def __call__(self,stage,output):
        a=self.a
        if stage.id=='soak.report':
            from .soak_report import report
            r=report(a.receipts,root=output.parent);write(output/'soak-report.json',r);return dict(r,stage_state='PASS' if r['sustained_operation']=='PASS' else 'FAIL' if r['sustained_operation']=='FAIL' else 'INCONCLUSIVE')
        from .benchmarks import legacy
        from .runner import CapabilityError
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        host=a.host(a.args.engine);source=digest(collect_snapshot(a.root));image=a.images.images[(a.args.engine,'gpu')]
        if stage.id=='soak.prepare':
            if not a.args.run or not a.args.inputs:raise CapabilityError('soak requires existing completed --run and approved --inputs')
            host.select_gpu();self.prepared=output/'prepared';self.prepared.mkdir()
            self.work=legacy(a.root).prepare(a.root/a.args.run,a.root/a.args.inputs,self.prepared,
                image=image,source=source,render_node=host.policy['render_node'],soak=True)
            self.work_hash=file_hash(self.prepared/'input/work.json')
            # Successful test temporary trees are owned by this invocation, excluded from receipts/bundles.
            for engine in a.args.pipeline_engines:
                root=output.parent/('cpu.'+engine+'.semantic')
                if a.receipts.get('cpu.'+engine+'.semantic',{}).get('state')=='PASS':
                    for path in root.rglob('pytest'):
                        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
            return dict(workload=self.work['identity'],data=self.work['data'],profile=PROFILES['m100'],fresh=True)
        if stage.id=='soak.reference':
            inputs=output/'input';inputs.mkdir();os.link(self.prepared/'input/work.json',inputs/'work.json')
            cfg=dict(version='pipeline-role/v1',source=source,contract=a.contract['identity'],mode='soak-reference',options=dict(seconds=570,memory_ceiling=host.policy['memory_bytes'],render_node=host.policy['render_node']))
            write(inputs/'expected.json',cfg)
            legacy(a.root).container_stage(host,image,inputs,output/'role',mode='soak-reference',seconds=600,gpu=False,argv=['python','-m','oeis_learn.pipeline_validation.role','soak-reference'])
            result=json.loads((output/'role/summary.json').read_text())
            if result.get('status')!='COMPLETE' or result.get('execution')!=EXECUTION:raise RuntimeError('full-size CPU reference incomplete')
            self.refroot=output/'role';self.refs={'reference.json':'reference.json','initial.pt':'private/initial.pt','oracle.pt':'private/oracle.pt'}
            self.ref_hashes={n:file_hash(self.refroot/r) for n,r in self.refs.items()}
            return result
        if stage.adapter=='soak-batch':
            batch=int(stage.id.split('.')[1][1:]);r=self.invoke(stage,output,kind='batch',batch=batch)
            return dict(r,stage_state='PASS' if r['status']=='COMPLETE' else 'PARTIAL')
        if stage.id=='soak.parity':return self.invoke(stage,output,kind='parity')
        if stage.id=='soak.freeze':
            rows={}
            for batch in (8,16,32):
                rows[str(batch)]=[]
                warm=a.receipts.get(f'soak.b{batch}.warmup',{})
                for i in range(3):
                    receipt=a.receipts.get(f'soak.b{batch}.calibration-{i}',{})
                    if not receipt:raise ValueError('all finite batch slots must be terminal before freezing')
                    if warm.get('state')!='PASS' or receipt['state']!='PASS':
                        rows[str(batch)].append(dict(complete=False));continue
                    r=receipt['observations']['result']['trainer']['engineering']
                    training=r['state']['components']['training'];rows[str(batch)].append(dict(complete=r['complete'],tokens=training['tokens'],seconds=training['seconds']))
            selected=select_batch(rows,expected_tokens=2*self.work['data']['tokens']['training'])
            self.choice=seal(dict(version='pipeline-soak-choice/v1',source=source,work=self.work['identity'],
                batches=rows,selected_training_batch=selected,profile=PROFILES['m100'],execution=EXECUTION,
                inference_batch=16,workers=4,envelope=self.envelope,adoption='EXPERIMENTAL_ONLY'))
            self.choice=seal(dict(self.choice,live_memory_upper_bytes=peak(selected)))
            write(output/'choice.json',self.choice);return self.choice
        if stage.id=='soak.joint':return self.invoke(stage,output,kind='joint',batch=self.choice['selected_training_batch'])
        if stage.id=='soak.observation':
            values=[]
            for label,observed in [('ordinary',False),('observed',True)]:
                target=output/label;target.mkdir()
                values.append(self.invoke(stage,target,kind='observation',batch=self.choice['selected_training_batch'],instrumented=observed,seconds=570))
            elapsed=[json.loads((output/label/'role/host-timing.json').read_text())['seconds_through_cleanup'] for label in ('ordinary','observed')]
            overhead=elapsed[1]/elapsed[0]-1
            self.observed=overhead<=.05
            result=seal(dict(version='pipeline-soak-observation/v1',overhead=overhead,primary='observed' if self.observed else 'ordinary',results=values))
            write(output/'observation-choice.json',result);return result
        if stage.id=='soak.persistent':
            from .receipts import Ledger
            from oeis_learn.synthesis.state import BudgetLedger
            parent=lease(source,self.work['identity'],self.choice['identity']);write(output/'lease.json',parent)
            budget=BudgetLedger(7800);budget.start('engineering-soak',7800)
            write(output/'budget.json',budget.snapshot())
            ledgers=[Ledger(output/f'window-{i}',f'soak.window-{i}',dict(source=source,lease=parent['identity'],choice=self.choice['identity'])) for i in (0,1)]
            results=[];role=output/'role'
            completed=False
            try:
                for window in (0,1):
                    if window:
                        archived=output/'leg-0';archived.mkdir()
                        for path in list(role.iterdir()):
                            if path.name not in {'private','.runtime'}:shutil.move(str(path),str(archived/path.name))
                    left=remaining(parent)
                    if left<3720:raise CapabilityError('remaining original lease cannot fit next complete work window')
                    ledger=ledgers[window];ledger.transition('READY');ledger.transition('RUNNING')
                    try:
                        result=self.invoke(stage,output,kind='window',batch=self.choice['selected_training_batch'],
                            instrumented=self.observed,seconds=min(3900,left),resume=bool(window),window=window,parent=parent,role_root=role)
                        write(output/f'window-{window}/result.json',result)
                        ledger.transition('PASS' if result['status']=='COMPLETE' else 'PARTIAL',
                            files={'result.json':file_hash(output/f'window-{window}/result.json')})
                        results.append(result)
                        if result['status']!='COMPLETE':raise RuntimeError('persistent window incomplete; no automatic restart/refund')
                    except BaseException as exc:
                        if ledger.read()['state']=='RUNNING':ledger.transition('PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__+': '+str(exc))
                        raise
                completed=True
            finally:
                for ledger in ledgers:
                    if ledger.read()['state']=='PENDING':ledger.transition('BLOCKED',reason='preceding window/reservation incomplete; original allowance not renewed')
                budget.finish('engineering-soak','completed' if completed else 'interrupted')
                write(output/'terminal-budget.json',budget.snapshot())
            write(output/'persistent.json',seal(dict(version='pipeline-soak-persistent/v1',lease=parent,choice=self.choice,results=results)))
            return dict(status='COMPLETE',lease=parent,choice=self.choice,results=results)
        raise ValueError('unknown soak stage')
