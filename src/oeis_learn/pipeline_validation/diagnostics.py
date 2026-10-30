"""Main CLI diagnostic DAG: shared qualification, serial read-only model roles."""
from collections import defaultdict
import copy
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
from .identity import seal,digest,file_hash,write,confined
from .diagnostics_contract import plan,parse
from .registry import Stage


def registry(contract,engine):
    collection=Stage('diagnostics.collect',1,seconds=300,adapter='diagnostics-collect')
    if contract['operation']=='collect':return [collection]
    from .registry import registry as original
    shared=[s for s in original([engine],engine) if s.adapter in {'host','engine','image','cpu','hip'}
        and (contract['device']=='hip' or s.adapter!='hip' and s.id!='image.'+engine+'.gpu')]
    result=[shared.pop(0),collection,Stage('diagnostics.import',1,('host.contracts',),seconds=300,adapter='diagnostics-import')]
    for stage in shared:
        if stage.adapter=='engine':stage=replace(stage,prerequisites=(*stage.prerequisites,'diagnostics.import'))
        result.append(stage)
    parents=('diagnostics.import','cpu.'+engine+'.semantic')+(('hip.preflight',) if contract['device']=='hip' else ())
    result.extend(Stage('diagnostics.snapshot-'+str(i),4,parents,seconds=11000 if contract['operation']=='evaluate' else 1800,adapter='diagnostics-model') for i in contract['snapshots'])
    return result


def summarize_rows(rows):
    groups=defaultdict(lambda:dict(rows=0,tokens=0,loss_sum=0.,correct_tokens=0,exact_programs=0,entropy_sum=0.))
    for r in rows:
        for key in ('all','split:'+r['split'],'split:'+r['split']+'|family:'+r['family'],'split:'+r['split']+'|stage:'+str(r['stage']),*[k+':'+str(v) for k,v in r.get('strata',{}).items()]):
            x=groups[key];x['rows']+=1;x['tokens']+=r['tokens'];x['loss_sum']+=r['loss_sum'];x['correct_tokens']+=r['correct_tokens'];x['exact_programs']+=r['exact_program'];x['entropy_sum']+=r['entropy_sum']
    for r in groups.values():
        r.update(loss=r['loss_sum']/r['tokens'],token_accuracy=r['correct_tokens']/r['tokens'],entropy=r['entropy_sum']/r['tokens'])
    return dict(groups)


class Diagnostics:
    def __init__(self,args,root,contract,adapters=None):
        self.args=args;self.root=Path(root);self.contract=contract;self.a=adapters;self.imported=None;self.receipts={}
    def stage_binding(self,stage,base):return self.a.stage_binding(stage,base) if self.a else base
    def __call__(self,stage,output):
        if stage.adapter=='diagnostics-collect':
            from .diagnostics_collect import Evidence,collect
            with Evidence(confined(self.root,self.args.run)) as reader:return collect(reader,output)
        if stage.adapter=='diagnostics-import':
            from .diagnostics_import import inspect
            if not self.args.inputs or not self.args.admission:raise ValueError('models require --inputs full producer and --admission retained evidence')
            self.imported=inspect(confined(self.root,self.args.run),confined(self.root,self.args.admission),confined(self.root,self.args.inputs),self.root,self.contract)
            write(output/'import.json',self.imported['public'])
            return dict(work=self.imported['manifest']['identity'],snapshots=self.contract['snapshots'],training=False)
        if stage.adapter!='diagnostics-model':
            self.a.receipts=self.receipts;return self.a(stage,output)
        ordinal=int(stage.id.rsplit('-',1)[1])
        if self.contract['operation']=='evaluate':
            from .followup_evaluation import evaluate
            return evaluate(self,output,ordinal)
        order=self.contract['snapshots'];position=order.index(ordinal)
        prior=order[position-1] if position else 0
        data=self.imported;p=self.a.pilot;p.purpose='diagnostics-'+self.contract['device'];p.manifest=data['manifest']
        public=lambda i:{k:v for k,v in data['snapshots'][i].items() if k not in ('path','descriptor')}
        settings=dict(plan=self.contract,work=data['manifest']['identity'],populations=data['populations'],weights=public(ordinal),initial=public(0),previous=public(prior))
        descriptor=output/'private/diagnostics.json';write(descriptor,settings)
        files={n:data['prepared']/n for n in ('prepared.json','examples.json','recipe.json')}
        files.update({'diagnostics.json':descriptor,'weights.pt':data['snapshots'][ordinal]['path'],
            'initial.pt':data['snapshots'][0]['path'],'previous.pt':data['snapshots'][prior]['path']})
        result,target=p.invoke(output,'snapshot-'+str(ordinal),'pilot-diagnostics',files,seconds=stage.seconds-10,ordinal=ordinal)
        return dict(stage_state='PASS' if result['status']=='COMPLETE' else 'PARTIAL',result=result)
    def finalize(self,output,invocation,receipts):
        points=[];missing=[];previous=None;changes=[]
        for i in self.contract['snapshots'] if self.contract['operation']=='models' else []:
            prefix=invocation/'stages'/('diagnostics.snapshot-'+str(i))/('snapshot-'+str(i))
            rows=prefix/'per-example.jsonl'
            if not rows.is_file():missing.append(i);continue
            raw=[json.loads(x) for x in rows.read_text().splitlines()]
            index={r['id']:r for r in raw}
            if len(index)!=len(raw):raise ValueError('duplicate per-example diagnostic')
            result=json.loads((prefix/'summary.json').read_text());planned=result['expected_rows']
            point=dict(ordinal=i,groups=summarize_rows(raw),completed_rows=len(raw),expected_rows=planned,
                missing_rows=result['missing_rows'],controls=json.loads((prefix/'matched-controls.json').read_text())['pairs'])
            points.append(point)
            if previous:
                oldordinal,old=previous;common=set(old)&set(index)
                changes.append(dict(from_ordinal=oldordinal,to_ordinal=i,complete_pairs=len(common),unresolved_pairs=planned-len(common),
                    loss_increased=sum(index[k]['loss_sum']/index[k]['tokens']>old[k]['loss_sum']/old[k]['tokens'] for k in common),
                    canonical_exact_gained=sum(not old[k]['exact_program'] and index[k]['exact_program'] for k in common),
                    canonical_exact_lost=sum(old[k]['exact_program'] and not index[k]['exact_program'] for k in common),
                    mean_loss_change=sum(index[k]['loss_sum']/index[k]['tokens']-old[k]['loss_sum']/old[k]['tokens'] for k in common)/len(common) if common else None))
            previous=i,index
        if self.contract['operation']=='evaluate':
            from .followup_evaluation import report
            return report(self,output,invocation,receipts)
        result=seal(dict(version='pipeline-pilot-diagnostic-report/v1',plan=self.contract['identity'],operation=self.contract['operation'],
            points=points,paired_changes=changes,unavailable_snapshots=missing,stages={k:v['state'] for k,v in receipts.items()},
            scope='POSTRUN_READ_ONLY',training=False,resume=False,final_access='SEALED',automatic_run_authorized=False,
            interpretation='Canonical teacher forcing/input interventions and tensor observations; not new semantic accuracy, proof of a causal mechanism, or learning-regime selection.'))
        from .pilot_runtime import bounded_write
        bounded_write(output/'diagnostics.json',result,16<<20);return dict(report=result['identity'],training=False,diagnostics='COMPLETE' if all(r['state']=='PASS' for r in receipts.values()) else 'PARTIAL')


def main(args,root,host_factory):
    contract=plan(operation=args.operation,device=args.device,snapshots=args.snapshots,training_rows=args.training_rows,spectra=args.spectra,include_selected=getattr(args,'include_selected',False),evaluation_policy=getattr(args,'evaluation_policy',None))
    root=Path(root);output=confined(root,args.output,exists=False)
    maximum=contract['budgets']['global_seconds']
    if args.max_seconds is not None:
        from .identity import number
        if not 0<number(args.max_seconds)<=maximum:raise ValueError('diagnostic ceiling may only be lowered')
        maximum=args.max_seconds
    engine=args.engine
    if engine=='auto':engine=next((n for n in ('docker','podman') if shutil.which(n)),'docker')
    stages=registry(contract,engine)
    if args.scope_report:
        print(json.dumps(dict(contract=contract,stages=[asdict(s) for s in stages],maximum_seconds=maximum,training=False),indent=2));return 0
    options=copy.copy(args);options.engine=engine;options.pipeline_engines=[engine];options.reuse=[]
    from .adapters import Adapters
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .closures import dependency_closure
    from .runner import execute
    snapshot=collect_snapshot(root)
    owner=Adapters(options,root,host_factory,contract) if args.operation in ('models','evaluate') else None
    return execute(stages,output,adapter=Diagnostics(options,root,contract,owner),
        binding=dict(source=digest(snapshot),contract=contract['identity'],dependency_closure=dependency_closure(snapshot,prerequisites={})),
        cap=maximum,reserve=120 if args.operation=='collect' else 1200,contract=contract)
