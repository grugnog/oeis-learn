"""Prospective paired SFT stages reuse source/data qualification and isolated scoring."""
import copy,json,shutil
from pathlib import Path
from .identity import seal,write,confined,file_hash,digest
from .diagnostics import Diagnostics,registry as diagnostic_registry
from .diagnostics_contract import plan as diagnostic_plan
from .followup_training import profile
from .registry import Stage


class Study(Diagnostics):
    def __call__(self,stage,output):
        if stage.adapter!='followup-train':return super().__call__(stage,output)
        arm=stage.id.split('.')[-1];data=self.imported;p=self.a.pilot;p.purpose='followup';p.manifest=data['manifest']
        policy=profile(arm,self.args.tokens,self.args.seed)
        descriptor=output/'private/study.json';write(descriptor,dict(profile=policy,work=data['manifest']['identity']))
        result,target=p.invoke(output,'train','pilot-followup-train',
            {**{n:data['prepared']/n for n in ('prepared.json','recipe.json','examples.json')},'study.json':descriptor},seconds=21800)
        weights=json.loads((target/'weights.json').read_text())
        ordinal=7+('uniform','family_balanced','decay').index(arm)
        data['snapshots'][ordinal]=dict(path=target/'private/best-weights.pt',weights_sha256=weights['best_weights'])
        from .followup_evaluation import evaluate
        evaluation=evaluate(self,output,ordinal)
        write(output/'evaluation.json',evaluation)
        return dict(stage_state='PASS' if result['status']=='COMPLETE' else 'PARTIAL',result=result,evaluation=evaluation)

    def finalize(self,output,invocation,receipts):
        rows=[]
        for arm in ('uniform','family_balanced','decay'):
            path=invocation/'stages'/('followup.'+arm)/'train/weights.json'
            if path.exists():rows.append(json.loads(path.read_text()))
        write(output/'study.json',seal(dict(version='pipeline-followup-study-result/v1',arms=rows,training=True,resume=False,
            final_access='SEALED',promotion='saved-weight evaluation and seed confirmation required')))
        return dict(study='COMPLETE' if len(rows)==3 and all(r['state']=='PASS' for r in receipts.values()) else 'PARTIAL')


def main(args,root,host_factory):
    from .adapters import Adapters
    from .runner import execute
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from .closures import dependency_closure
    contract=diagnostic_plan(operation='models',snapshots=[0],training_rows=32)
    contract.update(version='pipeline-followup-study/v1',training=True,profiles=[profile(a,args.tokens,args.seed) for a in ('uniform','family_balanced','decay')])
    contract['budgets'].update(global_seconds=129600,phase_seconds={'1':3600,'2':1800,'4':110000,'5':300})
    from .pilot_clocks import DRAIN
    contract['evaluation_policy']=DRAIN
    contract=seal(contract)
    engine=args.engine
    if engine=='auto':engine=next((n for n in ('docker','podman') if shutil.which(n)),'docker')
    stages=[s for s in diagnostic_registry(contract,engine) if s.adapter!='diagnostics-model']
    parent=('hip.preflight','cpu.'+engine+'.semantic','diagnostics.import')
    stages.extend(Stage('followup.'+a,4,parent,seconds=33000,adapter='followup-train') for a in ('uniform','family_balanced','decay'))
    if args.scope_report:
        from dataclasses import asdict
        print(json.dumps(dict(contract=contract,stages=[asdict(s) for s in stages]),indent=2));return 0
    options=copy.copy(args);options.engine=engine;options.pipeline_engines=[engine];options.reuse=[]
    current=collect_snapshot(root)
    return execute(stages,confined(root,args.output,exists=False),adapter=Study(options,root,contract,Adapters(options,root,host_factory,contract)),
        binding=dict(source=digest(current),contract=contract['identity'],dependency_closure=dependency_closure(current,prerequisites={})),
        cap=129600,reserve=1200,contract=contract)
