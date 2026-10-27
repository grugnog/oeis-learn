"""One saved-snapshot calibration scope; existing role isolation and supervision."""
import json
from pathlib import Path
import time
from .identity import seal,digest,file_hash,write,confined
from .pilot_clocks import POLICY


def manifest():
    from .pilot_contract import preflight_manifest
    m=preflight_manifest();m.update(version='pipeline-pilot-calibration-validation/v1',scope='engineering-pilot-calibration',
        stage_registry_version='pipeline-pilot-calibration-registry/v1',workloads=['saved-snapshot-calibration/v1'])
    m['engines']='selected'
    m['budgets']['phase_seconds']={'1':2400,'2':1200,'3':1200,'4':31000,'5':7200}
    return seal(m)


def registry(engines=('podman',),primary='podman'):
    from .registry import registry as original,Stage
    shared=[s for s in original(engines,primary) if s.adapter in {'host','engine','image','cpu','hip','retirement','operations','policy'}]
    # Read and verify all retained private inputs before a container engine starts.
    result=[shared.pop(0),Stage('calibration.import',1,('host.contracts',),seconds=300,adapter='calibration-import')]
    from dataclasses import replace
    for s in shared:
        if s.adapter=='engine':s=replace(s,prerequisites=(*s.prerequisites,'calibration.import'))
        result.append(s)
    for engine in engines:
        result.append(Stage('pilot.operations.'+engine,5,('image.'+engine+'.qualification',),seconds=180,adapter='soak-operations'))
    parents=('calibration.import','cpu.'+primary+'.semantic','hip.preflight','policy.compare',
        'operations.'+primary+'.workflow','operations.'+primary+'.retirement','pilot.operations.'+primary)
    result.append(Stage('calibration.session',4,parents,seconds=31000,adapter='calibration-session'))
    result.append(Stage('calibration.report',5,seconds=600,adapter='calibration-report'))
    return result


def load_snapshot_models(inputs,cfg):
    """A verified tensor-only view; never constructs a resumable CheckpointStore."""
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from .pilot_calibration_import import PRODUCER_SOURCE
    from .pilot_recipe import validate_effective
    d=cfg['diagnostic_snapshot']
    if set(d)!={'weights_sha256','pointer','original_source','ordinal','evaluation_policy'} or d['original_source']!=PRODUCER_SOURCE or d['evaluation_policy']!=POLICY:
        raise ValueError('registered diagnostic weight authority required')
    if d['pointer']!=cfg['checkpoint'] or d['ordinal'] not in (0,1,2) or (inputs/'checkpoints').exists():
        raise ValueError('weights-only nonresumable snapshot required')
    if file_hash(inputs/'weights.pt')!=d['weights_sha256']:raise ValueError('mounted snapshot weights changed')
    validate_effective(cfg['recipe'])
    payload=torch.load(inputs/'weights.pt',map_location='cpu',weights_only=True)
    if set(payload)!={'encoder','decoder'}:raise ValueError('encoder/decoder tensor-only payload required')
    encoder,decoder,counts=models(cfg['recipe'],'cuda')
    encoder.load_state_dict(payload['encoder']);decoder.load_state_dict(payload['decoder'])
    return encoder,decoder,counts


class CalibrationStages:
    def __init__(self,adapters):
        self.a=adapters;self.imported=None;self.observations=[];self.archives=[]

    def __call__(self,stage,output):
        if stage.id=='calibration.import':
            from .pilot_calibration_import import inspect
            if not self.a.args.inputs:raise ValueError('calibration --inputs must identify the preserved full producer directory')
            root=confined(self.a.root,self.a.args.inputs)
            self.imported=inspect(root,self.a.root,engine=self.a.args.engine)
            write(output/'import.json',self.imported['public'])
            return dict(status='COMPLETE',authority='evaluation only',work=self.imported['manifest']['identity'])
        if stage.id=='calibration.report':return self.report(output)
        if stage.id!='calibration.session':raise ValueError('registered calibration stage required')
        p=self.a.pilot;p.purpose='calibration';p.prepared=self.imported['prepared'];p.manifest=self.imported['manifest']
        p.deadline=time.monotonic()+stage.seconds
        from oeis_learn.fresh.common import write as replace_json
        for snapshot in self.imported['snapshots']:
            i=snapshot['ordinal'];raw=json.loads(snapshot['settings'].read_text())
            cfg={k:raw[k] for k in ('recipe','contract','contract_sha256','pilot_prompts_sha256','pilot_truth_sha256','pilot','checkpoint')}
            cfg.update(device='cuda',diagnostic_snapshot={k:snapshot[k] for k in ('weights_sha256','pointer','original_source','ordinal')})
            cfg['diagnostic_snapshot']['evaluation_policy']=POLICY
            descriptor=output/'private'/('settings-'+str(i)+'.json');replace_json(descriptor,cfg)
            started=time.monotonic()
            generated,gen=p.invoke(output,'generate-'+str(i),'pilot-generate',
                {'settings.json':descriptor,'weights.pt':snapshot['path'],'prompts.json':p.prepared/'prompts.json'},seconds=8600,ordinal=i)
            generation_seconds=time.monotonic()-started
            cfg['device']='cpu';cfg.pop('diagnostic_snapshot');cfg['pilot_selections_sha256']=file_hash(gen/'selections.json');replace_json(descriptor,cfg)
            began=time.monotonic()
            scored,score=p.invoke(output,'score-'+str(i),'pilot-score',
                {'settings.json':descriptor,'selections.json':gen/'selections.json','truth.json':p.prepared/'private/truth.json'},seconds=1500,ordinal=i)
            scoring_seconds=time.monotonic()-began
            clocks=json.loads((gen/'portfolio-clocks.json').read_text());boundary=json.loads((gen/'boundary.json').read_text())
            complete=generated['clock_complete'] and clocks['cost_complete'] and boundary['active']==boundary['reserved']==0
            row=seal(dict(complete=bool(complete),clock_complete=bool(complete),evaluation_policy=POLICY,portfolios=clocks['expected_portfolios'],
                scientific_complete=scored['complete'],seconds=generation_seconds+scoring_seconds,generation_seconds=generation_seconds,
                scoring_seconds=scoring_seconds,projected_all_selected_scoring_seconds=max(scoring_seconds,1776*scored['selected_program_cost_seconds']),
                rows=scored['rows'],source=scored['source'],work=scored['work'],checkpoint=snapshot['pointer'],clock_receipt=clocks['identity']))
            self.observations.append(row);write(output/('observation-'+str(i)+'.json'),row)
        return dict(status='COMPLETE',snapshots=3,learner_updates=0,reference_generated=0)

    def report(self,output):
        from .pilot_calibration_cost import forecast
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        source=digest(collect_snapshot(self.a.root));work=self.imported['manifest']['identity'] if self.imported else digest('unimported')
        if len(self.observations)==3 and not self.archives:
            from .supervision import launch
            import sys
            code='import sys;sys.path.insert(0,'+repr(str(self.a.root/'src'))+');from oeis_learn.pipeline_validation.pilot_archive import main;main()'
            try:
                launch([sys.executable,'-S','-c',code,'--output',str(output/'private/archive-probes'),'--source',source],output/'archive-supervision',seconds=540,stream_cap=8<<20)
                self.archives=json.loads((output/'private/archive-probes/observations.json').read_text())
            except (OSError,RuntimeError,TimeoutError) as exc:write(output/'archive-failure.json',dict(error=str(exc)))
        cost=forecast(self.observations,source=source,work=work,archives=self.archives)
        write(output/'candidate-cost.json',cost)
        required=[s.id for s in registry(self.a.args.pipeline_engines,self.a.args.engine) if s.id!='calibration.report']
        passed=all(self.a.receipts.get(k,{}).get('state') in ('PASS','REUSED') for k in required)
        result=seal(dict(version='pipeline-pilot-calibration-readiness/v1',source=source,work=work,
            primary_plumbing='PASS' if passed else 'BLOCKED',cost=cost['status'],snapshots=len(self.observations),
            readiness='CANDIDATE_PROFILE_REVIEW_REQUIRED' if passed and cost['within_budget'] else 'BLOCKED',
            long_run_admitted=False,automatic_run_authorized=False,final_access='SEALED'))
        write(output/'readiness.json',result);return result

    def finalize(self,output,invocation,receipts):
        self.a.receipts=dict(receipts)
        prior=invocation/'stages/calibration.report/readiness.json'
        if prior.exists():
            result=json.loads(prior.read_text());write(output/'report-reference.json',dict(path=str(prior),sha256=file_hash(prior)))
        else:result=self.report(output)
        return dict(result,engineering=result['primary_plumbing'],research='NOT_EXECUTED')
