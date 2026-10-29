"""Isolated pilot adapters reusing bigint_worker, synthesis and independent score."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from .identity import seal,digest,file_hash,write,validate
from .pilot_contract import SPEC,PROFILE,LEGACY_PROFILE,SPLITS
from .sizing_execution import sizing_execution,execution_observation
from .longrun_contract import PURPOSE,full_clock


def bounded_write(path,value,cap):
    if len((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())>cap:
        from .runner import ResourceBreach
        raise ResourceBreach('mandatory pilot evidence byte cap exceeded: '+str(path))
    write(path,value)


def options(config):
    value=config['options']
    if set(value)!={'seconds','memory_ceiling','render_node','purpose','ordinal','worker_seconds','image','interrupted'}:
        raise ValueError('closed pilot role options required')
    if value['purpose'] not in ('preflight','longrun','calibration','followup',PURPOSE) or type(value['ordinal']) is not int:
        raise ValueError('explicit pilot purpose/ordinal required')
    from .identity import number
    number(value['seconds'],positive=True);number(value['memory_ceiling'],positive=True);number(value['worker_seconds'])
    if value['seconds']>43200 or not 0<=value['ordinal']<=(6 if value['purpose']=='followup' else 5):raise ValueError('finite pilot role bound required')
    if value['purpose'] in ('preflight','calibration') and value['ordinal']>2:raise ValueError('three preflight snapshots only')
    return value


def visible_work(inputs,config):
    from .pilot_data import learner_view
    m=json.loads((inputs/'prepared.json').read_text())
    if m.get('version') not in ('pipeline-pilot-inputs/v1','pipeline-pilot-inputs/v2','pipeline-pilot-inputs/v3'):raise ValueError('registered pilot input format required')
    validate(m,{'version','source','profile','recipe','roster','audit','files'},m['version'])
    if m['version'] in ('pipeline-pilot-inputs/v2','pipeline-pilot-inputs/v3'):
        from .pilot_selection import POLICY
        if m['audit'].get('selection',{}).get('policy')!=POLICY:raise ValueError('registered selected-population policy required')
    current=m['version']=='pipeline-pilot-inputs/v3';profile=PROFILE if current else LEGACY_PROFILE
    if m['profile']!=profile['identity']:raise ValueError('pilot data/source/profile changed')
    if options(config)['purpose']==PURPOSE:
        from .longrun_import import validate_compatibility
        proof=json.loads((inputs/'compatibility.json').read_text())
        validate_compatibility(proof,source=config['source'])
        if m['source']!=proof['producer_source'] or m['identity']!=proof['work']:raise ValueError('pilot retained data binding changed')
    elif m['source']!=config['source']:raise ValueError('pilot data/source/profile changed')
    for name in ('examples.json','recipe.json'):
        if file_hash(inputs/name)!=m['files'][name]['sha256']:raise ValueError('pilot visible data/recipe changed')
    examples=learner_view(json.loads((inputs/'examples.json').read_text()))
    if {k:len(v) for k,v in examples.items()}!={k:SPEC[k] for k in examples}:raise ValueError('full pilot data required')
    from oeis_learn.fresh.bigint_config import resolve
    recipe=resolve(json.loads((inputs/'recipe.json').read_text()))
    if recipe['version']!=('bigint-bootstrap/v4' if current else 'bigint-bootstrap/v3') or digest(recipe)!=m['recipe']:raise ValueError('exact pilot recipe required')
    return m,recipe,examples


def settings(manifest,recipe,config):
    from oeis_learn.synthesis.state import ExperimentContract,BudgetLedger
    from oeis_learn.fresh.pilot_control import configuration
    opt=options(config);budget=3600 if opt['purpose']=='preflight' else 86400
    contract=ExperimentContract(data=manifest['identity'],split=digest(manifest['roster']),model=digest(recipe['model']),
        inputs=digest(recipe['inputs']),decoder=digest(recipe['model_profile']),solver=digest(recipe['fitting']),
        runtime=digest(dict(source=config['source'],image=opt['image'])),resources=digest(recipe['execution']['resources']),
        policy=digest(dict(profile=manifest['profile'],purpose=opt['purpose'],pipeline=recipe['pipeline'])),
        track='generic_only',foundation=None,seed=SPEC['seed'],budget=budget)
    ledger=BudgetLedger(budget);ledger.start('learning-pilot',budget)
    control=configuration(source=config['source'],work=manifest['identity'],purpose=opt['purpose'],
        ordinal=opt['ordinal'],seconds=opt['worker_seconds'],cadence_tokens=max(1,manifest['audit']['tokens']['training']//2),interrupted=opt['interrupted'])
    return dict(recipe=recipe,contract=asdict(contract),contract_sha256=contract.sha256,ledger=ledger.snapshot(),
        device='cuda',fixture=False,seconds=opt['seconds']-30,pilot=control,
        pilot_training_ids=manifest['roster']['training_panel'],pilot_prompts_sha256=manifest['files']['prompts.json']['sha256'],
        pilot_truth_sha256=manifest['files']['private/truth.json']['sha256'],pilot_roster=manifest['roster'])


@sizing_execution
def train(inputs,output,config):
    import torch,shutil
    from oeis_learn.fresh.common import write as replace_json
    from oeis_learn.fresh.bigint_worker import train as worker
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.synthesis.state import ExperimentContract
    opt=options(config)
    if opt['purpose']=='calibration':raise ValueError('calibration cannot train or resume')
    m,recipe,examples=visible_work(inputs,config)
    if (inputs/'truth.json').exists() or (inputs/'private').exists():raise ValueError('hidden data cannot enter pilot learner')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cfg=settings(m,recipe,config)
    from oeis_learn.fresh.common import digest as example_digest
    cfg['examples_sha256']=example_digest(examples)
    # The input bind is read-only; stage settings in the private writable worker view.
    view=output/'private/worker-input';view.mkdir(parents=True)
    shutil.copyfile(inputs/'examples.json',view/'examples.json');replace_json(view/'settings.json',cfg)
    if opt['ordinal']:
        if not (inputs/'checkpoints/latest.json').is_file():raise ValueError('complete prior pilot checkpoint required')
        shutil.copytree(inputs/'checkpoints',view/'checkpoints')
        if (inputs/'selected.json').exists():shutil.copyfile(inputs/'selected.json',view/'selected.json')
    elif (inputs/'checkpoints').exists():raise ValueError('fresh pilot cannot import trained state')
    root=output/'private/trainer'
    result=worker(view,root)
    # Worker-input copied state is no longer needed after the new checkpoint is durable.
    shutil.rmtree(view)
    store=CheckpointStore(root/'checkpoints',ExperimentContract(**cfg['contract']),policies=recipe['pipeline'])
    restored=store.load(result['checkpoint'])
    if restored['ledger'].snapshot()!=cfg['ledger']:raise ValueError('pilot cumulative debit changed')
    if restored['states']['data_order']['completed']!=result['update']:raise ValueError('pilot data cursor/update mismatch')
    checkpoint=json.loads((root/'checkpoints'/result['checkpoint']['path']).read_text())
    audit=seal(dict(version='pipeline-pilot-checkpoint-audit/v1',contract=cfg['contract_sha256'],
        checkpoint=result['checkpoint'],ledger=cfg['ledger'],states=sorted(restored['states']),
        payload_hashes={k:v['sha256'] for k,v in checkpoint['files'].items()},
        boundary=restored['states']['selection']['pipeline_boundary'],
        data_order={k:v for k,v in restored['states']['data_order'].items() if k not in {'rng','permutation'}},
        source=config['source'],work=m['identity']))
    write(output/'checkpoint-audit.json',audit)
    for name in ('pilot-validation.json','progress.json','checkpoint-identity.json'):
        if (root/name).exists():shutil.copyfile(root/name,output/name)
    history=[x for x in restored['states']['selection'].get('pilot_validations',[]) if x.get('ordinal')==opt['ordinal']]
    bounded_write(output/'validation-history.json',history,8<<20)
    state=result['pilot']['state']
    bounded_write(output/'training-bins.json',[x for x in state['bins'] if x['key'][0]==opt['ordinal']],2<<20)
    result['pilot']=dict(result['pilot'],state={k:v for k,v in state.items() if k not in {'seen','bins'}})
    result['pilot']['state'].update(unique_examples=len(state['seen']),seen_identity=digest(state['seen']),bins_identity=digest(state['bins']))
    write(output/'pilot.json',result['pilot'])
    write(root/'settings-for-evaluation.json',cfg)
    write(output/'trainer.json',result)
    return dict(status='COMPLETE' if result['pilot']['complete'] else 'PARTIAL',trainer=result,
        source=config['source'],work=m['identity'],execution=execution_observation(),device=dict(hip=torch.version.hip,name=torch.cuda.get_device_name()))


@sizing_execution
def generate(inputs,output,config):
    import torch
    from oeis_learn.fresh.bigint_worker import _load_models,client
    from oeis_learn.fresh.bigint_diagnostics import generate as evaluate
    from oeis_learn.fresh.common import read
    opt=options(config)
    if (inputs/'truth.json').exists() or (inputs/'private').exists():raise ValueError('hidden values in generator mount')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cfg=read(inputs/'settings.json');rows=read(inputs/'prompts.json')
    if file_hash(inputs/'prompts.json')!=cfg['pilot_prompts_sha256']:raise ValueError('frozen prompt file changed')
    if len(rows)!=296 or len({r['id'] for r in rows})!=296:raise ValueError('complete296-row control roster required')
    if any(set(r)-{'id','indices','values','horizon_indices','family','strata'} or len(r['values'])!=20 for r in rows):
        raise ValueError('visible-only generator prompts required')
    if opt['purpose']==PURPOSE:
        from oeis_learn.fresh.pilot_control import parse
        parse(cfg['pilot'])
        if cfg['pilot']['purpose']!=PURPOSE or cfg['pilot']['source']!=config['source'] or cfg['pilot']['ordinal']!=opt['ordinal']:
            raise ValueError('own full-state long-run snapshot required')
    if opt['purpose']=='followup':
        from .diagnostics_runtime import read_tensors
        from oeis_learn.fresh.bigint_worker import models
        if (inputs/'checkpoints').exists() or file_hash(inputs/'weights.pt')!=cfg['followup_weights']:raise ValueError('bound weights-only followup required')
        encoder,decoder,counts=models(cfg['recipe'],'cuda')
        weights=read_tensors(inputs/'weights.pt')
        encoder.load_state_dict(weights['encoder'],strict=True);decoder.load_state_dict(weights['decoder'],strict=True)
    elif opt['purpose']=='calibration':
        from .pilot_calibration import load_snapshot_models
        encoder,decoder,counts=load_snapshot_models(inputs,cfg)
    else:
        if 'diagnostic_snapshot' in cfg:raise ValueError('diagnostic weights cannot authorize ordinary evaluation/resume')
        encoder,decoder,counts=_load_models(inputs,cfg)
    from .model_sizes import PROFILES,parameters
    if counts['actual']!=parameters(PROFILES['m100']):raise ValueError('actual full104.5M model required')
    deadline=time.monotonic()+opt['seconds']-30;records={};controls={}
    from oeis_learn.fresh.pipeline_jobs import merge_boundary,JobLedger
    boundary=JobLedger().boundary(cfg['recipe']['pipeline'])
    from .pilot_evidence import Evidence
    policy=cfg.get('followup_policy') if opt['purpose']=='followup' else None
    if opt['purpose']=='followup':
        from .pilot_clocks import attempts
        attempts(policy)
    if full_clock(opt['purpose']) or opt['purpose']=='followup':
        from .pilot_clocks import FullClockEvidence,POLICY
        evidence=FullClockEvidence(output,rows,source=config['source'],work=cfg['pilot']['work'],**(dict(policy=policy) if policy else {}))
    else:evidence=Evidence(output,rows)
    with evidence, client(cfg,output/'private/native') as synthesis:
        for split in SPLITS:
            batch=[r for r in rows if r['strata']['split']==split]
            # A frozen proportional share prevents a slow split borrowing another's allowance.
            allowance=(opt['seconds']-60)*len(batch)/296
            result=evaluate(encoder,decoder,batch,cfg['recipe'],synthesis,
                deadline=deadline if full_clock(opt['purpose']) else min(deadline,time.monotonic()+allowance),
                diagnostic_attempts=attempts(policy) if policy else dict(correct=8,shuffled=1,absent=1,template=8),evidence=evidence,
                **(dict(evaluation_policy=policy or POLICY) if full_clock(opt['purpose']) or policy else {}))
            records.update({k:{n:r[n] for n in ('selections','arm_evidence','complete','family','strata')} for k,r in result['records'].items()})
            boundary=merge_boundary(boundary,result['pipeline_boundary']);controls[split]=result['controls']
    bounded_write(output/'selections.json',records,8<<20);write(output/'boundary.json',boundary);write(output/'controls.json',controls)
    return dict(status='COMPLETE',rows=len(records),checkpoint=cfg['checkpoint'],source=config['source'],
        work=cfg['pilot']['work'],execution=execution_observation(),counts=counts['actual'],
        **(dict(clock_complete=evidence.clock_summary['cost_complete'],evaluation_policy=policy or POLICY,
            portfolios=evidence.clock_summary['expected_portfolios']) if full_clock(opt['purpose']) or policy else {}))


def score(inputs,output,config):
    from oeis_learn.fresh.bigint_diagnostics import score as independent_score
    from oeis_learn.fresh.bigint_worker import client
    opt=options(config)
    if (inputs/'checkpoints').exists() or (inputs/'examples.json').exists():raise ValueError('scorer must not mount learner/model state')
    cfg=json.loads((inputs/'settings.json').read_text());truth=json.loads((inputs/'truth.json').read_text())
    selections=json.loads((inputs/'selections.json').read_text())
    if file_hash(inputs/'truth.json')!=cfg['pilot_truth_sha256']:raise ValueError('frozen independent truth changed')
    if file_hash(inputs/'selections.json')!=cfg['pilot_selections_sha256']:raise ValueError('sealed generation evidence changed')
    if len(truth)!=296 or set(truth)!=set(selections):raise ValueError('full independent scoring denominator required')
    from .pilot_scoring_cost import sentinels
    with client(cfg,output/'private/native') as synthesis:
        deadline=time.monotonic()+opt['seconds']-15
        cost=sentinels(synthesis,deadline=min(deadline,time.monotonic()+120),policies=cfg['recipe']['pipeline'])
        write(output/'cost-sentinels.json',cost)
        if not cost['complete']:raise ValueError('independent scoring-cost path incomplete')
        try:result=independent_score(selections,truth,synthesis,deadline=deadline)
        except RuntimeError as exc:
            from .runner import ResourceBreach
            raise ResourceBreach('independent exactness disagreement: '+str(exc)) from exc
    outcomes=result.pop('outcomes');bounded_write(output/'outcomes.json',outcomes,4<<20);write(output/'scoring.json',result)
    return dict(status='COMPLETE',rows=len(outcomes),complete=all(x['complete'] for row in outcomes.values() for x in row.values()),
        success=sum(x['success'] for row in outcomes.values() for x in row.values()),
        selected_program_cost_seconds=cost['maximum_selected_program_seconds'],source=config['source'],work=cfg['pilot']['work'],cpu_only=True)


def recover(inputs,output,config):
    """Audit retained complete state after acknowledged cleanup; never replay lost work."""
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.synthesis.state import ExperimentContract
    from oeis_learn.fresh.pilot_control import parse
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    options(config)
    cfg=json.loads((inputs/'settings.json').read_text());parse(cfg['pilot'])
    if cfg['pilot']['source']!=config['source'] or cfg['pilot']['purpose']!=config['options']['purpose'] or cfg['pilot']['purpose'] not in ('longrun',PURPOSE):raise ValueError('identical own pilot recovery only')
    pointer=json.loads((inputs/'checkpoints/latest.json').read_text())
    loaded=CheckpointStore(inputs/'checkpoints',ExperimentContract(**cfg['contract']),policies=cfg['recipe']['pipeline']).load(pointer)
    if loaded['ledger'].snapshot()!=cfg['ledger']:raise ValueError('recovered pilot debit changed')
    class State:
        def __init__(self,value):self.value=value
        def state_dict(self):return self.value
    weights=loaded['states']['weights']
    h=parameter_digest(State(weights['encoder']),State(weights['decoder']))
    state=loaded['states']['selection']['pilot']
    public={k:v for k,v in state.items() if k not in {'seen','bins'}}
    result=dict(update=loaded['states']['data_order']['completed'],checkpoint=pointer,final_parameter_digest=h,
        pilot=dict(complete=False,state=public),recovery='full-state audited; interrupted work remains incomplete and charged')
    write(output/'recovery.json',dict(result,ledger=cfg['ledger'],states=sorted(loaded['states'])))
    return dict(status='COMPLETE',trainer=result,source=config['source'],work=cfg['pilot']['work'])
