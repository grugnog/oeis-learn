"""Isolated references and worker supervision; actual updates stay in bigint_worker."""
import argparse
from dataclasses import asdict
import gc
import json
import sys
import time
from pathlib import Path
from .identity import seal,digest,file_hash,write,validate
from .model_sizes import PROFILES,parameters,EXECUTION
from .soak_contract import SPEC,data_contract
from .sizing_execution import sizing_execution,execution_observation,record_parity


def work_input(inputs,config):
    work=json.loads((inputs/'work.json').read_text())
    m=work['manifest']
    if set(work)!={'manifest','recipe','examples'}:raise ValueError('closed soak input required')
    validate(m,{'version','source','binding','data','recipe','spec','execution'},'pipeline-soak-input/v1')
    if (m!=seal(m) or m['version']!='pipeline-soak-input/v1' or m['source']!=config['source']
        or m['spec']!=SPEC or m['execution']!=EXECUTION or m['data']!=data_contract(work['examples'])
        or m['recipe']!=digest(work['recipe'])):raise ValueError('frozen full soak work/source mismatch')
    from oeis_learn.fresh.bigint_config import resolve
    recipe=resolve(work['recipe'])
    if recipe['version']!='bigint-bootstrap/v2' or recipe['model_profile']!=PROFILES['m100']:
        raise ValueError('exact fresh full-size profile required')
    return work,recipe


def role_options(config,*,reference=False):
    fields={'seconds','memory_ceiling','render_node'}
    if not reference:fields|={'kind','batch_size','instrumented','choice','resume','window','lease'}
    opt=config['options']
    if set(opt)!=fields:raise ValueError('closed soak role options required')
    from .identity import number
    number(opt['seconds'],positive=True);number(opt['memory_ceiling'],positive=True)
    if opt['seconds']>7800:raise ValueError('fixed role parent required')
    if not reference:
        if opt['kind'] not in {'parity','batch','joint','observation','window'}:raise ValueError('unknown soak role kind')
        if type(opt['batch_size']) is not int or opt['batch_size'] not in (8,16,32):raise ValueError('registered training batch required')
        if type(opt['instrumented']) is not bool or type(opt['resume']) is not bool:raise ValueError('explicit role flags required')
        if opt['kind']!='window' and (opt['resume'] or opt['window'] or opt['lease'] is not None):raise ValueError('only fixed windows recover state')
    return opt


@sizing_execution
def reference(inputs,output,config):
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from .replay import tensor_record
    role_options(config,reference=True)
    work,recipe=work_input(inputs,config);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    encoder,decoder,counts=models(recipe,'cpu');encoder.eval();decoder.eval()
    private=output/'private';private.mkdir()
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),private/'initial.pt')
    with torch.no_grad():
        memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in work['examples']['training'][:2]])
        expected={}
        for pos in (1,32,128,513,1024):
            tokens=torch.ones((2,pos),dtype=torch.long)
            expected[str(pos)]=decoder(tokens,memory)[:,-1:].cpu()
    torch.save(dict(memory=memory,expected=expected),private/'oracle.pt')
    ref=seal(dict(version='pipeline-soak-reference/v1',source=config['source'],contract=config['contract'],
        work=work['manifest']['identity'],profile=PROFILES['m100'],counts=counts['actual'],execution=execution_observation(),
        initial=file_hash(private/'initial.pt'),oracle=file_hash(private/'oracle.pt'),
        memory=tensor_record(memory),expected={k:tensor_record(v) for k,v in expected.items()}))
    write(output/'reference.json',ref)
    return dict(status='COMPLETE',reference=ref['identity'],counts=counts['actual'],execution=execution_observation())


def gpu_parity(inputs,output,config,work,recipe,*,long=False):
    import torch
    from oeis_learn.fresh.bigint_worker import models
    from .replay import tensor_record,verify_outputs
    ref=json.loads((inputs/'reference.json').read_text())
    if (ref!=seal(ref) or ref['source']!=config['source'] or ref['work']!=work['manifest']['identity']
        or ref['contract']!=config['contract'] or ref['profile']!=PROFILES['m100'] or ref['execution']!=EXECUTION
        or ref['initial']!=file_hash(inputs/'initial.pt') or ref['oracle']!=file_hash(inputs/'oracle.pt')):
        raise ValueError('exact CPU reference/source/initialization required')
    encoder,decoder,counts=models(recipe,'cuda');encoder.eval();decoder.eval();decoder.cache_bookkeeping='hoisted/v1'
    weights=torch.load(inputs/'initial.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict(weights['encoder']);decoder.load_state_dict(weights['decoder']);del weights
    oracle=torch.load(inputs/'oracle.pt',map_location='cpu',weights_only=True)
    if tensor_record(oracle['memory'])!=ref['memory'] or {k:tensor_record(v) for k,v in oracle['expected'].items()}!=ref['expected']:
        raise ValueError('CPU oracle payload changed')
    with torch.no_grad():
        memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in work['examples']['training'][:2]],device='cuda')
        record_parity(memory,oracle['memory'],output,'encoder')
        torch.testing.assert_close(memory.cpu(),oracle['memory'],atol=2e-5,rtol=2e-4)
        until=1024 if long else 128;tokens=torch.ones((2,until),dtype=torch.long,device='cuda')
        value,cache=decoder.prefill(tokens[:,:1],memory,capacity=until,backend='eager');actual={'1':value}
        deadline=time.monotonic()+500
        for pos in range(2,until+1):
            if time.monotonic()>=deadline:raise TimeoutError('long cached parity deadline')
            value=decoder.decode_step(tokens[:,pos-1],cache)
            if str(pos) in oracle['expected']:actual[str(pos)]=value
        for k,v in actual.items():record_parity(v,oracle['expected'][k],output,'decoder.p'+k)
        verify_outputs(actual,{k:oracle['expected'][k] for k in actual},deadline=deadline)
    if long:
        from oeis_learn.decoder.bigint_codec import encode_body
        from oeis_learn.fresh.bigint_worker import _batch_loss
        from oeis_learn.tracking.foundation_metrics import PhaseTimer
        source='(program (local $b0 bigint) '+('(set $b0 (int 1)) '*100)+'(return (get $b0)))'
        # A flat valid program stresses the declared context without changing corpus.
        tokens_long=encode_body(source)
        if len(tokens_long)<900:raise RuntimeError('long stress fixture must approach context limit')
        rows=[dict(values=r['values'],tokens=tokens_long) for r in work['examples']['training'][:32]]
        optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=.0001,foreach=False)
        _batch_loss(encoder,decoder,rows,'cuda',optimizer=optimizer,timer=PhaseTimer('cuda',synchronized=False),observe=False)
        write(output/'long-stress.json',dict(body_tokens=len(tokens_long),batch=32,actual_update=True,training_corpus=False))
        del optimizer,rows
    del encoder,decoder,oracle,memory,cache,actual,value
    gc.collect();torch.cuda.empty_cache()
    return ref


def worker_settings(work,recipe,config,*,fixture=False,resume=False,window=0,lease=None,mode=None,seconds=None):
    from oeis_learn.synthesis.state import ExperimentContract,BudgetLedger
    opt=config['options'];mode=mode or opt['kind'];batch=opt.get('batch_size',8)
    h=lambda x:digest(x)
    policy=h(dict(recipe=recipe,spec=SPEC,source=config['source'],choice=opt.get('choice',work['manifest']['identity']),
        mode=mode,batch=batch,lease=lease,instrumented=opt.get('instrumented',False)))
    contract=ExperimentContract(data=work['manifest']['data']['identity'],split=h(work['manifest']['data']['ids']),
        model=h(recipe['model']),inputs=h(recipe['inputs']),decoder=h(recipe['model_profile']),solver=h(recipe['fitting']),
        runtime=h(dict(source=config['source'],image=work['manifest']['binding']['image'])),resources=h(recipe['execution']['resources']),
        policy=policy,track='generic_only',foundation=None,seed=20261016,budget=7800)
    ledger=BudgetLedger(7800);ledger.start('engineering-soak',7800)
    examples=work['examples'] if not fixture else {k:v[:2] for k,v in work['examples'].items()}
    eng=seal(dict(version='pipeline-engineering-worker/v1',source=config['source'],work=work['manifest']['identity'],
        choice=opt.get('choice',work['manifest']['identity']),mode=mode,batch_size=batch,window=window,resume=resume,lease=lease,
        seconds=seconds or (3600 if mode=='window' else 270),instrumented=opt.get('instrumented',False),
        cadence_tokens=work['manifest']['data']['cadence_tokens'] if not fixture else 1,
        token_target=2*work['manifest']['data']['tokens']['training'] if not fixture else 1))
    return dict(recipe=recipe,contract=asdict(contract),contract_sha256=contract.sha256,ledger=ledger.snapshot(),
        examples_sha256=h(examples),device='cuda',fixture=fixture,seconds=eng['seconds']+120,
        engineering=eng,cost_scope='transfer-screen'),examples


def run_worker(inputs,output,work,recipe,config,**kw):
    from oeis_learn.fresh.common import write as replace_json
    from .supervision import launch
    settings,examples=worker_settings(work,recipe,config,**kw)
    source=output/'private/worker-input';source.mkdir(parents=True,exist_ok=True)
    replace_json(source/'examples.json',examples)
    replace_json(source/'settings.json',dict(settings,initial_weights_sha256=file_hash(inputs/'initial.pt')))
    import os
    if not (source/'initial.pt').exists():os.link(inputs/'initial.pt',source/'initial.pt')
    logs=output/('worker-'+('resume' if kw.get('resume') else 'fresh'))
    try:
        launch([sys.executable,'-m','oeis_learn.pipeline_validation.soak_runtime','--train',str(source),str(output/'private/trainer')],
            logs,seconds=min(config['options']['seconds'],settings['seconds']+90),stream_cap=8<<20)
    finally:
        export_worker_evidence(output,resume=kw.get('resume',False))
    report=json.loads((output/'private/trainer/worker-result.json').read_text())
    # Copy compact receipt only, never tensors/checkpoints/raw fitting payloads.
    write(output/('resumed-trainer.json' if kw.get('resume') else 'trainer.json'),report)
    trainer=output/'private/trainer'
    export=output/('resumed-evidence' if kw.get('resume') else 'fresh-evidence')
    pointer=report['checkpoint']
    checkpoint=json.loads((trainer/'checkpoints'/pointer['path']).read_text())
    write(export/'checkpoint-manifest.json',checkpoint)
    # Verify every complete state payload while it exists; public archive retains hashes only.
    from oeis_learn.fresh.bigint_state import CheckpointStore
    from oeis_learn.synthesis.state import ExperimentContract
    restored=CheckpointStore(trainer/'checkpoints',ExperimentContract(**settings['contract']),policies=recipe.get('pipeline')).load(pointer)
    if restored['ledger'].snapshot()!=settings['ledger']:raise ValueError('checkpoint ledger changed')
    write(export/'checkpoint-audit.json',dict(states=sorted(restored['states']),ledger=restored['ledger'].snapshot(),
        data_order={k:v for k,v in restored['states']['data_order'].items() if k!='rng'},
        engineering=restored['states']['selection']['engineering'],
        boundary=restored['states']['selection'].get('pipeline_boundary'),
        weights=checkpoint['binding']['weights'],optimizer=checkpoint['binding']['optimizer'],
        verified_files={k:v['sha256'] for k,v in checkpoint['files'].items()}))
    del restored
    return report


def export_worker_evidence(output,*,resume=False):
    """Also runs after child failure, before its private tensors are omitted."""
    import shutil
    output=Path(output);trainer=output/'private/trainer'
    export=output/('resumed-evidence' if resume else 'fresh-evidence');export.mkdir(parents=True,exist_ok=True)
    for name in ['engineering-attempts.jsonl','engineering.json','validation.json','progress.json','checkpoint-identity.json',
                 *[p.name for p in trainer.glob('native-pulse-*.json')],*[p.name for p in trainer.glob('cycle-plan-*.json')]]:
        if (trainer/name).is_file():shutil.copyfile(trainer/name,export/name)


@sizing_execution
def trial(inputs,output,config):
    import torch
    from oeis_learn.tracking.pipeline_profile import Recorder
    work,recipe=work_input(inputs,config);opt=role_options(config)
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one actual HIP owner required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with Recorder(output/'observations',instrumented=opt.get('instrumented',False),render_node=opt['render_node'],host_counters=False,sample_bytes=16<<10):
        if opt['kind'] in {'parity','joint'}:gpu_parity(inputs,output,config,work,recipe,long=opt['kind']=='joint')
        if opt['kind']=='parity':
            result=dict(status='COMPLETE',numerical_parity=True,counts=parameters(PROFILES['m100']))
        elif opt['kind']=='joint':
            from .soak_contract import lease
            l=lease(config['source'],work['manifest']['identity'],opt['choice'])
            first=run_worker(inputs,output,work,recipe,config,fixture=True,mode='window',seconds=60,lease=l)
            second=run_worker(inputs,output,work,recipe,config,fixture=True,mode='window',seconds=60,lease=l,resume=True,window=1)
            if not first['engineering']['complete'] or not second['engineering']['complete']:raise RuntimeError('bounded full-size restart fixture incomplete')
            result=dict(status='COMPLETE',first=first,second=second,fixture=True)
        else:
            result=run_worker(inputs,output,work,recipe,config,mode=opt['kind'],
                resume=opt.get('resume',False),window=opt.get('window',0),lease=opt.get('lease'))
            result=dict(status='COMPLETE' if result['engineering']['complete'] else 'PARTIAL',trainer=result)
    write(output/'execution.json',execution_observation())
    telemetry=json.loads((output/'observations/telemetry.json').read_text())
    if telemetry['collector_error'] or opt['instrumented'] and not telemetry['samples']:
        raise RuntimeError('declared observation mode did not collect bounded telemetry')
    return dict(result,execution=execution_observation(),profile=PROFILES['m100'],source=config['source'],
        work=work['manifest']['identity'],device=dict(hip=torch.version.hip,name=torch.cuda.get_device_name()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',nargs=2,required=True);args=parser.parse_args()
    from oeis_learn.fresh.bigint_worker import train
    from oeis_learn.fresh.common import write as replace_json
    result=train(*args.train)
    replace_json(Path(args.train[1])/'worker-result.json',result)


if __name__=='__main__':main()
