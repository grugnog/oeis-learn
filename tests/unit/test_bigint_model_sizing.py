"""Pinned-runtime shapes, frozen baseline compatibility and real numerical work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


def check_profiles(device,output):
    code='from tests.unit.test_bigint_model_sizing import _check_profiles;_check_profiles('+repr(device)+','+repr(str(output))+')'
    subprocess.run([sys.executable,'-c',code],check=True,timeout=180)


def _check_profiles(device,output):
    import gc
    import time
    import torch
    from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters
    from oeis_learn.experiments.bootstrap_models import construct_sizing_models,construct_bigint_models
    from oeis_learn.fresh.bigint_worker import _batch_loss
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.decoder.bigint_generation import GenerationRequest,sample_batch
    from oeis_learn.encoder.hybrid import ChunkHybridEncoder
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if device=='cuda':assert torch.version.hip and torch.cuda.is_available()
    torch.manual_seed(20261016);old_e,old_d,_=construct_bigint_models()
    new_e,new_d,_=construct_sizing_models(PROFILES['m8'])
    for old,new in [(old_e,new_e),(old_d,new_d)]:
        assert set(old.state_dict())==set(new.state_dict())
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    del old_e,old_d,new_e,new_d,old,new;gc.collect()
    try:ChunkHybridEncoder(d_model=512)
    except ValueError:pass
    else:raise AssertionError('production default shape guard changed')
    reports=[]
    for key,p in PROFILES.items():
        encoder,decoder,counts=construct_sizing_models(p);assert counts==parameters(p)
        encoder.eval();decoder.eval()
        rows=[[0,(1<<4096)-1,-1],[3,6,9]]
        with torch.no_grad():
            memory=encoder.forward_from_sequences(rows);tokens=torch.ones((2,32),dtype=torch.long)
            expected={i:decoder(tokens[:,:i],memory)[:,-1:].clone() for i in (1,8,32)}
        encoder=encoder.to(device);decoder=decoder.to(device);decoder.cache_bookkeeping='hoisted/v1'
        with torch.no_grad():
            actual_memory=encoder.forward_from_sequences(rows,device=device)
            torch.testing.assert_close(actual_memory.cpu(),memory,atol=2e-5,rtol=2e-4)
            value,cache=decoder.prefill(tokens[:,:1].to(device),actual_memory,capacity=32,backend='eager')
            for i in range(1,33):
                if i>1:value=decoder.decode_step(tokens[:,i-1].to(device),cache)
                if i in expected:
                    torch.testing.assert_close(value.cpu(),expected[i],atol=2e-5,rtol=2e-4)
                    assert torch.equal(value.cpu().argmax(-1),expected[i].argmax(-1))
            before=cache.tokens.clone();lengths=list(cache.lengths)
            decoder.decode_step(tokens[:,0].to(device),cache,active=[False,False])
            assert lengths==cache.lengths and torch.equal(before,cache.tokens)
        del cache,value,before,actual_memory,memory,expected
        optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=.0001)
        train=[dict(values=list(map(str,rows[0])),tokens=codec.encode_body('(program (return $index))'))]
        before=parameter_digest(encoder,decoder);encoder.train();decoder.train()
        _batch_loss(encoder,decoder,train,device,optimizer=optimizer)
        assert parameter_digest(encoder,decoder)!=before
        del optimizer
        encoder.eval();decoder.eval()
        with torch.no_grad():
            #32 distinct real memories across every registered batch.
            visible=[[i+j for j in range(3)] for i in range(32)]
            memory=encoder.forward_from_sequences(visible,device=device)
            for batch in (8,32):
                actual=[]
                for offset in range(0,32,batch):
                    requests=[GenerationRequest(memory[i:i+1],20261016+i,0,time.monotonic()+60,max_length=4) for i in range(offset,min(offset+batch,32))]
                    actual.extend(sample_batch(decoder,requests,batch_size=batch,capacity_qualified=batch==32,bookkeeping='hoisted/v1'))
                assert len(actual)==32 and all(len(r['tokens'])<=4 for r in actual)
        reports.append(dict(profile=key,counts=counts,device=device,hip=torch.version.hip if device=='cuda' else None,
            actual_update=True,cpu_full_cached_parity=True,batches=[8,32],body_cap=4,
            meaning='bounded runtime fixture; operator retains full1024 body cap'))
        del encoder,decoder,memory,tokens,train,actual;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'sizing-profiles.json').write_text(json.dumps(reports,indent=2)+'\n')


def test_actual_sizing_shapes_baseline_compatibility_and_cpu_work(tmp_path):
    check_profiles('cpu',tmp_path)


def test_actual_generated_sizing_native_path_preserves_clock_and_admission(tmp_path):
    import time
    from oeis_learn.pipeline_validation.model_sizes import PROFILES
    from oeis_learn.pipeline_validation.model_sizing import fit_generated
    source='(program (return (add (mul (param.get p0) $index) (param.get p1))))'
    rows=[dict(id=f'b8-p{i}',prompt=f'p{i}',seed=20261016+i,decoded=True,native=None,source=source) for i in range(4)]
    prompts=[dict(id=f'p{i}',indices=list(map(str,range(20))),values=[str(3*j+7+i) for j in range(20)]) for i in range(4)]
    fit_generated(tmp_path,rows,prompts,dict(contract='sha256:'+'1'*64,options={'profile':PROFILES['m8']}),time.monotonic()+30)
    ledger=json.loads((tmp_path/'native-jobs.json').read_text())
    assert ledger['boundary']['consumed']==4 and not ledger['boundary']['active']
    assert all(r['native']['verified'] and r['native']['complete'] for r in rows)
    for job in ledger['records'].values():
        clock=ledger['timing'][job['job']['prompt_id']]
        assert job['state']=='COMMITTED' and job['observed']<clock['deadline']==clock['admitted']+4
