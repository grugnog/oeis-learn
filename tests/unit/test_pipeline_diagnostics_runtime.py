"""Pinned CPU/HIP read-only diagnostic fixtures; no learning-quality threshold."""
import math
import time
import pytest


def test_scalar_losses_exclude_padding_and_keep_entropy_separate():
    import torch
    from oeis_learn.pipeline_validation.diagnostics_runtime import row_metrics
    logits=torch.zeros(2,3,4);targets=torch.tensor([[1,2,3],[2,0,0]])
    rows,_=row_metrics(logits,targets,0)
    assert [r['tokens'] for r in rows]==[3,1]
    assert abs(sum(r['loss_sum'] for r in rows)-4*math.log(4))<1e-5
    assert abs(sum(r['entropy_sum'] for r in rows)-4*math.log(4))<1e-5
    assert not any(r['exact_program'] for r in rows)
    logits[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):row_metrics(logits,targets,0)


def test_weight_statistics_zero_reference_buffers_and_restricted_loader(tmp_path):
    import torch
    from oeis_learn.pipeline_validation.diagnostics_runtime import tensor_summary,compare_tensor,read_tensors,tensor_comparisons
    a=torch.tensor([3.,4.]);z=torch.zeros(2)
    assert tensor_summary(a)['norm']==5
    assert compare_tensor(a,z)['relative_delta_norm'] is None
    assert compare_tensor(a,z)['delta_norm']==5
    assert compare_tensor(a,a)['cosine']==1
    payload=dict(encoder={'a':a,'mask':torch.tensor([True,False])},decoder={'identity':torch.tensor([1,2],dtype=torch.uint8)})
    torch.save(payload,tmp_path/'p.pt');restored=read_tensors(tmp_path/'p.pt')
    assert torch.equal(restored['encoder']['a'],a)
    result=tensor_comparisons(payload,payload,payload,spectra=True,deadline=time.monotonic()+10)
    assert all(r['versus_initial']['delta_norm']==0 for r in result)
    bad=dict(encoder={'a':a.double()},decoder={});torch.save(bad,tmp_path/'bad.pt')
    with pytest.raises(ValueError):read_tensors(tmp_path/'bad.pt')
    with pytest.raises(FloatingPointError):tensor_summary(torch.tensor([float('inf')]))
    with pytest.raises(ValueError):compare_tensor(a,torch.zeros(3))
    with pytest.raises(TimeoutError):tensor_comparisons(payload,payload,payload,spectra=False,deadline=0)


def test_paired_controls_do_not_compare_unequal_targets_or_fill_missing():
    from oeis_learn.pipeline_validation.diagnostics_runtime import paired_controls
    rows=[dict(id='a',mode='correct',tokens=2,loss_sum=2.),dict(id='a',mode='absent',tokens=2,loss_sum=4.),dict(id='b',mode='correct',tokens=1,loss_sum=1.)]
    r=paired_controls(rows)
    assert r['absent']['complete_pairs']==1 and r['absent']['unresolved_pairs']==1
    assert r['absent']['token_weighted_loss_increase']==1
    assert r['shuffled']['mean_per_example_loss_increase'] is None
    with pytest.raises(ValueError):paired_controls(rows+[rows[0]])
    missing=paired_controls([],expected_ids=['a','b'])
    assert missing['absent']['unresolved_pairs']==2
    rows[1]['tokens']=1
    with pytest.raises(ValueError):paired_controls(rows)


def actual_role_fixture(root,device):
    """Executed in a fresh subprocess so Torch thread policy is qualified exactly."""
    import json,os
    from pathlib import Path
    import torch
    from oeis_learn.pipeline_validation.pilot_recipe import recipe
    from oeis_learn.pipeline_validation.diagnostics_contract import plan,populations
    from oeis_learn.pipeline_validation.diagnostics_runtime import run
    from oeis_learn.pipeline_validation.identity import seal,write,file_hash,digest
    from oeis_learn.pipeline_validation.pilot_contract import PROFILE as DATA_PROFILE
    from oeis_learn.pipeline_validation.pilot_calibration_import import PRODUCER_SOURCE
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    from oeis_learn.decoder.bigint_codec import encode_body
    if device=='cuda':assert torch.version.hip and torch.cuda.is_available()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cfg=recipe();encoder,decoder,counts=models(cfg,device)
    root=Path(root);inputs=root/'inputs';inputs.mkdir();out=root/'output';out.mkdir()
    expected=parameter_digest(encoder,decoder)
    torch.save(dict(encoder={k:v.cpu() for k,v in encoder.state_dict().items()},decoder={k:v.cpu() for k,v in decoder.state_dict().items()}),inputs/'weights.pt')
    del encoder,decoder
    if device=='cuda':torch.cuda.empty_cache()
    for label in ('initial','previous'):os.link(inputs/'weights.pt',inputs/(label+'.pt'))
    source='(program (return $index))';tokens=encode_body(source)
    examples={split:[dict(id=split+':'+str(i),family='f'+str(i%4),stage='A',origin='generic_random',
        indices=list(map(str,range(20))),values=[str(n+i%3) for n in range(20)],
        horizon_indices=list(map(str,range(100))),tokens=tokens,source=source) for i in range(n)]
        for split,n in [('training',32768),('parameter_development',128),('structural_development',128)]}
    write(inputs/'examples.json',examples);write(inputs/'recipe.json',cfg)
    roster=dict(training_panel=[r['id'] for r in examples['training'][:32]],parameter_development=[r['id'] for r in examples['parameter_development']],
        structural_development=[r['id'] for r in examples['structural_development']],oeis_development=['oeis'+str(i) for i in range(8)])
    files={'examples.json':{'sha256':file_hash(inputs/'examples.json')}}
    m=seal(dict(version='pipeline-pilot-inputs/v3',source=PRODUCER_SOURCE,profile=DATA_PROFILE['identity'],recipe=digest(cfg),roster=roster,
        audit={'counts':{k:len(v) for k,v in examples.items()}},files=files));write(inputs/'prepared.json',m)
    policy=plan(operation='models',device='hip' if device=='cuda' else 'cpu',snapshots=[0],training_rows=32)
    pop=populations(examples,roster,training_rows=32)
    weight=dict(ordinal=0,weights_sha256=file_hash(inputs/'weights.pt'),parameter_digest=expected)
    write(inputs/'diagnostics.json',dict(plan=policy,work=m['identity'],populations=pop,weights=weight,initial=weight,previous=weight))
    config=dict(source=digest('explicit synthetic pinned fixture'),options=dict(seconds=450,memory_ceiling=32<<30,render_node=None,
        purpose='diagnostics-hip' if device=='cuda' else 'diagnostics-cpu',ordinal=0,worker_seconds=0,image=digest('fixture'),interrupted=[]))
    result=run(inputs,out,config)
    assert result['status']=='COMPLETE' and result['counts']['total']==104496662
    assert result['parameter_digest_before']==result['parameter_digest_after']==expected
    assert result['training'] is False and result['optimizer_loaded'] is False
    assert result['completed_rows']==288
    controls=json.loads((out/'matched-controls.json').read_text())
    assert len(controls['rows'])==864 and controls['pairs']['absent']['complete_pairs']==288
    activation=json.loads((out/'activations.json').read_text());assert activation['summaries']
    assert activation['completed_rows']==dict(correct=64,shuffled=64,absent=64)
    assert abs(activation['representations']['absent']['mean_off_diagonal_cosine']-1)<1e-6
    assert (out/'per-example.jsonl').is_file() and not (inputs/'checkpoints').exists()
    raw=(out/'per-example.jsonl').read_text()
    assert 'horizon_indices' not in raw and '"values"' not in raw and '"source"' not in raw


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.gpu)])
def test_read_only_diagnostic_role_actual_weights(tmp_path,device):
    import os,subprocess,sys
    if device=='cuda':
        import torch
        if not (torch.version.hip and torch.cuda.is_available()):
            if os.environ.get('OEIS_REQUIRE_BIGINT_GPU')=='1':pytest.fail('actual HIP required')
            pytest.skip('actual HIP required')
    command='from tests.unit.test_pipeline_diagnostics_runtime import actual_role_fixture;actual_role_fixture('+repr(str(tmp_path))+','+repr(device)+')'
    subprocess.run([sys.executable,'-c',command],check=True,timeout=540)
