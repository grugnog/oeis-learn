"""Real encoder/decoder and SFT updates; no substitute learner or fabricated loss."""
import copy
import pytest


def models():
    import torch
    from oeis_learn.encoder.integer_bytes import IntegerByteEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.decoder import bigint_codec as codec
    torch.manual_seed(7);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    encoder=IntegerByteEncoder(d_model=32,n_heads=4,n_encoder_layers=1,d_ff=64,dropout=0,byte_width=4)
    decoder=WatTransformerDecoder(vocab_size=codec.FOUNDATION_VOCAB_SIZE,codec_sha256=codec.codec_digest(),
        d_model=32,n_heads=4,n_decoder_layers=1,d_ff=64,dropout=0,max_seq_len=1024,pad_idx=codec.PAD_ID,
        position_style='sinusoidal',attention_backend='eager',input_version='decoder-inputs/v1',token_scale=1,position_scale=16)
    return encoder,decoder


def prepared(tmp_path,arm='C'):
    from tests.integration.test_generalization_preparation import pac_rows
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_pac_training
    from oeis_learn.pipeline_validation.generalization_training import PacPreparedBatches
    from oeis_learn.synthesis.reference import evaluate
    rows,probes=pac_rows()
    def native(s,i):
        r=evaluate(s,i);return dict(outcome=r.outcome,values=list(r.values))
    bundle=prepare_pac_training(rows,probes,output=tmp_path/'data',execute=native,final_exclusions=[],seed=7,batch_size=2,purpose='acceptance')
    return PacPreparedBatches(tmp_path/'data',bundle,arm=arm,batch_size=2,updates=100000),bundle


def test_real_panel_token_sums_and_rng_modes(tmp_path):
    import random,numpy as np,torch
    from oeis_learn.pipeline_validation.generalization_curriculum import observe_targets
    from oeis_learn.generalization.budgets import charge_tokens
    batches,_=prepared(tmp_path);encoder,decoder=models();encoder.train();decoder.eval()
    rng=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
    rows=list(batches.probe_rows.values())[:2]
    result=observe_targets(encoder,decoder,rows,device='cpu')
    assert result['tokens']==charge_tokens([r['tokens'] for r in rows],bos=1,pad=0)
    assert result['loss_sum']>0 and len(result['recall'])==2
    assert encoder.training and not decoder.training
    assert torch.equal(rng,torch.get_rng_state()) and python==random.getstate()
    assert np.array_equal(numpy[1],np.random.get_state()[1])
    # Sum/count, not mean-of-means: changing batch grouping preserves the measure.
    singles=[observe_targets(encoder,decoder,[r],device='cpu') for r in rows]
    assert sum(r['tokens'] for r in singles)==result['tokens']
    assert sum(r['loss_sum'] for r in singles)==pytest.approx(result['loss_sum'],rel=1e-6)


def test_real_contiguous_blocks_nonwarmup_scaler_and_failed_update(tmp_path):
    import torch
    from oeis_learn.pipeline_validation.generalization_curriculum import PacController
    from oeis_learn.pipeline_validation.generalization_training import torch_update
    from oeis_learn.generalization.budgets import Ledger
    from tests.integration.test_generalization_training import states
    batches,_=prepared(tmp_path);encoder,decoder=models()
    optimizer=torch.optim.AdamW([*encoder.parameters(),*decoder.parameters()],lr=.001)
    before=[p.detach().clone() for p in decoder.parameters()]
    controller=PacController(batches,encoder,decoder,device='cpu',seed=7,arm='C',cap=100000,observation_seconds=300)
    state=states();state['scheduler']={};seen=[]
    def save(s,l):seen.append((l.updates,copy.deepcopy(controller.policy.snapshot())))
    result=controller.run(state,Ledger(100000),update=torch_update(encoder,decoder,optimizer,device='cpu'),save=save,seconds=300,max_updates=64)
    assert result['ledger'].updates==64
    assert controller.policy.kernel.valid_updates==6 and controller.policy.scaler.seen==6
    assert controller.policy.last_observation['status'] not in {'warmup','degenerate','censored'}
    assert all(torch.isfinite(p).all() for p in decoder.parameters())
    assert any(not torch.equal(a,b) for a,b in zip(before,decoder.parameters()))
    for _,s in seen:
        b=s['block']
        if b:assert b['start_update']+b['completed']==s['updates']
    queue=controller.policy.review.snapshot()
    def fail(batch):raise RuntimeError('injected optimizer failure')
    with pytest.raises(RuntimeError,match='injected'):
        controller.run(state,result['ledger'],update=fail,save=save,seconds=300,max_updates=65)
    assert controller.policy.review.snapshot()==queue
