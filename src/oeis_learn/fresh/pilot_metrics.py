"""Token-weighted diagnostics using the existing teacher-forced model operation."""
import time


def tensor_metrics(logits,targets,pad_id):
    import torch
    from oeis_learn.rl.token_loss import token_cross_entropy
    with torch.no_grad():
        valid=targets.ne(pad_id);predicted=logits.argmax(-1)
        losses=token_cross_entropy(logits.detach(),targets)
        return dict(loss_sum=float(losses.masked_select(valid).sum()),tokens=int(valid.sum()),
            correct_tokens=int(((predicted==targets)&valid).sum()),
            exact_programs=int(((predicted==targets)|~valid).all(-1).sum()),programs=len(targets))


def measure(encoder,decoder,rows,device,*,deadline,timer=None):
    import torch
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.rl.sft_trainer import teacher_forced_logits
    encoder.eval();decoder.eval();totals=dict(loss_sum=0.,tokens=0,correct_tokens=0,exact_programs=0,programs=0)
    for offset in range(0,len(rows),16):
        if time.monotonic()>=deadline:break
        batch=rows[offset:offset+16];ids=[[codec.BOS_ID]+r['tokens'] for r in batch]
        tokens=torch.full((len(batch),max(map(len,ids))),codec.PAD_ID,dtype=torch.long,device=device)
        for i,x in enumerate(ids):tokens[i,:len(x)]=torch.tensor(x,device=device)
        with torch.no_grad():
            logits=teacher_forced_logits(encoder,decoder,[[int(v) for v in r['values']] for r in batch],tokens,
                device=device,pad_id=codec.PAD_ID,timer=timer)
            metrics=tensor_metrics(logits,tokens[:,1:],codec.PAD_ID)
        for k,x in metrics.items():totals[k]+=x
    if not totals['tokens']:loss=None;accuracy=None
    else:loss=totals['loss_sum']/totals['tokens'];accuracy=totals['correct_tokens']/totals['tokens']
    return dict(totals,loss=loss,token_accuracy=accuracy,denominator=len(rows),
        complete=totals['programs']==len(rows),normalization='nonpadding-token-weighted',
        exact_match_interpretation='canonical target agreement, not sequence semantic equivalence')


def validation(encoder,decoder,examples,device,*,deadline,timer=None,training_ids):
    ids=set(training_ids);panel=[r for r in examples['training'] if r['id'] in ids]
    if len(panel)!=len(training_ids) or len(ids)!=len(training_ids):raise ValueError('frozen training panel missing/duplicated')
    populations=dict(training_panel=panel,parameter_development=examples['parameter_development'],
        structural_development=examples['structural_development'])
    return {k:measure(encoder,decoder,rows,device,deadline=deadline,timer=timer) for k,rows in populations.items()}
