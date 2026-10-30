"""Bounded inference-only probes of actual saved tensors; no optimizer or fitter."""
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from .identity import seal,digest,file_hash,write
from .sizing_execution import sizing_execution,execution_observation


def tensor_summary(value):
    import torch
    a=value.detach().to(device='cpu').reshape(-1)
    total=squares=zeros=0;lo=hi=None
    for offset in range(0,a.numel(),1<<20):
        x=a[offset:offset+(1<<20)].double()
        if not bool(torch.isfinite(x).all()):raise FloatingPointError('nonfinite diagnostic tensor')
        total+=float(x.sum());squares+=float((x*x).sum());zeros+=int((x==0).sum())
        low=float(x.min());high=float(x.max());lo=low if lo is None else min(lo,low);hi=high if hi is None else max(hi,high)
    n=a.numel()
    return dict(elements=n,mean=total/n if n else None,rms=math.sqrt(squares/n) if n else None,
        norm=math.sqrt(squares),zero_fraction=zeros/n if n else None,minimum=lo,maximum=hi)


def compare_tensor(current,reference):
    import torch
    if current.shape!=reference.shape or current.dtype!=reference.dtype:raise ValueError('tensor comparison shape/dtype mismatch')
    a=current.detach().cpu().reshape(-1);b=reference.detach().cpu().reshape(-1);aa=bb=ab=dd=0.;maximum=0.
    for offset in range(0,a.numel(),1<<20):
        x=a[offset:offset+(1<<20)].double();y=b[offset:offset+(1<<20)].double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):raise FloatingPointError('nonfinite tensor comparison')
        delta=x-y;aa+=float((x*x).sum());bb+=float((y*y).sum());ab+=float((x*y).sum());dd+=float((delta*delta).sum())
        maximum=max(maximum,float(delta.abs().max()))
    return dict(delta_norm=math.sqrt(dd),relative_delta_norm=math.sqrt(dd/bb) if bb else None,
        cosine=ab/math.sqrt(aa*bb) if aa and bb else None,maximum_absolute_delta=maximum,
        undefined_reference_norm=bb==0)


def read_tensors(path):
    import torch
    p=Path(path)
    if not p.is_file() or p.is_symlink() or p.stat().st_size>512<<20:raise ValueError('bounded regular saved tensor file required')
    value=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or set(value)!={'encoder','decoder'}:raise ValueError('encoder/decoder tensor-only payload required')
    count=0
    for model in value.values():
        if not isinstance(model,dict):raise ValueError('state tensor mapping required')
        for name,tensor in model.items():
            if not isinstance(name,str) or not isinstance(tensor,torch.Tensor):raise ValueError('named state tensors required')
            if tensor.is_complex() or tensor.is_floating_point() and tensor.dtype!=torch.float32:raise ValueError('FP32 floating state tensors required')
            count+=tensor.numel()
    if count>128_000_000:raise ValueError('saved tensor capacity exceeded')
    return value


def tensor_comparisons(current,initial,previous,*,spectra,deadline):
    import torch
    result=[];spectral=0
    if set(current)!=set(initial) or set(current)!=set(previous):raise ValueError('model state groups changed')
    for group in sorted(current):
        if set(current[group])!=set(initial[group]) or set(current[group])!=set(previous[group]):raise ValueError('model state keys changed')
        for name,a in sorted(current[group].items()):
            if time.monotonic()>=deadline:raise TimeoutError('weight diagnostic deadline')
            if not a.is_floating_point() and (not torch.equal(a,initial[group][name]) or not torch.equal(a,previous[group][name])):raise ValueError('immutable identity buffer changed')
            row=dict(kind='floating_state' if a.is_floating_point() else 'immutable_identity_buffer',name=group+'.'+name,shape=list(a.shape),dtype=str(a.dtype),summary=tensor_summary(a),
                versus_initial=compare_tensor(a,initial[group][name]),versus_previous=compare_tensor(a,previous[group][name]))
            if spectra and a.ndim==2 and min(a.shape)>1 and spectral<32:
                sample=a[::max(1,math.ceil(a.shape[0]/64)),::max(1,math.ceil(a.shape[1]/64))].double()
                singular=torch.linalg.svdvals(sample);total=float(singular.sum())
                distribution=singular/total if total else singular
                entropy=-float((distribution[distribution>0]*distribution[distribution>0].log()).sum())
                row['sampled_spectrum']=dict(shape=list(sample.shape),top_singular_values=singular[:4].tolist(),
                    effective_rank=math.exp(entropy) if total else 0,interpretation='deterministic strided submatrix, not full-matrix spectrum/rank')
                spectral+=1
            result.append(row)
    return result


def row_metrics(logits,targets,pad):
    import torch
    from oeis_learn.rl.token_loss import token_cross_entropy
    mask=targets.ne(pad)
    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('nonfinite diagnostic logits')
    loss=token_cross_entropy(logits,targets);pred=logits.argmax(-1);logs=logits.float().log_softmax(-1)
    entropy=-(logs.exp()*logs).sum(-1)
    result=[]
    for i in range(len(targets)):
        valid=mask[i];n=int(valid.sum())
        if not n:raise ValueError('nonempty canonical target required')
        correct=int(((pred[i]==targets[i])&valid).sum())
        result.append(dict(tokens=n,loss_sum=float(loss[i][valid].sum()),correct_tokens=correct,
            exact_program=correct==n,entropy_sum=float(entropy[i][valid].sum())))
    return result,pred


class Activations:
    def __init__(self,encoder,decoder):
        import torch
        self.handles=[];self.rows=defaultdict(lambda:dict(calls=0,elements=0,sum=0.,squares=0.,zeros=0,maximum_absolute=0.));self.prefix='unset';self.vectors=defaultdict(list)
        def callback(name):
            def hook(module,inputs,output):self.add(name,output[0] if isinstance(output,tuple) else output)
            return hook
        for group,model in [('encoder',encoder),('decoder',decoder)]:
            for name,module in model.named_modules():
                if isinstance(module,(torch.nn.TransformerEncoderLayer,torch.nn.TransformerDecoderLayer)):
                    self.handles.append(module.register_forward_hook(callback(group+'.'+name)))
    def add(self,name,value):
        if not hasattr(value,'detach'):return
        if name=='memory':self.vectors[self.prefix].append(value.detach().mean(1).float().cpu())
        s=tensor_summary(value);r=self.rows[self.prefix+'|'+name];n=s['elements'];r['calls']+=1;r['elements']+=n
        r['sum']+=(s['mean'] or 0)*n;r['squares']+=s['norm']**2;r['zeros']+=(s['zero_fraction'] or 0)*n
        r['maximum_absolute']=max(r['maximum_absolute'],abs(s['minimum'] or 0),abs(s['maximum'] or 0))
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
    def representation_report(self):
        import torch
        result={}
        for mode,parts in self.vectors.items():
            values=torch.cat(parts);norms=values.norm(dim=1,keepdim=True);normalized=values/norms.clamp_min(1e-12)
            n=len(values);gram=normalized@normalized.T
            result[mode]=dict(rows=n,mean_feature_variance=float(values.var(0,unbiased=False).mean()),
                mean_off_diagonal_cosine=float((gram.sum()-gram.diag().sum())/(n*(n-1))) if n>1 else None,
                zero_norm_rows=int(norms.eq(0).sum()),interpretation='pooled memory observations; similarity alone is not proof of collapse')
        return result
    def report(self):
        return {k:dict(calls=r['calls'],elements=r['elements'],mean=r['sum']/r['elements'],
            rms=math.sqrt(r['squares']/r['elements']),zero_fraction=r['zeros']/r['elements'],maximum_absolute=r['maximum_absolute'])
            for k,r in self.rows.items() if r['elements']}


def evaluate_rows(encoder,decoder,rows,*,device,mode='correct',shuffled=None,deadline,hook=None):
    import torch
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.fresh.conditioning_controls import condition_memory
    result=[];categories=defaultdict(int)
    for offset in range(0,len(rows),16):
        if time.monotonic()>=deadline:break
        batch=rows[offset:offset+16];lists=[[codec.BOS_ID]+r['tokens'] for r in batch]
        if any(len(x)>1025 or len(x)<2 for x in lists):raise ValueError('bounded nonempty canonical tokens required')
        tokens=torch.full((len(batch),max(map(len,lists))),codec.PAD_ID,device=device,dtype=torch.long)
        for i,ids in enumerate(lists):tokens[i,:len(ids)]=torch.tensor(ids,device=device)
        source=batch if mode!='shuffled' else [shuffled[r['id']] for r in batch]
        memory=encoder.forward_from_sequences([[int(v) for v in r['values']] for r in source],device=device)
        padding=torch.zeros(memory.shape[:2],device=device,dtype=torch.bool)
        if mode=='absent':memory,padding=condition_memory(memory,padding,'absent')
        elif mode not in ('correct','shuffled'):raise ValueError('registered matched conditioning required')
        if hook:hook.add('memory',memory)
        logits=decoder(tokens[:,:-1],memory,tgt_key_padding_mask=tokens[:,:-1].eq(codec.PAD_ID),memory_key_padding_mask=padding)
        if hook:hook.add('logits',logits)
        metrics,predictions=row_metrics(logits,tokens[:,1:],codec.PAD_ID)
        if time.monotonic()>=deadline:break
        for i,(r,m) in enumerate(zip(batch,metrics)):
            # Export counts by token category, never raw targets/prompt values.
            valid=tokens[i,1:].ne(codec.PAD_ID);wrong=predictions[i].ne(tokens[i,1:])&valid
            for token in tokens[i,1:][wrong].tolist():
                kind='integer' if token in codec.DIGIT_IDS or token in (codec.INT_ID,codec.TERM_ID,codec.NEG_ID) else 'control_or_operator'
                categories[kind]+=1
            role_counts={}
            for kind in ('integer','control_or_operator'):
                flags=torch.tensor([('integer' if token in codec.DIGIT_IDS or token in (codec.INT_ID,codec.TERM_ID,codec.NEG_ID) else 'control_or_operator')==kind for token in tokens[i,1:].tolist()],device=device)&valid
                role_counts[kind]=dict(tokens=int(flags.sum()),errors=int((wrong&flags).sum()))
            positions=wrong.nonzero().flatten()
            selected_logits=logits[i][valid].float()
            top=selected_logits.topk(2,dim=-1).values
            margins=top[:,0]-top[:,1]
            target_logs=selected_logits.log_softmax(-1).gather(1,tokens[i,1:][valid,None]).flatten()
            result.append(dict(id=r['id'],family=r['family'],stage=r.get('stage'),mode=mode,target_identity=digest(r['tokens']),
                first_error_position=int(positions[0]) if len(positions) else None,token_roles=role_counts,
                mean_top_two_margin=float(margins.mean()),minimum_top_two_margin=float(margins.min()),
                mean_target_log_probability=float(target_logs.mean()),**m))
    return result,dict(categories)


def paired_controls(rows,expected_ids=None):
    index=defaultdict(dict)
    for r in rows:
        if r['mode'] not in ('correct','shuffled','absent'):raise ValueError('unknown matched conditioning mode')
        if r['mode'] in index[r['id']]:raise ValueError('duplicate matched diagnostic row')
        index[r['id']][r['mode']]=r
    expected=set(index) if expected_ids is None else set(expected_ids)
    if not set(index)<=expected:raise ValueError('matched row outside frozen population')
    result={}
    for mode in ('shuffled','absent'):
        pairs=[(r['correct'],r[mode]) for r in index.values() if 'correct' in r and mode in r]
        tokens=sum(a['tokens'] for a,b in pairs)
        if any(a['tokens']!=b['tokens'] or a.get('target_identity')!=b.get('target_identity') for a,b in pairs):raise ValueError('matched target normalization changed')
        result[mode]=dict(complete_pairs=len(pairs),unresolved_pairs=len(expected)-len(pairs),tokens=tokens,
            mean_per_example_loss_increase=sum(b['loss_sum']/b['tokens']-a['loss_sum']/a['tokens'] for a,b in pairs)/len(pairs) if pairs else None,
            token_weighted_loss_increase=sum(b['loss_sum']-a['loss_sum'] for a,b in pairs)/tokens if tokens else None,
            token_accuracy_change=sum(b['correct_tokens']-a['correct_tokens'] for a,b in pairs)/tokens if tokens and all('correct_tokens' in a and 'correct_tokens' in b for a,b in pairs) else None,
            interpretation='teacher-forced input intervention on identical canonical targets; not semantic continuation accuracy')
    return result


@sizing_execution
def run(inputs,output,config):
    began=time.monotonic()
    import torch,gc
    from oeis_learn.fresh.bigint_worker import models
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    from .pilot_recipe import validate_effective
    from .pilot_data import learner_view
    from .diagnostics_contract import parse,populations
    from .model_sizes import PROFILES,parameters
    from .pilot_runtime import bounded_write
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    opt=config['options'];device='cuda' if opt['purpose']=='diagnostics-hip' else 'cpu'
    if opt['purpose'] not in ('diagnostics-hip','diagnostics-cpu') or opt['worker_seconds'] or opt['interrupted']:raise ValueError('read-only diagnostic role required')
    if any((inputs/n).exists() for n in ('checkpoints','optimizer','truth.json','private')):raise ValueError('forbidden diagnostic mount')
    settings=json.loads((inputs/'diagnostics.json').read_text());plan=parse(settings['plan']);ordinal=opt['ordinal']
    if plan['device']!=('hip' if device=='cuda' else 'cpu') or ordinal not in plan['snapshots']:raise ValueError('frozen diagnostic device/ordinal changed')
    m=json.loads((inputs/'prepared.json').read_text())
    from .identity import validate
    from .pilot_contract import PROFILE as DATA_PROFILE,SPEC
    from .pilot_calibration_import import PRODUCER_SOURCE
    validate(m,{'version','source','profile','recipe','roster','audit','files'},'pipeline-pilot-inputs/v3')
    if m['source']!=PRODUCER_SOURCE or m['profile']!=DATA_PROFILE['identity']:raise ValueError('original generic data profile/source required')
    recipe=json.loads((inputs/'recipe.json').read_text());validate_effective(recipe)
    if m['identity']!=settings['work'] or file_hash(inputs/'examples.json')!=m['files']['examples.json']['sha256'] or digest(recipe)!=m['recipe']:
        raise ValueError('frozen diagnostic data/recipe changed')
    examples=learner_view(json.loads((inputs/'examples.json').read_text()))
    if {k:len(v) for k,v in examples.items()}!={k:SPEC[k] for k in examples}:raise ValueError('full original generic corpus required')
    population=populations(examples,m['roster'],training_rows=plan['training_rows'])
    if population!=settings['populations']:raise ValueError('diagnostic population changed')
    for label in ('weights','initial','previous'):
        if file_hash(inputs/(label+'.pt'))!=settings[label]['weights_sha256']:raise ValueError('saved diagnostic tensor changed')
    current=read_tensors(inputs/'weights.pt');initial=read_tensors(inputs/'initial.pt');previous=read_tensors(inputs/'previous.pt')
    deadline=began+opt['seconds']-30
    comparisons=tensor_comparisons(current,initial,previous,spectra=plan['spectra'],deadline=deadline)
    bounded_write(output/'tensor-comparisons.json',seal(dict(version='pipeline-diagnostic-tensors/v1',ordinal=ordinal,
        previous_ordinal=settings['previous']['ordinal'],initial_ordinal=0,tensors=comparisons)),2<<20)
    del initial,previous;gc.collect()
    encoder,decoder,counts=models(recipe,device)
    if counts['actual']!=parameters(PROFILES['m100']):raise ValueError('actual full-size diagnostic model required')
    encoder.load_state_dict(current['encoder'],strict=True);decoder.load_state_dict(current['decoder'],strict=True)
    del current;gc.collect()
    encoder.eval();decoder.eval();encoder.requires_grad_(False);decoder.requires_grad_(False)
    before=parameter_digest(encoder,decoder)
    expected_digest=settings['weights']['parameter_digest']
    if expected_digest is None:
        if ordinal!=6 or not plan.get('include_selected'):raise ValueError('missing original parameter digest')
    elif before!=expected_digest:raise ValueError('actual restored parameter digest changed')
    index={r['id']:dict(r,split=split) for split,rows in examples.items() for r in rows}
    shuffled={pid:index[target] for pid,target in population['shuffled'].items()}
    from .pilot_data import labels
    from oeis_learn.fresh.bigint_diagnostics import strata as program_strata
    strata={pid:{**labels(row),**program_strata(row)} for pid,row in index.items()}
    all_rows=[];matched=[];error_counts={};hook=None
    activation_completed={}
    with torch.inference_mode():
        for mode in ('correct','shuffled','absent'):
            values,errors=evaluate_rows(encoder,decoder,[index[k] for k in population['matched']],device=device,mode=mode,shuffled=shuffled,deadline=deadline)
            matched.extend(values);error_counts['matched:'+mode]=errors
        hook=Activations(encoder,decoder)
        try:
            for mode in ('correct','shuffled','absent'):
                hook.prefix=mode
                observed,_=evaluate_rows(encoder,decoder,[index[k] for k in population['activation']],device=device,mode=mode,shuffled=shuffled,deadline=deadline,hook=hook)
                activation_completed[mode]=len(observed)
        finally:hook.close()
        # Complete the small interventions first. Give development rows priority
        # over the larger training census if the original role clock expires.
        ordered=sorted(population['correct'],key=lambda k:(index[k]['split']=='training',k))
        correct,errors=evaluate_rows(encoder,decoder,[index[k] for k in ordered],device=device,deadline=deadline)
        all_rows.extend(dict(r,split=index[r['id']]['split'],strata=strata[r['id']]) for r in correct);error_counts['correct']=errors
    after=parameter_digest(encoder,decoder)
    if after!=before:raise RuntimeError('read-only diagnostic modified parameters')
    expected=len(population['correct']);seen={r['id'] for r in correct}
    # Compact scalar row stream; no full source, targets, logits or activations.
    payload=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in all_rows)
    if len(payload.encode())>32<<20:raise ValueError('diagnostic row stream exceeds32MiB')
    (output/'per-example.jsonl').write_text(payload)
    bounded_write(output/'matched-controls.json',seal(dict(version='pipeline-diagnostic-controls/v1',rows=matched,
        pairs=paired_controls(matched,population['matched']),expected_rows=3*len(population['matched']))),2<<20)
    bounded_write(output/'activations.json',seal(dict(version='pipeline-diagnostic-activations/v1',summaries=hook.report(),representations=hook.representation_report(),
        expected_rows_per_mode=len(population['activation']),completed_rows=activation_completed,interpretation='scalar forward observations; no causal localization claim')),1<<20)
    complete=len(seen)==expected and len(matched)==3*len(population['matched']) and all(n==len(population['activation']) for n in activation_completed.values())
    return dict(status='COMPLETE' if complete else 'PARTIAL',work=settings['work'],source=config['source'],ordinal=ordinal,
        execution=execution_observation(),device=device,counts=counts['actual'],parameter_digest_before=before,parameter_digest_after=after,
        completed_rows=len(seen),expected_rows=expected,missing_rows=sorted(set(population['correct'])-seen),error_token_counts=error_counts,
        training=False,optimizer_loaded=False,interpretation='Post-run teacher-forced diagnostic, not semantic success or training authority')
