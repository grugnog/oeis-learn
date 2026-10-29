"""Prospective, fresh SFT screens on an independently bound generic corpus."""
import math
from collections import defaultdict
from .identity import seal


def profile(arm='uniform',tokens=25_000_000,seed=20261016):
    if arm not in ('uniform','family_balanced','decay'):raise ValueError('registered followup arm required')
    if tokens not in (25_000_000,50_000_000) or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('bounded token budget and seed required')
    return seal(dict(version='pipeline-followup-training/v1',arm=arm,max_tokens=tokens,seed=seed,
        seconds=21600,batch=16,learning_rate=1e-4,validation_tokens=1_987_278,
        fresh=True,resume=False,assisted=False,final_access=False))


def validate(value):
    if value!=profile(value['arm'],value['max_tokens'],value['seed']):raise ValueError('changed training profile')
    return value


def sample(rows,rng,size,arm,groups=None):
    if arm not in ('uniform','family_balanced','decay'):raise ValueError('unknown sampling arm')
    if arm!='family_balanced':return rng.sample(rows,size)
    if groups is None:
        groups=defaultdict(list)
        for row in rows:groups[row['family']].append(row)
    families=sorted(groups)
    return [rng.choice(groups[rng.choice(families)]) for _ in range(size)]


def run(inputs,output,config):
    import random,time,json,torch
    from .identity import file_hash,write,digest
    from .sizing_execution import sizing_execution,execution_observation
    from .pilot_data import learner_view
    from oeis_learn.fresh.bigint_worker import models,_batch_loss
    from oeis_learn.fresh.pilot_metrics import validation
    from oeis_learn.fresh.pipeline_profile import parameter_digest
    @sizing_execution
    def execute():
        began=time.monotonic();opt=config['options']
        cfg=json.loads((inputs/'study.json').read_text());policy=validate(cfg['profile'])
        m=json.loads((inputs/'prepared.json').read_text())
        if (m['identity']!=cfg['work'] or file_hash(inputs/'examples.json')!=m['files']['examples.json']['sha256']
            or digest(json.loads((inputs/'recipe.json').read_text()))!=m['recipe']):raise ValueError('changed training inputs')
        if any((inputs/n).exists() for n in ('truth.json','checkpoints','weights.pt')):raise ValueError('fresh visible-only learner required')
        examples=learner_view(json.loads((inputs/'examples.json').read_text()))
        recipe=json.loads((inputs/'recipe.json').read_text());recipe['model']['seed']=policy['seed']
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        encoder,decoder,counts=models(recipe,'cuda');rng=random.Random(policy['seed'])
        optimizer=torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()),lr=policy['learning_rate'],foreach=False)
        deadline=began+min(policy['seconds'],opt['seconds'])-60
        groups=defaultdict(list)
        for row in examples['training']:groups[row['family']].append(row)
        tokens=updates=0;next_validation=policy['validation_tokens'];best=float('inf');history=[]
        checkpoint=output/'private';checkpoint.mkdir(parents=True,exist_ok=True)
        def save(label):
            torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),checkpoint/(label+'-weights.pt'))
            # Keep the entire learner state; no resume API is implied by this export.
            torch.save(dict(optimizer=optimizer.state_dict(),python=rng.getstate(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),tokens=tokens,updates=updates,profile=policy,
                source=config['source'],work=cfg['work']),checkpoint/(label+'-state.pt'))
        def measure():
            nonlocal best
            measured=validation(encoder,decoder,examples,'cuda',deadline=deadline,training_ids=m['roster']['training_panel'])
            if not all(r['complete'] for r in measured.values()):raise TimeoutError('followup validation incomplete')
            dev=[measured[k] for k in ('parameter_development','structural_development')]
            loss=sum(r['loss_sum'] for r in dev)/sum(r['tokens'] for r in dev)
            history.append(dict(tokens=tokens,updates=updates,splits=measured))
            if loss<best:best=loss;save('best')
            write(output/'validation-history.json',history)
        measure()
        while time.monotonic()<deadline and tokens<policy['max_tokens']:
            batch=sample(examples['training'],rng,policy['batch'],policy['arm'],groups)
            count=sum(len(r['tokens']) for r in batch)
            if tokens+count>policy['max_tokens']:break
            if policy['arm']=='decay':
                lr=policy['learning_rate']*(.1+.9*(1+math.cos(math.pi*tokens/policy['max_tokens']))/2)
                for group in optimizer.param_groups:group['lr']=lr
            encoder.train();decoder.train();_batch_loss(encoder,decoder,batch,'cuda',optimizer=optimizer,observe=False)
            tokens+=count;updates+=1
            if tokens>=next_validation:measure();next_validation+=policy['validation_tokens']
        measure();save('latest')
        write(output/'weights.json',seal(dict(profile=policy,work=cfg['work'],source=config['source'],tokens=tokens,updates=updates,
            best_weights=file_hash(checkpoint/'best-weights.pt'),latest_weights=file_hash(checkpoint/'latest-weights.pt'),
            resume=False,final_access='SEALED')))
        return dict(status='COMPLETE' if policy['max_tokens']-tokens<16*1024 else 'PARTIAL',source=config['source'],work=cfg['work'],
            training=True,resume=False,tokens=tokens,updates=updates,execution=execution_observation(),parameter_digest=parameter_digest(encoder,decoder))
    return execute()
