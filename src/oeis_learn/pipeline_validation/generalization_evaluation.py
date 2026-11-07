"""Checkpoint/visible-input I/O around the Phase B evaluation policies.

Callbacks are supplied by the qualified model/native role. This module never
opens a scorer target or reconstructs unavailable weights from public evidence.
"""
from oeis_learn.generalization.contracts import verify_artifact,shape
from oeis_learn.generalization.proposals import portfolio
from oeis_learn.generalization.diagnostics import join_attempts
from .identity import confined,seal


def evaluate_checkpoint(root,checkpoint,prompts,*,load,generate,fit,controls=('raw','rescue')):
    path=confined(root,checkpoint['path'],exists=False)
    if not path.exists():
        return dict(availability='unavailable',checkpoint=checkpoint['sha256'],attempts=[],reason='MISSING_ARTIFACT')
    verify_artifact(root,checkpoint)
    model=load(path)
    attempts=[]
    for prompt in prompts:
        shape('visible-prompt',prompt)
        draws=[];records=[];fits={}
        for control in controls:
            for draw in generate(model,dict(prompt),control):
                draws.append(draw)
                portfolio(draws)
                id=f"{prompt['id']}:{len(draws)-1}"
                records.append(dict(id=id,prompt=prompt['id'],control=control,source=draw.get('source')))
                fits[id]=fit(draw.get('source'),dict(prompt))
        attempts.extend(join_attempts(records,fits))
    return seal(dict(version='generalization-checkpoint-evaluation/v1',availability='available',checkpoint=checkpoint['sha256'],attempts=attempts))


def load_retained(root,checkpoint,recipe,*,device):
    """Load exactly the requested weights-only payload, with no optimizer restore."""
    import io,torch
    from oeis_learn.fresh.bigint_worker import models
    payload=verify_artifact(root,checkpoint)
    states=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if set(states)!={'encoder','decoder'}:raise ValueError('exact encoder/decoder weights required')
    encoder,decoder,counts=models(recipe,device)
    encoder.load_state_dict(states['encoder'],strict=True);decoder.load_state_dict(states['decoder'],strict=True)
    encoder.eval();decoder.eval()
    return encoder,decoder,counts


def build_bank(decoder,checkpoint,seeds,*,deadline):
    """Recipient-independent absent marker; all eight draw slots remain charged."""
    import time,torch
    from oeis_learn.decoder.bigint_generation import GenerationRequest,sample_batch
    from oeis_learn.generalization.proposals import make_bank
    # Validate the frozen seed roster before model work.
    make_bank(checkpoint,seeds,lambda *_:'validation-only',elapsed_seconds=0)
    param=next(decoder.parameters());memory=param.new_zeros((1,1,decoder.d_model));memory[:,:,0]=1
    began=time.monotonic();limit=min(deadline,began+60);sources={}
    for ordinal,seed in enumerate(seeds):
        if time.monotonic()>=limit:sources[seed]=None;continue
        result=sample_batch(decoder,[GenerationRequest(memory,seed,0 if ordinal==0 else .8,limit)],batch_size=1)[0]
        sources[seed]=result['source'] if result['complete'] else None
    elapsed=time.monotonic()-began
    return make_bank(checkpoint,seeds,lambda _model,seed,_mode:sources[seed],elapsed_seconds=elapsed)


def evaluate_retained(root,checkpoint,prompts,recipe,output,*,client,device,study,source,seconds=21600,bank_seeds=None,pac=False,compare_selectors=False):
    """Existing matched/drained production portfolios, plus the shared bank control."""
    import time
    deadline=time.monotonic()+seconds
    from oeis_learn.generalization.budgets import remaining_seconds
    from pathlib import Path
    from oeis_learn.fresh.bigint_diagnostics import generate,_seal
    from oeis_learn.synthesis.admission import VisiblePrompt,source_identity
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.generalization.proposals import apply_bank,bank_cost
    from .pilot_clocks import DRAIN,FullClockEvidence,attempts
    from .identity import write
    from .identity import number
    number(seconds,positive=True)
    if not pac and seconds>21600:raise ValueError('bounded evaluation required')
    output=Path(output)
    if output.exists():raise ValueError('fresh evaluation output required')
    output.mkdir(parents=True,exist_ok=False)
    for prompt in prompts:shape('visible-prompt',prompt)
    if len({p['id'] for p in prompts})!=len(prompts):raise ValueError('unique visible roster required')
    path=confined(root,checkpoint['path'],exists=False)
    if not path.is_file():
        result=seal(dict(status='PARTIAL',availability='unavailable',checkpoint=checkpoint['sha256'],reason='MISSING_ARTIFACT'))
        write(output/'evaluation.json',result);return result
    encoder,decoder,counts=load_retained(root,checkpoint,recipe,device=device)
    remaining_seconds(deadline)
    bank=build_bank(decoder,checkpoint['sha256'],bank_seeds,deadline=deadline) if bank_seeds is not None else None
    evidence=FullClockEvidence(output/'attempts',prompts,source=source,work=study,policy=DRAIN)
    try:
        result=generate(encoder,decoder,prompts,recipe,client,deadline=deadline,
                        diagnostic_attempts=attempts(DRAIN),evaluation_policy=DRAIN,evidence=evidence,
                        **(dict(retain_selector_candidates=True) if compare_selectors else {}))
    finally:evidence.finish()
    if bank is not None:
        write(output/'proposal-bank.json',bank)
        for prompt in prompts:
            visible=VisiblePrompt(tuple(prompt['indices']),tuple(prompt['values']))
            limit=min(deadline,time.monotonic()+max(0,60-bank['generation_seconds']))
            details=[]
            def fit(candidate,indices,values):
                began=time.monotonic()
                if time.monotonic()>=limit:
                    value=dict(complete=False,admitted=[],outcome='budget_exhausted')
                else:
                    try:value=client.fit(candidate,visible,policy=fitting_policy(candidate),deadline=limit)
                    except ValueError:value=dict(complete=True,admitted=[],outcome='unsupported_source')
                # The sealer needs the exact fitted bank proposal, not its
                # transformed parameter source. Do not mutate fitter/cache records.
                if any(item.get('original_source_sha256')!=source_identity(candidate)
                       for item in value.get('admitted',[])):
                    raise ValueError('bank original source identity mismatch')
                value=dict(value,admitted=[dict(item,_original_source=candidate)
                                          for item in value.get('admitted',[])])
                details.append(dict(complete=value.get('complete',False),outcome=value.get('outcome'),
                    service_seconds=None,queue_seconds=None,cleanup_seconds=None,elapsed_seconds=time.monotonic()-began))
                return value
            fitted=apply_bank(bank,prompt,fit)
            admitted=[item for value in fitted if value is not None for item in value.get('admitted',[])]
            record=result['records'][prompt['id']]
            record['selections']['bank']=_seal(admitted,visible,cost=sum(d['elapsed_seconds'] for d in details))
            if compare_selectors:
                from oeis_learn.fresh.bigint_diagnostics import selector_candidates
                record['selector_candidates']['bank']=selector_candidates(admitted,visible,cost=sum(d['elapsed_seconds'] for d in details))
            record['arm_evidence']['bank']=dict(complete=all(v is not None and v.get('complete') is True for v in fitted),
                reason='bank_application',expected=8,charged_attempts=8,details=details,**bank_cost(bank,len(prompts)))
    if compare_selectors:
        import copy
        from oeis_learn.generalization.selector_validation import CONTROLS,rerank
        alternatives={};observations={}
        for prompt in prompts:
            record=result['records'][prompt['id']];pools=record.pop('selector_candidates')
            if set(pools)!=set(CONTROLS):raise ValueError('all seven selector comparison controls required')
            alternative=dict(selections={},arm_evidence=copy.deepcopy(record['arm_evidence']))
            visible=VisiblePrompt(tuple(prompt['indices']),tuple(prompt['values']));observations[prompt['id']]={}
            for control in CONTROLS:
                chosen,observation=rerank(pools[control],record['selections'][control],visible,deadline=deadline)
                alternative['selections'][control]=chosen
                observations[prompt['id']][control]=observation
            alternatives[prompt['id']]=alternative
        write(output/'alternative-selections.json',alternatives)
        write(output/'selector-observations.json',observations)
    # Scorer input contains generated candidates and visible verification only.
    write(output/'selections.json',result['records'])
    report=seal(dict(status='PASS' if evidence.clock_summary['cost_complete'] else 'PARTIAL',availability='available',
        checkpoint=checkpoint['sha256'],study=study,source=source,counts=counts,policy=DRAIN,
        primary_attempts=attempts(DRAIN),raw_rescue_attempts=1,bank=bank['identity'] if bank else None,
        scientific_status='UNMEASURED',clock=evidence.clock_summary['identity']))
    write(output/'evaluation.json',report);return report
