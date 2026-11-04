"""Recipient-independent bank authority and a shared visible-only portfolio cap."""
from oeis_learn.pipeline_validation.identity import seal, number


def make_bank(model, seeds, generate, *, elapsed_seconds):
    if len(seeds)!=8 or len(set(seeds))!=8 or any(type(s) is not int or not 0<=s<2**32 for s in seeds):
        raise ValueError('eight frozen generation seeds required')
    number(elapsed_seconds)
    proposals=[]
    for ordinal, seed in enumerate(seeds):
        try:
            source=generate(model,seed,'absent-memory')
            if not isinstance(source,str): raise ValueError('invalid source result')
            proposals.append(dict(ordinal=ordinal,seed=seed,source=source,status='generated'))
        except (ValueError,RuntimeError) as exc:
            proposals.append(dict(ordinal=ordinal,seed=seed,source=None,status='generation_failed',reason=type(exc).__name__))
    return seal(dict(version='generalization-bank/v1',model=model,intervention='absent-memory',seeds=list(seeds),proposals=proposals,attempts=8,generation_seconds=elapsed_seconds))


def bank_cost(bank, recipients):
    if type(recipients) is not int or recipients<1 or seal(bank)!=bank:
        raise ValueError('invalid bank reuse')
    return dict(cold_seconds=bank['generation_seconds'],amortized_seconds=bank['generation_seconds']/recipients)


def portfolio(draws, *, attempt_cap=8, seconds_cap=60):
    if type(attempt_cap) is not int or not 1<=attempt_cap<=8 or not 0<seconds_cap<=60:
        raise ValueError('invalid portfolio caps')
    elapsed=0
    for draw in draws:
        if set(draw)-{'source','seconds','control','status','visible_score'} or draw.get('control') not in {'raw','rescue','correct','shuffled','absent','bank'}:
            raise ValueError('visible-only portfolio fields required')
        elapsed+=number(draw['seconds'])
    if len(draws)>attempt_cap or elapsed>seconds_cap:
        raise ValueError('aggregate portfolio exhausted')
    return dict(charged_attempts=len(draws),seconds=elapsed,remaining_attempts=attempt_cap-len(draws),remaining_seconds=seconds_cap-elapsed)


def apply_bank(bank, visible, fit):
    """Fit identical ordered sources to the recipient's own visible prefix only."""
    from .contracts import shape
    shape('visible-prompt',visible)
    if seal(bank)!=bank: raise ValueError('bank identity mismatch')
    return tuple(fit(p['source'], tuple(visible['indices']), tuple(visible['values'])) if p['source'] is not None else None for p in bank['proposals'])
