import pytest
from oeis_learn.generalization.proposals import make_bank, portfolio, bank_cost
from oeis_learn.generalization.diagnostics import signature, join_attempts

def test_bank_independence_and_every_draw_charged():
    calls=[]
    def generate(model, seed, intervention):
        calls.append((model,seed,intervention)); return 'bad syntax'
    bank=make_bank('weights', tuple(range(8)), generate, elapsed_seconds=4)
    assert len(calls)==8 and bank['attempts']==8
    assert 'prompt' not in bank and 'recipient' not in bank
    assert bank_cost(bank, 2) == {'cold_seconds':4, 'amortized_seconds':2}
    draws=[dict(source='bad', seconds=1, control='raw')]*4+[dict(source='bad', seconds=1, control='rescue')]*4
    assert portfolio(draws)['charged_attempts']==8
    with pytest.raises(ValueError): portfolio(draws+[draws[0]])
    with pytest.raises(ValueError): portfolio([dict(source='x',seconds=61,control='raw')])

def test_failed_signatures_and_joins_survive():
    assert signature('bad')['parse_status']=='invalid'
    s=signature('(program (return (add $index (param.get p0))))')
    assert s['parameters']==1
    joined=join_attempts([{'id':'a','source':'bad'}, {'id':'b','source':'bad'}], {'a':{'status':'rejected','reason':'parse'}})
    assert len(joined)==2 and joined[1]['fit']['status']=='unavailable'
    assert joined[0]['duplicate_group']==joined[1]['duplicate_group']
