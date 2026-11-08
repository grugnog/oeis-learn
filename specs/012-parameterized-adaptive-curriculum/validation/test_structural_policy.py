"""Independent structural identity/quota counterexamples; no native PASS claim."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).parent
PRIOR=HERE.parents[1]/'011-controlled-generalization/validation'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture(scope='module')
def policy():return module(HERE/'structural_policy.py','pac_structural')

@pytest.fixture(scope='module')
def design(policy):
    return policy.design(json.loads((PRIOR/'observable-evaluation-protocol.json').read_text()))


def test_original_draws_are_identical_and_old_scope_stays_256(policy,design):
    old=module(PRIOR/'prospective_evaluation_policy.py','prior_structural')
    prior=json.loads((PRIOR/'observable-evaluation-protocol.json').read_text())
    original=old.design(prior)
    assert sum(c['count'] for c in original['cells'])==256
    assert policy.draw(design)==[r for r in old.draw(original) if r['holdout']=='structural']
    assert len(policy.draw(design))==2288
    assert design['parameter']=='DEFERRED' and design['assessment']=='EXCLUSION_ONLY'
    assert len(design['controls'])==7 and design['bank_seeds']==list(range(20261101,20261109))
    for key,value in [('seed',42),('identity_capacity',4096),('inference',dict(design['inference'],seconds=61))]:
        changed=copy.deepcopy(design);changed[key]=value
        from oeis_learn.pipeline_validation.identity import seal
        with pytest.raises(ValueError):policy.draw(seal(changed))


def admitted_fixture(policy,design):
    draws=policy.draw(design);selected=[]
    for cell in design['cells']:
        selected.extend([r for r in draws if r['family']==cell['family']][:cell['count']])
    # Explicit fake execution inputs test pure comparison only, never hardware proof.
    results={r['id']:dict(reference=[str(i) for i in range(100)],native=[str(i) for i in range(100)]) for r in selected}
    return draws,selected,results


def test_all_16_exclusions_and_native_agreement_are_mandatory(policy,design):
    draws,selected,results=admitted_fixture(policy,design)
    receipts=policy.original_cells()
    assert len(receipts)==16
    got=policy.admit(design,draws,selected,results,exclusion_cells=receipts,rejected_ids=set())
    assert len(got)==128
    for missing in receipts:
        with pytest.raises(ValueError,match='16 original'):
            policy.admit(design,draws,selected,results,exclusion_cells=[r for r in receipts if r!=missing],rejected_ids=set())
    changed=copy.deepcopy(results);changed[selected[0]['id']]['native'][50]='-1'
    with pytest.raises(ValueError,match='native'):
        policy.admit(design,draws,selected,changed,exclusion_cells=receipts,rejected_ids=set())
    changed=copy.deepcopy(draws);changed[0]['id']='redrawn'
    with pytest.raises(ValueError,match='draw'):
        policy.admit(design,changed,selected,results,exclusion_cells=receipts,rejected_ids=set())


def test_completed_exclusion_shortage_is_not_timeout_or_retry(policy,design):
    draws,selected,results=admitted_fixture(policy,design)
    with pytest.raises(policy.QuotaDeficit) as error:
        policy.admit(design,draws,selected,results,exclusion_cells=policy.original_cells(),rejected_ids={selected[0]['id']})
    assert not isinstance(error.value,TimeoutError)
    assert error.value.retryable is False
    assert sum(error.value.deficits.values())==1
