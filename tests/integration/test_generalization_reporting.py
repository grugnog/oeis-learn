import copy
import pytest


def test_paired_coverage_and_actual_exposure_have_no_private_source():
    from oeis_learn.pipeline_validation.generalization_reporting import pac_representation_report
    inventory={'rows':[dict(id='a',family='f',target_stratum='supported'),dict(id='b',family='f',target_stratum='fallback')]}
    literal=[dict(id='a',indices=['0'],values=['1'],source='private literal',tokens=[3,2]),dict(id='b',indices=['0'],values=['2'],source='private guard',tokens=[4,2])]
    parameterized=copy.deepcopy(literal);parameterized[0]['tokens']=[5,2];parameterized[0]['source']='private parameterized'
    report=pac_representation_report(inventory,literal,parameterized,dict(updates=4,tokens=16,learner_seconds=9))
    assert report['families']['f']==dict(rows=2,supported=1,fallback=1,supported_fraction='1/2')
    assert report['exposure']['examples'] is None and report['exposure']['updates']==4
    assert 'private' not in str(report) and 'equal_data' not in report
    parameterized.pop()
    with pytest.raises(ValueError):pac_representation_report(inventory,literal,parameterized,{})


def test_fitter_parity_requires_identical_limits_and_availability():
    from oeis_learn.pipeline_validation.generalization_scoring import pac_fitter_parity
    recipe={'fitting':{'attempts':8,'parameters':8},'execution':{'integer':'exact'}}
    assert pac_fitter_parity({'B':recipe,'C':copy.deepcopy(recipe)})['matched']
    changed=copy.deepcopy(recipe);changed['fitting']['parameters']=9
    with pytest.raises(ValueError):pac_fitter_parity({'B':recipe,'C':changed})
