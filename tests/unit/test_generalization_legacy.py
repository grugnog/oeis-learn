import copy,json
import pytest
from oeis_learn.pipeline_validation.identity import seal,digest


def records():
    from oeis_learn.pipeline_validation.followup_training import profile
    recipe=dict(version='legacy-test/v1',track=dict(kind='generic_only',assisted_manifest=None,foundation=None),learning=dict(replay=False,target_view='literal'),model=dict(precision='float32'))
    prepared=seal(dict(recipe=digest(recipe)))
    history=[dict(tokens=i*10,updates=i,splits={name:dict(complete=True,loss_sum=loss,tokens=10) for name in ('parameter_development','structural_development')}) for i,loss in enumerate((10.,2.,3.))]
    training=seal(dict(profile=profile(),work=prepared['identity'],source=digest('source'),tokens=20,updates=2,best_weights=digest('best'),latest_weights=digest('latest'),resume=False,final_access='SEALED'))
    stage=seal(dict(version='pipeline-stage/v1',state='PASS',stage='followup.uniform',binding=dict(source=training['source']),observations=dict(result=dict(result=dict(source=training['source'],work=training['work'],tokens=20,updates=2,status='COMPLETE',training=True,resume=False)))))
    return training,prepared,recipe,history,stage


def test_legacy_best_exposure_is_derived_from_original_pooled_loss():
    from oeis_learn.generalization.legacy import audit_uniform
    got=audit_uniform(*records())
    assert got['best']==dict(tokens=10,updates=1)
    assert got['latest']==dict(tokens=20,updates=2)
    assert got['selection']=='historical_continuous_pooled_nonpadding_dev_loss'


@pytest.mark.parametrize('change',['corpus','assisted','resume','stage','history'])
def test_legacy_rejects_mixed_or_ineligible_provenance(change):
    from oeis_learn.generalization.legacy import audit_uniform
    t,p,r,h,s=records()
    if change=='corpus':t=seal(dict(t,work=digest('other')))
    if change=='assisted':r['track']['kind']='assisted';p=seal(dict(p,recipe=digest(r)));t=seal(dict(t,work=p['identity']))
    if change=='resume':t=seal(dict(t,resume=True))
    if change=='stage':s=seal(dict(s,state='REUSED'))
    if change=='history':h[1]['splits']['structural_development']['complete']=False
    with pytest.raises(ValueError):audit_uniform(t,p,r,h,s)
