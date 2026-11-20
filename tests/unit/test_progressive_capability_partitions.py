"""Pure reconciliation fixtures do not qualify hardware or prove model progress."""
import copy
import pytest

from oeis_learn.generalization.progressive import readiness
from oeis_learn.generalization.progressive.contracts import atomic_write,digest,reference,seal
from oeis_learn.generalization.progressive.coverage import audit
from tests.unit.test_progressive_cohorts import fixture


def inputs(root):
    cohort,exposure=fixture(root,role='development',counts=(2,2,2,2),panel='development')
    cohort=seal(dict(cohort,payload=dict(cohort['payload'],required_per_stratum=2,minimum_groups=2)))
    coverage=[audit(root,row['id'],None,created_at=cohort['created_at']) for row in cohort['payload']['rows']]
    return cohort,exposure,coverage


def producers(root,cohort,*,parts=1):
    result=[]
    ids=[r['id'] for r in cohort['payload']['rows']]
    for model in ('retained_D','random'):
        for i in range(parts):
            members=ids[i::parts]
            candidates=seal(dict(version='discovery-candidates/v1',model_control=model,
                model=digest(model),checkpoint=digest(model+'checkpoint'),recipe=digest('recipe'),
                methods=['correct','shuffled','absent','template','polynomial','linear_recurrence'],
                records={key:dict(costs={'measured_us':10}) for key in members},elapsed_us=10*len(members)))
            path=root/('parts'+str(parts))/(model+str(i)+'-candidates.json');atomic_write(path,candidates)
            cref=reference(root,path)
            scores=seal(dict(version='discovery-scoring/v1',candidates=cref['sha256'],outcomes={
                key:{m:dict(success=True,complete=True,outcome='success') for m in
                    ('raw','rescue','equal_budget','shuffled','absent','template','polynomial','linear_recurrence')}
                for key in members}))
            path=root/('parts'+str(parts))/(model+str(i)+'-scores.json');atomic_write(path,scores)
            result.append(dict(candidates=cref,scores=reference(root,path)))
    return result


def test_partitioned_originals_reconcile_without_reducing_panel_denominators(tmp_path):
    c,e,coverage=inputs(tmp_path)
    one=readiness.build(tmp_path,c,e,coverage,producers(tmp_path,c))
    parts=producers(tmp_path,c,parts=2)
    split=readiness.build(tmp_path,c,e,coverage,parts)
    assert one['status']==split['status']=='COMPLETE'
    assert one['strata']==split['strata']
    for model in split['costs']:
        assert one['costs'][model]['total_elapsed_us']==split['costs'][model]['total_elapsed_us']
        assert len(split['costs'][model]['producers'])==2
    short=readiness.build(tmp_path,c,e,coverage,[parts[0],*parts[2:]])
    assert short['status']=='INCOMPLETE'
    assert short['missing']==['capability_member_inventory:retained_D']
    for stratum in short['strata'].values():
        assert stratum['methods']['retained_D:equal_budget']['yield_fraction']==.5
        assert stratum['methods']['retained_D:equal_budget']['expected_tasks']==2


def test_duplicate_or_foreign_model_partition_is_rejected(tmp_path):
    from oeis_learn.generalization.progressive.contracts import load
    c,e,coverage=inputs(tmp_path)
    parts=producers(tmp_path,c,parts=2)
    with pytest.raises(ValueError,match='duplicate partitioned'):
        readiness.build(tmp_path,c,e,coverage,[*parts,parts[0]])
    bad=copy.deepcopy(parts)
    original=load(tmp_path/parts[1]['candidates']['path'])
    path=tmp_path/'foreign-candidates.json'
    atomic_write(path,seal(dict(original,checkpoint=digest('foreign-checkpoint'))))
    bad[1]['candidates']=reference(tmp_path,path)
    original=load(tmp_path/parts[1]['scores']['path'])
    path=tmp_path/'foreign-scores.json'
    atomic_write(path,seal(dict(original,candidates=bad[1]['candidates']['sha256'])))
    bad[1]['scores']=reference(tmp_path,path)
    with pytest.raises(ValueError,match='model scope'):
        readiness.build(tmp_path,c,e,coverage,bad)
