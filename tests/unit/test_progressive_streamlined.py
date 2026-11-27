"""Prospective token-preserving scheduling; old studies stay reproducible."""
import copy
import pytest
from oeis_learn.generalization.progressive.contracts import seal
from oeis_learn.generalization.progressive.research_roster import construct
from oeis_learn.generalization.progressive.research_budget import reservation
from oeis_learn.pipeline_validation.progressive_research_design import scientific_round
from tests.unit.test_progressive_research_roster import fixture


def roster():
    _,panels,screen=fixture()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    return construct(scientific_round(data,expanded=True,streamlined=True),panels,screen)


def test_linked_confirmations_preserve_tokens_and_fit_original_total():
    r=roster(); budget=reservation(r)
    assert budget['status']=='READY' and budget['training_seconds']==536*3600
    assert budget['total_seconds']<budget['selected_maximum_seconds']
    byid={t['id']:t for t in r['tasks']}
    for position in range(4):
        for seed in (17,19):
            end=byid[f'confirmation-{position}-{seed}'];first=byid[end['parent']]
            assert end['token_target']==25000000 and first['token_target']==12500000
            assert first['parent'] is None and first['seed']==end['seed']
            assert first['domain']==end['domain'] and first['horizon']==end['horizon']==400000000
            assert first['seconds']==end['seconds']==21*3600
    assert all(t['seconds']<=86400 for t in r['tasks'] if t['operation']=='train')


def test_only_sizing_intermediate_evaluations_are_removed():
    r=roster()
    for ordinal in range(16):
        selected=[t for t in r['tasks'] if t['trial'] and t['trial']['ordinal']==ordinal and t['trial']['phase'] not in {'confirmation','selection'}]
        assert len(selected)==(18 if ordinal<12 else 10)
        scoring=[t for t in selected if t['trial']['phase']=='scoring']
        assert {t['trial']['coordinate'] for t in scoring}==({1000000,2000000,4000000} if ordinal<12 else {4000000})
        training=[t for t in selected if t['operation']=='train']
        assert [t['token_target'] for t in training]==[1000000,2000000,4000000]
        assert all(t['seconds']==(2 if ordinal<12 else 4)*3600 for t in training)
    assert len(r['study_evaluations'])==40
    assert len(r['tasks'])==352


def test_old_design_and_roster_unchanged():
    _,panels,screen=fixture()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    old=construct(scientific_round(data,expanded=True),panels,screen)
    assert len(old['tasks'])==376 and reservation(old)['training_seconds']==480*3600
    with pytest.raises(ValueError):scientific_round(data,streamlined=True)
    broken=copy.deepcopy(roster());broken['tasks'][0]['seconds']+=1
    # Changing a train allocation is independently rejected on reservation replay.
    t=next(t for t in broken['tasks'] if t['operation']=='train');t['seconds']+=1
    with pytest.raises(ValueError):reservation(seal(broken))


def test_round014_slowest_confirmation_fits_each_segment_with_fourfold_margin(tmp_path):
    from tests.unit.test_progressive_training_timing import package_fixture
    from oeis_learn.pipeline_validation.progressive_runtime_timing import case_package,feasibility_scopes
    from oeis_learn.pipeline_validation.progressive_trajectory import estimate
    p=case_package(package_fixture(tmp_path),next(c for c in feasibility_scopes() if c['id']=='larger-b16-static'))
    p['update_policy']['schedule_horizon_tokens']=400000000
    def row(shape,batch,arm,tokens,elapsed,fixed,checkpoint):
        return dict(case=dict(id=f'{shape}-b{batch}-{arm}',kind='learner',shape=shape,batch=batch,arm=arm),
            measured_tokens=tokens,measured_steady_us=elapsed,fixed_us=fixed,checkpoint_us=checkpoint)
    # Original round014 measurements, recorded here without private training rows.
    rows=[row('larger',16,'static',115309,104835822.789,52512832,22933965),
          row('reference',16,'static',115309,51147543.187,18275719,4370681),
          row('reference',16,'refreshed',116974,59420883.56,67199597,28200562)]
    producer=dict(measurements=dict(update_us=1693719,checkpoint_us=158982258))
    context=dict(trajectory=dict(cases=rows),baseline={('larger',16,400000000):1693719/1.0788630693077506})
    old=estimate(p,producer,context,delta=25000000,target=25000000,available=84599000000,arm='refreshed')
    assert old['estimated_work_us']>old['available_work_us']
    for target in (12500000,25000000):
        split=estimate(p,producer,context,delta=12500000,target=target,available=73799000000,arm='refreshed')
        assert split['estimated_work_us']<split['available_work_us']
        assert split['estimated_work_us']>4*split['expected_work_us']
